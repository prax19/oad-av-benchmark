import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.adapters import OADMethodAdapter
from core.common import setup_dataset
from core.evaluation import evaluate_model
from core.logger import Logger

from utils.torch_scripts import get_device, autocast_for, is_cuda_device


def _initialize_adapter_head_if_needed(model, adapter: OADMethodAdapter, loader: DataLoader, device) -> None:
    """
    Ensure lazy adapter layers (e.g. model._adapter_head) are created before optimizer init.
    Otherwise such parameters are excluded from optimizer.param_groups and never trained.
    """
    try:
        x, y, _ = next(iter(loader))
    except StopIteration:
        return

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.no_grad():
        logits = adapter.forward_logits(model, x, y, device)
        _ = adapter.normalize_logits(logits, y, model, device)


def _cfg_get(cfg, path: str, default=None):
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(part)
        else:
            node = getattr(node, part, None)
    return default if node is None else node


def _extract_model_timing(cfg) -> dict[str, float | int | None]:
    return {
        "fps": _cfg_get(cfg, "DATA.FPS"),
        "long_memory_seconds": _cfg_get(cfg, "MODEL.LSTR.LONG_MEMORY_SECONDS"),
        "long_memory_sample_rate": _cfg_get(cfg, "MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE"),
        "work_memory_seconds": _cfg_get(cfg, "MODEL.LSTR.WORK_MEMORY_SECONDS"),
        "work_memory_sample_rate": _cfg_get(cfg, "MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE"),
    }


def _collect_class_stats(loader: DataLoader, device) -> tuple[torch.Tensor | None, float]:
    pos = None
    total = 0.0

    for _, y, ann in loader:
        y = y.to(device, non_blocking=True)
        ann = ann.to(device, non_blocking=True).unsqueeze(-1)

        if pos is None:
            pos = torch.zeros((y.shape[-1],), dtype=torch.float32, device=device)

        y_sel = y * ann
        pos += y_sel.sum(dim=(0, 1))
        total += float(ann.sum().item())

    return pos, total


def estimate_pos_weight(
    loader: DataLoader,
    device,
    max_pos_weight: float = 24.0,
    min_pos_weight: float = 1.0,
    power: float = 0.75,
) -> torch.Tensor | None:
    """Estimate class-wise pos_weight from annotated train frames."""
    pos, total = _collect_class_stats(loader=loader, device=device)

    if pos is None or total <= 0:
        return None

    eps = 1.0
    neg = total - pos
    raw = (neg + eps) / (pos + eps)
    pw = torch.pow(raw, power)
    pw = torch.clamp(pw, min=min_pos_weight, max=max_pos_weight)

    print(f"[train] pos_count={pos.detach().cpu().tolist()}")
    print(f"[train] raw_pos_weight={raw.detach().cpu().tolist()}")
    print(f"[train] used_pos_weight={pw.detach().cpu().tolist()}")
    return pw


def estimate_class_weight(
    loader: DataLoader,
    device,
    max_class_weight: float = 8.0,
    min_class_weight: float = 0.25,
    power: float = 1.0,
) -> torch.Tensor | None:
    """Estimate per-class multipliers using effective-number reweighting."""
    pos, total = _collect_class_stats(loader=loader, device=device)

    if pos is None or total <= 0:
        return None

    beta = 0.999
    effective_num = 1.0 - torch.pow(torch.full_like(pos, beta), pos.clamp_min(1.0))
    cw = (1.0 - beta) / effective_num
    cw = torch.pow(cw, power)
    cw = cw / cw.mean().clamp_min(1e-6)
    cw = torch.clamp(cw, min=min_class_weight, max=max_class_weight)

    print(f"[train] class_weight={cw.detach().cpu().tolist()}")
    return cw


def estimate_positive_class_weight(
    loader: DataLoader,
    device,
    max_class_weight: float = 20.0,
    min_class_weight: float = 0.2,
    power: float = 1.0,
) -> torch.Tensor | None:
    """Estimate positive-label multipliers to favor rare classes."""
    pos, total = _collect_class_stats(loader=loader, device=device)

    if pos is None or total <= 0:
        return None

    eps = 1.0
    inv = (total + eps) / (pos + eps)
    pos_w = torch.pow(inv, power)
    pos_w = pos_w / pos_w.mean().clamp_min(1e-6)
    pos_w = torch.clamp(pos_w, min=min_class_weight, max=max_class_weight)

    print(f"[train] positive_class_weight={pos_w.detach().cpu().tolist()}")
    return pos_w


def estimate_negative_class_weight(
    loader: DataLoader,
    device,
    min_class_weight: float = 0.02,
    max_class_weight: float = 1.0,
    power: float = 0.5,
) -> torch.Tensor | None:
    """
    Downweight negative terms for ultra-rare classes.
    For rare class prior p, weight ~ p^power -> tiny negatives, preventing collapse-to-all-negative.
    """
    pos, total = _collect_class_stats(loader=loader, device=device)

    if pos is None or total <= 0:
        return None

    eps = 1.0
    prior = (pos + eps) / (total + eps)
    neg_w = torch.pow(prior, power)
    neg_w = torch.clamp(neg_w, min=min_class_weight, max=max_class_weight)

    print(f"[train] negative_class_weight={neg_w.detach().cpu().tolist()}")
    return neg_w


def _set_finetune_mode(model: torch.nn.Module) -> None:
    """Permanently freeze backbone; train classification heads only."""
    for name, param in model.named_parameters():
        is_head = ("classifier" in name) or ("_adapter_head" in name)
        param.requires_grad = is_head


def train_epoch(
    model,
    adapter: OADMethodAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device,
    epoch: int,
    epochs: int,
    max_grad_norm: float | None = None,
    use_amp: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    class_weight: torch.Tensor | None = None,
    positive_class_weight: torch.Tensor | None = None,
    negative_class_weight: torch.Tensor | None = None,
    focal_gamma_pos: float = 1.0,
    focal_gamma_neg: float = 4.0,
):
    model.train()
    running = 0.0
    n = 0

    batch_pbar = tqdm(
        loader,
        desc=f"epochs {epoch}/{epochs}",
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    for x, y, ann in batch_pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ann = ann.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device=device, enabled=use_amp):
            logits = adapter.forward_logits(model, x, y, device)
            logits = adapter.normalize_logits(logits, y, model, device)

            loss_raw = criterion(logits, y)

            # Asymmetric focal modulation: suppress easy negatives, keep positives stronger.
            probs = torch.sigmoid(logits)
            pt = torch.where(y > 0.5, probs, 1.0 - probs)
            gamma = torch.where(
                y > 0.5,
                torch.full_like(pt, focal_gamma_pos),
                torch.full_like(pt, focal_gamma_neg),
            )
            focal_factor = torch.pow((1.0 - pt).clamp_min(1e-6), gamma)
            loss_raw = loss_raw * focal_factor

            if class_weight is not None:
                loss_raw = loss_raw * class_weight.view(1, 1, -1)

            pos_mask = (y > 0.5).to(loss_raw.dtype)
            neg_mask = 1.0 - pos_mask

            if positive_class_weight is not None:
                pos_scale = positive_class_weight.view(1, 1, -1)
                loss_raw = loss_raw * (1.0 + pos_mask * (pos_scale - 1.0))

            if negative_class_weight is not None:
                neg_scale = negative_class_weight.view(1, 1, -1)
                loss_raw = (loss_raw * pos_mask) + (loss_raw * neg_mask * neg_scale)

            frame_loss = loss_raw.mean(dim=-1)
            mask = ann.to(frame_loss.dtype)
            denom = mask.sum().clamp_min(1.0)
            loss = (frame_loss * mask).sum() / denom

        optimizer_step_done = False
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            # If scale decreased, optimizer.step() was skipped due to inf/nan gradients.
            optimizer_step_done = new_scale >= prev_scale
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer_step_done = True

        if scheduler is not None and optimizer_step_done:
            scheduler.step()

        loss_val = float(loss.detach().item())
        running += loss_val
        n += 1

        # batch tqdm: only current loss
        batch_pbar.set_postfix(loss=f"{loss_val:.4f}")

    return running / max(1, n), n


def train_model(
    adapter: OADMethodAdapter,
    cfg: dict,
    name: str | None = None,
    device=get_device(),
    epochs=5,
    batch_size: int = 16,
    split_variant: int = 1,
    dataset_root: str = "data/road",
    dataset_variant: str = "features-tsn-kinetics-400-4hz",
    dataset_long: int = 512,
    dataset_work: int = 8,
    dataset_stride: int = 4,
    shuffle: bool = True,
    max_grad_norm: float | None = 1.0,
    lr=5e-5,
    wd=1e-3,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    use_amp: bool = True,
    min_lr_ratio: float = 0.1,
    pos_weight_power: float = 0.75,
    pos_weight_min: float = 1.0,
    pos_weight_max: float = 24.0,
    class_weight_power: float = 1.0,
    class_weight_min: float = 0.25,
    class_weight_max: float = 8.0,
    positive_class_weight_power: float = 1.0,
    positive_class_weight_min: float = 0.2,
    positive_class_weight_max: float = 20.0,
    negative_class_weight_power: float = 0.5,
    negative_class_weight_min: float = 0.02,
    negative_class_weight_max: float = 1.0,
    focal_gamma_pos: float = 1.0,
    focal_gamma_neg: float = 4.0,
    use_weighted_sampler: bool = True,
    weighted_sampler_rarity_power: float = 1.2,
    weighted_sampler_min_weight: float = 1.0,
    weighted_sampler_max_weight: float = 30.0,
    logger: Logger | None = None,
):
    if dataset_long < 1 or dataset_work < 1 or dataset_stride < 1:
        raise ValueError(
            "Invalid dataset window params: "
            f"dataset_long={dataset_long}, dataset_work={dataset_work}, dataset_stride={dataset_stride}. "
            "All must be >= 1."
        )

    # If adapter exposes recommended windowing derived from model temporal config,
    # use it when caller left defaults unchanged.
    default_window = (dataset_long, dataset_work, dataset_stride) == (512, 8, 4)
    if default_window and hasattr(adapter, "recommend_dataset_window"):
        try:
            rec = adapter.recommend_dataset_window()
        except Exception:
            rec = None
        if isinstance(rec, dict):
            dataset_long = int(rec.get("dataset_long", dataset_long))
            dataset_work = int(rec.get("dataset_work", dataset_work))
            dataset_stride = int(rec.get("dataset_stride", dataset_stride))
            print(
                "[train] using adapter-recommended window params: "
                f"long={dataset_long}, work={dataset_work}, stride={dataset_stride}"
            )

    train_loader, train_dataset = setup_dataset(
        batch_size=batch_size,
        split_type="train",
        split_variant=split_variant,
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        long=dataset_long,
        work=dataset_work,
        stride=dataset_stride,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        use_weighted_sampler=use_weighted_sampler,
        weighted_sampler_rarity_power=weighted_sampler_rarity_power,
        weighted_sampler_min_weight=weighted_sampler_min_weight,
        weighted_sampler_max_weight=weighted_sampler_max_weight,
    )

    val_loader, val_dataset = setup_dataset(
        batch_size=batch_size,
        split_type="val",
        split_variant=split_variant,
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        long=dataset_long,
        work=dataset_work,
        stride=dataset_stride,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        use_weighted_sampler=False,
    )

    sample_x, sample_y, _ = train_dataset[0]
    if sample_x.ndim < 2 or sample_y.ndim < 2 or sample_x.shape[0] < 1 or sample_y.shape[0] < 1:
        raise ValueError(
            "Invalid sample temporal dimensions from dataset: "
            f"x.shape={tuple(sample_x.shape)}, y.shape={tuple(sample_y.shape)}. "
            "Check dataset_long/dataset_work/dataset_stride and extracted feature files."
        )

    model = adapter.build_model(
        cfg=cfg,
        num_classes=train_dataset[0][1].shape[-1],
        device=device,
    )

    try:
        _initialize_adapter_head_if_needed(model=model, adapter=adapter, loader=train_loader, device=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Adapter/model warmup forward failed. This often means a mismatch between model config and "
            "dataset window sizes. Ensure dataset_work > 0 and that model cfg temporal settings match "
            f"dataset_long={dataset_long}, dataset_work={dataset_work}. Original error: {exc}"
        ) from exc

    if is_cuda_device(device):
        torch.backends.cudnn.benchmark = True

    _set_finetune_mode(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after freezing backbone.")

    amp_enabled = bool(use_amp and (str(device).startswith("cuda") or str(device).startswith("xpu")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and is_cuda_device(device))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    # Compute class-imbalance statistics on the natural train distribution
    # (no weighted sampler, no shuffle) to avoid feeding sampler-induced priors
    # back into class-weight estimation.
    stats_loader, _ = setup_dataset(
        batch_size=batch_size,
        split_type="train",
        split_variant=split_variant,
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        long=dataset_long,
        work=dataset_work,
        stride=dataset_stride,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        use_weighted_sampler=False,
    )

    pos_weight = estimate_pos_weight(
        loader=stats_loader,
        device=device,
        power=pos_weight_power,
        min_pos_weight=pos_weight_min,
        max_pos_weight=pos_weight_max,
    )
    class_weight = estimate_class_weight(
        loader=stats_loader,
        device=device,
        power=class_weight_power,
        min_class_weight=class_weight_min,
        max_class_weight=class_weight_max,
    )
    positive_class_weight = estimate_positive_class_weight(
        loader=stats_loader,
        device=device,
        power=positive_class_weight_power,
        min_class_weight=positive_class_weight_min,
        max_class_weight=positive_class_weight_max,
    )
    negative_class_weight = estimate_negative_class_weight(
        loader=stats_loader,
        device=device,
        power=negative_class_weight_power,
        min_class_weight=negative_class_weight_min,
        max_class_weight=negative_class_weight_max,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    # Hardcoded best direction from quick ablations: cosine scheduler + BCE
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=lr * min_lr_ratio,
    )

    if logger is None:
        run_name = name if name is not None else f"{adapter.name}_"
        ds_info = getattr(train_dataset, "dataset_info", {})
        model_timing = _extract_model_timing(cfg)
        logger = Logger(
            name=run_name,
            metadata={
                "method": adapter.name,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
                "lr_initial": lr,
                "weight_decay": wd,
                "scheduler": scheduler.__class__.__name__,
                "loss": criterion.__class__.__name__,
                "weight_calibration": {
                    "pos_weight": {"power": pos_weight_power, "min": pos_weight_min, "max": pos_weight_max},
                    "class_weight": {"power": class_weight_power, "min": class_weight_min, "max": class_weight_max},
                    "positive_class_weight": {
                        "power": positive_class_weight_power,
                        "min": positive_class_weight_min,
                        "max": positive_class_weight_max,
                    },
                    "negative_class_weight": {
                        "power": negative_class_weight_power,
                        "min": negative_class_weight_min,
                        "max": negative_class_weight_max,
                    },
                    "focal": {"gamma_pos": focal_gamma_pos, "gamma_neg": focal_gamma_neg},
                    "sampler": {
                        "use_weighted_sampler": use_weighted_sampler,
                        "rarity_power": weighted_sampler_rarity_power,
                        "min_weight": weighted_sampler_min_weight,
                        "max_weight": weighted_sampler_max_weight,
                    },
                },
                "dataset": {
                    "name": ds_info.get("name", train_dataset.__class__.__name__),
                    "backbone": ds_info.get("backbone", "unknown"),
                    "backbone_dataset": ds_info.get("backbone_dataset", "unknown"),
                    "hz": ds_info.get("hz", None),
                    "min_lr_ratio": min_lr_ratio,
                    "split_variant": split_variant,
                    "root": dataset_root,
                    "variant": dataset_variant,
                    "class_names": getattr(train_dataset, "class_names", {}),
                    "window": {
                        "dataset_long": dataset_long,
                        "dataset_work": dataset_work,
                        "dataset_stride": dataset_stride,
                    },
                    "model_timing": model_timing,
                    "clips_ignored_train": train_dataset.ignored_clips,
                    "frames_ignored_train": train_dataset.ignored_frames,
                    "clips_ignored_val": val_dataset.ignored_clips,
                    "frames_ignored_val": val_dataset.ignored_frames
                },
            },
        )

    if hasattr(model, "_adapter_head"):
        print("[train] optimizer includes _adapter_head parameters")

    epoch_pbar = tqdm(
        range(1, epochs + 1),
        desc=f"train epochs | adapter={adapter.name}",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    prev_train_loss: float | None = None
    prev_val_loss: float | None = None
    prev_map_macro: float | None = None
    steps = 0
    best_map_macro = float("-inf")
    best_epoch = 0
    final_metrics: dict = {}

    try:
        for epoch in epoch_pbar:
            # epoch tqdm: only previous train loss, val loss and macro mAP
            epoch_pbar.set_postfix(
                prev_train_loss=f"{prev_train_loss:.4f}" if prev_train_loss is not None else "n/a",
                prev_val_loss=f"{prev_val_loss:.4f}" if prev_val_loss is not None else "n/a",
                prev_map_macro=f"{prev_map_macro:.4f}" if prev_map_macro is not None else "n/a",
            )

            avg_train_loss, epoch_steps = train_epoch(
                model=model,
                adapter=adapter,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                epochs=epochs,
                max_grad_norm=max_grad_norm,
                use_amp=amp_enabled,
                scaler=scaler if scaler.is_enabled() else None,
                scheduler=scheduler,
                class_weight=class_weight,
                positive_class_weight=positive_class_weight,
                negative_class_weight=negative_class_weight,
                focal_gamma_pos=focal_gamma_pos,
                focal_gamma_neg=focal_gamma_neg,
            )
            steps += epoch_steps

            metrics = evaluate_model(
                adapter=adapter,
                model=model,
                loader=val_loader,
            )

            avg_val_loss = float(metrics.get("bce_loss", 0.0))
            map_macro = float(metrics.get("map_macro", 0.0))

            metadata = {
                "epoch": epoch,
                "steps": steps,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(avg_train_loss),
                "val_loss": avg_val_loss,
            }
            logger.log_event(metrics, metadata)

            current_map = float(metrics.get("map_macro", 0.0))
            if current_map > best_map_macro:
                best_map_macro = current_map
                best_epoch = epoch

            final_metrics = metrics
            prev_train_loss = float(avg_train_loss)
            prev_val_loss = avg_val_loss
            prev_map_macro = map_macro
    finally:
        logger.finalize()

    return {
        "best_map_macro": best_map_macro if best_map_macro > float("-inf") else 0.0,
        "best_epoch": best_epoch,
        "final_metrics": final_metrics,
    }


from core.adapters import *


def main():
    # adapter = MiniROADAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    # adapter = MATAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    # adapter = CMeRTAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    # Sweep loop prepared for LR/WD/cosine floor experiments.
    ds_variants = ['features-tsn-kinetics-400-8hz']
    lr_values = [1e-4]
    wd_values = [5e-4]
    min_lr_ratio_values = [0.05]

    ds_memory_seconds = [96, 128, 160]
    ds_memory_sample_rate = [12, 16]
    ds_work_seconds = [2, 4, 8]
    ds_work_sample_rate = [1, 2, 4]
    fps = 8

    for long_mem in ds_memory_seconds:
        for long_sample_rate in ds_memory_sample_rate:
            for work_mem in ds_work_seconds:
                for work_sample_rate in ds_work_sample_rate:
                    long_memory_length = long_mem * fps
                    work_memory_length = work_mem * fps
                    if (long_memory_length % long_sample_rate) != 0:
                        print(
                            "[sweep] skip invalid config: "
                            f"LONG_MEMORY_LENGTH={long_memory_length} not divisible by "
                            f"LONG_MEMORY_SAMPLE_RATE={long_sample_rate}"
                        )
                        continue
                    if (work_memory_length % work_sample_rate) != 0:
                        print(
                            "[sweep] skip invalid config: "
                            f"WORK_MEMORY_LENGTH={work_memory_length} not divisible by "
                            f"WORK_MEMORY_SAMPLE_RATE={work_sample_rate}"
                        )
                        continue

                    adapter = TeSTrAAdapter(
                        fps=fps,
                        long_memory_seconds=long_mem,
                        long_memory_sample_rate=long_sample_rate,
                        work_memory_seconds=work_mem,
                        work_memory_sample_rate=work_sample_rate,
                    )
                    cfg = adapter.get_cfg(
                        adapter.default_cfg,
                        opts=[
                            "DATA.DATA_INFO", str(adapter.default_data_info.resolve()),
                        ],
                    )

                    for variant in ds_variants:
                        for lr in lr_values:
                            for wd in wd_values:
                                for min_lr_ratio in min_lr_ratio_values:
                                    train_model(
                                        adapter=adapter,
                                        cfg=cfg,
                                        epochs=20,
                                        batch_size=16,
                                        num_workers=8,
                                        prefetch_factor=4,
                                        lr=lr,
                                        wd=wd,
                                        min_lr_ratio=min_lr_ratio,
                                        dataset_variant=variant,
                                        dataset_root="data/road"
                                    )


if __name__ == "__main__":
    main()
