import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.adapters import OADMethodAdapter
from core.common import setup_dataset
from core.evaluation import evaluate_model

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


def estimate_pos_weight(
    loader: DataLoader,
    device,
    max_pos_weight: float = 8.0,
    min_pos_weight: float = 0.5,
    power: float = 0.5,
) -> torch.Tensor | None:
    """Estimate class-wise pos_weight from annotated train frames with smoothing."""
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
            frame_loss = loss_raw.mean(dim=-1)
            mask = ann.to(frame_loss.dtype)
            denom = mask.sum().clamp_min(1.0)
            loss = (frame_loss * mask).sum() / denom

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        loss_val = float(loss.detach().item())
        running += loss_val
        n += 1

        batch_pbar.set_postfix(batch_loss=f"{loss_val:.4f}", running_loss=f"{running / n:.4f}")

    return running / max(1, n)


def train_model(
    adapter: OADMethodAdapter,
    cfg: dict,
    device=get_device(),
    epochs=5,
    batch_size: int = 16,
    split_type: str = "train",
    split_variant: int = 1,
    dataset_root: str = "data/road",
    dataset_variant: str = "features-tsn-kinetics-400",
    shuffle: bool = True,
    max_grad_norm: float | None = 1.0,
    lr=5e-5,
    wd=1e-3,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    use_amp: bool = True,
):
    # TODO: make it parametrizable
    train_loader, train_dataset = setup_dataset(
        batch_size=batch_size,
        split_type=split_type,
        split_variant=split_variant,
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    val_loader, _ = setup_dataset(
        batch_size=batch_size,
        split_type="val",
        split_variant=split_variant,
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    model = adapter.build_model(
        cfg=cfg,
        num_classes=train_dataset[0][1].shape[-1],
        device=device,
    )

    _initialize_adapter_head_if_needed(model=model, adapter=adapter, loader=train_loader, device=device)

    if is_cuda_device(device):
        torch.backends.cudnn.benchmark = True

    _set_finetune_mode(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after freezing backbone.")

    amp_enabled = bool(use_amp and (str(device).startswith("cuda") or str(device).startswith("xpu")))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and is_cuda_device(device))

    # TODO: make it parametrizable
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    pos_weight = estimate_pos_weight(loader=train_loader, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    if hasattr(model, "_adapter_head"):
        print("[train] optimizer includes _adapter_head parameters")

    epoch_pbar = tqdm(
        range(1, epochs + 1),
        desc=f"train epochs | adapter={adapter.name}",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    prev_loss = None
    best_map = float("-inf")
    best_epoch = 0

    for epoch in epoch_pbar:
        epoch_pbar.set_postfix(epoch=f"{epoch}/{epochs}", prev_loss=f"{prev_loss:.4f}" if prev_loss is not None else "n/a")

        avg_loss = train_epoch(
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
        )

        prev_loss = avg_loss
        epoch_pbar.set_postfix(epoch=f"{epoch}/{epochs}", train_loss=f"{avg_loss:.4f}")

        metrics = evaluate_model(
            adapter=adapter,
            model=model,
            loader=val_loader,
        )

        current_map = float(metrics.get("map_macro", 0.0))
        if current_map > best_map:
            best_map = current_map
            best_epoch = epoch

        metrics["best_map_macro"] = best_map
        metrics["best_epoch"] = best_epoch

        print(metrics)


from core.adapters import *


def main():
    # adapter = MiniROADAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    # adapter = TeSTrAAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    # adapter = MATAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    # adapter = CMeRTAdapter()
    # cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    # train_model(adapter=adapter, cfg=cfg, epochs=1)

    adapter = TeSTrAAdapter()
    cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    train_model(adapter=adapter, cfg=cfg, epochs=20, batch_size=32, num_workers=12, prefetch_factor = 8, lr=1e-4, wd=1e-3)

    adapter = TeSTrAAdapter()
    cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    train_model(adapter=adapter, cfg=cfg, epochs=20, batch_size=32, num_workers=12, prefetch_factor = 8, lr=5e-5, wd=1e-3)


if __name__ == "__main__":
    main()