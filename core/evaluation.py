import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score

from core.adapters import OADMethodAdapter
from core.common import setup_dataset
from utils.torch_scripts import get_device


def _flatten_time_batch(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() != 3 or targets.dim() != 3:
        raise ValueError(
            f"Expected [B, T, C] tensors, got logits={tuple(logits.shape)} and targets={tuple(targets.shape)}"
        )
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have identical shapes after adapter normalization, got {tuple(logits.shape)} vs {tuple(targets.shape)}"
        )

    c = logits.shape[-1]
    return logits.reshape(-1, c), targets.reshape(-1, c)


@torch.no_grad()
def evaluate_model(
    adapter: OADMethodAdapter,
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module | None = None,
    device=get_device(),
    threshold=0.3
):

    if model is None:
        pass
        # model = adapter.build_model(
        #     cfg=cfg,
        #     num_classes=dataset[0][1].shape[-1],
        #     device=device,
        # )

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    losses: list[float] = []
    probs_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []

    eval_pbar = tqdm(
        loader,
        desc=f"eval batches | adapter={adapter.name}",
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    for x, y in eval_pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = adapter.forward_logits(model, x, y, device)
        logits = adapter.normalize_logits(logits, y, model, device)

        flat_logits, flat_targets = _flatten_time_batch(logits, y)
        loss = criterion(flat_logits, flat_targets)
        losses.append(float(loss.detach().item()))

        probs_all.append(torch.sigmoid(flat_logits).cpu())
        targets_all.append(flat_targets.cpu())

        eval_pbar.set_postfix(batch_loss=f"{losses[-1]:.4f}", running_loss=f"{sum(losses) / len(losses):.4f}")

    y_prob = torch.cat(probs_all, dim=0).numpy()
    y_true = torch.cat(targets_all, dim=0).numpy()
    y_pred = (y_prob >= threshold).astype(int)

    per_class_ap = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].max() == 0:
            continue
        per_class_ap.append(average_precision_score(y_true[:, c], y_prob[:, c]))

    map_macro = float(sum(per_class_ap) / len(per_class_ap)) if per_class_ap else 0.0

    metrics = {
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "bce_loss": float(sum(losses) / max(1, len(losses))),
        "map_macro": map_macro,
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics
