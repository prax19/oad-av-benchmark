import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.adapters import OADMethodAdapter
from core.common import setup_dataset

from utils.torch_scripts import get_device


def train_epoch(
    model,
    adapter: OADMethodAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device,
    epoch: int,
    epochs: int,
):
    model.train()
    running = 0.0
    n = 0

    batch_pbar = tqdm(
        loader,
        desc=f"batches {epoch}/{epochs}",
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    for x, y in batch_pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = adapter.forward_logits(model, x, y, device)
        logits = adapter.normalize_logits(logits, y, model, device)

        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        running += loss_val
        n += 1

        batch_pbar.set_postfix(loss=f"{loss_val:.4f}", avg=f"{running / n:.4f}")

    return running / max(1, n)


def train_model(
    adapter: OADMethodAdapter,
    cfg: dict,
    device=get_device(),
    epochs=5,
    batch_size: int = 16,
    split_type: str = "train",
    split_variant: int = 2,
    dataset_root: str = "data/road",
    dataset_variant: str = "features-tsn-kinetics-400",
    shuffle: bool = True,
):
    # TODO: make it parametrizable
    loader, dataset = setup_dataset(
        batch_size=batch_size,
        split_type=split_type,
        split_variant=split_variant,
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        shuffle=shuffle,
    )

    model = adapter.build_model(
        cfg=cfg,
        num_classes=dataset[0][1].shape[-1],
        device=device,
    )

    # TODO: make it parametrizable
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    epoch_pbar = tqdm(
        range(1, epochs + 1),
        desc="epoch",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )

    prev_loss = None
    for epoch in epoch_pbar:
        epoch_pbar.set_postfix(prev=f"{prev_loss:.4f}" if prev_loss is not None else "n/a")

        avg_loss = train_epoch(
            model=model,
            adapter=adapter,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )

        prev_loss = avg_loss
        epoch_pbar.set_description(f"epoch {epoch}/{epochs}")
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")



from pathlib import Path
from core.adapters import *

def main():
    adapter = MiniROADAdapter()
    cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    train_model(adapter=adapter, cfg=cfg, epochs=1)

    adapter = TeSTrAAdapter()
    cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    train_model(adapter=adapter, cfg=cfg, epochs=1)

    adapter = MATAdapter()
    cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    train_model(adapter=adapter, cfg=cfg, epochs=1)

    adapter = CMeRTAdapter()
    cfg = adapter.get_cfg(adapter.default_cfg, opts=["DATA.DATA_INFO", str(adapter.default_data_info)])
    train_model(adapter=adapter, cfg=cfg, epochs=1)

if __name__ == "__main__":
    main()
