import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.datasets import PreExtractedDataset
from core.adapters import OADMethodAdapter
from utils.torch_scripts import get_device


def setup_dataset():
    dataset = PreExtractedDataset(
        dataset_root="data/road",
        dataset_variant="features-tsn-kinetics-400",
        split_variant=2,
        split_type="train",
        cache_mmap="r",
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )

    return loader, dataset


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
):
    # TODO: make it parametrizable
    loader, dataset = setup_dataset()

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
from core.adapters import MiniROADAdapter

def main():
    adapter = MiniROADAdapter()
    data_info = adapter.method_root / "data_info" / "video_list.json"
    cfg = adapter.get_cfg(
        Path("methods", "MiniROAD", "configs", "miniroad_thumos_kinetics.yaml"),
        opts=["DATA.DATA_INFO", str(data_info)],
    )
    train_model(adapter=adapter, cfg=cfg)

if __name__ == "__main__":
    main()
