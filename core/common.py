from core.datasets import PreExtractedDataset
from torch.utils.data import DataLoader
import torch

def setup_dataset(
    batch_size=32,
    split_type: str = "train",
    split_variant: int = 2,
    dataset_root: str = "data/road",
    dataset_variant: str = "features-tsn-kinetics-400-4hz",
    shuffle: bool = True,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
):
    dataset = PreExtractedDataset(
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        split_variant=split_variant,
        split_type=split_type,
        cache_mmap="r",
    )

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
    )

    return loader, dataset
