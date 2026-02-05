import torch
from torch.utils.data import DataLoader

from core.datasets import PreExtractedDataset

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
