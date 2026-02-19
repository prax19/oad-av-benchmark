from core.datasets import PreExtractedDataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
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
    use_weighted_sampler: bool = False,
    weighted_sampler_rarity_power: float = 1.0,
    weighted_sampler_min_weight: float = 1.0,
    weighted_sampler_max_weight: float = 25.0,
    long: int = 512,
    work: int = 8,
    stride: int = 4,
):
    if long < 1 or work < 1 or stride < 1:
        raise ValueError(
            f"Invalid window parameters: long={long}, work={work}, stride={stride}. All must be >= 1."
        )

    dataset = PreExtractedDataset(
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        split_variant=split_variant,
        split_type=split_type,
        long=long,
        work=work,
        stride=stride,
        cache_mmap="r",
    )

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset produced zero windows for split='{split_type}', variant='{dataset_variant}'. "
            f"Try lowering long/work or stride (current: long={long}, work={work}, stride={stride})."
        )

    sampler = None
    loader_shuffle = shuffle

    if use_weighted_sampler:
        sample_weights = dataset.build_sample_weights(
            rarity_power=weighted_sampler_rarity_power,
            min_weight=weighted_sampler_min_weight,
            max_weight=weighted_sampler_max_weight,
        )
        if len(sample_weights) == 0:
            raise ValueError("Weighted sampler cannot be built for an empty dataset.")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader_shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
    )

    return loader, dataset
