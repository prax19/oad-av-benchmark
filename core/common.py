from core.datasets import PreExtractedDataset
from torch.utils.data import DataLoader

def setup_dataset(
    batch_size = 32,
    split_type: str = "train",
    split_variant: int = 2,
    dataset_root: str = "data/road",
    dataset_variant: str = "features-tsn-kinetics-400",
    shuffle: bool = True,
):
    dataset = PreExtractedDataset(
        dataset_root=dataset_root,
        dataset_variant=dataset_variant,
        split_variant=split_variant,
        split_type=split_type,
        cache_mmap="r",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=False,
    )

    return loader, dataset