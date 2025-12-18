from torchvision import datasets
from pathlib import Path

from abc import abstractmethod, ABC

class ExtractionDataset(ABC, datasets.VisionDataset):
    """
    An example dataset class for video extraction. This class should just output the paths to video files.
    """

    @abstractmethod
    def __init__(self, root: str): pass

    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __getitem__(self, idx: int): pass

    @abstractmethod
    def get_extraction_directory(self, backbone: str) -> Path: pass

class RoadExtractionDataset(datasets.VisionDataset):

    def __init__(self, root: str):
        super().__init__(root, transform=None)
        self.root = Path(root)

        self.video_index = list(self.root.glob('videos/*.mp4'))

    def __len__(self):
        return len(self.video_index)

    def __getitem__(self, idx: int):
        return Path(self.video_index[idx])
    
    def get_extraction_directory(self, backbone: str) -> Path:
        out_dir = Path(self.root, f"features-{backbone}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return Path(out_dir)