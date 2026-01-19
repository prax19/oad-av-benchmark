from torch.utils.data import Dataset
from pathlib import Path
import json

from abc import abstractmethod, ABC

from extraction.utils.json_filtering import extract_labels_per_video

class ExtractionDataset(ABC, Dataset):
    """
    An example dataset class for video extraction. This class should just output the paths to video files.
    """

    @abstractmethod
    def __init__(self, root: str, dataset_config: str): pass

    @abstractmethod
    def __len__(self): pass

    @abstractmethod
    def __getitem__(self, idx: int): pass

    @abstractmethod
    def get_extraction_directory(self, backbone: str) -> Path: pass

class RoadExtractionDataset(Dataset):

    def __init__(self, root: str, dataset_config: str):
        self.root = Path(root)
        self.cfg = Path(root, dataset_config)
        if (not self.root.exists()) or (not self.cfg.exists()):
            raise FileNotFoundError('Invalid dataset path.')

        self.video_index = list(self.root.glob('videos/*.mp4'))

    def __len__(self):
        return len(self.video_index)

    def __getitem__(self, idx: int):
        video_meta = extract_labels_per_video(dataset_cfg=self.cfg, video_id=self.video_index[idx].stem)
        return Path(self.video_index[idx]), video_meta
    
    def get_extraction_directory(self, backbone: str) -> Path:
        out_dir = Path(self.root, f"features-{backbone}/features")
        out_dir.mkdir(parents=True, exist_ok=True)
        return Path(out_dir)