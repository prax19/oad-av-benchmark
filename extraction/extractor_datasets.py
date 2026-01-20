from torch.utils.data import Dataset
from pathlib import Path
import json

from extraction.utils.json_filtering import pack_av_cls_from_frames

class RoadExtractionDataset(Dataset):
    def __init__(self, root: str, dataset_config: str):
        self.root = Path(root)
        self.cfg_path = self.root / dataset_config
        if (not self.root.exists()) or (not self.cfg_path.exists()):
            raise FileNotFoundError("Invalid dataset path.")

        with open(self.cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.class_names = cfg["av_action_labels"]
        self.num_classes = len(self.class_names)
        self.db = cfg["db"]

        self.video_index = list((self.root / "videos").glob("*.mp4"))

    def __len__(self):
        return len(self.video_index)

    def __getitem__(self, idx: int):
        vid_path = self.video_index[idx]
        video_id = vid_path.stem

        video_meta = self.db[video_id]
        frames = video_meta["frames"]

        labels = pack_av_cls_from_frames(frames, num_classes=self.num_classes)

        return vid_path, labels
    
    def get_extraction_directory(self, backbone: str) -> Path:
        out_dir = self.root / f"features-{backbone}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
