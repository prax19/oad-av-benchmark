import numpy as np
import torch.utils.data as data
from pathlib import Path
import json
from typing import Literal

SplitStr = Literal["all", "train", "val", "test"]

class PreExtractedROADDataset(data.Dataset):

    def __init__(self, 
        dataset_root: str, 
        dataset_variant: str, 
        split_type: str | SplitStr = 'all',
        split_variant: int = 0
    ):
        self.dataset_root = Path(dataset_root)
        self.sessions_dir = Path(dataset_root, dataset_variant)
        if not self.sessions_dir.exists():
            raise FileNotFoundError("Invalid dataset directory.")
        
        self.config = list(self.dataset_root.glob('*.json'))
        if len(self.config) > 1:
            raise FileExistsError("Invalid dataset config. There is multiple configs in the root direction.")
        self.config = self.config[0]

        if not split_variant in range(0, 4):
            raise ValueError("Split variant value should be in range of [0, 3].")

        with open(self.config) as json_config:
            cfg = json.load(json_config)
            self.action_labels = cfg['av_action_labels']

            def _norm_split(s):
                if s is None:
                    return None
                s = str(s)
                return s.split("_", 1)[0]
            
            db = cfg["db"]
            self.split_lut = {
                vid: _norm_split(
                    meta.get("split_ids", [None] * (split_variant + 1))[split_variant]
                    if len(meta.get("split_ids", [])) > split_variant else None
                )
                for vid, meta in db.items()
            }
        
        self.sessions = []
        for session in list(self.sessions_dir.glob('*.npz')):
            split = self.split_lut[session.stem]
            if (split.strip() == 'all') or (split_type == 'all'):
                self.sessions.append(session)
            elif split.strip() == split_type.strip():
                self.sessions.append(session)
        print(len(self.sessions))
        
    def __getitem__(self, index):
        archive = np.load(self.sessions[index])
        return archive['x'], archive['y']
    
    def __len__(self):
        return len(self.sessions)
    
# dataset = PreExtractedROADDataset(
#     dataset_root='data/road', 
#     dataset_variant='features-tsn-kinetics-400',
#     split_variant=2,
#     split_type='train'
# )
# x, y = dataset[0]