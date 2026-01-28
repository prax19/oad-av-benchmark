import numpy as np
import torch.utils.data as data
from pathlib import Path
import yaml
from typing import Literal
import warnings

from utils.caching import _NPZCache

SplitStr = Literal["all", "train", "val", "test"]

class PreExtractedDataset(data.Dataset):

    def __init__(self, 
        dataset_root: str, 
        dataset_variant: str, 
        split_type: str | SplitStr = 'all',
        split_variant: int = 0,
        long: int = 512,    # context memory
        work: int = 8,      # prediction space
        stride: int = 4,    # window stride,
        cache_size: int = 8,
        cache_mmap: bool = 'r'
    ):
        self.dataset_root = Path(dataset_root)
        self.sessions_dir = Path(dataset_root, dataset_variant)

        required = ["rgb", "target_perframe"]
        missing = [name for name in required if not (self.sessions_dir / name).exists()]

        if missing:
            raise FileNotFoundError(f"Missing subdirs in {self.sessions_dir}: {missing}")
        
        self.long = long
        self.work = work

        # # Caching
        # self._npz_cache = _NPZCache(max_items=cache_size, mmap_mode=cache_mmap)
        
        # Clip splitting
        self.config = Path(self.sessions_dir, 'meta.yaml')
        if not self.config.exists():
            raise FileNotFoundError("No `meta.yaml` file.")

        if not split_variant in range(0, 4):
            raise ValueError("Split variant value should be in range of [0, 3].")

        with open(self.config) as yaml_config:
            cfg = yaml.safe_load(yaml_config)

            def _norm_split(s):
                if s is None:
                    return None
                s = str(s)
                return s.split("_", 1)[0]
            
            db = cfg['dataset']['videos']
            self.split_lut = {
                vid: _norm_split(
                    meta.get("split_ids", [None] * (split_variant + 1))[split_variant]
                    if len(meta.get("split_ids", [])) > split_variant else None
                )
                for vid, meta in db.items()
            }
        
        self.sessions = []
        for session_name in db.keys():
            session_pth = Path(self.sessions_dir, 'rgb', f'{session_name}.npy')
            split = self.split_lut[session_name]
            if (split.strip() == 'all') or (split_type == 'all') or (split.strip() == split_type.strip()):
                self.sessions.append((session_pth, Path(self.sessions_dir, 'target_perframe', f'{session_name}.npy')))

        # Windowing map preparation
        T = long + work
        self.samples = []
        ignored_clips = 0
        ignored_frames = 0
        for session_x_pth, session_y_pth in self.sessions:
            x = np.load(session_x_pth); y = np.load(session_y_pth)

            if len(x) != len(y):
                raise ValueError(f"Length mismatch: len(x)={len(x)} len(y)={len(y)}")

            N = len(x)
            if N < T:
                ignored_clips = ignored_clips + 1
                ignored_frames = ignored_frames + N
                continue

            for start in range(0, N - T + 1, stride):
                self.samples.append((session_x_pth, session_y_pth, start))
        
        if ignored_clips != 0:
            warnings.warn(f'Ignored {ignored_clips} clip(s) containing {ignored_frames} frames.', RuntimeWarning)
    
    def __getitem__(self, index):
        session_x_pth, session_y_pth, start = self.samples[index]
        session_x = np.load(session_x_pth)
        session_y = np.load(session_y_pth)
        end = start + self.long + self.work
        x = session_x[start:end]
        y = session_y[end-self.work:end]
        return x, y
    
    def __len__(self):
        return len(self.samples)
    
dataset = PreExtractedDataset(
    dataset_root='data/road', 
    dataset_variant='features-tsn-kinetics-400',
    split_variant=2,
    split_type='train'
)