import numpy as np
import torch.utils.data as data
from pathlib import Path
import json
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
        if not self.sessions_dir.exists():
            raise FileNotFoundError("Invalid dataset directory.")
        
        self.long = long
        self.work = work

        # Caching
        self._npz_cache = _NPZCache(max_items=cache_size, mmap_mode=cache_mmap)
        
        # Clip splitting
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
        for session_pth in list(self.sessions_dir.glob('*.npz')):
            split = self.split_lut[session_pth.stem]
            if (split.strip() == 'all') or (split_type == 'all'):
                self.sessions.append(session_pth)
            elif split.strip() == split_type.strip():
                self.sessions.append(session_pth)
        
        # Windowing map preparation
        T = long + work
        self.samples = []
        ignored_clips = 0
        ignored_frames = 0
        for session_pth in self.sessions:
            session = np.load(session_pth)
            try:
                x = session['x']; y = session['y']
            finally:
                session.close()

            if len(x) != len(y):
                raise ValueError(f"Length mismatch: len(x)={len(x)} len(y)={len(y)}")

            N = len(x)
            if N < T:
                ignored_clips = ignored_clips + 1
                ignored_frames = ignored_frames + N
                continue

            for start in range(0, N - T + 1, stride):
                self.samples.append((session_pth, start))
        
        if ignored_clips != 0:
            warnings.warn(f'Ignored {ignored_clips} clip(s) containing {ignored_frames} frames.', RuntimeWarning)
    
    def __getitem__(self, index):
        session_pth, start = self.samples[index]
        session = self._npz_cache.get(session_pth) # cached npz load
        end = start + self.long + self.work
        x = session['x'][start:end]
        y = session['y'][end-self.work:end]
        return x, y
    
    def __len__(self):
        return len(self.samples)