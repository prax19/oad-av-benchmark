import json
import re
import warnings
from pathlib import Path
from typing import Literal

import torch
import torch.utils.data as data
import yaml

from utils.caching import _NPYCache

SplitStr = Literal["all", "train", "val", "test"]


def _infer_dataset_info(dataset_variant: str) -> dict[str, str | float | None]:
    """Infer backbone metadata from variant name like features-tsn-kinetics-400-4hz."""
    info: dict[str, str | float | None] = {
        "backbone": "unknown",
        "backbone_dataset": "unknown",
        "hz": None,
    }

    m = re.match(r"^features-(.+)-(\d+)hz$", str(dataset_variant).strip())
    if not m:
        return info

    backbone_full, hz_str = m.group(1), m.group(2)
    parts = backbone_full.split("-")
    if parts:
        info["backbone"] = parts[0]
        info["backbone_dataset"] = "-".join(parts[1:]) if len(parts) > 1 else "unknown"

    try:
        info["hz"] = float(hz_str)
    except Exception:
        info["hz"] = None

    return info


class PreExtractedDataset(data.Dataset):

    def __init__(
        self,
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
        self.annotated_dir = self.sessions_dir / "annotated_perframe"
        self.has_annotation_mask = self.annotated_dir.exists()
        missing = [name for name in required if not (self.sessions_dir / name).exists()]

        if missing:
            raise FileNotFoundError(f"Missing subdirs in {self.sessions_dir}: {missing}")

        self.long = long
        self.work = work

        # Caching
        self._npz_cache = _NPYCache(max_items=cache_size, mmap_mode=cache_mmap)

        # Class names (ROAD)
        self.class_names = self._load_class_names()

        # Clip splitting
        self.config = Path(self.sessions_dir, 'meta.yaml')
        if not self.config.exists():
            raise FileNotFoundError("No `meta.yaml` file.")

        if split_variant not in range(0, 4):
            raise ValueError("Split variant value should be in range of [0, 3].")

        with open(self.config) as yaml_config:
            cfg = yaml.safe_load(yaml_config)

            def _norm_split(s):
                if s is None:
                    return None
                s = str(s)
                return s.split("_", 1)[0]

            dataset_meta = cfg.get('dataset', {})
            inferred_info = _infer_dataset_info(dataset_variant=str(dataset_variant))

            self.dataset_info = {
                'name': self.__class__.__name__,
                'backbone': dataset_meta.get('backbone', inferred_info['backbone']),
                'backbone_dataset': dataset_meta.get('backbone_dataset', inferred_info['backbone_dataset']),
                'hz': dataset_meta.get('hz', inferred_info['hz']),
            }

            metadata_vid = dataset_meta.get('videos', {})
            self.split_lut = {
                vid: _norm_split(
                    meta.get("split_ids", [None] * (split_variant + 1))[split_variant]
                    if len(meta.get("split_ids", [])) > split_variant else None
                )
                for vid, meta in metadata_vid.items()
            }

        self.sessions = []
        for session_name in metadata_vid.keys():
            split = self.split_lut[session_name]
            if (split.strip() == 'all') or (split_type == 'all') or (split.strip() == split_type.strip()):
                self.sessions.append(session_name)

        # Windowing map preparation
        T = long + work
        self.samples = []
        self.ignored_clips = 0
        self.ignored_frames = 0
        for vid, meta in {k: metadata_vid[k] for k in self.sessions if k in metadata_vid}.items():
            session_x_pth = Path(self.sessions_dir, 'rgb', f'{vid}.npy')
            session_y_pth = Path(self.sessions_dir, 'target_perframe', f'{vid}.npy')
            session_a_pth = Path(self.annotated_dir, f'{vid}.npy') if self.has_annotation_mask else None
            has_ann_file = bool(session_a_pth is not None and session_a_pth.exists())
            N = meta['num_steps']
            if N < T:
                self.ignored_clips = self.ignored_clips + 1
                self.ignored_frames = self.ignored_frames + N
                continue

            for start in range(0, N - T + 1, stride):
                self.samples.append((session_x_pth, session_y_pth, session_a_pth, has_ann_file, start))

        if self.ignored_clips != 0:
            warnings.warn(f'Ignored {self.ignored_clips} clip(s) containing {self.ignored_frames} frames.', RuntimeWarning)

    @staticmethod
    def _is_synthetic_class_names(names: dict[int, str]) -> bool:
        if not names:
            return True
        for k, v in names.items():
            if str(v) != f"class_{k}":
                return False
        return True

    def _load_class_names(self) -> dict[int, str]:
        # 1) Prefer class names persisted with extracted features metadata.
        meta_path = self.sessions_dir / "meta.yaml"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_cfg = yaml.safe_load(f) or {}
                meta_labels = meta_cfg.get("dataset", {}).get("class_names")

                if isinstance(meta_labels, list) and meta_labels:
                    if all(isinstance(x, str) for x in meta_labels):
                        parsed = {i: str(name) for i, name in enumerate(meta_labels)}
                        if not self._is_synthetic_class_names(parsed):
                            return parsed

                if isinstance(meta_labels, dict) and meta_labels:
                    out: dict[int, str] = {}
                    for k, v in meta_labels.items():
                        try:
                            idx = int(k)
                        except Exception:
                            continue
                        out[idx] = str(v)
                    if out:
                        parsed = {k: out[k] for k in sorted(out.keys())}
                        if not self._is_synthetic_class_names(parsed):
                            return parsed
            except Exception:
                pass

        # 2) Fall back to dataset jsons from multiple likely locations.
        candidates = [
            self.dataset_root / "road_trainval_v1.0.json",
            self.dataset_root / "road_waymo_trainval_v1.0.json",
        ]
        candidates.extend(sorted(self.dataset_root.glob("*trainval*.json")))

        def _labels_from_config(cfg: dict) -> dict[int, str]:
            labels = cfg.get("av_action_labels")

            # Format A: ["name0", "name1", ...]
            if isinstance(labels, list) and labels:
                if all(isinstance(x, str) for x in labels):
                    return {i: str(name) for i, name in enumerate(labels)}

                # Format B: [{"id": 0, "name": "..."}, ...]
                if all(isinstance(x, dict) for x in labels):
                    out: dict[int, str] = {}
                    for i, item in enumerate(labels):
                        idx = item.get("id", i)
                        name = item.get("name", item.get("label", f"class_{idx}"))
                        try:
                            out[int(idx)] = str(name)
                        except Exception:
                            continue
                    if out:
                        return {k: out[k] for k in sorted(out.keys())}

            # Format C: {"0": "name", "1": "name", ...} or nested values.
            if isinstance(labels, dict) and labels:
                out: dict[int, str] = {}
                for k, v in labels.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue

                    if isinstance(v, str):
                        out[idx] = v
                    elif isinstance(v, dict):
                        out[idx] = str(v.get("name", v.get("label", f"class_{idx}")))
                    else:
                        out[idx] = f"class_{idx}"

                if out:
                    return {k: out[k] for k in sorted(out.keys())}

            return {}

        seen: set[Path] = set()
        for cfg_path in candidates:
            cfg_path = cfg_path.resolve()
            if cfg_path in seen:
                continue
            seen.add(cfg_path)
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                parsed = _labels_from_config(cfg)
                if parsed:
                    return parsed
            except Exception:
                continue

        # 3) Final fallback: infer class count from targets and create synthetic names.
        tgt_files = sorted((self.sessions_dir / "target_perframe").glob("*.npy"))
        if tgt_files:
            try:
                import numpy as np
                y = np.load(tgt_files[0], mmap_mode="r")
                num_classes = int(y.shape[-1]) if y.ndim >= 2 else 0
                if num_classes > 0:
                    return {i: f"class_{i}" for i in range(num_classes)}
            except Exception:
                pass

        return {}

    def build_sample_weights(
        self,
        rarity_power: float = 1.0,
        min_weight: float = 1.0,
        max_weight: float = 25.0,
    ) -> torch.DoubleTensor:
        """
        Build per-sample weights for WeightedRandomSampler by upweighting windows
        that contain rare positive classes.
        """
        import numpy as np

        if len(self.samples) == 0:
            return torch.ones((0,), dtype=torch.double)

        class_pos = None
        sample_presence: list[np.ndarray] = []

        for _, y_path, a_path, has_ann_file, start in self.samples:
            y_full = self._npz_cache.get(y_path)
            end = start + self.long + self.work
            y = y_full[end - self.work:end]

            if has_ann_file:
                a_full = self._npz_cache.get(a_path)
                ann = a_full[end - self.work:end].astype(bool)
                if ann.ndim == 1:
                    valid = ann
                else:
                    valid = ann.reshape(-1).astype(bool)
            else:
                valid = np.ones((y.shape[0],), dtype=bool)

            if valid.any():
                y_valid = y[valid]
                present = (y_valid.max(axis=0) > 0).astype(np.float32)
            else:
                present = np.zeros((y.shape[-1],), dtype=np.float32)

            sample_presence.append(present)
            if class_pos is None:
                class_pos = present.copy()
            else:
                class_pos += present

        assert class_pos is not None
        n_samples = float(len(sample_presence))
        inv = np.power((n_samples + 1.0) / (class_pos + 1.0), rarity_power)

        weights = []
        for present in sample_presence:
            if present.sum() <= 0:
                w = 1.0
            else:
                w = 1.0 + float((inv * present).sum())
            w = max(min_weight, min(max_weight, w))
            weights.append(w)

        return torch.tensor(weights, dtype=torch.double)

    def __getitem__(self, index):
        session_x_pth, session_y_pth, session_a_pth, has_ann_file, start = self.samples[index]
        session_x = self._npz_cache.get(session_x_pth)
        session_y = self._npz_cache.get(session_y_pth)
        end = start + self.long + self.work
        x = session_x[start:end]
        y = session_y[end-self.work:end]

        if has_ann_file:
            session_a = self._npz_cache.get(session_a_pth)
            ann = session_a[end-self.work:end].copy()
        else:
            ann = None

        ann_t = torch.from_numpy(ann).bool() if ann is not None else torch.ones((self.work,), dtype=torch.bool)
        # Arrays loaded with mmap_mode="r" may be non-writable; copy to avoid UB warning in torch.from_numpy.
        x_t = torch.from_numpy(x.copy()).float()
        y_t = torch.from_numpy(y.copy()).float()
        return x_t, y_t, ann_t

    def __len__(self):
        return len(self.samples)
