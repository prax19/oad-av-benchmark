import numpy as np
import torch
from pathlib import Path
from mmengine import Config
from mmengine.dataset import Compose
from mmaction.apis import init_recognizer

from weights import load_by_key
from utils import torch_scripts

import cv2
from pathlib import Path

from copy import deepcopy

def force_opencv_decode(cfg):
    cfg = deepcopy(cfg)
    pipe = cfg.test_dataloader.dataset.pipeline
    for t in pipe:
        if t.get("type") == "DecordInit":
            t["type"] = "OpenCVInit"
        elif t.get("type") == "DecordDecode":
            t["type"] = "OpenCVDecode"
    return cfg

@torch.no_grad()
def extract_features(backbone: str, dataset_root: Path):
    cfg_path, ckpt_path = load_by_key(backbone)

    out_dir = dataset_root / f"features-{backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    cfg = force_opencv_decode(cfg)

    pack = cfg.test_dataloader.dataset.pipeline[-1]
    if pack.get("type") != "PackActionInputs":
        raise RuntimeError(f"Last pipeline step is not PackActionInputs: {pack.get('type')}")
    mk = list(pack.get("meta_keys", ()))
    for k in ("frame_inds", "frame_interval", "clip_len", "num_clips", "total_frames", "avg_fps"):
        if k not in mk:
            mk.append(k)
    pack["meta_keys"] = tuple(mk)

    device = torch_scripts.get_device()
    model = init_recognizer(cfg, str(ckpt_path), device=device).eval()
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)

    for vid in sorted((dataset_root / "videos").glob("*.mp4")):
        out_path = out_dir / f"{vid.stem}.npz"
        if out_path.exists():
            continue

        data = pipeline(dict(filename=str(vid), label=-1, start_index=0))

        batch = dict(inputs=[data["inputs"]], data_samples=[data["data_samples"]])
        batch = model.data_preprocessor(batch, training=False)

        inputs = batch["inputs"]  # [N, T, C, H, W]
        N, T, C, H, W = inputs.shape

        feats = model.backbone(inputs.reshape(N * T, C, H, W))
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))  # [N*T, D]

        x = feats.reshape(N, T, -1)[0].detach().cpu().numpy().astype(np.float32)  # [T, D]

        ds0 = batch["data_samples"][0]
        frame_inds = ds0.metainfo.get("frame_inds", None)
        if frame_inds is None:
            frame_inds = data.get("frame_inds", None)
        if frame_inds is None:
            raise RuntimeError("Pipeline didn't produce frame_inds; can't align labels later.")

        frame_inds = np.asarray(frame_inds).reshape(-1).astype(np.int64)  # [T]

        meta = ds0.metainfo
        np.savez_compressed(
            out_path,
            x=x,
            frame_inds=frame_inds,
            frame_interval=int(meta.get("frame_interval", 1)),
            clip_len=int(meta.get("clip_len", 1)),
            num_clips=int(meta.get("num_clips", 1)),
            total_frames=int(meta.get("total_frames", -1)),
            avg_fps=float(meta.get("avg_fps", -1.0)),
        )
        print("saved:", out_path, x.shape, frame_inds.shape)

extract_features("slowfast-kinetics-400", Path("data/road"))