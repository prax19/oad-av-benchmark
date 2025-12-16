from pathlib import Path
from tqdm import tqdm
import gc

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T

from mmengine import Config
from mmaction.apis import init_recognizer
from mmaction.apis import init_recognizer

from weights import load_by_key
from utils.torch_scripts import get_device

def sample_frame_ids(fps: float, total_frames: int, sample_hz: float):
    duration = (total_frames - 1) / max(fps, 1e-6)
    step = 1.0 / sample_hz
    times = np.arange(0.0, duration + 1e-9, step)
    ids = np.rint(times * fps).astype(np.int64)
    ids = np.clip(ids, 0, total_frames - 1)
    ids = np.unique(ids)
    return ids

def read_frames_by_index(video_path: Path, frame_ids: np.ndarray, transforms):
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if frame_ids.size == 0:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    cur = 0

    for want in tqdm(frame_ids, desc="Preprocessing video", leave=False):
        want = int(want)
        if want < cur:
            cap.release()
            raise RuntimeError("frame_ids must be sorted increasing")

        # fast skipping
        while cur < want:
            if not cap.grab():
                cap.release()
                raise RuntimeError(f"Failed grab at frame {cur} for {video_path}")
            cur += 1

        # decode valid frames
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed read at frame {want} for {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transforms(frame))
        cur += 1

    cap.release()
    return torch.stack(frames, dim=0)

@torch.no_grad()
def extract_2d_features(model, x_tchw: torch.Tensor, micro=64):
    device = next(model.parameters()).device
    feats = []

    for s in tqdm(range(0, x_tchw.shape[0], micro), desc='Extracting features', leave=False):
        inp = x_tchw[s:s+micro].to(device, non_blocking=True)

        out = model.backbone(inp)

        if hasattr(model, "neck") and model.neck is not None:
            out = model.neck(out)

        # global avg pool -> [B,C]
        if out.ndim == 4:
            out = out.mean(dim=(2, 3))
        elif out.ndim != 2:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(out.shape)}")

        feats.append(out.detach().cpu())

    return torch.cat(feats, dim=0)

def extract_features(
    backbone = "tsn-kinetics-400",
    dataset_root = Path("data/road"),
    device = get_device(),
    sample_hz = 4
):
    # preparing config files and directories
    cfg_path, ckpt_path = load_by_key(backbone)
    out_dir = dataset_root / f"features-{backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # model initialization
    cfg = Config.fromfile(str(cfg_path))
    model = init_recognizer(cfg, str(ckpt_path), device=device)
    model.eval()

    # preparing preprocessing
    dp = cfg.model.data_preprocessor
    mean = [m/255.0 for m in dp.mean] if max(dp.mean) > 1 else list(dp.mean)
    std  = [s/255.0 for s in dp.std]  if max(dp.std)  > 1 else list(dp.std)

    transforms = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=mean, std=std),
    ])

    # full dataset pass
    vids = sorted(Path(dataset_root, "videos").glob("*.mp4"))
    for vid in tqdm(vids, desc='Processing dataset', leave=False):
        feat_path = Path(out_dir, f"{vid.stem}.npz")
        if feat_path.exists():
            continue

        vid_cap = cv2.VideoCapture(vid)
        fps = float(vid_cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_cap.release()

        frame_ids = sample_frame_ids(fps=fps, total_frames=total_frames, sample_hz=sample_hz)
        frames = read_frames_by_index(video_path=vid, frame_ids=frame_ids, transforms=transforms)
        
        feats = extract_2d_features(model, frames, micro=64)
        del frames
        gc.collect()

        np.savez(
            feat_path,
            x=feats.numpy().astype(np.float32),
            frame_ids=frame_ids.astype(np.int64),
            fps=float(fps),
            hz=float(sample_hz),
            total_frames=int(total_frames),
        )
        del feats, frame_ids
        gc.collect()

extract_features()