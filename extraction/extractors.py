import cv2
from matplotlib.pylab import overload
import torch
import numpy as np
from torchvision.transforms import v2 as T

from abc import ABC, abstractmethod
from typing import overload
from tqdm import tqdm

from utils.torch_scripts import get_device

class Extractor(ABC):

    def __init__(self, transforms):
        self.transforms = transforms

    @torch.no_grad()
    @abstractmethod
    def extract_timestamp_features(self, model: torch.nn.Module, x_tchw: torch.Tensor, micro: int, device: str) -> torch.Tensor:
        """
        Extracts features for single timestamp.
        """
        pass
    
    @abstractmethod
    def preprocess_video_frames(self, video_capture: cv2.VideoCapture, timestamp_frames: np.ndarray):
        pass

    @staticmethod
    def sample_timestamp_frames(fps: float, total_frames: int, sample_hz: float):
        duration = (total_frames - 1) / max(fps, 1e-6)
        step = 1.0 / sample_hz
        times = np.arange(0.0, duration + 1e-9, step)
        ids = np.rint(times * fps).astype(np.int64)
        ids = np.clip(ids, 0, total_frames - 1)
        ids = np.unique(ids)
        return ids
    
    @staticmethod
    def compose_transforms(mean, std, resize, center_crop):
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(resize, antialias=True),
            T.CenterCrop(center_crop),
            T.Normalize(mean=mean, std=std),
        ])

class Extractor_2D(Extractor):

    def __init__(self, transforms):
        super().__init__(transforms=transforms)

    @torch.no_grad()
    def extract_timestamp_features(
        self, 
        model,
        x_tchw, 
        device = get_device(),
        micro = 64
    ):
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
    
    def preprocess_video_frames(self, video_capture, timestamp_frames):
        timestamp_frames = np.asarray(timestamp_frames, dtype=np.int64)
        if timestamp_frames.size == 0:
            return []

        if not video_capture.isOpened():
            raise RuntimeError(f"Video not opened.")

        frames = []
        cur = 0

        for want in tqdm(timestamp_frames, desc="Preprocessing video", leave=False):
            want = int(want)
            if want < cur:
                video_capture.release()
                raise RuntimeError("timestamp_frames must be sorted increasing")

            # fast skipping
            while cur < want:
                if not video_capture.grab():
                    video_capture.release()
                    raise RuntimeError(f"Failed grab at frame {cur}.")
                cur += 1

            # decode valid frames
            ok, frame = video_capture.read()
            if not ok:
                video_capture.release()
                raise RuntimeError(f"Failed read at frame {want}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transforms(frame))
            cur += 1

        return torch.stack(frames, dim=0)
    
    @staticmethod
    def compose_transforms(mean, std, resize=256, center_crop=224):
        return Extractor.compose_transforms(mean, std, resize, center_crop)