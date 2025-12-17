import cv2
import torch
import numpy as np
from torchvision.transforms import v2 as T

from abc import ABC, abstractmethod
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
    
class Extractor_3D(Extractor):
    def __init__(self, transforms, clip_len=16, frame_interval=4):
        super().__init__(transforms=transforms)
        self.clip_len = int(clip_len)
        self.frame_interval = int(frame_interval)

    @torch.no_grad()
    def extract_timestamp_features(
        self,
        model,
        x_ncthw: torch.Tensor,   # [N, 3, T, H, W]
        device=get_device(),
        micro=2
    ):
        feats = []

        for s in tqdm(range(0, x_ncthw.shape[0], micro), desc="Extracting features", leave=False):
            inp = x_ncthw[s:s+micro].to(device, non_blocking=True)

            out = model.backbone(inp)

            if hasattr(model, "neck") and model.neck is not None:
                out = model.neck(out)

            if isinstance(out, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in out):
                pooled = []
                for t in out:
                    while t.ndim > 2:
                        t = t.mean(dim=-1)   # redukuje T/H/W
                    pooled.append(t)
                out = torch.cat(pooled, dim=1)  # [B, D_total]
            else:
                while out.ndim > 2:
                    out = out.mean(dim=-1)      # [B, D]

            feats.append(out.detach().cpu())

        return torch.cat(feats, dim=0)  # [N, D]

    def preprocess_video_frames(self, video_capture: cv2.VideoCapture, timestamp_frames: np.ndarray):
        timestamp_frames = np.asarray(timestamp_frames, dtype=np.int64)
        if timestamp_frames.size == 0:
            return torch.empty((0, 3, self.clip_len, 224, 224), dtype=torch.float32)

        if not video_capture.isOpened():
            raise RuntimeError("Video not opened.")

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Could not read total frame count.")

        L = self.clip_len
        I = self.frame_interval

        # [N, L]
        ar = np.arange(L, dtype=np.int64)
        clip_ids = timestamp_frames[:, None] - I * (L - 1 - ar[None, :])
        clip_ids = np.clip(clip_ids, 0, total_frames - 1).astype(np.int64)

        all_ids = np.unique(clip_ids.reshape(-1))
        frames = []
        cur = 0

        for want in tqdm(all_ids, desc="Preprocessing video", leave=False):
            want = int(want)
            if want < cur:
                video_capture.release()
                raise RuntimeError("Internal error: all_ids not sorted increasing")

            while cur < want:
                if not video_capture.grab():
                    video_capture.release()
                    raise RuntimeError(f"Failed grab at frame {cur}.")
                cur += 1

            ok, frame = video_capture.read()
            if not ok:
                video_capture.release()
                raise RuntimeError(f"Failed read at frame {want}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transforms(frame))  # [3,H,W]
            cur += 1

        frames_all = torch.stack(frames, dim=0)  # [M,3,H,W]

        pos = np.searchsorted(all_ids, clip_ids)                  # [N,L]
        clips = frames_all[pos]                                   # [N,L,3,H,W]
        clips = clips.permute(0, 2, 1, 3, 4).contiguous()         # [N,3,T,H,W]

        return clips

    @staticmethod
    def compose_transforms(mean, std, resize=256, center_crop=224):
        return Extractor.compose_transforms(mean, std, resize, center_crop)
