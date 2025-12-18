import cv2
import torch
import numpy as np
from torchvision.transforms import v2 as T

from abc import ABC, abstractmethod
from tqdm import tqdm

from utils.torch_scripts import get_device

class Extractor(ABC):

    def __init__(
        self,
        model: torch.nn.Module,
        sampling_hz: float,
        transforms: T.Compose
    ):
        self.model = model
        self.transforms = transforms
        self.sample_hz = sampling_hz

    @torch.no_grad()
    @abstractmethod
    def extract_video_features(
        self,
        video_capture: cv2.VideoCapture,
        timestamp_frames: np.ndarray,
        micro: int,
        device: str
    ):
        """
        Extracts features from a single video.
        """
        pass

    def extract_timestamp_frames(self, video_capture: cv2.VideoCapture) -> np.ndarray:
        """
        Extracts timestamp frames from a single video.
        """
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.sample_timestamp_frames(fps=fps, total_frames=total_frames, sample_hz=self.sample_hz)

    @staticmethod
    def compose_transforms(mean, std, resize, center_crop):
        """
        Prepares the transformation pipeline.
        """
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(resize, antialias=True),
            T.CenterCrop(center_crop),
            T.Normalize(mean=mean, std=std),
        ])

    @staticmethod
    def sample_timestamp_frames(fps: float, total_frames: int, sample_hz: float) -> np.ndarray:
        """
        Calculates timestamp frames to sample from the video.
        """
        duration = (total_frames - 1) / max(fps, 1e-6)
        step = 1.0 / sample_hz
        times = np.arange(0.0, duration + 1e-9, step)
        ids = np.rint(times * fps).astype(np.int64)
        ids = np.clip(ids, 0, total_frames - 1)
        ids = np.unique(ids)
        return ids

class Extractor_2D(Extractor):

    def __init__(self, model, transforms, sampling_hz: float = 4.0):
        super().__init__(model=model, transforms=transforms, sampling_hz=sampling_hz)
    
    @torch.no_grad()
    def extract_video_features(
        self,
        video_capture: cv2.VideoCapture,
        micro: int = 16,
        device: str = get_device()
    ):
        timestamp_frames = self.extract_timestamp_frames(video_capture)
        if timestamp_frames.size == 0:
            return []

        if not video_capture.isOpened():
            raise RuntimeError(f"Video not opened.")
        
        feats = []
        batch = []
        cur = 0

        for want in tqdm(timestamp_frames, desc="Feature extraction", leave=False):
            want = int(want)

            while cur < want:
                if not video_capture.grab():
                    raise RuntimeError(f"Failed grab at frame {cur}.")
                cur += 1

            ok, frame = video_capture.read()
            if not ok:
                raise RuntimeError(f"Failed read at frame {want}.")
            cur += 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = self.transforms(frame)          # [3,H,W]
            batch.append(x)

            if len(batch) >= micro:
                inp = torch.stack(batch, dim=0).to(device, non_blocking=True)  # [B,3,H,W]
                out = self.model.backbone(inp)
                if hasattr(self.model, "neck") and self.model.neck is not None:
                    out = self.model.neck(out)

                if out.ndim == 4:
                    out = out.mean(dim=(2, 3))  # [B,C]
                elif out.ndim != 2:
                    raise RuntimeError(f"Unexpected backbone output shape: {tuple(out.shape)}")

                feats.append(out.detach().cpu())
                batch.clear()

        if batch:
            inp = torch.stack(batch, dim=0).to(device, non_blocking=True)
            out = self.model.backbone(inp)
            if hasattr(self.model, "neck") and self.model.neck is not None:
                out = self.model.neck(out)

            if out.ndim == 4:
                out = out.mean(dim=(2, 3))
            elif out.ndim != 2:
                raise RuntimeError(f"Unexpected backbone output shape: {tuple(out.shape)}")

            feats.append(out.detach().cpu())

            del inp, out
            batch.clear()

        feats = torch.cat(feats, dim=0) if feats else torch.empty((0, 0), dtype=torch.float32)
        return feats, timestamp_frames
    
    @staticmethod
    def compose_transforms(mean, std, resize=256, center_crop=224):
        return Extractor.compose_transforms(mean, std, resize, center_crop)
    
class Extractor_3D(Extractor):
    
    def __init__(
            self, 
            model, 
            transforms,
            clip_len=16, 
            frame_interval=4, 
            sampling_hz: float = 4.0
    ):
        super().__init__(model=model, transforms=transforms, sampling_hz=sampling_hz)
        self.clip_len = int(clip_len)
        self.frame_interval = int(frame_interval)
    
    @torch.no_grad()
    def extract_video_features(
        self,
        video_capture: cv2.VideoCapture,
        micro: int = 8,
        device: str = get_device()
    ):
        timestamp_frames = self.extract_timestamp_frames(video_capture)
        if not video_capture.isOpened():
            raise RuntimeError("Video not opened.")
        
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        L = self.clip_len
        I = self.frame_interval
        ar = np.arange(L, dtype=np.int64)
        span = I * (L - 1)

        # iterate through video and extract only needed frames
        need = np.zeros(total_frames, dtype=np.bool_)
        for end in timestamp_frames:
            ids = end - I * (L - 1 - ar)
            ids = np.clip(ids, 0, total_frames - 1)
            need[ids] = True

        feats = []
        clip_batch = []
        buf = {} # preprocessed frames buffer

        ti = 0
        next_end = int(timestamp_frames[ti])
        last_end = int(timestamp_frames[-1])

        # full video pass
        for cur in tqdm(range(0, last_end + 1), desc="Feature extraction", leave=False):
            if need[cur]:
                ok, frame = video_capture.read()
                if not ok:
                    raise RuntimeError(f"Failed read at frame {cur}.")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buf[cur] = self.transforms(frame)  # [3,H,W]
            else:
                if not video_capture.grab():
                    raise RuntimeError(f"Failed grab at frame {cur}.")

            if cur == next_end:
                ids = next_end - I * (L - 1 - ar)
                ids = np.clip(ids, 0, total_frames - 1).astype(np.int64)

                clip = torch.stack([buf[int(i)] for i in ids], dim=0)          # [T,3,H,W]
                clip = clip.permute(1, 0, 2, 3).contiguous()                   # [3,T,H,W]
                clip_batch.append(clip)

                # excluding old frames from buffer
                cutoff = next_end - span
                for k in list(buf.keys()):
                    if k < cutoff:
                        del buf[k]

                ti += 1
                if ti < len(timestamp_frames):
                    next_end = int(timestamp_frames[ti])

                if len(clip_batch) >= micro:
                    inp = torch.stack(clip_batch, dim=0).to(device, non_blocking=True)  # [B,3,T,H,W]
                    out = self.model.backbone(inp)
                    if hasattr(self.model, "neck") and self.model.neck is not None:
                        out = self.model.neck(out)

                    if isinstance(out, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in out):
                        pooled = []
                        for t in out:
                            while t.ndim > 2:
                                t = t.mean(dim=-1)
                            pooled.append(t)
                        out = torch.cat(pooled, dim=1)
                    else:
                        while out.ndim > 2:
                            out = out.mean(dim=-1)

                    feats.append(out.detach().cpu())
                    clip_batch.clear()

        # flush
        if clip_batch:
            inp = torch.stack(clip_batch, dim=0).to(device, non_blocking=True)
            out = self.model.backbone(inp)
            if hasattr(self.model, "neck") and self.model.neck is not None:
                out = self.model.neck(out)

            if isinstance(out, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in out):
                pooled = []
                for t in out:
                    while t.ndim > 2:
                        t = t.mean(dim=-1)
                    pooled.append(t)
                out = torch.cat(pooled, dim=1)
            else:
                while out.ndim > 2:
                    out = out.mean(dim=-1)

            feats.append(out.detach().cpu())
            clip_batch.clear()

        feats = torch.cat(feats, dim=0) if feats else torch.empty((0, 0), dtype=torch.float32)
        return feats, timestamp_frames

    @staticmethod
    def compose_transforms(mean, std, resize=256, center_crop=224):
        return Extractor.compose_transforms(mean, std, resize, center_crop)
