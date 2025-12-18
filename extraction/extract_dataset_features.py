from pathlib import Path
from tqdm import tqdm
from torchvision import datasets

import cv2
import numpy as np

from mmengine import Config
from mmaction.apis import init_recognizer
from mmaction.apis import init_recognizer

from weights import load_by_key
from utils.torch_scripts import get_device
from extraction import extractors
from extraction.extractor_datasets import RoadExtractionDataset

def extract_dataset_features(
    backbone = "tsn-kinetics-400",
    dataset = RoadExtractionDataset("data/road"),
    device = get_device(),
    sample_hz = 4
):
    # preparing config files and directories
    cfg_path, ckpt_path = load_by_key(backbone)
    out_dir = dataset.get_extraction_directory(backbone=backbone)

    # model initialization
    cfg = Config.fromfile(str(cfg_path))
    model = init_recognizer(cfg, str(ckpt_path), device=device)
    model.eval()

    # preparing preprocessing
    dp = cfg.model.data_preprocessor
    mean = [m/255.0 for m in dp.mean] if max(dp.mean) > 1 else list(dp.mean)
    std  = [s/255.0 for s in dp.std]  if max(dp.std)  > 1 else list(dp.std)
    resize = cfg.val_dataloader.dataset.pipeline[3].scale[1]
    crop_size = cfg.val_dataloader.dataset.pipeline[4].crop_size

    # preparing extraction config
    model_type = cfg.model.type
    extractor: extractors.Extractor = None
    if model_type == 'Recognizer2D':
        transforms = extractors.Extractor_2D.compose_transforms(
            mean=mean, 
            std=std,
            resize=resize,
            center_crop=crop_size
        )
        extractor = extractors.Extractor_2D(
            model=model, 
            transforms=transforms, 
            sampling_hz=sample_hz
        )
    elif model_type == 'Recognizer3D':
        frame_interval = cfg.val_dataloader.dataset.pipeline[1].frame_interval
        clip_len = cfg.val_dataloader.dataset.pipeline[1].clip_len
        transforms = extractors.Extractor_3D.compose_transforms(
            mean=mean, 
            std=std,
            resize=resize,
            center_crop=crop_size
        )
        extractor = extractors.Extractor_3D(
            model=model, 
            transforms=transforms, 
            sampling_hz=sample_hz, 
            frame_interval=frame_interval, 
            clip_len=clip_len
        )
    else:
        raise(NotImplementedError(f'{model_type} model type not supported.'))

    for vid in tqdm(dataset, desc='Processing dataset', leave=False):
        feat_path = Path(out_dir, f"{vid.stem}.npz")
        if feat_path.exists():
            continue

        vid_cap = cv2.VideoCapture(vid)

        fps = float(vid_cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        feats, timestamp_frames = extractor.extract_video_features(video_capture=vid_cap)

        vid_cap.release()

        np.savez(
            feat_path,
            x=feats.numpy().astype(np.float32),
            frame_ids=timestamp_frames.astype(np.int64),
            fps=float(fps),
            hz=float(sample_hz),
            total_frames=int(total_frames),
        )

extract_dataset_features(backbone='tsn-kinetics-400')