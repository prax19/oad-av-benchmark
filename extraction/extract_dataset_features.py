from pathlib import Path
from tqdm import tqdm
import yaml

import cv2
import numpy as np

from mmengine import Config
from mmaction.apis import init_recognizer

from weights import load_by_key
from utils.torch_scripts import get_device
from extraction import extractors
from extraction.extractor_datasets import RoadExtractionDataset

def extract_dataset_features(
    backbone = "tsn-kinetics-400",
    dataset = RoadExtractionDataset(root='data/road', dataset_config='road_trainval_v1.0.json'),
    device = get_device(),
    sample_hz = 4
):
    # preparing config files and directories
    cfg_path, ckpt_path = load_by_key(backbone)
    out_dir = dataset.get_extraction_directory(backbone=backbone)
    tgt_dir = dataset.get_target_directory(backbone=backbone)
    ann_dir = dataset.get_annotated_directory(backbone=backbone)
    dump_root = dataset.get_dump_directory(backbone=backbone)

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

    # Metadata config / loading
    meta_path = Path(dump_root, 'meta.yaml')
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
    else:
        meta = { # metadata template
            "dataset": {
                "hz": float(sample_hz),
                "paths": {
                    "feats_dir": str(Path(out_dir)),
                    "targets_dir": str(Path(tgt_dir)),
                    "annotated_dir": str(Path(ann_dir))
                },
                "videos": {}
            }
        }

    try:
        for vid, label, annotated, split in tqdm(dataset, desc='Processing dataset', leave=False):
            # Path handling
            feat_path = Path(out_dir, f"{vid.stem}.npy")
            tgt_path = Path(tgt_dir, f"{vid.stem}.npy")
            ann_path = Path(ann_dir, f"{vid.stem}.npy")
            paths = [feat_path, tgt_path, ann_path]

            feat_exists = feat_path.exists()
            tgt_exists  = tgt_path.exists()
            ann_exists  = ann_path.exists()
            meta_exists = vid.stem in meta["dataset"]["videos"]

            exists = [feat_exists, tgt_exists, ann_exists, meta_exists]
            if all(exists):
                continue
            elif any(exists):
                for p in paths:
                    p.unlink(missing_ok=True)
                meta["dataset"]["videos"].pop(vid.stem, None)

            # Video handling and extraction
            vid_cap = cv2.VideoCapture(vid)

            fps = float(vid_cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            feats, timestamp_frames = extractor.extract_video_features(video_capture=vid_cap, device=device, micro=32)
            timestamp_frames = np.asarray(timestamp_frames, dtype=np.int64)

            if timestamp_frames.size and len(label) > 0:
                valid_mask = timestamp_frames < len(label)
                if not np.all(valid_mask):
                    timestamp_frames = timestamp_frames[valid_mask]
                    feats = feats[valid_mask]

            x = feats.numpy().astype(np.float32)
            y = label[timestamp_frames].astype(np.uint8)
            a = annotated[timestamp_frames].astype(np.bool_)

            vid_cap.release()

            # Saving
            np.save(feat_path, x)
            np.save(tgt_path, y)
            np.save(ann_path, a)

            # Metadata saving
            meta["dataset"]["videos"][vid.stem] = {
                "num_steps": int(x.shape[0]),
                "total_frames": total_frames,
                "fps": float(fps),
                "split_ids": list(split)
            }
        print("\nExtraction finished.")
    except KeyboardInterrupt:
        print("\nExtraction stopped. Saving metadata...")
    finally:
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

extract_dataset_features(backbone='tsn-kinetics-400')