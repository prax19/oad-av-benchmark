import json
import numpy as np

def pick(d: dict, keys):
    return {k: d[k] for k in keys if k in d}

def filter_video_entry_opencv(video_meta: dict) -> dict:
    out = pick(video_meta, ["split_ids", "numf"])

    frames_in = video_meta["frames"]
    frames_out = {}

    for fk_str, fr in frames_in.items():
        fk0 = int(fk_str) - 1
        frames_out[fk0] = pick(fr, ["annotated", "av_action_ids"])

    out["frames"] = dict(sorted(frames_out.items(), key=lambda kv: kv[0]))
    return out


def filter_video_frames_by_id(video_meta: dict, timestamp_frames: np.ndarray) -> dict:
    out = pick(video_meta, ["split_ids", "numf"])

    frames_in = video_meta["frames"]
    frames0 = {int(k) - 1: pick(v, ["annotated", "av_action_ids"]) for k, v in frames_in.items()}

    ts = np.asarray(timestamp_frames, dtype=np.int64)
    out["frames"] = {int(t): frames0.get(int(t), {"annotated": False, "av_action_ids": []}) for t in ts}

    return out

def extract_labels_per_video(dataset_cfg: str, video_id: str):
    with open(dataset_cfg, "r", encoding="utf-8") as json_cfg:
        cfg = json.load(json_cfg)
    video_meta = cfg["db"][video_id]
    return filter_video_entry_opencv(video_meta)
