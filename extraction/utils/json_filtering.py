import numpy as np

def pack_av_multihot_from_frames(frames: dict, num_classes: int):
    numf = len(frames)
    y = np.zeros((numf, num_classes), dtype=np.uint8)
    annotated = np.zeros((numf,), dtype=np.bool_)

    for fk_str, fr in frames.items():
        t0 = int(fk_str) - 1
        if t0 < 0 or t0 >= numf:
            continue
        annotated[t0] = bool(fr.get("annotated", False))
        for c in fr.get("av_action_ids", []):
            c = int(c)
            if 0 <= c < num_classes:
                y[t0, c] = 1

    return y, annotated

def pack_av_cls_from_frames(frames: dict, num_classes: int = 7, default: int = -1):
    numf = len(frames)
    y = np.full((numf,), default, dtype=np.int16)

    for fk_str, fr in frames.items():
        t0 = int(fk_str) - 1
        if t0 < 0 or t0 >= numf:
            continue

        if not bool(fr.get("annotated", False)):
            continue

        ids = fr.get("av_action_ids", [])
        if not ids:
            continue

        v = int(ids[0])
        if 0 <= v < num_classes:
            y[t0] = v

    return y