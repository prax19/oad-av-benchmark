import torch
import torch.nn as nn

import inspect

def patch_lstr_3072_to_2048(model: nn.Module, device) -> nn.Module:
    """
    LSTR include 2 streams. This function serves to patch method to work in RGB-only mode.
    """
    for head_name in ["feature_head_long", "feature_head_work"]:
        head = getattr(model, head_name)
        old = head.input_linear[0]

        if not isinstance(old, nn.Linear):
            raise TypeError(f"{head_name}.input_linear[0] is {type(old)}")

        if old.in_features == 2048:
            continue

        new = nn.Linear(2048, old.out_features, bias=(old.bias is not None)).to(device)

        with torch.no_grad():
            new.weight.copy_(old.weight[:, :2048])
            if old.bias is not None:
                new.bias.copy_(old.bias)

        head.input_linear[0] = new

    print("[patch] changed feature heads in_features: 3072 -> 2048")
    return model

def unwrap_logits(logits):
    preferred_keys = ("logits", "pred", "preds", "scores", "output", "outputs")
    while isinstance(logits, (list, tuple, dict)):
        if isinstance(logits, dict):
            for k in preferred_keys:
                if k in logits:
                    logits = logits[k]
                    break
            else:
                logits = next(iter(logits.values()))
        else:
            logits = logits[0]
    return logits

def call_model(model, x):
    params = list(inspect.signature(model.forward).parameters.values())
    num_args = len([p for p in params if p.name != "self"])
    if num_args == 1:
        return model(x)
    if num_args == 2:
        return model(x, None)
    return model(x, None, None)