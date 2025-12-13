import torch

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    return device