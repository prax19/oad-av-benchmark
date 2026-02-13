import torch

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    return device

def is_cuda_device(device) -> bool:
    return str(device).startswith("cuda")

def autocast_for(device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    dev = "cuda" if is_cuda_device(device) else str(device)
    if dev not in {"cuda", "xpu"}:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=dev, enabled=True)