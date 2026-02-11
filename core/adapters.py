import torch
from torch import nn

from utils.torch_scripts import get_device

from core.pkg_scope import use_method_src
from utils.model_scripts import patch_lstr_3072_to_2048, unwrap_logits

from yacs.config import CfgNode as CN
from typing import Protocol, Any, Dict
from pathlib import Path
import inspect

class OADMethodAdapter(Protocol):
    name: str
    def build_model(self, cfg: Dict[str, Any], num_classes, device) -> torch.nn.Module: ...
    def get_cfg(self, cfg_file: Path, gpu="0", opts=None) -> Dict[str, Any]: ...

class LSTRAdapter(OADMethodAdapter, Protocol):

    def __init__(
        self,
        repo_root: Path = Path(__file__).resolve().parents[1]
    ): ...

    def build_model(self, cfg, num_classes, device) -> torch.nn.Module:
        with use_method_src(self.module_src):
            from rekognition_online_action_detection.models import build_model
            out = build_model(cfg, get_device())

        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict) and "model" in out:
            out = out["model"]
        
        out = out.to(device)

        out = patch_lstr_3072_to_2048(out, device)
        if out.classifier.out_features != num_classes:
            in_f = out.classifier.in_features
            out.classifier = torch.nn.Linear(in_f, num_classes).to(device)

        return out
    
    def get_cfg(self, cfg_file: Path, gpu="0", opts=None):
        with use_method_src(self.module_src):
            from types import SimpleNamespace
            from rekognition_online_action_detection.config.defaults import get_cfg
            from rekognition_online_action_detection.utils.parser import assert_and_infer_cfg

            cfg = get_cfg()
            cfg.set_new_allowed(True)
            cfg.merge_from_file(str(cfg_file))
            if opts:
                cfg.merge_from_list(opts)

            args = SimpleNamespace(config_file=str(cfg_file), gpu=gpu, opts=opts or [])
            assert_and_infer_cfg(cfg, args)
            return cfg
        
    def forward_logits(self, model, x: torch.Tensor, y: torch.Tensor | None, device):
        B, T, _ = x.shape

        motion_dim = getattr(self, "motion_dim", 0)
        obj_dim    = getattr(self, "obj_dim", 0)

        use_mask        = getattr(self, "use_mask", False)
        third_arg_none  = getattr(self, "third_arg_none", False)
        tail_none       = getattr(self, "tail_none", False)

        motion = x.new_zeros((B, T, motion_dim))

        if use_mask:
            mask = x.new_zeros((B, T))
            return model(x, motion, mask)

        if third_arg_none:
            return model(x, motion, None)

        obj = x.new_zeros((B, T, obj_dim))
        if tail_none:
            return model(x, motion, obj, None)

        return model(x, motion, obj)
    
    def normalize_logits(self, logits, y: torch.Tensor, model: nn.Module, device) -> torch.Tensor:
        logits = unwrap_logits(logits)

        if not torch.is_tensor(logits):
            raise TypeError(f"{self.name}: logits is not a Tensor after unwrap: {type(logits)}")

        if logits.dim() == 3 and y.dim() == 3:
            work = y.shape[1]
            if logits.shape[1] > work:
                logits = logits[:, -work:, :]
            elif logits.shape[1] < work:
                raise ValueError(f"{self.name}: logits T={logits.shape[1]} < work={work}")

        if logits.dim() == 3 and y.dim() == 3 and logits.shape[-1] != y.shape[-1]:
            if not hasattr(model, "_adapter_head"):
                model._adapter_head = nn.Linear(logits.shape[-1], y.shape[-1]).to(device)
                print(f"[adapter] created projection head: {logits.shape[-1]} -> {y.shape[-1]}")
            logits = model._adapter_head(logits)

        return logits

class TeSTrAAdapter(LSTRAdapter):
    name = "TeSTrA"
    motion_dim = 0
    obj_dim = 0
    tail_none = True

    def __init__(
        self,
        repo_root: Path = Path(__file__).resolve().parents[1]
    ):
        self.method_root = repo_root / "methods" / "TeSTrA"
        self.module_src = self.method_root / 'src'
        self.default_cfg = self.method_root / "configs" / "THUMOS" / "TESTRA" / "testra_lite_long_512_work_8_kinetics_1x_box.yaml"
        self.default_data_info = self.method_root / "data" / "data_info.json"

class CMeRTAdapter(LSTRAdapter):
    name = "CMeRT"
    motion_dim = 1024
    third_arg_none = True

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "CMeRT"
        self.module_src = self.method_root / 'src'
        self.default_cfg = self.method_root / "configs" / "THUMOS" / "cmert_long256_work4_kinetics_1x.yaml"
        self.default_data_info = self.method_root / "data" / "data_info.json"

class MATAdapter(LSTRAdapter):
    name = "MAT"
    motion_dim = 1024
    use_mask = True

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "MAT"
        self.module_src = self.method_root / 'src'
        self.default_cfg = self.method_root / "configs" / "THUMOS" / "MAT" / "mat_long_256_work_8_kinetics_1x.yaml"
        self.default_data_info = self.method_root / "data" / "data_info.json"

class MiniROADAdapter(OADMethodAdapter):
    name = "MiniROAD"

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "MiniROAD"
        self.module_src = self.method_root
        self.default_cfg = self.method_root / "configs" / "miniroad_thumos_kinetics.yaml"
        self.default_data_info = self.method_root / "data_info" / "video_list.json"

    def build_model(self, cfg, num_classes, device):
        with use_method_src(
            self.module_src,
            pkg=None,
            purge_extra=("utils", "model", "datasets", "trainer", "criterions", "evaluation"),
            restore_extra=True,
        ):
            from model.model_builder import build_model
            out = build_model(cfg, get_device())

        out = out.to(device)
        if hasattr(out, "classifier") and out.classifier.out_features != num_classes:
            in_f = out.classifier.in_features
            out.classifier = torch.nn.Linear(in_f, num_classes).to(device)

        return out
        
    def apply_opts(self, cfg: dict, opts: list[str]) -> dict:
        if not opts:
            return cfg
        assert len(opts) % 2 == 0
        it = iter(opts)
        for k, v in zip(it, it):
            keys = k.split(".")
            d = cfg
            for kk in keys[:-1]:
                d = d.setdefault(kk, {})
            d[keys[-1]] = v
        return cfg
    
    def get_cfg(self, cfg_file: Path, opts=None):
        import yaml
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("no_rgb", False)
        cfg.setdefault("no_flow", True)

        if opts:
            cfg = self.apply_opts(cfg, opts)

        return cfg
    
    def forward_logits(self, model, x: torch.Tensor, y: torch.Tensor | None, device):
        if not hasattr(model, "_adapter_num_args"):
            params = list(inspect.signature(model.forward).parameters.values())
            model._adapter_num_args = len([p for p in params if p.name != "self"])

        n = model._adapter_num_args
        if n == 1:
            return model(x)
        if n == 2:
            return model(x, None)
        return model(x, None, None)
    
    def normalize_logits(self, logits, y: torch.Tensor, model: torch.nn.Module, device) -> torch.Tensor:
        return LSTRAdapter.normalize_logits(self, logits, y, model, device)