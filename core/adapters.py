import torch
from utils.torch_scripts import get_device

from core.pkg_scope import use_method_src
from utils.model_scripts import patch_lstr_3072_to_2048

from yacs.config import CfgNode as CN
from typing import Protocol, Any, Dict
from pathlib import Path

class OADMethodAdapter(Protocol):
    name: str
    def build_model(self, cfg: Dict[str, Any], num_classes, device) -> torch.nn.Module: ...
    def get_default_cfg(self) -> Dict[str, Any]: ...

class LSTRAdapter(OADMethodAdapter, Protocol):

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]): ...

    def build_model(self, cfg, num_classes, device) -> torch.nn.Module:
        with use_method_src(self.module_src):
            from rekognition_online_action_detection.models import build_model
            out = build_model(cfg, get_device())

        if isinstance(out, tuple):
            return out[0]
        if isinstance(out, dict) and "model" in out:
            return out["model"]
        
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

class TeSTrAAdapter(LSTRAdapter):
    name = "TeSTrA"

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "TeSTrA"
        self.module_src = self.method_root / 'src'

class CMeRTAdapter(LSTRAdapter):
    name = "CMeRT"

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "CMeRT"
        self.module_src = self.method_root / 'src'

class MATAdapter(LSTRAdapter):
    name = "MAT"

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "MAT"
        self.module_src = self.method_root / 'src'

class MiniROADAdapter(OADMethodAdapter):
    name = "MiniROAD"

    def __init__(self, repo_root: Path = Path(__file__).resolve().parents[1]):
        self.method_root = repo_root / "methods" / "MiniROAD"
        self.module_src = self.method_root

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