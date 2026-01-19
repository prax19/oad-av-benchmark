from typing import Protocol, Any, Dict
import torch
from utils.torch_scripts import get_device

from pathlib import Path
from pkg_scope import use_method_src

from yacs.config import CfgNode as CN

class OADMethodAdapter(Protocol):
    name: str
    def build_model(self, cfg: Dict[str, Any]) -> torch.nn.Module: ...
    def get_default_cfg(self) -> Dict[str, Any]: ...


class TeSTrAAdapter:
    name = "TeSTrA"

    def __init__(self, repo_root: Path):
        self.method_root = repo_root / "methods" / "TeSTrA"
        self.module_src = self.method_root / 'src'

    def build_model(self, cfg) -> torch.nn.Module:
        with use_method_src(self.module_src):
            from rekognition_online_action_detection.models import build_model
            out = build_model(cfg, get_device())

        if isinstance(out, tuple):
            return out[0]
        if isinstance(out, dict) and "model" in out:
            return out["model"]
        return out
    
    def get_cfg(self, cfg_file: Path, gpu="0", opts=None):
        with use_method_src(self.module_src):
            from types import SimpleNamespace
            from rekognition_online_action_detection.config.defaults import get_cfg
            from rekognition_online_action_detection.utils.parser import assert_and_infer_cfg

            cfg = get_cfg()
            cfg.merge_from_file(str(cfg_file))
            if opts:
                cfg.merge_from_list(opts)

            args = SimpleNamespace(config_file=str(cfg_file), gpu=gpu, opts=opts or [])
            assert_and_infer_cfg(cfg, args)
            return cfg


repo_root = Path(__file__).resolve().parents[1]
adapter = TeSTrAAdapter(repo_root)

data_info = adapter.method_root / "data" / "data_info.json"
cfg = adapter.get_cfg(
    Path('methods', 'TeSTrA', 'configs', 'THUMOS', 'TESTRA', 'testra_lite_long_512_work_8_kinetics_1x_box.yaml'),
    opts=['DATA.DATA_INFO', str(data_info)]
)
model = adapter.build_model(cfg)

