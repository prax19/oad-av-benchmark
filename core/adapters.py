import torch
from torch import nn

from utils.torch_scripts import get_device

from core.pkg_scope import use_method_src
from utils.model_scripts import patch_lstr_3072_to_2048, unwrap_logits

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

    def _default_cfg_opts(self) -> list[str]:
        return list(getattr(self, "cfg_default_opts", []))

    def recommend_dataset_window(self) -> dict[str, int] | None:
        return None

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

            merged_opts: list[str] = []
            merged_opts.extend(self._default_cfg_opts())
            if opts:
                merged_opts.extend(opts)
            if merged_opts:
                cfg.merge_from_list(merged_opts)

            args = SimpleNamespace(config_file=str(cfg_file), gpu=gpu, opts=merged_opts)
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
        repo_root: Path = Path(__file__).resolve().parents[1],
        fps: int = 8,
        long_memory_seconds: int = 16,
        long_memory_sample_rate: int = 1,
        work_memory_seconds: int = 2,
        work_memory_sample_rate: int = 4,
    ):
        self.method_root = repo_root / "methods" / "TeSTrA"
        self.module_src = self.method_root / 'src'
        self.default_cfg = self.method_root / "configs" / "THUMOS" / "TESTRA" / "testra_lite_long_512_work_8_kinetics_1x_box.yaml"
        self.default_data_info = self.method_root / "data" / "data_info.json"

        def _as_positive_int(name: str, value: Any) -> int:
            try:
                f = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.name}: {name} must be a positive integer, got {value!r}") from exc

            if not f.is_integer() or int(f) <= 0:
                raise ValueError(f"{self.name}: {name} must be a positive integer, got {value!r}")
            return int(f)

        self.fps = _as_positive_int("fps", fps)
        self.long_memory_seconds = _as_positive_int("long_memory_seconds", long_memory_seconds)
        self.long_memory_sample_rate = _as_positive_int("long_memory_sample_rate", long_memory_sample_rate)
        self.work_memory_seconds = _as_positive_int("work_memory_seconds", work_memory_seconds)
        self.work_memory_sample_rate = _as_positive_int("work_memory_sample_rate", work_memory_sample_rate)

        long_memory_length = self.long_memory_seconds * self.fps
        if long_memory_length % self.long_memory_sample_rate != 0:
            raise ValueError(
                f"{self.name}: invalid long-memory timing: "
                f"LONG_MEMORY_LENGTH={long_memory_length} (seconds={self.long_memory_seconds}, fps={self.fps}) "
                f"is not divisible by LONG_MEMORY_SAMPLE_RATE={self.long_memory_sample_rate}."
            )

        work_memory_length = self.work_memory_seconds * self.fps
        if work_memory_length % self.work_memory_sample_rate != 0:
            raise ValueError(
                f"{self.name}: invalid work-memory timing: "
                f"WORK_MEMORY_LENGTH={work_memory_length} (seconds={self.work_memory_seconds}, fps={self.fps}) "
                f"is not divisible by WORK_MEMORY_SAMPLE_RATE={self.work_memory_sample_rate}."
            )

        self.cfg_default_opts = [
            "DATA.FPS", str(self.fps),
            "MODEL.LSTR.LONG_MEMORY_SECONDS", str(self.long_memory_seconds),
            "MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE", str(self.long_memory_sample_rate),
            "MODEL.LSTR.WORK_MEMORY_SECONDS", str(self.work_memory_seconds),
            "MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE", str(self.work_memory_sample_rate),
        ]

    def recommend_dataset_window(self) -> dict[str, int] | None:
        fps = max(1, self.fps)
        long_sr = max(1, self.long_memory_sample_rate)
        long_steps = max(1, int(round((self.long_memory_seconds * fps) / long_sr)))
        # Keep full work horizon in frame units for targets.
        work_steps = max(1, int(round(self.work_memory_seconds * fps)))
        stride = max(1, work_steps // 2)
        return {"dataset_long": long_steps, "dataset_work": work_steps, "dataset_stride": stride}

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
