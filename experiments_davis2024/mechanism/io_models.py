"""Checkpoint resolution and model loaders for M-series experiments."""
from __future__ import annotations

from pathlib import Path

from . import constants, paths, runtime


def get_latest_ckpt(exp_dir: Path) -> Path | None:
    """Prefer model.ckpt.pt symlink; fall back to highest-numbered ckpt."""
    final = exp_dir / "model.ckpt.pt"
    if final.exists():
        return final
    ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def get_model_exp_dir(model_name: str, fold_id: int) -> str:
    """Checkpoint directory name, e.g. exp7a_fold2."""
    if model_name not in constants.MODEL_CONFIGS:
        raise KeyError(model_name)
    return f"{model_name}_fold{fold_id}"


def resolve_ckpt(exp_dir: Path) -> Path:
    """Prefer model.ckpt-{CKPT_STEP}.pt when CKPT_STEP is set; else latest."""
    if runtime.CKPT_STEP is not None:
        pinned = exp_dir / f"model.ckpt-{runtime.CKPT_STEP}.pt"
        if pinned.exists():
            return pinned
        print(f"  Warning: pinned checkpoint {pinned.name} missing, falling back to latest")
    ckpt = get_latest_ckpt(exp_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {exp_dir}")
    return ckpt


def load_property_model(model_name: str, fold_id: int | None = None, *, no_jit: bool = False):
    from deepmd.pt.infer.deep_eval import DeepProperty
    cfg = constants.MODEL_CONFIGS[model_name]
    sub = get_model_exp_dir(model_name, fold_id) if fold_id is not None else cfg["exp_dir"]
    ckpt = resolve_ckpt(paths.ROOT / sub)
    kwargs = {"head": cfg["property_head"]} if cfg["property_head"] else {}
    if no_jit:
        kwargs["no_jit"] = True
    model = DeepProperty(str(ckpt), **kwargs)
    print(f"Loaded property model {model_name} from {ckpt.name}" + (" (no_jit)" if no_jit else ""))
    return model


def load_descriptor_model(model_name: str, fold_id: int | None = None):
    from deepmd.infer import DeepPot
    cfg = constants.MODEL_CONFIGS[model_name]
    sub = get_model_exp_dir(model_name, fold_id) if fold_id is not None else cfg["exp_dir"]
    ckpt = resolve_ckpt(paths.ROOT / sub)
    kwargs = {"head": cfg["descriptor_head"]} if cfg["descriptor_head"] else {}
    dp = DeepPot(str(ckpt), **kwargs)
    print(f"Loaded descriptor model {model_name} from {ckpt.name}")
    return dp
