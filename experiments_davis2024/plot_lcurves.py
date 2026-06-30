#!/usr/bin/env python3
"""
Plot training curves from DeepMD lcurve.out files.

Auto-detects which experiments have lcurve.out and plots them.
Handles single-task lcurve files plus multi-head DeePMD logs by parsing the
header row to find the property head that actually has validation metrics.
Log-log scale, EMA smoothed, tiled subplots.

Output: [`experiments_davis2024/training_curves.png`](experiments_davis2024/training_curves.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_plot_style import EXP_COLORS, save_png_pdf, setup_nature_style, style_axes

ROOT = Path(__file__).resolve().parent

# Experiment registry — maps exp_dir_name → (label, lcurve_type)
# lcurve_type: "single" = 6-col property lcurve
#              "multi"  = multi-head DeePMD lcurve with header-based parsing
EXP_DIRS: dict[str, tuple[str, str]] = {
    "exp1a_crystal_dpa32":   ("Exp1a: Crystal + DPA-3.2", "single"),
    "exp1b_molecule_dpa32":  ("Exp1b: Molecule + DPA-3.2", "single"),
    # exp2_theory2exp_dpa32 — deprecated (renamed to _exp2_theory2exp_dpa32)
    "exp3a_crystal_deepems": ("Exp3a: Crystal + deepems-lam", "single"),
    "exp3b_molecule_deepems":("Exp3b: Molecule + deepems-lam", "single"),
    # exp4a_multitask_deepems — deprecated (renamed to _exp4a_multitask_deepems)
    # exp4b_multitask_deepems — deprecated (renamed to _exp4b_multitask_deepems)
    "exp4c_multitask_deepems_mol":("Exp4c: MT deepems mol (3-head)", "multi"),
    "exp4d_multitask_deepems_mol":("Exp4d: MT deepems mol (2-head)", "multi"),
    **{f"exp5v1_fold{i}": (f"Exp5v1 Fold{i}: PEMs 2-head ablation", "multi") for i in range(5)},
    **{f"exp5v2_fold{i}": (f"Exp5v2 Fold{i}: PEMs 3-head CV", "multi") for i in range(5)},
    **{f"exp5v2fair_fold{i}": (f"Exp5v2fair Fold{i}: PEMs 3-head epoch-eq", "multi") for i in range(5)},
    **{f"exp5v3_fold{i}": (f"Exp5v3 Fold{i}: PEMs single-task finetune", "single") for i in range(5)},
    **{f"exp5v4b_fold{i}": (f"Exp5v4b Fold{i}: David wt=0.10", "multi") for i in range(5)},
    **{f"exp5v4d_fold{i}": (f"Exp5v4d Fold{i}: David wt=0.40", "multi") for i in range(5)},
    **{f"exp5v4e_fold{i}": (f"Exp5v4e Fold{i}: David wt=0.45", "multi") for i in range(5)},
    **{f"exp5v5_fold{i}": (f"Exp5v5 Fold{i}: PEMs train-from-scratch", "single") for i in range(5)},
    **{f"exp7a_fold{i}": (f"Exp7a Fold{i}: 2-head unified-LR", "multi") for i in range(5)},
    **{f"exp7b_fold{i}": (f"Exp7b Fold{i}: 3-head unified-LR", "multi") for i in range(5)},
    **{f"exp7c_fold{i}": (f"Exp7c Fold{i}: single-task finetune unified-LR", "single") for i in range(5)},
    **{f"exp7d_fold{i}": (f"Exp7d Fold{i}: from-scratch unified-LR", "single") for i in range(5)},
    **{f"ablation/exp7a_lr5e6_fold{i}": (f"Exp7a-lr5e6 Fold{i}: 2-head LR=5e-6", "multi") for i in range(5)},
    **{f"ablation/exp7b_lr5e6_fold{i}": (f"Exp7b-lr5e6 Fold{i}: 3-head LR=5e-6", "multi") for i in range(5)},
    **{f"ablation/exp7a_lr1e4_fold{i}": (f"Exp7a-lr1e4 Fold{i}: 2-head LR=1e-4", "multi") for i in range(5)},
    **{f"ablation/exp7b_lr1e4_fold{i}": (f"Exp7b-lr1e4 Fold{i}: 3-head LR=1e-4", "multi") for i in range(5)},
    **{f"ablation/exp7c_lr1e4_fold{i}": (f"Exp7c-lr1e4 Fold{i}: single-task LR=1e-4", "single") for i in range(5)},
    **{f"ablation/exp7d_lr1e4_fold{i}": (f"Exp7d-lr1e4 Fold{i}: from-scratch LR=1e-4", "single") for i in range(5)},
    "ablation/exp7a_200k": ("Exp7a-200k: 2-head numb_steps=200k (decay 2.5k)", "multi"),
    "ablation/exp7a_800k": ("Exp7a-800k: 2-head numb_steps=800k (decay 10k)", "multi"),
    "exp6v1_allpems": ("Exp6v1: All PEMs (2-head)", "multi"),
    "exp6_allpems": ("Exp6: All PEMs (3-head)", "multi"),
}

PREFERRED_PROP_HEADS = (
    "pems_vdet_kj",
    "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis",
)

HEAD_LABELS = {
    "pems_vdet_kj": "PEMs vdet (kJ)",
    "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis": "David2024 vdet (kJ)",
    "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis": "David2024 vdet (exp)",
}

SINGLE_TASK_LABEL = "single-task property"

LEGACY_COLOR_FALLBACKS = {
    "blue": "exp7a",
    "orange": "exp7c",
    "teal": "exp7b",
}


def _color(name: str) -> str:
    """Support older plotting color keys against the canonical palette."""
    if name in EXP_COLORS:
        return EXP_COLORS[name]
    fallback = LEGACY_COLOR_FALLBACKS.get(name)
    if fallback and fallback in EXP_COLORS:
        return EXP_COLORS[fallback]
    return EXP_COLORS["charcoal"]


def ema(values: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Exponential moving average."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def load_lcurve_single(path: Path) -> dict[str, np.ndarray] | None:
    """Load single-task lcurve.out (6 columns):
    step | mae_val | mae_trn | rmse_val | rmse_trn | lr
    """
    if not path.exists():
        return None
    try:
        data = np.loadtxt(path, comments="#")
    except Exception:
        return None
    if data.ndim != 2 or len(data) < 5 or data.shape[1] < 6:
        return None
    return {
        "step":     data[:, 0],
        "mae_val":  data[:, 1],
        "mae_trn":  data[:, 2],
        "lr":       data[:, 5],
    }


def _read_lcurve_header(path: Path) -> list[str]:
    """Return the tokenized '# step ...' header line."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("# step"):
                    return line[1:].strip().split()
    except OSError:
        return []
    return []


def _summarize_validation_systems(systems: list[str]) -> str:
    """Return a compact human-readable description of validation_data.systems."""
    if not systems:
        return "validation_data (0 systems)"

    parent_names = {Path(s).parent.name for s in systems}
    n_sys = len(systems)

    if parent_names == {"crystal_systems"}:
        return f"crystal_systems ({n_sys} systems)"
    if parent_names == {"molecule_systems"}:
        return f"molecule_systems ({n_sys} systems)"
    if parent_names == {"exp_val_systems"}:
        return f"exp_val_systems ({n_sys} systems)"
    if parent_names <= {"pems_cluster_n1_systems", "pems_cluster_n2_systems", "pems_cluster_n3_systems"}:
        return f"held-out PEMs clusters n1/n2/n3 ({n_sys} systems)"
    if all("dataset_vanilla/train_cleaned" in s for s in systems):
        return f"deepems DFT validation ({n_sys} systems)"
    if len(parent_names) == 1:
        return f"{next(iter(parent_names))} ({n_sys} systems)"
    return f"validation_data ({n_sys} systems)"


def _load_monitor_metadata(exp_name: str, lcurve_type: str, prop_head: str | None) -> dict[str, str]:
    """Describe which head and validation set the plotted val curve comes from."""
    input_path = ROOT / exp_name / "input.json"
    if not input_path.exists():
        return {"head_label": SINGLE_TASK_LABEL if lcurve_type == "single" else "property head", "val_desc": "validation_data"}

    cfg = json.loads(input_path.read_text(encoding="utf-8"))
    if lcurve_type == "single":
        val_systems = cfg.get("training", {}).get("validation_data", {}).get("systems", [])
        return {
            "head_label": SINGLE_TASK_LABEL,
            "val_desc": _summarize_validation_systems(val_systems),
        }

    data_dict = cfg.get("training", {}).get("data_dict", {})
    head_key = prop_head if prop_head in data_dict else None
    if head_key is None:
        for candidate in PREFERRED_PROP_HEADS:
            if candidate in data_dict:
                head_key = candidate
                break
    if head_key is None and data_dict:
        head_key = next(iter(data_dict))

    head_cfg = data_dict.get(head_key or "", {})
    val_systems = head_cfg.get("validation_data", {}).get("systems", [])
    return {
        "head_label": HEAD_LABELS.get(head_key or "", head_key or "property head"),
        "val_desc": _summarize_validation_systems(val_systems),
    }


def load_lcurve_multi(path: Path) -> dict[str, np.ndarray] | None:
    """Load a multi-head lcurve.out by parsing metric names from the header."""
    if not path.exists():
        return None
    header = _read_lcurve_header(path)
    if not header:
        return None
    try:
        data = np.loadtxt(path, comments="#")
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or len(data) < 5 or data.shape[1] != len(header):
        return None

    col_idx = {name: i for i, name in enumerate(header)}
    candidate_heads = [
        name[len("mae_val_"):]
        for name in header
        if name.startswith("mae_val_") and f"mae_trn_{name[len('mae_val_'):]}" in col_idx
    ]
    if not candidate_heads:
        return None

    prop_head = next((head for head in PREFERRED_PROP_HEADS if head in candidate_heads), candidate_heads[0])
    mae_val_key = f"mae_val_{prop_head}"
    mae_trn_key = f"mae_trn_{prop_head}"
    if mae_val_key not in col_idx or mae_trn_key not in col_idx:
        return None

    lr_key = "lr" if "lr" in col_idx else header[-1]
    return {
        "step": data[:, col_idx["step"]],
        "rmse_e_trn": data[:, col_idx["rmse_e_trn_deepems_vanilla"]] if "rmse_e_trn_deepems_vanilla" in col_idx else None,
        "rmse_f_trn": data[:, col_idx["rmse_f_trn_deepems_vanilla"]] if "rmse_f_trn_deepems_vanilla" in col_idx else None,
        "mae_val": data[:, col_idx[mae_val_key]],
        "mae_trn": data[:, col_idx[mae_trn_key]],
        "lr": data[:, col_idx[lr_key]],
        "prop_head": prop_head,
        "prop_head_label": HEAD_LABELS.get(prop_head, prop_head),
    }


def main() -> None:
    setup_nature_style()
    # Collect available experiments
    available = {}
    for exp_name, (label, lcurve_type) in EXP_DIRS.items():
        lcurve_path = ROOT / exp_name / "lcurve.out"
        if lcurve_type == "multi":
            data = load_lcurve_multi(lcurve_path)
        else:
            data = load_lcurve_single(lcurve_path)
        if data is not None:
            available[exp_name] = (label, lcurve_type, data)

    if not available:
        print("No lcurve.out files found with sufficient data. Nothing to plot.")
        return

    n = len(available)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)

    for idx, (exp_name, (label, lcurve_type, data)) in enumerate(available.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        step = data["step"]
        trn = data["mae_trn"]
        val = data["mae_val"]

        # Clip extreme values for multi-task (oscillation can reach very large)
        if lcurve_type == "multi":
            trn = np.clip(trn, 0, 1e5)
            val = np.clip(val, 0, 1e5)

        alpha = 0.01
        trn_smooth = ema(trn, alpha)
        val_smooth = ema(val, alpha)
        monitor_meta = _load_monitor_metadata(exp_name, lcurve_type, data.get("prop_head"))

        ax.plot(step, trn, alpha=0.12, color=_color("blue"))
        ax.plot(step, val, alpha=0.12, color=_color("orange"))
        ax.plot(step, trn_smooth, color=_color("blue"), linewidth=1.4, label="Train MAE")
        ax.plot(step, val_smooth, color=_color("orange"), linewidth=1.4, label="Val MAE")

        # For multi-task, also show energy RMSE on secondary axis
        if lcurve_type == "multi" and data.get("rmse_e_trn") is not None:
            ax2 = ax.twinx()
            rmse_e = np.clip(data["rmse_e_trn"], 0, 1e3)
            ax2.plot(step, ema(rmse_e, alpha), color=_color("teal"), linewidth=1,
                     linestyle="--", alpha=0.7, label="Ener RMSE trn")
            ax2.set_ylabel("Energy RMSE (eV/atom)", color=_color("teal"), fontsize=8)
            ax2.tick_params(axis="y", labelcolor=_color("teal"), labelsize=7)
            ax2.spines["top"].set_visible(False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e2, max(step.max() * 2, 1e6))
        ax.set_ylim(1e1, 1e5 if lcurve_type == "multi" else 1e4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Property MAE (m/s)")
        title_suffix = f"step {int(step[-1])}, val MAE {val_smooth[-1]:.0f}"
        ax.set_title(f"{label}\n{monitor_meta['head_label']} | {monitor_meta['val_desc']}\n{title_suffix}")
        ax.legend(fontsize=8, loc="upper right")
        style_axes(ax, grid=True)

        if lcurve_type == "multi":
            ax.text(0.02, 0.05,
                    "NOTE: Multi-task oscillation normal\n(property head trains ~50% of steps)",
                    transform=ax.transAxes, fontsize=7, color="gray",
                    va="bottom", ha="left")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("David2024 training curves", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = ROOT / "training_curves.png"
    save_png_pdf(fig, out_path, dpi=300)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
