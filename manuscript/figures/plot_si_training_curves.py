#!/usr/bin/env python3
"""Fig S4 — training curves for all five folds of every adaptation variant.

Mirrors the EMA smoothing (alpha = 0.01) used by
``experiments_davis2024/plot_lcurves.py`` so the SI figure matches what the
training-side audit notebooks already show.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import TRAIN_COLOR as CANONICAL_TRAIN_COLOR, VALIDATION_COLOR, save_figure, setup_style, style_axes

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
EXP = ROOT / "experiments_davis2024"
OUT = THIS / "_si_training_curves"

VARIANTS = [
    ("MT-FT",            "exp7a", "multi"),
    ("MT-FT-aux",      "exp7b", "multi"),
    ("ST-FT",         "exp7c", "single"),
    ("ST-TFS",            "exp7d", "single"),
    ("MT-FT-crystal",       "exp8a", "multi"),
]
N_FOLDS = 5

PREFERRED_HEADS = (
    "pems_vdet_kj",
    "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis",
)

TRAIN_COLOR = CANONICAL_TRAIN_COLOR
VAL_COLOR = VALIDATION_COLOR
ALPHA_RAW   = 0.10


def ema(values: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _read_header(path: Path) -> list[str]:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("# step"):
            return line[1:].split()
    return []


def _load_curve(path: Path, kind: str):
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if kind == "single":
        if data.shape[1] < 6:
            return None
        return data[:, 0], data[:, 2], data[:, 1]
    header = _read_header(path)
    if not header or len(header) != data.shape[1]:
        return None
    idx = {name: i for i, name in enumerate(header)}
    heads = [n[len("mae_val_"):] for n in header if n.startswith("mae_val_")]
    if not heads:
        return None
    head = next((h for h in PREFERRED_HEADS if h in heads), heads[0])
    val_key, trn_key = f"mae_val_{head}", f"mae_trn_{head}"
    if val_key not in idx or trn_key not in idx:
        return None
    return data[:, idx["step"]], data[:, idx[trn_key]], data[:, idx[val_key]]


def _plot_panel(ax, label: str, exp_dir: str, kind: str) -> None:
    train_first = val_first = True
    for f in range(N_FOLDS):
        path = EXP / f"{exp_dir}_fold{f}" / "lcurve.out"
        loaded = _load_curve(path, kind)
        if loaded is None:
            continue
        step, trn, val = loaded
        cap = 1e5 if kind == "multi" else 1e4
        trn = np.clip(trn, 1, cap)
        val = np.clip(val, 1, cap)
        ax.plot(step, trn, color=TRAIN_COLOR, alpha=ALPHA_RAW, lw=0.5)
        ax.plot(step, val, color=VAL_COLOR,   alpha=ALPHA_RAW, lw=0.5)
        ax.plot(
            step, ema(trn), color=TRAIN_COLOR, lw=1.0, alpha=0.85,
            label="train MAE" if train_first else None,
        )
        ax.plot(
            step, ema(val), color=VAL_COLOR, lw=1.1,
            label="validation MAE" if val_first else None,
        )
        train_first = val_first = False

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 1e6)
    ax.set_ylim(1e1, 1e5 if kind == "multi" else 1e4)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=8)
    ax.set_ylabel("Property MAE (m$\\cdot$s$^{-1}$)", fontsize=8)
    style_axes(ax, grid=True)


def main() -> None:
    setup_style()
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, 5.6))
    axes = axes.ravel()

    for ax, (label, exp_dir, kind) in zip(axes, VARIANTS):
        _plot_panel(ax, label, exp_dir, kind)

    for j in range(len(VARIANTS), nrows * ncols):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.06),
            ncol=2, frameon=False, fontsize=8,
        )


    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(fig, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
