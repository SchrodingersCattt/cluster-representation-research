#!/usr/bin/env python3
"""Supplementary OOD-holdout evaluation figure.

Outputs:
    manuscript/figures/_si_ood_heldout.{png,pdf}

The figure expands the aggregate hyperparameter-ablation table with a compact
visual view of per-material OOD-holdout behavior:

    a) AE heatmap across all ablation families and OOD-holdout materials, grouped by
       model family (MT, MT aux, ST pretrained, TFS).
    b) Default-learning-rate predicted-vs-reference parity scatter for the same
       model families, avoiding overplotting the full ablation sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np

from figure_style import (
    CHARCOAL,
    FAINT_GRID,
    MID_GRAY,
    MID_GRAY_DARK,
    MODEL_FAMILY_COLORS,
    heatmap_text_color,
    save_figure,
    setup_style,
    display_material,
)


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments"
SUMMARY_JSON = EXP_DIR / "ablation_full_eval_summary.json"
OUT = THIS_DIR / "_si_ood_heldout"

MATERIALS = ["DAC-4", "TAP-2", "EAP-4", "SY", "DAI-1_0.5 4_0.5"]

# Rows are grouped by model family so white separators encode model family, not
# the hyperparameter category used to create each ablation.
ROW_ORDER: list[tuple[str, str, str]] = [
    ("MT", "exp7a", "Default"),
    ("MT", "exp7a_lr1e4", r"$10^{-4}$"),
    ("MT", "exp7a_lr5e6", r"$5{\times}10^{-6}$"),
    ("MT", "exp7a_200k", "200k"),
    ("MT", "exp7a_800k", "800k"),
    ("MT", "exp7a_seed7", "seed7"),
    ("MT", "exp7a_seed13", "seed13"),
    ("MT", "exp7a_decay200", "1/200"),
    ("MT aux", "exp7b", "Default"),
    ("MT aux", "exp7b_lr1e4", r"$10^{-4}$"),
    ("MT aux", "exp7b_lr5e6", r"$5{\times}10^{-6}$"),
    ("ST pretrained", "exp7c", "Default"),
    ("ST pretrained", "exp7c_lr1e4", r"$10^{-4}$"),
    ("ST pretrained", "exp7c_seed7", "seed7"),
    ("ST pretrained", "exp7c_seed13", "seed13"),
    ("ST pretrained", "exp7c_decay200", "1/200"),
    ("TFS", "exp7d", "Default"),
    ("TFS", "exp7d_lr1e4", r"$10^{-4}$"),
]

DEFAULT_FAMILIES = {"exp7a", "exp7b", "exp7c", "exp7d"}

MODEL_COLORS = MODEL_FAMILY_COLORS

MATERIAL_MARKERS = {
    "DAC-4": "o",
    "TAP-2": "s",
    "EAP-4": "^",
    "SY": "D",
    "DAI-1_0.5 4_0.5": "X",
}


def _load_rows() -> tuple[np.ndarray, list[str], list[str], list[dict[str, object]]]:
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    families = data["families"]
    heat = np.full((len(ROW_ORDER), len(MATERIALS)), np.nan, dtype=float)
    scatter_rows: list[dict[str, object]] = []
    labels: list[str] = []
    groups: list[str] = []

    for i, (model_family, family, label) in enumerate(ROW_ORDER):
        labels.append(label)
        groups.append(model_family)
        held = families[family]["OOD_heldout"]["per_material"]
        for j, material in enumerate(MATERIALS):
            rec = held[material]
            ae = float(rec["ae"])
            pred = float(rec["pred"])
            exp = float(rec["exp"])
            heat[i, j] = max(ae, 0.1)
            scatter_rows.append(
                {
                    "model_family": model_family,
                    "family": family,
                    "label": label,
                    "material": material,
                    "pred": pred,
                    "exp": exp,
                    "ae": ae,
                    "is_default": family in DEFAULT_FAMILIES,
                }
            )
    return heat, labels, groups, scatter_rows


def _draw_group_labels(ax: plt.Axes, groups: list[str]) -> None:
    start = 0
    while start < len(groups):
        group = groups[start]
        end = start
        while end + 1 < len(groups) and groups[end + 1] == group:
            end += 1
        center = (start + end) / 2
        ax.text(
            -1.18,
            center,
            group,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.0,
            fontweight="bold",
            color=MODEL_COLORS[group],
            clip_on=False,
        )
        start = end + 1


def _draw_heatmap(ax: plt.Axes, heat: np.ndarray, labels: list[str], groups: list[str]) -> None:
    cmap = plt.get_cmap("viridis").copy()
    norm = mcolors.LogNorm(vmin=1.0, vmax=max(heat.max(), 1500.0))
    im = ax.imshow(heat, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(MATERIALS)))
    ax.set_xticklabels([display_material(m) for m in MATERIALS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.0)
    ax.tick_params(length=0)
    ax.set_title("a  OOD-holdout AE heatmap", loc="left", fontweight="bold", pad=5)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            color = heatmap_text_color(cmap, norm, val)
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8.0, color=color)

    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=1.4)
    _draw_group_labels(ax, groups)

    # Separate the double-perovskite fail case visually because it is excluded from aggregate MAE.
    ax.axvline(3.5, color=CHARCOAL, lw=0.8)
    ax.text(
        4,
        -1.15,
        "fail case",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=CHARCOAL,
    )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.045)
    cbar.set_label("AE (m$\\cdot$s$^{-1}$, log scale)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def _draw_parity(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    rows = [row for row in rows if bool(row["is_default"])]
    x_vals = np.array([float(r["exp"]) for r in rows], dtype=float)
    y_vals = np.array([float(r["pred"]) for r in rows], dtype=float)
    lo = min(x_vals.min(), y_vals.min()) - 250
    hi = max(x_vals.max(), y_vals.max()) + 250

    ax.plot([lo, hi], [lo, hi], color=CHARCOAL, lw=0.8, zorder=1)
    ax.plot([lo, hi], [lo + 200, hi + 200], color=MID_GRAY_DARK, lw=0.7, ls="--", zorder=1)
    ax.plot([lo, hi], [lo - 200, hi - 200], color=MID_GRAY_DARK, lw=0.7, ls="--", zorder=1)

    for row in rows:
        model_family = str(row["model_family"])
        material = str(row["material"])
        ax.scatter(
            float(row["exp"]),
            float(row["pred"]),
            s=30 if material != "DAI-1_0.5 4_0.5" else 36,
            marker=MATERIAL_MARKERS[material],
            facecolor=MODEL_COLORS[model_family],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.88,
            zorder=3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Reference $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.set_ylabel(r"Predicted $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.grid(True, color=FAINT_GRID, lw=0.45, zorder=0)
    ax.set_title("b  Default-LR OOD-holdout parity", loc="left", fontweight="bold", pad=5)
    ax.text(
        0.03,
        0.97,
        "20 points\n4 model families × 5 materials",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=MID_GRAY, alpha=0.92),
    )
    ax.annotate(
        r"$\pm$200 m$\cdot$s$^{-1}$",
        xy=(0.76, 0.70),
        xycoords="axes fraction",
        xytext=(0.96, 0.08),
        textcoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=MID_GRAY_DARK,
        arrowprops=dict(
            arrowstyle="-",
            color=MID_GRAY_DARK,
            lw=0.55,
            shrinkA=2,
            shrinkB=1,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
        zorder=4,
    )

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.8,
               markerfacecolor=color, markeredgecolor="white", label=model_family)
        for model_family, color in MODEL_COLORS.items()
    ]
    material_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", markersize=5.5,
               markerfacecolor=MID_GRAY, markeredgecolor="white", label=display_material(material))
        for material, marker in MATERIAL_MARKERS.items()
    ]
    leg1 = ax.legend(
        handles=model_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.19),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=material_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.40),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )


def main() -> None:
    setup_style()
    heat, labels, groups, rows = _load_rows()
    fig = plt.figure(figsize=(8.8, 7.9))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.22, 1.0],
        left=0.12,
        right=0.985,
        top=0.95,
        bottom=0.24,
        wspace=0.50,
    )
    _draw_heatmap(fig.add_subplot(gs[0, 0]), heat, labels, groups)
    _draw_parity(fig.add_subplot(gs[0, 1]), rows)
    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
