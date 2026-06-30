#!/usr/bin/env python3
"""Lollipop companion for the Extended Data periodic-control heatmap."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import CHARCOAL, FAINT_GRID, MID_GRAY_DARK, MODEL_COLORS, save_figure, setup_style, style_axes
from _uq_lollipop import draw_lollipop

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments"
CROSS_JSON = EXP_DIR / "cross_infer_rep.json"
OUT = THIS_DIR / "_si_uq_periodic_control"

FAMILIES = [
    ("MT baseline", "exp7a"),
    ("MT aux", "exp7b"),
    ("ST pretrained", "exp7c"),
    ("TFS", "exp7d"),
    ("Periodic-crystal control", "exp8a"),
]
INPUTS = [
    (r"Cluster input ($n_1$)", "cluster_n1"),
    ("Periodic crystal input", "crystal"),
]


def _load_records() -> list[dict[str, object]]:
    cross = json.loads(CROSS_JSON.read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []
    for family_label, family_key in FAMILIES:
        for input_label, input_key in INPUTS:
            rec = cross[family_key][input_key]
            fold_maes = [float(v) for v in rec["fold_maes"]]
            records.append(
                {
                    "family_label": family_label,
                    "family_key": family_key,
                    "input_label": input_label,
                    "input_key": input_key,
                    "label": f"{family_label} | {input_label}",
                    "mean": float(rec["mean_mae"]),
                    "std": float(rec["std_mae"]),
                    "fold_maes": fold_maes,
                }
            )
    return records


def main() -> None:
    setup_style()
    records = _load_records()

    values = [float(r["mean"]) for r in records]
    errors = [float(r["std"]) for r in records]
    labels = [str(r["label"]) for r in records]
    realizations = [list(r["fold_maes"]) for r in records]
    colors = [MODEL_COLORS[str(r["family_key"])] for r in records]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.subplots_adjust(left=0.37, right=0.96, top=0.94, bottom=0.16)

    coords = draw_lollipop(
        ax,
        values,
        errors,
        labels,
        ref=0.0,
        realizations=realizations,
        color=colors,
        dot_size=32,
        realization_size=12,
    )
    ax.invert_yaxis()

    for boundary in np.arange(1.5, len(records) - 0.5, 2.0):
        ax.axhline(boundary, color=FAINT_GRID, lw=0.7, zorder=0)

    xmax = max(max(v + e for v, e in zip(values, errors)), max(max(r) for r in realizations))
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel(r"Five-fold MAE (m$\cdot$s$^{-1}$)")

    for y, value in zip(coords, values):
        ax.text(value + xmax * 0.018, y, f"{value:.0f}", ha="left", va="center", fontsize=7.8, color=CHARCOAL)

    style_axes(ax, grid=True, grid_axis="x")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=CHARCOAL,
               markersize=5.5, label="Mean MAE"),
        Line2D([0], [0], marker="o", linestyle="none", color=CHARCOAL, alpha=0.35,
               markersize=4.0, label="Fold MAE"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
