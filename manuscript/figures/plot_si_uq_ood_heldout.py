#!/usr/bin/env python3
"""Lollipop companion for the Extended Data OOD-holdout AE heatmap."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import CHARCOAL, MATERIAL_COLORS, save_figure, setup_style, style_axes, display_material
from _uq_lollipop import draw_lollipop

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments_davis2024"
SUMMARY_JSON = EXP_DIR / "ablation_full_eval_summary.json"
OUT = THIS_DIR / "_si_uq_ood_heldout"

MATERIALS = ["DAC-4", "TAP-2", "EAP-4", "SY", "DAI-1_0.5 4_0.5"]
ROW_ORDER = [
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


def _load_records() -> list[dict[str, object]]:
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    families = data["families"]
    records: list[dict[str, object]] = []
    for material in MATERIALS:
        aes: list[float] = []
        labels: list[str] = []
        for model_family, family_key, label in ROW_ORDER:
            rec = families[family_key]["OOD_heldout"]["per_material"][material]
            aes.append(float(rec["ae"]))
            labels.append(f"{model_family} {label}")
        values = np.asarray(aes, dtype=float)
        records.append(
            {
                "material": material,
                "label": display_material(material),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "aes": aes,
                "labels": labels,
            }
        )
    return records


def main() -> None:
    setup_style()
    records = _load_records()

    values = [float(r["mean"]) for r in records]
    errors = [float(r["std"]) for r in records]
    labels = [str(r["label"]) for r in records]
    realizations = [list(r["aes"]) for r in records]
    colors = [MATERIAL_COLORS.get(str(r["material"]), CHARCOAL) for r in records]

    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    fig.subplots_adjust(left=0.26, right=0.965, top=0.93, bottom=0.30)

    coords = draw_lollipop(
        ax,
        values,
        errors,
        labels,
        ref=0.0,
        realizations=realizations,
        color=colors,
        dot_size=34,
        realization_size=11,
    )
    ax.invert_yaxis()
    ax.axvspan(0, 200, color="#D6D6D6", alpha=0.16, zorder=0)

    xmax = max(max(v + e for v, e in zip(values, errors)), max(max(row) for row in realizations))
    ax.set_xlim(0, xmax * 1.12)
    ax.set_xlabel(r"Absolute error (m$\cdot$s$^{-1}$)")

    for y, value in zip(coords, values):
        ax.text(value + xmax * 0.018, y, f"{value:.0f}", ha="left", va="center", fontsize=7.8, color=CHARCOAL)

    style_axes(ax, grid=True, grid_axis="x")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=CHARCOAL,
               markersize=5.5, label="Mean AE"),
        Line2D([0], [0], marker="o", linestyle="none", color=CHARCOAL, alpha=0.35,
               markersize=4.0, label="Ablation cell AE"),
        Line2D([0], [0], color="#D6D6D6", lw=6, alpha=0.35, label=r"$\leq$200 m$\cdot$s$^{-1}$ band"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, -0.18), ncol=3, frameon=False, fontsize=7.8)

    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
