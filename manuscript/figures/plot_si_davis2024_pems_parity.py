#!/usr/bin/env python3
"""Plot Davis2024-trained zero-shot parity on PEMs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import CHARCOAL, FAINT_GRID, MID_GRAY_DARK, save_figure, setup_style, style_axes

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments_davis2024"
PRED_JSON = EXP_DIR / "davis2024_pems_zeroshot_predictions.json"
SUMMARY_JSON = EXP_DIR / "davis2024_pems_zeroshot_summary.json"
OUT = THIS_DIR / "_si_davis2024_pems_parity"

BASELINES = [
    "CHNO-only DPA crystal",
    "CHNO-only DPA molecule",
    "CHNO-only DeepEMs crystal",
    "CHNO-only DeepEMs molecule",
    "CHNO+DFT+experiment transfer",
    "CHNO+DFT transfer",
]
DISPLAY_TITLES = {
    "CHNO-only DPA crystal": "CHNO-only DPA\ncrystal input",
    "CHNO-only DPA molecule": "CHNO-only DPA\nmolecular input",
    "CHNO-only DeepEMs crystal": "CHNO-only DeepEMs\ncrystal input",
    "CHNO-only DeepEMs molecule": "CHNO-only DeepEMs\nmolecular input",
    "CHNO+DFT+experiment transfer": "CHNO+DFT+experiment\ntransfer",
    "CHNO+DFT transfer": "CHNO+DFT\ntransfer",
}
DATASET_STYLE = {
    "pems_crystal": {"label": "crystal", "marker": "o", "color": CHARCOAL},
    "pems_cluster_n1": {"label": r"cluster $n_1$", "marker": "^", "color": MID_GRAY_DARK},
}


def _axis_limits(rows: list[dict[str, object]]) -> tuple[float, float]:
    vals = []
    for row in rows:
        vals.append(float(row["ground_truth_m_s"]))
        vals.append(float(row["predicted_m_s"]))
    lo = min(vals) - 350
    hi = max(vals) + 350
    return lo, hi


def main() -> None:
    setup_style()
    rows = json.loads(PRED_JSON.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))["baselines"]
    lo, hi = _axis_limits(rows)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2), sharex=True, sharey=True)
    for ax, baseline in zip(axes.flat, BASELINES):
        ax.plot([lo, hi], [lo, hi], color=CHARCOAL, lw=0.7, ls="--", zorder=1)
        for dataset, style in DATASET_STYLE.items():
            subset = [r for r in rows if r["baseline"] == baseline and r["dataset"] == dataset]
            x = np.array([float(r["ground_truth_m_s"]) for r in subset])
            y = np.array([float(r["predicted_m_s"]) for r in subset])
            ax.scatter(
                x,
                y,
                s=20,
                marker=str(style["marker"]),
                facecolor=str(style["color"]),
                edgecolor="white",
                linewidth=0.35,
                alpha=0.78,
                zorder=3,
            )
        c_mae = summary[baseline]["datasets"]["pems_crystal"]["mae_m_s"]
        n1_mae = summary[baseline]["datasets"]["pems_cluster_n1"]["mae_m_s"]
        ax.set_title(f"{DISPLAY_TITLES[baseline]}\nMAE: crystal {c_mae:.0f}; cluster $n_1$ {n1_mae:.0f} m$\\cdot$s$^{{-1}}$", pad=4)
        style_axes(ax, grid=True, grid_axis="both")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Predicted $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Reference $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")

    handles = [
        Line2D([0], [0], marker=str(style["marker"]), linestyle="none", markersize=5.5,
               markerfacecolor=str(style["color"]), markeredgecolor="white", label=str(style["label"]))
        for style in DATASET_STYLE.values()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.045, 1, 1], h_pad=1.5, w_pad=0.8)
    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
