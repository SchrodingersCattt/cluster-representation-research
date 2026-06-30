#!/usr/bin/env python3
"""Lollipop companion for the Fig. 3d DAP-core pretrained-domain test."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import CHARCOAL, MODEL_COLORS, save_figure, setup_style, style_axes
from _uq_lollipop import draw_lollipop

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments_davis2024" / "exp_ood_pretrained_domain"
RESULTS_JSON = EXP_DIR / "pretrained_domain_results.json"
OUT = THIS_DIR / "_si_uq_pretrained_domain"

MODEL_ORDER = [
    ("pretrained_domain_mt", "MT baseline", MODEL_COLORS["exp7a"]),
    ("pretrained_domain_st", "ST pretrained", MODEL_COLORS["exp7c"]),
    ("pretrained_domain_sd", "TFS", MODEL_COLORS["exp7d"]),
]
MATERIAL_ORDER = ["DAP-1", "DAP-2", "DAP-3", "DAP-4", "DAP-5", "DAP-6", "DAP-7", "DAP-M4"]


def _load_results() -> dict[str, object]:
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


def _model_records(results: dict[str, object], model_key: str) -> list[dict[str, object]]:
    model = results[model_key]
    mats = model["dap_material_results"]
    rows: list[dict[str, object]] = []
    for material in MATERIAL_ORDER:
        rec = mats[material]
        exp = float(rec["exp_m_s"])
        per_cluster_errors = [float(v) - exp for v in rec["per_cluster_means"].values()]
        rows.append(
            {
                "material": material,
                "error": float(rec["error_m_s"]),
                "std": float(rec["grand_std_m_s"]),
                "cluster_errors": per_cluster_errors,
                "abs_error": float(rec["abs_error_m_s"]),
            }
        )
    return rows


def main() -> None:
    setup_style()
    results = _load_results()

    all_errors: list[float] = []
    all_spans: list[float] = []
    panels: list[tuple[str, str, str, list[dict[str, object]]]] = []
    for model_key, model_label, color in MODEL_ORDER:
        rows = _model_records(results, model_key)
        panels.append((model_key, model_label, color, rows))
        for row in rows:
            err = float(row["error"])
            std = float(row["std"])
            all_errors.append(err)
            all_spans.extend([err - std, err + std, *[float(v) for v in row["cluster_errors"]]])

    lim = max(120.0, float(np.nanmax(np.abs(all_spans))) * 1.18)

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.95), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.93, bottom=0.28, wspace=0.18)

    for ax, (_model_key, model_label, color, rows) in zip(axes, panels):
        values = [float(row["error"]) for row in rows]
        errors = [float(row["std"]) for row in rows]
        realizations = [list(row["cluster_errors"]) for row in rows]
        labels = [str(row["material"]) for row in rows]
        coords = draw_lollipop(
            ax,
            values,
            errors,
            labels,
            ref=0.0,
            realizations=realizations,
            color=color,
            dot_size=30,
            realization_size=12,
        )
        ax.invert_yaxis()
        ax.axvspan(-200, 200, color="#D6D6D6", alpha=0.14, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(r"Signed error (m$\cdot$s$^{-1}$)")
        ax.set_title(model_label, loc="left", fontweight="bold", pad=5)
        style_axes(ax, grid=True, grid_axis="x")
        for y, val in zip(coords, values):
            ha = "left" if val >= 0 else "right"
            offset = 0.025 * lim if val >= 0 else -0.025 * lim
            ax.text(val + offset, y, f"{abs(val):.0f}", ha=ha, va="center", fontsize=7.4, color=CHARCOAL)

    axes[0].set_ylabel("DAP test material")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=CHARCOAL,
               markersize=5.5, label="Mean signed error"),
        Line2D([0], [0], marker="o", linestyle="none", color=CHARCOAL, alpha=0.35,
               markersize=4.0, label="Cluster error"),
        Line2D([0], [0], color="#D6D6D6", lw=6, alpha=0.35, label=r"$\pm$200 m$\cdot$s$^{-1}$ band"),
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=7.8)

    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
