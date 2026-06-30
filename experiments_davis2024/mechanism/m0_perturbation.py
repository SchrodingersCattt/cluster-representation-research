"""M0 -- Perturbation sensitivity from existing CV inference data.

Reads ``pems_sensitivity_summary.json`` (produced by ``infer_pems.py cv``)
and emits a compact summary plus a 2-panel figure (DeltaMAE bars + scatter).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import paths
from .plot_helpers import (
    COLORS,
    add_panel_label,
    save_figure,
    style_axes,
)


def run_m0(output_dir: Path) -> None:
    print("\n" + "=" * 60 + "\nM0: Perturbation sensitivity from existing data\n" + "=" * 60)
    data = json.loads(paths.SENSITIVITY_PATH.read_text(encoding="utf-8"))
    exp7_names = {"exp7a_5fold_cv", "exp7b_5fold_cv", "exp7c_5fold_cv", "exp7d_5fold_cv"}
    table: dict[str, dict[str, dict]] = {}
    for r in data["sensitivity_comparison"]:
        if r["experiment"] in exp7_names:
            exp = r["experiment"].replace("_5fold_cv", "")
            table.setdefault(exp, {})[r["perturbation"]] = {
                "baseline_mae": r["baseline_mae_mean"],
                "delta_mae": r["delta_mae"],
                "delta_mae_pct": r["delta_mae_pct"],
            }
    (output_dir / "mechanism_m0_results.json").write_text(json.dumps({"table": table}, indent=2))
    plot_m0(table, output_dir)


def plot_m0(table: dict, output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    exp_order = ["exp7a", "exp7b", "exp7c", "exp7d"]
    perts = ["rotation", "translation", "dap4_template"]
    pcol = {
        "rotation": COLORS["exp7a"],
        "translation": "#A85E32",
        "dap4_template": "#8D8D8D",
    }
    plab = {"rotation": "Rotation", "translation": "Translation", "dap4_template": "Template"}
    markers = {"rotation": "o", "translation": "s", "dap4_template": "D"}

    x = np.arange(len(exp_order))
    w = 0.22
    for i, pt in enumerate(perts):
        vals = [table.get(e, {}).get(pt, {}).get("delta_mae", 0) for e in exp_order]
        ax1.bar(
            x + (i - 1) * w,
            vals,
            w,
            label=plab[pt],
            color=pcol[pt],
            edgecolor="white",
            linewidth=0.5,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_order)
    ax1.set_ylabel("\u0394MAE (m/s)")
    ax1.legend(frameon=False, loc="upper left")
    ax1.axhline(0, color=COLORS["ref"], lw=1.0, ls="--")
    style_axes(ax1, grid=True)
    add_panel_label(ax1, "A")

    for e in exp_order:
        for pt in perts:
            info = table.get(e, {}).get(pt)
            if not info:
                continue
            ax2.scatter(
                info["baseline_mae"],
                info["delta_mae"],
                color=pcol[pt],
                marker=markers[pt],
                s=28,
                edgecolors="white",
                linewidths=0.4,
                zorder=4,
            )
            ax2.annotate(
                e,
                (info["baseline_mae"], info["delta_mae"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#444444",
            )
    for pt in perts:
        ax2.scatter([], [], color=pcol[pt], marker=markers[pt], s=24, label=plab[pt])
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_xlabel("Baseline MAE (m/s)")
    ax2.set_ylabel("\u0394MAE (m/s)")
    ax2.axhline(0, color=COLORS["ref"], lw=1.0, ls="--")
    style_axes(ax2, grid=True)
    add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M0_perturbation_sensitivity_main",
        supplementary=False,
        legacy_png_name="figure_m0_perturbation_sensitivity.png",
    )
    plt.close(fig)
    print("Saved Figure M0")
