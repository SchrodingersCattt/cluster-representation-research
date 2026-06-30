#!/usr/bin/env python3
"""SI ABX4 OOD figure.

Two stacked panels:

 (a) ABX3 template + ClO4- counterfactual predictions for the four
     ABX4 OOD perchlorate hits (PEP / PEP-M / PEP-H / DEP) across four
     model-checkpoint families: exp6v1 (canonical single model),
     exp7a (multi-task 5-fold), exp7c (single-task pretrained 5-fold),
     exp7d (from-scratch 5-fold).

 (b) Own-CIF predictions for the same four materials under the three
     5-fold checkpoint families (exp7a / exp7c / exp7d).  The
     multi-task numbers are also reported in the main figure but are
     reproduced here so the three model variants can be compared on
     the same axis.

Sources:
  manuscript/figures/_si_abx4_ood_predictions.json
    -> ABX3-template predictions (newly run, 16 checkpoints x 4 mats x 3 variants)
  experiments_davis2024/pems_ood_5fold_exp{7a,7c,7d}.json
    -> own-CIF predictions (cached; 5 folds x 3 cluster variants per material)

Outputs: manuscript/figures/figure_si_abx4_ood.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    CHARCOAL,
    ERRORBAR_KW,
    FAINT_GRID,
    MID_GRAY_DARK,
    MODEL_COLORS,
    save_figure,
    setup_style,
    style_axes,
    display_material,
)

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
EXP_DIR = ROOT / "experiments_davis2024"

ABX3_TPL_JSON = THIS_DIR / "_si_abx4_ood_predictions.json"

OWN_JSONS = {
    "exp7a": EXP_DIR / "pems_ood_5fold_exp7a.json",
    "exp7c": EXP_DIR / "pems_ood_5fold_exp7c.json",
    "exp7d": EXP_DIR / "pems_ood_5fold_exp7d.json",
}

MATERIALS = ["PEP", "MPEP", "HPEP", "SY"]
DISPLAY_TITLES = {mat: display_material(mat) for mat in MATERIALS}

# Reference experimental V_det (m·s⁻¹; from main Fig 5 / NEW_MATERIALS table)
EXP_VDET = {"PEP": 9090, "MPEP": 8729, "HPEP": 8764, "SY": 8867}

FAMILY_ORDER_PANEL_A = ("exp6v1", "exp7a", "exp7c", "exp7d")
FAMILY_ORDER_PANEL_B = ("exp7a", "exp7c", "exp7d")

FAMILY_LABELS = {
    "exp6v1": "MT-FT-full",
    "exp7a":  "MT-FT (5-fold)",
    "exp7c":  "ST-FT (5-fold)",
    "exp7d":  "ST-TFS (5-fold)",
}

FAMILY_COLORS = {key: MODEL_COLORS[key] for key in ("exp6v1", "exp7a", "exp7c", "exp7d")}
FONT = 8.0


def _load_abx3_tpl() -> dict[str, dict[str, dict[str, float]]]:
    """Returns {family: {material: {"mean_m_s","std_m_s","n_samples"}}}."""
    blob = json.loads(ABX3_TPL_JSON.read_text(encoding="utf-8"))
    return {
        fam: {mat: dat[mat]["abx3_template"] for mat in MATERIALS}
        for fam, dat in blob["aggregated"].items()
    }


def _load_own_cif() -> dict[str, dict[str, dict[str, float]]]:
    """Returns {family: {material: {"mean_m_s","std_m_s","n_samples"}}}."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for fam, fp in OWN_JSONS.items():
        rows = json.loads(fp.read_text(encoding="utf-8"))
        idx = {row["material"]: row for row in rows if row["material"] in MATERIALS}
        per_mat: dict[str, dict[str, float]] = {}
        for mat in MATERIALS:
            row = idx.get(mat)
            if row is None:
                per_mat[mat] = {"mean_m_s": float("nan"), "std_m_s": 0.0, "n_samples": 0}
                continue
            samples: list[float] = []
            for fold_key, fold_data in row.get("predictions", {}).items():
                for cv_key in ("n1", "n2", "n3"):
                    val = fold_data.get(cv_key)
                    if isinstance(val, (int, float)):
                        samples.append(float(val))
            arr = np.asarray(samples, dtype=float)
            per_mat[mat] = {
                "mean_m_s": float(np.mean(arr)) if arr.size else float("nan"),
                "std_m_s":  float(np.std(arr)) if arr.size > 1 else 0.0,
                "n_samples": int(arr.size),
            }
        out[fam] = per_mat
    return out


def _draw_panel(
    ax: plt.Axes,
    data: dict[str, dict[str, dict[str, float]]],
    family_order: tuple[str, ...],
    title: str,
) -> None:
    n_fams = len(family_order)
    n_mats = len(MATERIALS)
    width = 0.8 / n_fams
    x_centers = np.arange(n_mats, dtype=float)

    # Bars
    for fi, fam in enumerate(family_order):
        offset = (fi - (n_fams - 1) / 2.0) * width
        means = np.array([data[fam][mat]["mean_m_s"] for mat in MATERIALS])
        stds  = np.array([data[fam][mat]["std_m_s"]  for mat in MATERIALS])
        ax.bar(
            x_centers + offset, means, width,
            yerr=stds, color=FAMILY_COLORS[fam],
            edgecolor="white", linewidth=0.5,
            error_kw=ERRORBAR_KW,
            capsize=ERRORBAR_KW["capsize"], label=FAMILY_LABELS[fam], zorder=3,
        )
    # Reference experimental V_det horizontal ticks per material
    for xi, mat in enumerate(MATERIALS):
        v = EXP_VDET[mat]
        ax.hlines(
            v, x_centers[xi] - 0.45, x_centers[xi] + 0.45,
            colors=MID_GRAY_DARK, linestyles="--", linewidth=0.8, zorder=4,
        )
        ax.text(
            x_centers[xi] - 0.43,
            v + 24,
            f"{v}",
            ha="left",
            va="bottom",
            fontsize=8.0,
            color=MID_GRAY_DARK,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=5,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels([DISPLAY_TITLES[m] for m in MATERIALS])
    ax.set_ylabel(r"Predicted $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.set_title(title, loc="left", pad=4, fontweight="bold")
    ax.set_xlim(-0.65, n_mats - 0.35)
    ax.set_ylim(7800, 10600)
    style_axes(ax)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color=FAINT_GRID)
    ax.set_axisbelow(True)


def main() -> None:
    setup_style()

    abx3_data = _load_abx3_tpl()
    own_data = _load_own_cif()

    fig_w_in, fig_h_in = 7.8, 6.8
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(
        2, 1,
        left=0.10, right=0.86, top=0.95, bottom=0.07,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    _draw_panel(
        ax_a, abx3_data, FAMILY_ORDER_PANEL_A,
        title=r"a  ABX$_3$ template + ClO$_4^-$ (DAP-4 scaffold): counterfactual chemistry",
    )
    _draw_panel(
        ax_b, own_data, FAMILY_ORDER_PANEL_B,
        title="b  Own-CIF clusters (target's experimental crystal structure)",
    )

    handles_a = [
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=7,
                      color=FAMILY_COLORS[f], label=FAMILY_LABELS[f])
        for f in FAMILY_ORDER_PANEL_A
    ]
    handles_a.append(mlines.Line2D(
        [], [], color=MID_GRAY_DARK, linestyle="--", linewidth=0.9, label="Reference $V_\\mathrm{det}$",
    ))
    fig.legend(
        handles=handles_a,
        loc="upper right", bbox_to_anchor=(0.985, 0.98),
        frameon=False, fontsize=8.0,
    )

    out_base = THIS_DIR / "figure_si_abx4_ood"
    save_figure(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


if __name__ == "__main__":
    main()
