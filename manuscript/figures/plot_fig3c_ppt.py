#!/usr/bin/env python3
"""Figure 3c PPT version -- OOD predictions, MT only, with error bars.

Shows only multi-task (exp7a) predictions for DAC-4, TAP-2, EAP-4, and DEP (SY).
No legend, DEP labeled as "DEP (SY)".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
EXP_DIR = REPO_ROOT / "experiments"
OOD_HELDOUT_PATH = EXP_DIR / "pems_ood_heldout_exp7_all.json"
OUT_PATH = THIS_DIR / "_ppt_figure3c.png"

sys.path.insert(0, str(EXP_DIR))
from paper_plot_style import save_png_pdf, style_axes, setup_nature_style

# Colors
MT_COLOR = "#4A6274"  # exp7a color
CHARCOAL = "#2F2F2F"

# Font sizes for PPT
FS_LABEL = 11
FS_TICK = 10


def load_ood_data() -> dict:
    """Load OOD heldout comparison data."""
    with open(OOD_HELDOUT_PATH, "r") as f:
        return json.load(f)


def plot_fig3c_ppt(ax: plt.Axes, ood: dict) -> None:
    """Plot OOD predictions for MT only with error bars.
    
    Materials: DAC-4, TAP-2, EAP-4, DEP (SY)
    Only show exp7a (multi-task) predictions with model_std as error bars.
    """
    materials = ("DAC-4", "TAP-2", "EAP-4", "SY")
    display_labels = ["DAC-4", "TAP-2", "EAP-4", "DEP (SY)"]
    
    # Extract exp7a data
    exp7a_data = None
    for family in ood["families"]:
        if family["series"] == "exp7a":
            exp7a_data = family
            break
    
    if exp7a_data is None:
        raise ValueError("exp7a data not found in OOD file")
    
    # Build material lookup
    mat_dict = {m["material"]: m for m in exp7a_data["materials"]}
    
    x = np.arange(len(materials), dtype=float)
    
    # Reference values (experimental)
    ref_vals = np.array([float(mat_dict[mat]["exp_m_s"]) for mat in materials], dtype=float)
    
    # Predicted values and std
    pred_vals = np.array([float(mat_dict[mat]["pred_m_s"]) for mat in materials], dtype=float)
    pred_stds = np.array([float(mat_dict[mat]["model_std_m_s"]) for mat in materials], dtype=float)
    
    # Plot reference lines
    for xi, ref in zip(x, ref_vals):
        ax.hlines(
            ref,
            xi - 0.35,
            xi + 0.35,
            color=CHARCOAL,
            linewidth=1.2,
            zorder=2,
            label="Reference" if xi == 0 else None,
        )
    
    # Plot MT predictions with error bars
    ax.errorbar(
        x,
        pred_vals,
        yerr=pred_stds,
        fmt="o",
        color=MT_COLOR,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        elinewidth=1.0,
        capsize=3.5,
        capthick=1.0,
        zorder=4,
    )
    
    ax.set_xticks(x, display_labels)
    ax.set_ylabel(r"$V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)", fontsize=FS_LABEL)
    ax.set_ylim(6500.0, 10000.0)
    ax.set_xlim(-0.6, len(materials) - 0.4)
    ax.set_xlabel("OOD-holdout materials", fontsize=FS_LABEL)
    
    ax.tick_params(labelsize=FS_TICK)
    
    style_axes(ax, grid=True)


def main() -> None:
    setup_nature_style()
    
    ood = load_ood_data()
    
    # Single panel figure for PPT
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.94, bottom=0.14)
    
    plot_fig3c_ppt(ax, ood)
    
    save_png_pdf(fig, OUT_PATH, dpi=300)
    plt.close(fig)
    
    print(f"Saved {OUT_PATH}")
    print(f"Saved {OUT_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
