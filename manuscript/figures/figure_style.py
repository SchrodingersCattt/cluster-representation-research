from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

CHARCOAL = "#2F2F2F"
MID_GRAY_DARK = "#8A8A8A"
MID_GRAY = "#D6D6D6"
LIGHT_GRAY = "#F2F2F2"
FAINT_GRID = "#ECECEC"

MODEL_COLORS = {
    "exp7a": "#4A6274",
    "exp7b": "#657217",
    "exp7c": "#7A4B58",
    "exp7d": "#6B6B6B",
    "exp8a": "#9E8B5E",
    "exp6v1": "#6E7F95",
    "zero_shot": "#C8B8A8",
    "mean": "#C8B8A8",
    "composition": "#C8B8A8",
}

MODEL_FAMILY_COLORS = {
    "MT": MODEL_COLORS["exp7a"],
    "MT aux": MODEL_COLORS["exp7b"],
    "ST pretrained": MODEL_COLORS["exp7c"],
    "TFS": MODEL_COLORS["exp7d"],
}

SITE_COLORS = {
    "A": "#8A5A67",
    "B": "#6B7C4E",
    "X": "#5A6D7B",
}

X_FAMILY_COLORS = {
    "ClO4": SITE_COLORS["X"],
    "ClO4-": SITE_COLORS["X"],
    "NO3": "#9E8B5E",
    "NO3-": "#9E8B5E",
    "IO4": "#7A6B8A",
    "IO4-": "#7A6B8A",
    "H4IO6-": "#7A6B8A",
    "IO4-family": "#7A6B8A",
    "other": MID_GRAY,
}

MATERIAL_COLORS = {
    "SY": "#3B4F7A",
    "PEP": "#A0522D",
    "MPEP": "#6B5B7B",
    "HPEP": "#2E7D6A",
    "DAP-4": "#B88A3A",
    "EAP-4": "#4E8C9A",
    "DAC-4": "#5A6D7B",
    "TAP-2": "#9E8B5E",
    "DPPE-1": "#7A4B58",
    "DAI-1_0.5 4_0.5": "#7A4B58",
}

MATERIAL_DISPLAY_LABELS = {
    "SY": "DEP",
    "SY-1": "DEP-1",
    "MPEP": "PEP-M",
    "HPEP": "PEP-H",
    "DPPE-1": r"DAI-1$_{0.5}$4$_{0.5}$",
    "DAI-1_0.5 4_0.5": r"DAI-1$_{0.5}$4$_{0.5}$",
}


def display_material(material: str) -> str:
    """Return the canonical display label for legacy material IDs."""
    return MATERIAL_DISPLAY_LABELS.get(material, material)


TRAIN_COLOR = MODEL_COLORS["exp7a"]
VALIDATION_COLOR = "#B35C44"
LINEAR_COLOR = MODEL_COLORS["exp7a"]
NONLINEAR_COLOR = VALIDATION_COLOR

ERRORBAR_KW: dict[str, Any] = {
    "elinewidth": 0.55,
    "capthick": 0.55,
    "capsize": 1.8,
    "ecolor": CHARCOAL,
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titlesize": 10,
            "figure.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.color": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "axes.edgecolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "axes.linewidth": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes, *, grid: bool = False, grid_axis: str = "y") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_edgecolor(CHARCOAL)
    ax.tick_params(direction="out", length=2.5, width=0.5, pad=2, color=CHARCOAL)
    if grid:
        if grid_axis in {"x", "both"}:
            ax.xaxis.grid(True, color=FAINT_GRID, linewidth=0.4, linestyle=":")
        if grid_axis in {"y", "both"}:
            ax.yaxis.grid(True, color=FAINT_GRID, linewidth=0.4, linestyle=":")
        ax.set_axisbelow(True)


def heatmap_text_color(cmap: Any, norm: Any, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.45 else CHARCOAL


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = 300) -> None:
    fig.savefig(Path(out_base).with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(Path(out_base).with_suffix(".png"), dpi=dpi, bbox_inches="tight")
