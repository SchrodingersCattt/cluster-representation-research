from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Canonical color palette — single source of truth for all figures.
# ---------------------------------------------------------------------------
EXP_COLORS = {
    # Experiment model colors
    "exp7a": "#205C77",   # dark teal  — multi-task baseline
    "exp7b": "#657217",   # olive      — multi-task auxiliary-head variant
    "exp7c": "#931143",   # crimson    — single-task pretrained variant
    "exp7d": "#474747",   # dark gray  — single-task scratch baseline
    # Reference / neutral
    "gray":       "#474747",
    "ref":        "#474747",
    "mid_gray":   "#D6D6D6",
    "light_gray": "#F2F2F2",
    # Nature-quality additions
    "charcoal":   "#2F2F2F",   # softer-than-black for text / axis elements
    "faint_grid": "#ECECEC",   # extremely faint grid lines (0.4 pt dotted)
}

# ---------------------------------------------------------------------------
# Failure conditions for style consistency (see AGENTS.md):
#   1. exp7a ≠ #205C77, exp7c ≠ #931143, or exp7b ≠ #657217
#   2. Font family is not Arial/DejaVu Sans
#   3. Top or right spines are visible
#   4. Panel labels are missing or not bold uppercase
#   5. PDF fonttype ≠ 42
# ---------------------------------------------------------------------------


def setup_nature_style() -> None:
    """Apply Nature-journal rcParams.

    Key choices (aligned with INVAR FIGURE.md reference):
    - Axis labels 8 pt, tick labels 7 pt, legend 6.5 pt  (down from 9/8/7.5)
    - Axis linewidth 0.5 pt, tick width 0.5 pt           (down from 0.8)
    - Charcoal #2F2F2F for all text / axis elements       (softer than pure black)
    - PDF fonttype 42 (vector font embedding)
    """
    plt.rcParams.update(
        {
            # --- Font ---
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            # --- Background ---
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # --- Font sizes (Nature-quality: compact, information-dense) ---
            "axes.titlesize": 9,
            "figure.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            # --- Charcoal text / axis color (softer than pure black) ---
            "text.color":        "#2F2F2F",
            "axes.labelcolor":   "#2F2F2F",
            "axes.edgecolor":    "#2F2F2F",
            "xtick.color":       "#2F2F2F",
            "ytick.color":       "#2F2F2F",
            # --- Axis lines & ticks (thinner = more elegant) ---
            "axes.linewidth":    0.5,
            "xtick.direction":   "out",
            "ytick.direction":   "out",
            "xtick.major.size":  2.5,
            "ytick.major.size":  2.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            # --- Font embedding (vector PDF) ---
            "pdf.fonttype": 42,
            "ps.fonttype":  42,
        }
    )


def style_axes(ax, grid: bool = False) -> None:
    """Remove top/right spines; set charcoal linewidth; optional faint grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_edgecolor(EXP_COLORS["charcoal"])
    ax.spines["bottom"].set_edgecolor(EXP_COLORS["charcoal"])
    ax.tick_params(direction="out", length=2.5, width=0.5, pad=2,
                   color=EXP_COLORS["charcoal"])
    if grid:
        ax.yaxis.grid(
            True,
            color=EXP_COLORS["faint_grid"],
            linewidth=0.4,
            linestyle=":",
        )
        ax.set_axisbelow(True)


def add_panel_label(ax, label: str) -> None:
    """Bold lowercase panel label (a, b, c …) at top-left of axes."""
    ax.text(
        -0.12,
        1.03,
        label.lower(),
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=EXP_COLORS["charcoal"],
        ha="left",
        va="bottom",
    )


def save_png_pdf(fig, png_path: Path, *, dpi: int = 300) -> None:
    png_path = Path(png_path)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
