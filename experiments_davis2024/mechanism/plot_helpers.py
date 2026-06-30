"""Mechanism-specific plotting helpers.

Style baseline imports from ``paper_plot_style`` (single source of truth).
This module only adds a thin ``save_figure`` wrapper that mirrors the
historical behaviour of writing both a legacy PNG under
``mechanism_results/`` and a paper-style PNG+PDF under ``paper_figures/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# paper_plot_style lives at experiments_davis2024/paper_plot_style.py.
# Modules in this package live one level deeper, so make sure the parent
# directory is importable when this module is imported.
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import paper_plot_style as _pp  # noqa: E402

from . import constants, paths, runtime  # noqa: E402

# Re-export canonical style helpers so experiment modules can do
#     from .plot_helpers import setup_nature_style, style_axes, add_panel_label
# without needing to know about paper_plot_style directly.
setup_nature_style = _pp.setup_nature_style
style_axes = _pp.style_axes
add_panel_label = _pp.add_panel_label
save_png_pdf = _pp.save_png_pdf

# Convenience re-exports (formerly module-level globals in run_mechanism_analysis.py).
COLORS = constants.COLORS
PERTURBATION_STYLE = constants.PERTURBATION_STYLE
disp = constants.disp


def save_figure(
    fig,
    paper_name: str,
    *,
    supplementary: bool = False,
    legacy_png_name: str | None = None,
) -> None:
    """Save a polished paper figure and optionally a legacy PNG too.

    Behaviour preserved from run_mechanism_analysis.py::save_figure:
      - When PLOT_STYLE == "paper", writes <paper_name>.png and .pdf to
        PAPER_FIG_DIR (or SUPP_FIG_DIR when supplementary=True).
      - When legacy_png_name is given, also writes that legacy PNG into
        OUTPUT_DIR for back-compat with downstream consumers.
    """
    if runtime.PLOT_STYLE == "paper":
        target_dir = paths.SUPP_FIG_DIR if supplementary else paths.PAPER_FIG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        base = target_dir / paper_name
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    if legacy_png_name is not None:
        paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.OUTPUT_DIR / legacy_png_name, dpi=300, bbox_inches="tight")


def rounded_limits(values: list[float], *, step: float = 250.0, pad: float = 0.04) -> tuple[float, float]:
    """Round axis limits outward to a multiple of ``step`` after applying padding."""
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi > lo else 1.0
    lo -= span * pad
    hi += span * pad
    return step * np.floor(lo / step), step * np.ceil(hi / step)


def plot_mean_with_individuals(
    ax,
    series_by_material: dict[str, dict[str, float]],
    *,
    mean_color: str,
    mean_label: str,
    ref_series: list[float] | None = None,
    ref_label: str | None = None,
    ref_color: str | None = None,
) -> None:
    """Plot per-material faint lines + mean curve + optional reference series."""
    if ref_color is None:
        ref_color = COLORS["kj"]
    materials = sorted(series_by_material)
    all_x = sorted(set(float(s) for v in series_by_material.values() for s in v))
    for mat in materials:
        sc = sorted(series_by_material[mat].keys(), key=float)
        ax.plot(
            [float(s) for s in sc],
            [series_by_material[mat][s] for s in sc],
            color=COLORS["ref"],
            alpha=0.25,
            lw=0.7,
        )
    mean_y = []
    for sx in all_x:
        ss = f"{sx:.2f}"
        vals = [series_by_material[m][ss] for m in materials if ss in series_by_material[m]]
        mean_y.append(float(np.mean(vals)) if vals else np.nan)
    ax.plot(all_x, mean_y, color=mean_color, lw=2.25, label=mean_label, zorder=10)
    if ref_series is not None:
        ax.plot(all_x, ref_series, color=ref_color, lw=1.2, ls="--", label=ref_label or "Reference", zorder=9)
    ax.axvline(1.0, color=COLORS["ref"], lw=1.0, ls=":")
    style_axes(ax)
