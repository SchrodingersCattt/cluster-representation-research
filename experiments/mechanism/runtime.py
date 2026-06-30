"""Mutable runtime configuration set by run_mechanism_analysis.py::main().

Modules in this package import these as ``from mechanism import runtime``
and reference ``runtime.CKPT_STEP`` (etc.) at call time, never at import
time, so that ``configure()`` propagates correctly.
"""
from __future__ import annotations

CKPT_STEP: int | None = 400000
PLOT_STYLE: str = "paper"
ACTIVE_FOLD_IDS: list[int] = [0, 1, 2, 3, 4]


def configure(
    *,
    ckpt_step: int | None,
    plot_style: str,
    fold_ids: list[int],
) -> None:
    """Set runtime state. Called once by the CLI dispatcher before any run_mX."""
    global CKPT_STEP, PLOT_STYLE, ACTIVE_FOLD_IDS
    CKPT_STEP = ckpt_step
    PLOT_STYLE = plot_style
    ACTIVE_FOLD_IDS = list(fold_ids)
