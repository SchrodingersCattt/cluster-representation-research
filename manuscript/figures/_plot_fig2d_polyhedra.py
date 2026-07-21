#!/usr/bin/env python3
"""Compatibility wrapper for the Figure 2d/Figure 5a polyhedra renderer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "manuscript" / "figures"))

from stoich_cluster_learning.viz.polyhedra import main, render_all_strategies, SELECTION_STRATEGIES  # noqa: E402


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        action="store_true",
        help="Render one figure per selection strategy for comparison.",
    )
    parser.add_argument(
        "--strategy",
        choices=SELECTION_STRATEGIES,
        default=None,
        help="Set the active selection strategy for a single render.",
    )
    args = parser.parse_args()
    if args.strategy:
        import stoich_cluster_learning.viz.polyhedra as polyhedra

        polyhedra.SELECTION_STRATEGY = args.strategy
    if args.strategies:
        render_all_strategies()
    else:
        main()
