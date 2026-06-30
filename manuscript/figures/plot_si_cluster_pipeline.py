#!/usr/bin/env python3
"""Fig S3 — stoichiometric vacancy-cluster construction pipeline.

Rendered as a Nature-style flat flow diagram in matplotlib so the asset is a
plain PDF/PNG that can be dropped into a Word manuscript later. No LaTeX.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "_si_cluster_pipeline"

NODE_FACE = "#F5F7FA"
NODE_EDGE = "#1F3A5F"
ARROW_COL = "#1F3A5F"
TITLE_COL = "#0F1F3D"
BODY_COL = "#37475A"

NODES = [
    ("Clean CIF",      "Resolve disorder.\nIdentify molecules."),
    ("Supercell",      "Try $2{\\times}2{\\times}2 \\to 4{\\times}4{\\times}4$\nuntil $\\geq 3$ spread seeds."),
    ("Stoichiometry",  "Topology-based species.\nGCD simplest unit."),
    ("Vacancy cluster","Seeded spatial removal.\nGrow by nearest molecule."),
    ("DeepMD input",   "Minimum-image wrap.\n100 \u00c5 non-periodic box."),
]


def _draw_node(ax, x, y, w, h, title, body):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.045",
        facecolor=NODE_FACE,
        edgecolor=NODE_EDGE,
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h - 0.18 * h,
        title,
        ha="center", va="top",
        fontsize=10, fontweight="bold", color=TITLE_COL,
    )
    ax.plot(
        [x + 0.10 * w, x + 0.90 * w],
        [y + h - 0.34 * h, y + h - 0.34 * h],
        color=NODE_EDGE, lw=0.6, solid_capstyle="butt",
    )
    ax.text(
        x + w / 2, y + 0.10 * h,
        body,
        ha="center", va="bottom",
        fontsize=8.4, color=BODY_COL,
        linespacing=1.3,
    )


def _draw_arrow(ax, x0, x1, y):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y),
            arrowstyle="-|>", mutation_scale=12,
            lw=1.0, color=ARROW_COL,
            shrinkA=0, shrinkB=0,
        )
    )


def main() -> None:
    n = len(NODES)
    box_w = 1.45
    box_h = 1.35
    gap = 0.55
    pitch = box_w + gap

    width_in = 1.05 * (n * box_w + (n - 1) * gap) + 0.5
    height_in = 2.3

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    ax.set_xlim(0, n * box_w + (n - 1) * gap)
    ax.set_ylim(-0.55, box_h + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")

    y0 = 0.0
    centers_y = y0 + box_h / 2

    for i, (title, body) in enumerate(NODES):
        x = i * pitch
        _draw_node(ax, x, y0, box_w, box_h, title, body)
        if i < n - 1:
            _draw_arrow(ax, x + box_w + 0.06, x + pitch - 0.06, centers_y)

    ax.text(
        (n * box_w + (n - 1) * gap) / 2, -0.42,
        "Cluster variants use deterministic spread-seed offsets (n1 / n2 / n3) "
        "and no coordinate perturbation.",
        ha="center", va="center",
        fontsize=8.2, color=BODY_COL,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
