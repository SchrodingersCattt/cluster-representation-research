#!/usr/bin/env python3
"""Extended Data periodic-control heatmap."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import CHARCOAL, heatmap_text_color, save_figure, setup_style

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments_davis2024"
CROSS_JSON = EXP_DIR / "cross_infer_rep.json"
OUT = THIS_DIR / "_ed_periodic_control_heatmap"

ROWS = [
    ("Cluster-trained baseline", "exp7a"),
    ("Crystal-trained control", "exp8a"),
]
COLUMNS = [
    (r"Cluster input ($n_1$)", "cluster_n1"),
    ("Periodic crystal input", "crystal"),
]


def _load_matrix() -> np.ndarray:
    cross = json.loads(CROSS_JSON.read_text(encoding="utf-8"))
    return np.array(
        [
            [float(cross[key][col_key]["mean_mae"]) for _, col_key in COLUMNS]
            for _, key in ROWS
        ],
        dtype=float,
    )


def main() -> None:
    matrix = _load_matrix()

    setup_style()

    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    fig.subplots_adjust(left=0.34, right=0.86, top=0.82, bottom=0.20)

    vmax = max(1500.0, float(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(COLUMNS)))
    ax.set_xticklabels([label for label, _ in COLUMNS])
    ax.set_yticks(np.arange(len(ROWS)))
    ax.set_yticklabels([label for label, _ in ROWS])
    ax.set_title("Cluster vs periodic-crystal control", loc="left", fontweight="bold", pad=5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = heatmap_text_color(im.cmap, im.norm, value)
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    def draw_arrow(
        xytext: tuple[float, float],
        xy: tuple[float, float],
        *,
        linestyle: str = "-",
        arrowstyle: str = "-|>",
    ) -> None:
        ax.annotate(
            "",
            xy=xy,
            xytext=xytext,
            arrowprops={
                "arrowstyle": arrowstyle,
                "color": "white",
                "lw": 1.55,
                "linestyle": linestyle,
                "mutation_scale": 10.5,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=6,
        )

    draw_arrow((0.38, 0.0), (0.62, 0.0), linestyle="-")
    draw_arrow((0.62, 1.0), (0.38, 1.0), linestyle="-")
    draw_arrow((0.18, 0.20), (0.78, 0.80), linestyle=(0, (3, 2)), arrowstyle="<->")

    for x in np.arange(-0.5, matrix.shape[1], 1):
        ax.axvline(x, color="white", lw=1.0)
    for y in np.arange(-0.5, matrix.shape[0], 1):
        ax.axhline(y, color="white", lw=1.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE (m$\\cdot$s$^{-1}$)")
    cbar.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color=CHARCOAL, lw=1.55, linestyle="-", marker=">", markersize=5, label="Cross-representation transfer"),
        Line2D([0], [0], color=CHARCOAL, lw=1.55, linestyle=(0, (3, 2)), marker="|", markersize=6, label="Matched-input comparison"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        fontsize=8.0,
    )

    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Extended Data periodic-control heatmap."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import CHARCOAL, heatmap_text_color, save_figure, setup_style

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments_davis2024"
CROSS_JSON = EXP_DIR / "cross_infer_rep.json"
OUT = THIS_DIR / "_ed_periodic_control_heatmap"

ROWS = [
    ("Cluster-trained baseline", "exp7a"),
    ("Crystal-trained control", "exp8a"),
]
COLUMNS = [
    (r"Cluster input ($n_1$)", "cluster_n1"),
    ("Periodic crystal input", "crystal"),
]


def _load_matrix() -> np.ndarray:
    cross = json.loads(CROSS_JSON.read_text(encoding="utf-8"))
    return np.array(
        [
            [float(cross[key][col_key]["mean_mae"]) for _, col_key in COLUMNS]
            for _, key in ROWS
        ],
        dtype=float,
    )


def main() -> None:
    matrix = _load_matrix()

    setup_style()

    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    fig.subplots_adjust(left=0.34, right=0.86, top=0.82, bottom=0.20)

    vmax = max(1500.0, float(np.nanmax(matrix)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(COLUMNS)))
    ax.set_xticklabels([label for label, _ in COLUMNS])
    ax.set_yticks(np.arange(len(ROWS)))
    ax.set_yticklabels([label for label, _ in ROWS])
    ax.set_title("Cluster vs periodic-crystal control", loc="left", fontweight="bold", pad=5)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = heatmap_text_color(im.cmap, im.norm, value)
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)

    def draw_arrow(
        xytext: tuple[float, float],
        xy: tuple[float, float],
        *,
        linestyle: str = "-",
        arrowstyle: str = "-|>",
    ) -> None:
        ax.annotate(
            "",
            xy=xy,
            xytext=xytext,
            arrowprops={
                "arrowstyle": arrowstyle,
                "color": "white",
                "lw": 1.55,
                "linestyle": linestyle,
                "mutation_scale": 10.5,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=6,
        )

    draw_arrow((0.38, 0.0), (0.62, 0.0), linestyle="-")
    draw_arrow((0.62, 1.0), (0.38, 1.0), linestyle="-")
    draw_arrow((0.18, 0.20), (0.78, 0.80), linestyle=(0, (3, 2)), arrowstyle="<->")

    for x in np.arange(-0.5, matrix.shape[1], 1):
        ax.axvline(x, color="white", lw=1.0)
    for y in np.arange(-0.5, matrix.shape[0], 1):
        ax.axhline(y, color="white", lw=1.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE (m$\\cdot$s$^{-1}$)")
    cbar.ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D([0], [0], color=CHARCOAL, lw=1.55, linestyle="-", marker=">", markersize=5, label="Cross-representation transfer"),
        Line2D([0], [0], color=CHARCOAL, lw=1.55, linestyle=(0, (3, 2)), marker="|", markersize=6, label="Matched-input comparison"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        fontsize=8.0,
    )

    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
