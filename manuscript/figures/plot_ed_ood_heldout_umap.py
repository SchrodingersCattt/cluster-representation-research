#!/usr/bin/env python3
"""Supplementary UMAP for OOD-holdout cluster embeddings."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_fig5 import (  # noqa: E402
    COLORS,
    THIS_DIR,
    plot_cluster_umap_panel,
    save_png_pdf,
    setup_nature_style,
)
from figure_style import MATERIAL_COLORS  # noqa: E402

OOD_HELDOUT_MATERIALS = ("DAC-4", "TAP-2", "DPPE-1", "EAP-4", "SY")
OOD_HELDOUT_COLORS = {material: MATERIAL_COLORS.get(material, COLORS.get(material, "#8A8A8A")) for material in OOD_HELDOUT_MATERIALS}


def main() -> None:
    setup_nature_style()
    plt.rcParams.update({"axes.titlesize": 8.0, "legend.fontsize": 8.0})
    fig, ax = plt.subplots(figsize=(180 / 25.4, 92 / 25.4))
    fig.subplots_adjust(left=0.080, right=0.985, top=0.900, bottom=0.145)
    plot_cluster_umap_panel(
        ax,
        focus_materials=OOD_HELDOUT_MATERIALS,
        focus_set="ood_heldout",
        focus_label="OOD-holdout",
        focus_colors=OOD_HELDOUT_COLORS,
        show_ylabel=True,
        label_offsets_pt={
            "DAC-4": (28.0, -12.0),
            "TAP-2": (26.0, 12.0),
            "DPPE-1": (30.0, -8.0),
            "EAP-4": (30.0, -14.0),
            "SY": (24.0, 14.0),
        },
    )
    save_png_pdf(fig, THIS_DIR / "_si_ood_heldout_umap")
    plt.close(fig)


if __name__ == "__main__":
    main()
