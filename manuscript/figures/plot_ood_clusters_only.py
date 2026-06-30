#!/usr/bin/env python3
"""Render only the OOD cluster figure (TRN already saved).

This wrapper skips the heavier TRN run (75 panels) and only renders
the 12 OOD panels, so we can recover from a Kaleido watchdog hang
without redoing the work.
"""
from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import plot_trn_clusters as pt  # noqa: E402

if __name__ == "__main__":
    print("Rendering OOD only (4 rows x 3 cols: n1/n2/n3)", flush=True)
    pt.render_grid(pt.OOD_MATERIALS, pt.ood_cif_path, "_ood_clusters")
