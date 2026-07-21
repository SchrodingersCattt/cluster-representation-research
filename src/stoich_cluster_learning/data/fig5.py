"""Curated Figure 5 data assets for the standalone release."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from stoich_cluster_learning.paths import repo_root

ABX4_DATA_DIR = repo_root() / "data" / "abx4"
ABX4_CIF_DIR = ABX4_DATA_DIR / "cifs"
ABX4_PXRD_DIR = ABX4_DATA_DIR / "pxrd"
ABX4_PROPERTIES_CSV = ABX4_DATA_DIR / "properties.csv"


def load_abx4_properties() -> dict[str, dict[str, str]]:
    """Load curated ABX4 material properties keyed by material name."""
    rows: dict[str, dict[str, str]] = {}
    with ABX4_PROPERTIES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material = (row.get("material") or "").strip()
            if material:
                rows[material] = dict(row)
    return rows


def load_pxrd_specs() -> dict[str, dict[str, tuple[Path, int, int]]]:
    """Return PXRD input specs in the format expected by ``plot_fig5.py``.

    The release stores every PXRD trace as a two-column CSV with header
    ``two_theta,intensity``. Column indices are therefore always ``(0, 1)``.
    """
    manifest_path = ABX4_PXRD_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs: dict[str, dict[str, tuple[Path, int, int]]] = {}
    for material, entries in manifest.items():
        specs[material] = {
            role: (ABX4_PXRD_DIR / filename, 0, 1)
            for role, filename in entries.items()
        }
    return specs
