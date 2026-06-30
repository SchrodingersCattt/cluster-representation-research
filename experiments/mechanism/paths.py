"""Path constants for M-series experiments."""
from __future__ import annotations

from pathlib import Path

# experiments/mechanism/paths.py -> experiments/
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "00_data_prep"
MECHANISM_DIR = DATA_ROOT / "pems_mechanism_systems"
CLUSTER_N1_DIR = DATA_ROOT / "pems_cluster_n1_systems"
TEMPLATE_N1_DIR = DATA_ROOT / "pems_dap4_template_systems" / "cluster_n1"
MANIFEST_PATH = DATA_ROOT / "pems_manifest.json"
M1_SPLITS_PATH = DATA_ROOT / "pems_5fold_splits_v2.json"
SENSITIVITY_PATH = ROOT / "pems_sensitivity_summary.json"
PEMS_CSV = ROOT.parent / "data" / "pems" / "pems.csv"
OUTPUT_DIR = ROOT / "mechanism_results"
PAPER_FIG_DIR = OUTPUT_DIR / "paper_figures"
SUPP_FIG_DIR = PAPER_FIG_DIR / "supplementary"
