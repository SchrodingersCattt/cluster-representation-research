#!/usr/bin/env python3
"""
Unified PEMs inference script.

Replaces:
  - infer_and_plot_PEMs.py   (IND 5-fold CV + sensitivity)
  - infer_ood_from_cleaned_cifs.py  (OOD single-model)
  - infer_ood_5fold_exp7a_7c.py     (OOD 5-fold ensemble)
  - infer_ood_model_deviation.py    (UQ / model deviation)
  - infer_dac4.py                   (single-material)

Usage
-----
# IND 5-fold CV (all registered families):
python infer_pems.py cv

# IND 5-fold CV for a specific series:
python infer_pems.py cv --series exp7a
python infer_pems.py cv --series exp8a
python infer_pems.py cv --series exp9a

# OOD inference (build clusters from cleaned CIFs, run 5-fold ensemble):
python infer_pems.py ood --series exp7a
python infer_pems.py ood --series exp7a_lr1e4

# UQ / model deviation (IND + OOD, 5-fold × 3 cluster variants):
python infer_pems.py uq --series exp7a_lr1e4

# Perturbation sensitivity (fold-0 model on rotation/translation/template datasets):
python infer_pems.py sensitivity --series exp7a

# Single material from pre-built npy:
python infer_pems.py single --material DAC-4 --series exp6v2_allpems exp7a_lr1e4

Checkpoint resolution (all subcommands):
  1. model.ckpt.pt  (symlink to latest — works for in-progress training)
  2. highest model.ckpt-*.pt
  3. Error if neither found

Output files (written to experiments/):
  cv      → pems_inference_summary.json, pems_predictions.json
            pems_parity_plots.png/.pdf, pems_ood_heatmap.png/.pdf
            pems_sensitivity_summary.json, pems_sensitivity_heatmap.png/.pdf
  ood     → pems_ood_5fold_{series}.json
  uq      → pems_ood_model_deviation_{series}.json, pems_uq_calibration_{series}.json
  single  → printed to stdout (no JSON)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from paper_plot_style import EXP_COLORS, save_png_pdf, setup_nature_style, style_axes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "00_data_prep"
CLEANED_CIF_DIR = DATA_ROOT / "pems_cleaned_cifs"
CSV_PATH = ROOT.parent / "data" / "pems" / "pems.csv"
BRANCH_META = ROOT / "branch_meta.json"
OOD_EXP_VALUES = ROOT / "ood_experimental_values.json"

# Ablation experiments live under ROOT/ablation/ (LR sweeps + step-length sweeps).
ABLATION_FAMILIES: set[str] = {
    "exp7a_lr1e4", "exp7b_lr1e4", "exp7c_lr1e4", "exp7d_lr1e4",
    "exp7a_lr5e6", "exp7b_lr5e6",
    "exp7a_200k", "exp7a_800k",
    "exp7a_seed7", "exp7a_seed13", "exp7c_seed7", "exp7c_seed13",
    "exp7a_decay200", "exp7c_decay200",
}


def _exp_base_dir(family: str) -> Path:
    """Return the parent directory holding ``{family}_fold{i}`` (or single-model ``{family}``)."""
    return ROOT / "ablation" if family in ABLATION_FAMILIES else ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYPE_MAP = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og',
]
TYPE_TO_ID: dict[str, int] = {sym: i for i, sym in enumerate(TYPE_MAP)}

_HEAVY = ["I", "Na", "K", "Rb", "Ba", "Ag"]
PEM_BOND_THRESHOLDS: dict[tuple[str, str], float] = {}
_HEAVY_NONO_LIMITS: dict[str, float] = {"I": 2.10, "Na": 2.30, "K": 2.50, "Rb": 2.60, "Ba": 2.60, "Ag": 2.30}
_HEAVY_O_LIMITS: dict[str, float] = {"I": 2.05, "Na": 2.20, "K": 2.30, "Rb": 2.40, "Ba": 2.40, "Ag": 2.20}
for _metal, _limit in _HEAVY_NONO_LIMITS.items():
    for _org in ["C", "H", "N", "Cl"]:
        PEM_BOND_THRESHOLDS[(_metal, _org)] = _limit
        PEM_BOND_THRESHOLDS[(_org, _metal)] = _limit
for _metal, _limit in _HEAVY_O_LIMITS.items():
    PEM_BOND_THRESHOLDS[(_metal, "O")] = _limit
    PEM_BOND_THRESHOLDS[("O", _metal)] = _limit
for _i, _m1 in enumerate(_HEAVY):
    for _m2 in _HEAVY[_i:]:
        PEM_BOND_THRESHOLDS[(_m1, _m2)] = 3.2
        if _m1 != _m2:
            PEM_BOND_THRESHOLDS[(_m2, _m1)] = 3.2

CLUSTER_BOX = np.eye(3, dtype=np.float64) * 100.0
CLUSTER_RANDOM_SEEDS = {"cluster_n1": 101, "cluster_n2": 202, "cluster_n3": 303}
CLUSTER_VARIANT_OFFSETS = {"cluster_n1": 0, "cluster_n2": 1, "cluster_n3": 2}
SUPERCELL_SCHEDULE = [(2, 2, 2), (3, 3, 3), (2, 3, 3), (3, 3, 4), (4, 4, 4)]
MIN_DISTINCT_SEEDS = 3

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Each entry: exp_dir, head (None = single-task), splits_file, data_type
# data_type: "cluster_n1n2n3" | "crystal"
# splits_file: path to JSON with {"folds": {"0": [...], "1": [...], ...}}

FOLD_FAMILIES: dict[str, dict[str, Any]] = {
    # --- exp7 random 5-fold ---
    "exp7a": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7b": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7c": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7d": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    # --- split-seed robustness ablations (same exp7a/exp7c configs, alternate PEMs 5-fold splits) ---
    "exp7a_seed7": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_seed7.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7a_seed13": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_seed13.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7c_seed7": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_seed7.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7c_seed13": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_seed13.json",
        "data_type": "cluster_n1n2n3",
    },
    # Low-LR sweep retained for exp7b auxiliary-head ablation.
    "exp7b_lr5e6": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7a_lr1e4": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7b_lr1e4": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7c_lr1e4": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7d_lr1e4": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    # --- step-length ablations (5-fold CV on exp7a splits, dirs live under
    #     ablation/<family>_fold{i}) ---
    "exp7a_200k": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7a_800k": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    # --- LR decay-ratio ablations (same exp7a/exp7c configs, decay_steps = numb_steps / 200) ---
    "exp7a_decay200": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    "exp7c_decay200": {
        "head": None,
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "cluster_n1n2n3",
    },
    # --- exp8a crystal 5-fold (random splits, same as exp7) ---
    "exp8a": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_5fold_splits_v2.json",
        "data_type": "crystal",
    },
    # --- exp9a cluster DD-ranked 5-fold ---
    "exp9a": {
        "head": "pems_vdet_kj",
        "splits_file": DATA_ROOT / "pems_9a_dd_splits.json",
        "data_type": "cluster_n1n2n3",
    },
}

# Single-model experiments (no fold structure)
SINGLE_MODEL_EXPS: dict[str, dict[str, Any]] = {
    # exp6v1_allpems uses the production LR (start_lr=2e-5) and is the only
    # exp6 variant consumed by manuscript figures (Fig. 4 ABX grid).
    "exp6v1_allpems": {"head": "pems_vdet_kj"},
    # exp6v2_allpems was an LR=1e-4 ablation matching exp7a_lr1e4 style; not
    # used in any manuscript figure. Kept on disk but excluded from default
    # registries to avoid stale-output churn.
    # "exp6v2_allpems": {"head": "pems_vdet_kj"},
}

# Perturbation datasets for sensitivity analysis
SENSITIVITY_DATASETS: list[str] = [
    "pems_cluster_n1", "pems_cluster_n2", "pems_cluster_n3",
    "pems_mod_rotation_n1", "pems_mod_rotation_n2", "pems_mod_rotation_n3",
    "pems_mod_translation_n1", "pems_mod_translation_n2", "pems_mod_translation_n3",
    "pems_dap4_template_n1", "pems_dap4_template_n2", "pems_dap4_template_n3",
]
SENSITIVITY_DATASET_DIRS: dict[str, Path] = {
    "pems_cluster_n1": DATA_ROOT / "pems_cluster_n1_systems",
    "pems_cluster_n2": DATA_ROOT / "pems_cluster_n2_systems",
    "pems_cluster_n3": DATA_ROOT / "pems_cluster_n3_systems",
    "pems_mod_rotation_n1": DATA_ROOT / "pems_mod_rotation_systems" / "cluster_n1",
    "pems_mod_rotation_n2": DATA_ROOT / "pems_mod_rotation_systems" / "cluster_n2",
    "pems_mod_rotation_n3": DATA_ROOT / "pems_mod_rotation_systems" / "cluster_n3",
    "pems_mod_translation_n1": DATA_ROOT / "pems_mod_translation_systems" / "cluster_n1",
    "pems_mod_translation_n2": DATA_ROOT / "pems_mod_translation_systems" / "cluster_n2",
    "pems_mod_translation_n3": DATA_ROOT / "pems_mod_translation_systems" / "cluster_n3",
    "pems_dap4_template_n1": DATA_ROOT / "pems_dap4_template_systems" / "cluster_n1",
    "pems_dap4_template_n2": DATA_ROOT / "pems_dap4_template_systems" / "cluster_n2",
    "pems_dap4_template_n3": DATA_ROOT / "pems_dap4_template_systems" / "cluster_n3",
}

# Families that run sensitivity analysis
SENSITIVITY_FAMILIES: tuple[str, ...] = (
    "exp7a", "exp7b", "exp7c", "exp7d",
    "exp7a_lr1e4", "exp7b_lr1e4", "exp7b_lr5e6", "exp7c_lr1e4", "exp7d_lr1e4",
    "exp8a", "exp9a",
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_exp_values() -> dict[str, float]:
    """Load experimental Vdet from pems.csv (km/s → m/s), plus OOD overrides."""
    exp: dict[str, float] = {}
    with CSV_PATH.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mat = (row.get("material") or "").strip()
            d = (row.get("D_km_s") or "").strip()
            if mat and d:
                try:
                    exp[mat] = float(d) * 1000.0
                except ValueError:
                    pass

    # OOD values may come from manually curated literature entries not present
    # in pems.csv, or from rows whose commas are not CSV-escaped reliably.
    if OOD_EXP_VALUES.exists():
        data = json.loads(OOD_EXP_VALUES.read_text(encoding="utf-8"))
        for mat, value in data.get("values_m_s", {}).items():
            try:
                exp[str(mat)] = float(value)
            except (TypeError, ValueError):
                pass
    return exp


def resolve_checkpoint(exp_dir: Path) -> Path | None:
    """Resolve checkpoint for an experiment directory.

    Priority:
      1. model.ckpt.pt  (symlink to latest — works for in-progress training)
      2. highest-step model.ckpt-*.pt
    Returns None if no checkpoint found.
    """
    symlink = exp_dir / "model.ckpt.pt"
    if symlink.exists():
        return symlink
    ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    if ckpts:
        return ckpts[-1]
    return None


def load_splits(splits_file: Path) -> dict[str, list[str]]:
    """Load fold splits from JSON. Returns {fold_idx_str: [material, ...]}."""
    data = json.loads(splits_file.read_text(encoding="utf-8"))
    return data.get("folds", {})


def get_val_systems(family: str, fold_idx: int, data_type: str, splits_file: Path) -> list[Path]:
    """Return held-out system paths for a given fold."""
    splits = load_splits(splits_file)
    val_materials = splits.get(str(fold_idx), [])
    systems: list[Path] = []
    if data_type == "cluster_n1n2n3":
        for cluster_dir in ("pems_cluster_n1_systems", "pems_cluster_n2_systems", "pems_cluster_n3_systems"):
            for material in val_materials:
                p = DATA_ROOT / cluster_dir / material
                if p.exists():
                    systems.append(p)
    elif data_type == "crystal":
        for material in val_materials:
            p = DATA_ROOT / "pems_crystal_systems" / material
            if p.exists():
                systems.append(p)
    return sorted(systems)


def get_heldout_systems_from_dataset(
    fold_idx: int, splits_file: Path, dataset_dir: Path
) -> list[Path]:
    """Return held-out systems for one fold from a specific dataset directory."""
    splits = load_splits(splits_file)
    val_materials: set[str] = set(splits.get(str(fold_idx), []))
    if not val_materials or not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name in val_materials])


def load_cluster_system(system_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a DeepMD npy system. Returns (coords, cells_or_None, symbols)."""
    import dpdata
    vs = dpdata.LabeledSystem(str(system_dir), fmt="deepmd/npy")
    data_type_map = vs.data["atom_names"]
    coords = vs.data["coords"]  # (nframes, natoms, 3)
    cells = None if (system_dir / "nopbc").exists() else vs.data["cells"]
    symbols = [data_type_map[t] for t in vs.data["atom_types"]]
    return coords, cells, symbols


def predict_system(
    model: Any,
    model_type_map: list[str],
    coords: np.ndarray,
    cells: np.ndarray | None,
    symbols: list[str],
) -> np.ndarray:
    """Run DeepProperty inference. Returns array of predictions (one per frame)."""
    unique = sorted(set(symbols), key=lambda s: TYPE_TO_ID.get(s, 999))
    local = np.array([unique.index(s) for s in symbols], dtype=np.int32)
    model_ids = np.array([int(np.where(np.array(model_type_map) == t)[0][0]) for t in unique], dtype=np.int32)
    at = np.array([model_ids[l] for l in local], dtype=np.int32)
    return model.eval(coords=coords, atom_types=at, cells=cells)[0].reshape(-1)


def run_inference_on_systems(
    ckpt: Path,
    systems: list[Path],
    head: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run inference on a list of npy system directories.

    Returns (gt_array, pred_array, system_names).
    """
    from deepmd.pt.infer.deep_eval import DeepProperty
    import dpdata

    kwargs: dict = {}
    if head is not None:
        kwargs["head"] = head
    model = DeepProperty(str(ckpt), **kwargs)
    model_type_map = model.get_type_map()

    gt_list: list[float] = []
    pred_list: list[float] = []
    names: list[str] = []

    for sys_path in systems:
        try:
            vs = dpdata.LabeledSystem(str(sys_path), fmt="deepmd/npy")
            data_type_map = vs.data["atom_names"]
            coords = vs.data["coords"]
            cells = None if (sys_path / "nopbc").exists() else vs.data["cells"]
            atom_types = np.array([
                np.where(np.array(model_type_map) == data_type_map[t])[0][0]
                for t in vs.data["atom_types"]
            ], dtype=np.int32)
            gt = vs.data["energies"]
            pred = model.eval(coords=coords, atom_types=atom_types, cells=cells)[0].reshape(-1)
            gt_list.extend(gt)
            pred_list.extend(pred)
            names.extend([sys_path.name] * len(pred))
        except Exception as e:
            print(f"  WARNING: failed on {sys_path.name}: {e}")

    return np.array(gt_list), np.array(pred_list), names


# ---------------------------------------------------------------------------
# Cluster build pipeline (for OOD / UQ subcommands)
# ---------------------------------------------------------------------------

def _select_spread_seeds(supercell: Any, analyzer: Any, seed_species_id: str, n: int = 3) -> list[int]:
    """Pick n seed molecules maximally spread in space. Returns local indices."""
    candidates = list(analyzer.species_map.get(seed_species_id, []))
    if len(candidates) <= n:
        return candidates
    coms = np.array([
        np.array(supercell.molecules[idx].get_center_of_mass(), dtype=np.float64)
        for idx in candidates
    ])
    centroid = coms.mean(axis=0)
    dists_to_centroid = np.linalg.norm(coms - centroid, axis=1)
    first = int(np.argmin(dists_to_centroid))
    selected = [first]
    for _ in range(n - 1):
        best_idx, best_min_dist = -1, -1.0
        for ci in range(len(candidates)):
            if ci in selected:
                continue
            dists = np.array([np.linalg.norm(coms[ci] - coms[si]) for si in selected])
            min_d = float(dists.min())
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = ci
        if best_idx < 0:
            break
        selected.append(best_idx)
    return [candidates[i] for i in selected]


def build_cluster_from_crystal(crystal: Any, variant_name: str, seed: int) -> Any:
    """Build stoichiometric cluster from MolecularCrystal. Returns cluster MolecularCrystal."""
    from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
    from molcrys_kit.operations.defects import VacancyGenerator

    offset = CLUSTER_VARIANT_OFFSETS[variant_name]
    for sc_dims in SUPERCELL_SCHEDULE:
        supercell = crystal.get_supercell(*sc_dims)
        analyzer = StoichiometryAnalyzer(supercell)
        simplest_unit = analyzer.get_simplest_unit()
        seed_species_id = next(iter(simplest_unit.keys()))
        seed_candidates = analyzer.species_map.get(seed_species_id, [])
        if not seed_candidates:
            continue
        spread_seeds = _select_spread_seeds(supercell, analyzer, seed_species_id, n=MIN_DISTINCT_SEEDS)
        if len(spread_seeds) >= MIN_DISTINCT_SEEDS:
            seed_index = spread_seeds[offset % len(spread_seeds)]
            gen = VacancyGenerator(supercell)
            _, cluster = gen.generate_vacancy(
                target_spec=simplest_unit, seed_index=seed_index,
                return_removed_cluster=True, random_seed=seed,
            )
            return cluster
    # Fallback: largest supercell
    supercell = crystal.get_supercell(*SUPERCELL_SCHEDULE[-1])
    analyzer = StoichiometryAnalyzer(supercell)
    simplest_unit = analyzer.get_simplest_unit()
    seed_species_id = next(iter(simplest_unit.keys()))
    seed_candidates = analyzer.species_map.get(seed_species_id, [])
    seed_index = seed_candidates[offset % len(seed_candidates)]
    gen = VacancyGenerator(supercell)
    _, cluster = gen.generate_vacancy(
        target_spec=simplest_unit, seed_index=seed_index,
        return_removed_cluster=True, random_seed=seed,
    )
    return cluster


def prepare_cluster_atoms(cluster_crystal: Any) -> Any:
    """Convert cluster MolecularCrystal → centered ASE Atoms (nopbc, 100Å box)."""
    from molcrys_kit.structures.crystal import MolecularCrystal
    unwrapped = cluster_crystal.get_unwrapped_molecules()
    molecules = [mol.copy() for mol in unwrapped]
    if not molecules:
        raise RuntimeError("No molecules in cluster")
    ref = molecules[0].get_center_of_mass()
    inv_lat = np.linalg.inv(np.array(cluster_crystal.lattice, dtype=np.float64))
    for mol in molecules[1:]:
        shift = mol.get_center_of_mass() - ref
        frac = np.dot(shift, inv_lat)
        frac -= np.round(frac)
        cart = np.dot(frac, cluster_crystal.lattice)
        mol.positions += cart - shift
    combined = MolecularCrystal(
        lattice=np.array(cluster_crystal.lattice, dtype=np.float64).copy(),
        molecules=molecules, pbc=cluster_crystal.pbc,
    )
    atoms = combined.to_ase()
    atoms.positions = atoms.get_positions() - atoms.get_positions().mean(axis=0, keepdims=True) + 50.0
    atoms.set_cell(CLUSTER_BOX)
    atoms.set_pbc(False)
    return atoms


def predict_atoms(model: Any, model_type_map: list[str], atoms: Any) -> float:
    """Run DeepProperty inference on ASE Atoms. Returns scalar prediction."""
    symbols = atoms.get_chemical_symbols()
    unique = sorted(set(symbols), key=lambda s: TYPE_TO_ID.get(s, 999))
    local = np.array([unique.index(s) for s in symbols], dtype=np.int32)
    model_ids = np.array([int(np.where(np.array(model_type_map) == t)[0][0]) for t in unique], dtype=np.int32)
    at = np.array([model_ids[l] for l in local], dtype=np.int32)
    coords = atoms.get_positions().reshape(1, -1, 3)
    return float(model.eval(coords=coords, atom_types=at, cells=None)[0].reshape(-1)[0])


def build_ood_clusters(cif_dir: Path, skip_materials: set[str]) -> dict[str, dict[str, Any]]:
    """Build cluster_n1/n2/n3 for all CIFs in cif_dir not in skip_materials.

    Returns {material: {variant_name: ASE Atoms}}.
    """
    from ase.io import read as ase_read
    from molcrys_kit.structures.crystal import MolecularCrystal

    ood_cifs = sorted(p for p in cif_dir.glob("*.cif") if p.stem not in skip_materials)
    print(f"Building clusters for {len(ood_cifs)} OOD materials...")
    material_clusters: dict[str, dict[str, Any]] = {}
    for cif_path in ood_cifs:
        material = cif_path.stem
        try:
            atoms = ase_read(str(cif_path))
            crystal = MolecularCrystal.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS)
        except Exception as e:
            print(f"  ERROR {material}: {e}")
            continue
        clusters: dict[str, Any] = {}
        for vname, seed in CLUSTER_RANDOM_SEEDS.items():
            try:
                cc = build_cluster_from_crystal(crystal, vname, seed)
                clusters[vname] = prepare_cluster_atoms(cc)
            except Exception as e:
                print(f"  ERROR {material}/{vname}: {e}")
        if clusters:
            material_clusters[material] = clusters
    print(f"Built clusters for {len(material_clusters)} materials")
    return material_clusters


# ---------------------------------------------------------------------------
# Subcommand: cv
# ---------------------------------------------------------------------------

def cmd_cv(args: argparse.Namespace) -> None:
    """IND 5-fold CV inference + parity plots + sensitivity."""
    requested = set(args.series) if args.series else set(FOLD_FAMILIES.keys())
    families_to_run = [f for f in FOLD_FAMILIES if f in requested]
    if not families_to_run:
        print(f"No matching families for series={args.series}. Available: {list(FOLD_FAMILIES.keys())}")
        sys.exit(1)

    summary: dict[str, dict] = {}
    prediction_rows: list[dict] = []
    # fold_results[family][fold_idx] = (gt, pred, names, ckpt_name)
    fold_results: dict[str, dict[int, tuple]] = {f: {} for f in families_to_run}

    for family in families_to_run:
        cfg = FOLD_FAMILIES[family]
        splits_file: Path = cfg["splits_file"]
        head: str | None = cfg["head"]
        data_type: str = cfg["data_type"]
        fold_indices: list[int] = list(cfg.get("fold_indices", range(5)))
        dir_template: str = cfg.get("dir_template", "{family}_fold{fold_idx}")

        if not splits_file.exists():
            print(f"  SKIP {family}: splits file not found: {splits_file}")
            continue

        for fold_idx in fold_indices:
            exp_name = dir_template.format(family=family, fold_idx=fold_idx)
            exp_dir = _exp_base_dir(family) / exp_name
            if not exp_dir.exists():
                print(f"  SKIP {exp_name}: directory not found")
                continue
            ckpt = resolve_checkpoint(exp_dir)
            if ckpt is None:
                print(f"  SKIP {exp_name}: no checkpoint found")
                continue

            systems = get_val_systems(family, fold_idx, data_type, splits_file)
            if not systems:
                print(f"  SKIP {exp_name}: no held-out systems found")
                continue

            print(f"=== {exp_name} fold {fold_idx} ({ckpt.name}, {len(systems)} systems) ===")
            gt, pred, names = run_inference_on_systems(ckpt, systems, head=head)
            if len(gt) == 0:
                continue

            fold_results[family][fold_idx] = (gt, pred, names, ckpt.name)
            # Cluster-pool by material name within this fold; the raw arrays
            # contain 3 cluster realizations per material (n1/n2/n3) which
            # share an identical ground truth.
            mat_to_preds: dict[str, list[float]] = {}
            mat_to_gt: dict[str, float] = {}
            for n, g, p in zip(names, gt, pred):
                mat_to_preds.setdefault(n, []).append(float(p))
                mat_to_gt[n] = float(g)
            pooled_pairs = [
                (mat_to_gt[m], float(np.mean(mat_to_preds[m]))) for m in sorted(mat_to_preds)
            ]
            cluster_stds = [
                float(np.std(mat_to_preds[m], ddof=1)) if len(mat_to_preds[m]) >= 2 else 0.0
                for m in sorted(mat_to_preds)
            ]
            gt_p = np.array([g for g, _ in pooled_pairs])
            pr_p = np.array([p for _, p in pooled_pairs])
            errs_pool = np.abs(gt_p - pr_p)
            summary_key = f"{exp_name}__pems_fold{fold_idx}_val"
            summary[summary_key] = {
                "checkpoint": ckpt.name,
                "n_materials": int(len(pooled_pairs)),
                "n_samples_raw": int(len(gt)),
                "mae_m_s": round(float(np.mean(errs_pool)), 1),
                "median_ae_m_s": round(float(np.median(errs_pool)), 1),
                "rmse_m_s": round(float(np.sqrt(np.mean((gt_p - pr_p) ** 2))), 1),
                "mean_cluster_std_m_s": round(float(np.mean(cluster_stds)), 2),
                "head": head,
                "dataset": f"pems_fold{fold_idx}_val",
                "fold": fold_idx,
                "family": family,
                "data_type": data_type,
                "metric_convention": "cluster-pooled (3 cluster realizations averaged per material before MAE/RMSE)",
            }
            for name, y_true, y_pred in zip(names, gt, pred):
                prediction_rows.append({
                    "experiment": exp_name,
                    "dataset": f"pems_fold{fold_idx}_val",
                    "system": name,
                    "ground_truth_m_s": float(y_true),
                    "predicted_m_s": float(y_pred),
                    "abs_error_m_s": float(abs(y_true - y_pred)),
                    "checkpoint": ckpt.name,
                    "head": head,
                    "fold": fold_idx,
                    "family": family,
                })

        # Print per-fold and mean MAE for this family
        fold_maes = [
            summary[k]["mae_m_s"]
            for i in fold_indices
            for k in (f"{dir_template.format(family=family, fold_idx=i)}__pems_fold{i}_val",)
            if k in summary
        ]
        if fold_maes:
            n = len(fold_maes)
            label = "CV MAE" if n > 1 else f"fold{fold_indices[0]} MAE"
            print(f"\n{family} {label}: {fold_maes} → mean={np.mean(fold_maes):.1f} m/s\n")

    # Sensitivity analysis
    if not args.no_sensitivity:
        _run_sensitivity(families_to_run, summary, prediction_rows)

    # Save JSON — merge with existing file so multiple --series runs accumulate.
    # Skip when --no-save (monitoring runs of in-progress checkpoints).
    if not args.no_save:
        summary_path = ROOT / "pems_inference_summary.json"
        if summary_path.exists():
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            existing_summary.update(summary)   # new keys overwrite, old keys preserved
            summary = existing_summary
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        predictions_path = ROOT / "pems_predictions.json"
        if predictions_path.exists():
            existing_preds = json.loads(predictions_path.read_text(encoding="utf-8"))
            new_families = {r["family"] for r in prediction_rows if "family" in r}
            existing_preds = [r for r in existing_preds if r.get("family") not in new_families]
            prediction_rows = existing_preds + prediction_rows
        predictions_path.write_text(json.dumps(prediction_rows, indent=2), encoding="utf-8")
        print(f"Saved pems_inference_summary.json ({len(summary)} entries)")

        # Legacy sensitivity summary consumed by mechanism/m0_perturbation.py
        # and manuscript/figures/plot_fig3.py panel c.
        _write_sensitivity_summary(summary, ROOT / "pems_sensitivity_summary.json")
    else:
        print(f"[monitor] skipping JSON save; {len(summary)} summary entries computed")

    # Plots
    if not args.no_plot:
        _plot_cv_parity(fold_results, ROOT / "pems_parity_plots.png")
        _plot_sensitivity_heatmap(summary, prediction_rows, ROOT / "pems_sensitivity_heatmap.png")


def _write_sensitivity_summary(summary: dict[str, dict], out_path: Path) -> None:
    """Derive ``pems_sensitivity_summary.json`` from the merged CV summary.

    Replicates the legacy ``sensitivity_comparison`` rows expected by
    ``mechanism/m0_perturbation.py`` and ``manuscript/figures/plot_fig3.py``.
    """
    comparisons = [
        ("pems_mod_rotation_n", "pems_cluster_n", "rotation"),
        ("pems_mod_translation_n", "pems_cluster_n", "translation"),
        ("pems_dap4_template_n", "pems_cluster_n", "dap4_template"),
    ]
    exp_names = sorted({k.split("__")[0] for k in summary if "_5fold_cv__" in k})

    rows: list[dict] = []
    for exp in exp_names:
        for pert_prefix, base_prefix, pert_label in comparisons:
            pert_maes, base_maes = [], []
            for n in ("1", "2", "3"):
                base_key = f"{exp}__{base_prefix}{n}"
                pert_key = f"{exp}__{pert_prefix}{n}"
                if base_key in summary and pert_key in summary:
                    base_maes.append(summary[base_key]["mae_m_s"])
                    pert_maes.append(summary[pert_key]["mae_m_s"])
            if not base_maes:
                continue
            mean_base = float(np.mean(base_maes))
            mean_pert = float(np.mean(pert_maes))
            rows.append({
                "experiment": exp,
                "perturbation": pert_label,
                "baseline_mae_mean": round(mean_base, 1),
                "perturbed_mae_mean": round(mean_pert, 1),
                "delta_mae": round(mean_pert - mean_base, 1),
                "delta_mae_pct": round(100.0 * (mean_pert - mean_base) / max(mean_base, 1.0), 1),
            })

    if not rows:
        print("No sensitivity comparison rows derived; skipping summary write.")
        return

    # Merge with any existing rows for experiments outside the current run so
    # the file remains a complete record across incremental --series invocations.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            existing_rows = existing.get("sensitivity_comparison", [])
            new_exp_set = {r["experiment"] for r in rows}
            existing_rows = [r for r in existing_rows if r.get("experiment") not in new_exp_set]
            rows = existing_rows + rows
        except Exception:
            pass

    rows.sort(key=lambda r: abs(r["delta_mae"]), reverse=True)
    out_path.write_text(
        json.dumps({"sensitivity_comparison": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {out_path.name} ({len(rows)} sensitivity comparison rows)")


def _run_sensitivity(
    families: list[str],
    summary: dict[str, dict],
    prediction_rows: list[dict],
) -> None:
    """Run sensitivity analysis for families in SENSITIVITY_FAMILIES."""
    for family in families:
        if family not in SENSITIVITY_FAMILIES:
            continue
        cfg = FOLD_FAMILIES[family]
        splits_file: Path = cfg["splits_file"]
        head: str | None = cfg["head"]

        for dataset_name in SENSITIVITY_DATASETS:
            dataset_dir = SENSITIVITY_DATASET_DIRS.get(dataset_name)
            if dataset_dir is None or not dataset_dir.exists():
                continue
            fold_maes: list[float] = []
            fold_rmses: list[float] = []
            fold_medians: list[float] = []
            ckpt_ref = None
            gt0, pred0, names0 = None, None, None

            for fold_idx in range(5):
                fold_name = f"{family}_fold{fold_idx}"
                exp_dir = _exp_base_dir(family) / fold_name
                if not exp_dir.exists():
                    continue
                ckpt = resolve_checkpoint(exp_dir)
                if ckpt is None:
                    continue
                systems = get_heldout_systems_from_dataset(fold_idx, splits_file, dataset_dir)
                if not systems:
                    continue
                print(f"=== {family} sensitivity on {dataset_name} fold {fold_idx} ({ckpt.name}) ===")
                gt, pred, names = run_inference_on_systems(ckpt, systems, head=head)
                if len(gt) == 0:
                    continue
                errors = np.abs(gt - pred)
                fold_maes.append(float(np.mean(errors)))
                fold_rmses.append(float(np.sqrt(np.mean((gt - pred) ** 2))))
                fold_medians.append(float(np.median(errors)))
                if fold_idx == 0:
                    ckpt_ref = ckpt
                    gt0, pred0, names0 = gt, pred, names

            if not fold_maes or ckpt_ref is None or gt0 is None:
                continue
            key = f"{family}_5fold_cv__{dataset_name}"
            summary[key] = {
                "checkpoint": ckpt_ref.name,
                "n_samples": int(len(gt0)),
                "mae_m_s": round(float(np.mean(fold_maes)), 1),
                "mae_std_m_s": round(float(np.std(fold_maes)), 1),
                "median_ae_m_s": round(float(np.mean(fold_medians)), 1),
                "rmse_m_s": round(float(np.mean(fold_rmses)), 1),
                "rmse_std_m_s": round(float(np.std(fold_rmses)), 1),
                "head": head,
                "dataset": dataset_name,
                "family": family,
                "n_folds_averaged": len(fold_maes),
            }
            for name, y_true, y_pred in zip(names0, gt0, pred0):
                prediction_rows.append({
                    "experiment": f"{family}_5fold_cv",
                    "dataset": dataset_name,
                    "system": name,
                    "ground_truth_m_s": float(y_true),
                    "predicted_m_s": float(y_pred),
                    "abs_error_m_s": float(abs(y_true - y_pred)),
                    "checkpoint": ckpt_ref.name,
                    "head": head,
                })


# ---------------------------------------------------------------------------
# Subcommand: ood
# ---------------------------------------------------------------------------

def cmd_ood(args: argparse.Namespace) -> None:
    """OOD 5-fold ensemble predictions from cleaned CIFs."""
    from deepmd.pt.infer.deep_eval import DeepProperty

    requested = set(args.series) if args.series else set(FOLD_FAMILIES.keys())
    families_to_run = [f for f in FOLD_FAMILIES if f in requested]
    if not families_to_run:
        print(f"No matching families. Available: {list(FOLD_FAMILIES.keys())}")
        sys.exit(1)

    exp_values = load_exp_values()

    # Determine training materials (union of all fold train sets for each family)
    # For OOD: skip materials that appear in ALL folds' training sets
    # We use the branch_meta.json to find training materials per family
    branch_meta: dict = {}
    if BRANCH_META.exists():
        branch_meta = json.loads(BRANCH_META.read_text(encoding="utf-8"))

    for family in families_to_run:
        cfg = FOLD_FAMILIES[family]
        splits_file: Path = cfg["splits_file"]
        head: str | None = cfg["head"]

        if not splits_file.exists():
            print(f"  SKIP {family}: splits file not found")
            continue

        # Collect all materials that appear in any fold's val set = all 25 training materials
        splits = load_splits(splits_file)
        all_train_materials: set[str] = set()
        for fold_mats in splits.values():
            all_train_materials.update(fold_mats)

        # Build OOD clusters (CIFs not in training set)
        material_clusters = build_ood_clusters(CLEANED_CIF_DIR, skip_materials=all_train_materials)
        if not material_clusters:
            print(f"  No OOD materials found for {family}")
            continue

        # Run 5-fold ensemble
        predictions: dict[str, dict[str, dict[str, float]]] = {m: {} for m in material_clusters}
        for fold_idx in range(5):
            exp_name = f"{family}_fold{fold_idx}"
            exp_dir = _exp_base_dir(family) / exp_name
            if not exp_dir.exists():
                print(f"  SKIP {exp_name}: directory not found")
                continue
            ckpt = resolve_checkpoint(exp_dir)
            if ckpt is None:
                print(f"  SKIP {exp_name}: no checkpoint")
                continue
            print(f"\n  Loading fold{fold_idx}: {ckpt.name}")
            if head:
                model = DeepProperty(str(ckpt), head=head)
            else:
                model = DeepProperty(str(ckpt))
            model_tm = model.get_type_map()
            for material, clusters in material_clusters.items():
                fp: dict[str, float] = {}
                for vname, catoms in clusters.items():
                    try:
                        fp[vname] = predict_atoms(model, model_tm, catoms)
                    except Exception as e:
                        print(f"    [{material}/{vname}/fold{fold_idx}] ERROR: {e}")
                predictions[material][f"fold{fold_idx}"] = fp
            del model

        # Compute stats and save
        results = []
        for material in sorted(material_clusters.keys()):
            exp_val = exp_values.get(material)
            mat_preds = predictions[material]
            fold_means: list[float] = []
            fold_details: dict[str, dict] = {}
            for fn in sorted(mat_preds.keys()):
                vp = mat_preds[fn]
                if not vp:
                    continue
                vals = list(vp.values())
                fm = float(np.mean(vals))
                fold_means.append(fm)
                fold_details[fn] = {
                    "n1": vp.get("cluster_n1"), "n2": vp.get("cluster_n2"),
                    "n3": vp.get("cluster_n3"), "mean": fm,
                    "cluster_std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                }
            if not fold_means:
                continue
            # Convention: pool clusters within each fold (-> fold mean), then report
            # ddof=1 sample std across the 5 folds. This separates model deviation
            # from within-fold cluster noise; cluster noise is already absorbed
            # into each fold mean and reported separately as `cluster_std`.
            gm = float(np.mean(fold_means))
            ms = float(np.std(fold_means, ddof=1)) if len(fold_means) >= 2 else 0.0
            err = gm - exp_val if exp_val is not None else None
            results.append({
                "material": material,
                "exp_m_s": exp_val,
                "predictions": fold_details,
                "grand_mean_m_s": gm,
                "model_std_m_s": ms,
                "error_m_s": err,
                "abs_error_m_s": abs(err) if err is not None else None,
                "n_folds": len(fold_means),
            })

        out = ROOT / f"pems_ood_5fold_{family}.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved: {out}")

        # Print table
        print(f"\n{'Material':<16} {'Exp':>8} {'Grand':>8} {'ModelStd':>9} {'|Error|':>8}")
        print("-" * 55)
        for r in results:
            e = f"{r['exp_m_s']:.0f}" if r.get("exp_m_s") else "N/A"
            ae = f"{r['abs_error_m_s']:.1f}" if r.get("abs_error_m_s") is not None else "N/A"
            print(f"{r['material']:<16} {e:>8} {r['grand_mean_m_s']:>8.1f} {r['model_std_m_s']:>9.1f} {ae:>8}")


# ---------------------------------------------------------------------------
# Subcommand: uq
# ---------------------------------------------------------------------------

def cmd_uq(args: argparse.Namespace) -> None:
    """Model deviation UQ: IND + OOD, 5 folds × 3 cluster variants."""
    from deepmd.pt.infer.deep_eval import DeepProperty

    requested = set(args.series) if args.series else {"exp7a_lr1e4"}
    families_to_run = [f for f in FOLD_FAMILIES if f in requested]
    if not families_to_run:
        print(f"No matching families. Available: {list(FOLD_FAMILIES.keys())}")
        sys.exit(1)

    exp_values = load_exp_values()

    for family in families_to_run:
        cfg = FOLD_FAMILIES[family]
        splits_file: Path = cfg["splits_file"]
        head: str | None = cfg["head"]

        if not splits_file.exists():
            print(f"  SKIP {family}: splits file not found")
            continue

        splits = load_splits(splits_file)
        all_train_materials: set[str] = set()
        for fold_mats in splits.values():
            all_train_materials.update(fold_mats)

        # Build OOD clusters
        ood_clusters = build_ood_clusters(CLEANED_CIF_DIR, skip_materials=all_train_materials)

        # Load IND systems (pre-built npy)
        ind_systems: dict[str, dict[str, Path]] = {}
        for material in all_train_materials:
            variants: dict[str, Path] = {}
            for vname in ("cluster_n1", "cluster_n2", "cluster_n3"):
                p = DATA_ROOT / f"pems_{vname}_systems" / material
                if p.exists():
                    variants[vname] = p
            if variants:
                ind_systems[material] = variants

        # Determine which fold holds out each IND material
        material_to_fold: dict[str, int] = {}
        for fold_idx_str, mats in splits.items():
            for m in mats:
                material_to_fold[m] = int(fold_idx_str)

        # Run 5-fold models
        # ood_preds[material][fold_name][variant] = value
        ood_preds: dict[str, dict[str, dict[str, float]]] = {m: {} for m in ood_clusters}
        # ind_preds[material][fold_name][variant] = value
        ind_preds: dict[str, dict[str, dict[str, float]]] = {m: {} for m in ind_systems}

        for fold_idx in range(5):
            exp_name = f"{family}_fold{fold_idx}"
            exp_dir = _exp_base_dir(family) / exp_name
            if not exp_dir.exists():
                continue
            ckpt = resolve_checkpoint(exp_dir)
            if ckpt is None:
                continue
            print(f"\n  Loading fold{fold_idx}: {ckpt.name}")
            if head:
                model = DeepProperty(str(ckpt), head=head)
            else:
                model = DeepProperty(str(ckpt))
            model_tm = model.get_type_map()
            fold_key = f"fold{fold_idx}"

            for material, clusters in ood_clusters.items():
                fp: dict[str, float] = {}
                for vname, catoms in clusters.items():
                    try:
                        fp[vname] = predict_atoms(model, model_tm, catoms)
                    except Exception as e:
                        print(f"    OOD [{material}/{vname}/fold{fold_idx}] ERROR: {e}")
                ood_preds[material][fold_key] = fp

            for material, variants in ind_systems.items():
                fp2: dict[str, float] = {}
                for vname, sys_dir in variants.items():
                    try:
                        coords, cells, symbols = load_cluster_system(sys_dir)
                        preds = predict_system(model, model_tm, coords, cells, symbols)
                        fp2[vname] = float(np.mean(preds))
                    except Exception as e:
                        print(f"    IND [{material}/{vname}/fold{fold_idx}] ERROR: {e}")
                ind_preds[material][fold_key] = fp2
            del model

        # Compute stats
        def _compute_stats(
            material: str,
            preds_by_fold: dict[str, dict[str, float]],
            exp_val: float | None,
            heldout_fold: int | None,
        ) -> dict:
            fold_means: list[float] = []
            fold_details: dict[str, dict] = {}
            for fn in sorted(preds_by_fold.keys()):
                vp = preds_by_fold[fn]
                if not vp:
                    continue
                vals = list(vp.values())
                fm = float(np.mean(vals))
                fold_means.append(fm)
                fold_details[fn] = {
                    "mean": fm,
                    "cluster_std": float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0,
                    **vp,
                }
            if not fold_means:
                return {}
            # Convention: pool clusters within each fold (-> fold mean), then report
            # ddof=1 sample std across the 5 folds. Pure model deviation; cluster
            # noise lives in `cluster_std`.
            gm = float(np.mean(fold_means))
            ms = float(np.std(fold_means, ddof=1)) if len(fold_means) >= 2 else 0.0
            err = gm - exp_val if exp_val is not None else None
            # For IND: use only the held-out fold's prediction as the "honest" error
            honest_pred = None
            if heldout_fold is not None:
                hk = f"fold{heldout_fold}"
                if hk in fold_details:
                    honest_pred = fold_details[hk]["mean"]
            honest_err = (honest_pred - exp_val) if (honest_pred is not None and exp_val is not None) else None
            return {
                "material": material,
                "exp_m_s": exp_val,
                "predictions": fold_details,
                "grand_mean_m_s": gm,
                "model_std_m_s": ms,
                "error_m_s": err,
                "abs_error_m_s": abs(err) if err is not None else None,
                "honest_pred_m_s": honest_pred,
                "honest_error_m_s": honest_err,
                "honest_abs_error_m_s": abs(honest_err) if honest_err is not None else None,
                "heldout_fold": heldout_fold,
                "n_folds": len(fold_means),
                "is_ood": heldout_fold is None,
            }

        all_results = []
        for material in sorted(ood_clusters.keys()):
            r = _compute_stats(material, ood_preds[material], exp_values.get(material), None)
            if r:
                all_results.append(r)
        for material in sorted(ind_systems.keys()):
            hf = material_to_fold.get(material)
            r = _compute_stats(material, ind_preds[material], exp_values.get(material), hf)
            if r:
                all_results.append(r)

        out = ROOT / f"pems_ood_model_deviation_{family}.json"
        out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nSaved: {out}")

        # UQ calibration -- consolidated single JSON keyed by family.
        with_exp = [r for r in all_results if r.get("abs_error_m_s") is not None]
        if len(with_exp) >= 3:
            try:
                from scipy import stats as sp
                stds = np.array([r["model_std_m_s"] for r in with_exp])
                errs = np.array([r["abs_error_m_s"] for r in with_exp])
                rho, p = sp.spearmanr(stds, errs)
                print(f"\nUQ: Spearman ρ(model_std, |error|) = {rho:.3f} (p={p:.3e})")
                calib_entry = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "mean_model_std": float(stds.mean()),
                    "mean_abs_error": float(errs.mean()),
                    "n_materials": int(len(with_exp)),
                }
                calib_path = ROOT / "pems_uq_calibration.json"
                calib_db: dict[str, dict] = {}
                if calib_path.exists():
                    try:
                        calib_db = json.loads(calib_path.read_text(encoding="utf-8"))
                    except Exception:
                        calib_db = {}
                calib_db[family] = calib_entry
                calib_path.write_text(json.dumps(calib_db, indent=2), encoding="utf-8")
                print(f"Saved UQ calibration ({family}) → {calib_path.name}")
            except ImportError:
                print("scipy not available, skipping UQ calibration")


# ---------------------------------------------------------------------------
# Subcommand: single
# ---------------------------------------------------------------------------

def cmd_single(args: argparse.Namespace) -> None:
    """Predict a single material from pre-built npy systems."""
    from deepmd.pt.infer.deep_eval import DeepProperty

    material = args.material
    exp_values = load_exp_values()
    exp_val = exp_values.get(material)

    requested_series = args.series or list(FOLD_FAMILIES.keys()) + list(SINGLE_MODEL_EXPS.keys())

    print(f"\n=== Single-material inference: {material} ===")
    if exp_val is not None:
        print(f"Experimental Vdet: {exp_val:.0f} m/s")

    for series in requested_series:
        # Determine if fold family or single model
        if series in FOLD_FAMILIES:
            cfg = FOLD_FAMILIES[series]
            head = cfg["head"]
            fold_preds: list[float] = []
            for fold_idx in range(5):
                exp_dir = _exp_base_dir(series) / f"{series}_fold{fold_idx}"
                if not exp_dir.exists():
                    continue
                ckpt = resolve_checkpoint(exp_dir)
                if ckpt is None:
                    continue
                # Try all 3 cluster variants
                variant_preds: list[float] = []
                for vname in ("cluster_n1", "cluster_n2", "cluster_n3"):
                    sys_dir = DATA_ROOT / f"pems_{vname}_systems" / material
                    if not sys_dir.exists():
                        continue
                    try:
                        if head:
                            model = DeepProperty(str(ckpt), head=head)
                        else:
                            model = DeepProperty(str(ckpt))
                        model_tm = model.get_type_map()
                        coords, cells, symbols = load_cluster_system(sys_dir)
                        preds = predict_system(model, model_tm, coords, cells, symbols)
                        variant_preds.append(float(np.mean(preds)))
                        del model
                    except Exception as e:
                        print(f"  [{series}_fold{fold_idx}/{vname}] ERROR: {e}")
                if variant_preds:
                    fold_preds.append(float(np.mean(variant_preds)))
            if fold_preds:
                gm = float(np.mean(fold_preds))
                ms = float(np.std(fold_preds))
                err = f" | error={gm - exp_val:+.0f}" if exp_val else ""
                print(f"  {series}: {gm:.1f} ± {ms:.1f} m/s (5-fold){err}")
        elif series in SINGLE_MODEL_EXPS:
            cfg = SINGLE_MODEL_EXPS[series]
            head = cfg["head"]
            exp_dir = _exp_base_dir(series) / series
            if not exp_dir.exists():
                print(f"  SKIP {series}: directory not found")
                continue
            ckpt = resolve_checkpoint(exp_dir)
            if ckpt is None:
                print(f"  SKIP {series}: no checkpoint")
                continue
            variant_preds = []
            for vname in ("cluster_n1", "cluster_n2", "cluster_n3"):
                sys_dir = DATA_ROOT / f"pems_{vname}_systems" / material
                if not sys_dir.exists():
                    continue
                try:
                    if head:
                        model = DeepProperty(str(ckpt), head=head)
                    else:
                        model = DeepProperty(str(ckpt))
                    model_tm = model.get_type_map()
                    coords, cells, symbols = load_cluster_system(sys_dir)
                    preds = predict_system(model, model_tm, coords, cells, symbols)
                    variant_preds.append(float(np.mean(preds)))
                    del model
                except Exception as e:
                    print(f"  [{series}/{vname}] ERROR: {e}")
            if variant_preds:
                gm = float(np.mean(variant_preds))
                ms = float(np.std(variant_preds))
                err = f" | error={gm - exp_val:+.0f}" if exp_val else ""
                print(f"  {series}: {gm:.1f} ± {ms:.1f} m/s (3 variants){err}")
        else:
            print(f"  SKIP {series}: not in FOLD_FAMILIES or SINGLE_MODEL_EXPS")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_cv_parity(
    fold_results: dict[str, dict[int, tuple]],
    out_path: Path,
) -> None:
    """Parity plot: per-family, cluster-pooled within each fold.

    Convention (matches manuscript Table 2 / `model_std_m_s`):
      1. Within each held-out fold, group raw predictions by material name and
         average the 3 cluster realizations (n1/n2/n3) into a single fold-level
         prediction. The y-axis error bar is the within-material cluster std
         (ddof=1, n=3 cluster realizations).
      2. MAE and RMSE are computed on the cluster-pooled values (1 point per
         held-out material), not on the raw 3-per-material samples.
    """
    setup_nature_style()
    families = [f for f, folds in fold_results.items() if folds]
    if not families:
        return

    ncols = min(3, len(families))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), squeeze=False)

    for idx, family in enumerate(families):
        ax = axes[idx // ncols][idx % ncols]
        folds = fold_results[family]
        all_gt: list[float] = []
        all_pred: list[float] = []
        all_yerr: list[float] = []
        for fold_idx in sorted(folds.keys()):
            gt, pred, names, _ = folds[fold_idx]
            if len(gt) == 0:
                continue
            # Cluster-pool by material name within this fold
            mat_to_preds: dict[str, list[float]] = {}
            mat_to_gt: dict[str, float] = {}
            for n, g, p in zip(names, gt, pred):
                mat_to_preds.setdefault(n, []).append(float(p))
                mat_to_gt[n] = float(g)  # gt is identical across cluster realizations
            for mat in sorted(mat_to_preds.keys()):
                vals = mat_to_preds[mat]
                all_gt.append(mat_to_gt[mat])
                all_pred.append(float(np.mean(vals)))
                all_yerr.append(float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0)
        if not all_gt:
            ax.set_visible(False)
            continue
        gt_arr = np.asarray(all_gt)
        pred_arr = np.asarray(all_pred)
        yerr_arr = np.asarray(all_yerr)
        errors = np.abs(gt_arr - pred_arr)
        mae = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean((gt_arr - pred_arr) ** 2)))
        mean_cluster_std = float(np.mean(yerr_arr))

        color = EXP_COLORS.get(family, EXP_COLORS.get("exp7a", "#205C77"))
        ax.errorbar(
            gt_arr, pred_arr, yerr=yerr_arr,
            fmt="o", markersize=4.5, mfc=color, mec="white", mew=0.4,
            ecolor=color, elinewidth=0.7, capsize=2.0, capthick=0.6,
            alpha=0.75, zorder=3,
        )
        vmin = min(gt_arr.min(), pred_arr.min())
        vmax = max(gt_arr.max(), pred_arr.max())
        span = max(vmax - vmin, 1.0)
        pad = 0.08 * span
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "--", lw=1.0, color="#474747", zorder=1)
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xlabel("Ground Truth (m/s)", fontsize=9)
        ax.set_ylabel("Predicted (m/s)", fontsize=9)
        ax.set_title(f"{family}\n5-fold CV (cluster-pooled)", fontsize=9)
        ax.text(
            0.04, 0.96,
            f"MAE={mae:.1f}\nRMSE={rmse:.1f}\n"
            r"$\overline{\sigma}_{\rm cluster}$=" + f"{mean_cluster_std:.1f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CFCFCF", alpha=0.9),
        )
        style_axes(ax, grid=True)
        ax.set_aspect("equal", adjustable="box")

    for idx in range(len(families), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("PEMs 5-fold CV parity (cluster-pooled, error bar = within-material cluster σ)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_png_pdf(fig, out_path, dpi=300)
    plt.close(fig)
    print(f"Saved parity plot: {out_path}")


def _plot_sensitivity_heatmap(
    summary: dict[str, dict],
    prediction_rows: list[dict],
    out_path: Path,
) -> None:
    """Sensitivity heatmap: ΔMAE (perturbed − original cluster)."""
    setup_nature_style()
    comparisons = [
        ("pems_mod_rotation_n", "pems_cluster_n", "rotation"),
        ("pems_mod_translation_n", "pems_cluster_n", "translation"),
        ("pems_dap4_template_n", "pems_cluster_n", "dap4_template"),
    ]
    # Collect families that have sensitivity data
    families_with_data = sorted({
        k.split("__")[0].replace("_5fold_cv", "")
        for k in summary
        if "__pems_cluster_n" in k or "__pems_mod_" in k or "__pems_dap4_" in k
    })
    if not families_with_data:
        print("No sensitivity data found, skipping heatmap")
        return

    delta_mat = np.full((len(families_with_data), len(comparisons)), np.nan)
    for i, family in enumerate(families_with_data):
        for j, (pert_prefix, base_prefix, _label) in enumerate(comparisons):
            pert_maes, base_maes = [], []
            for n in ("1", "2", "3"):
                base_key = f"{family}_5fold_cv__{base_prefix}{n}"
                pert_key = f"{family}_5fold_cv__{pert_prefix}{n}"
                if base_key in summary and pert_key in summary:
                    base_maes.append(summary[base_key]["mae_m_s"])
                    pert_maes.append(summary[pert_key]["mae_m_s"])
            if base_maes:
                delta_mat[i, j] = float(np.mean(pert_maes)) - float(np.mean(base_maes))

    valid = delta_mat[~np.isnan(delta_mat)]
    if valid.size == 0:
        print("No valid sensitivity data, skipping heatmap")
        return
    abs_max = max(float(np.abs(valid).max()), 1.0)
    pert_labels = [label for _, _, label in comparisons]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, max(3.0, 0.5 * len(families_with_data) + 1.5)))
    delta_norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    im = ax.imshow(delta_mat, cmap="RdBu_r", norm=delta_norm, aspect="auto")
    ax.set_xticks(np.arange(len(pert_labels)))
    ax.set_xticklabels(pert_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(families_with_data)))
    ax.set_yticklabels(families_with_data, fontsize=9)
    ax.set_title("Sensitivity ΔMAE (perturbed − original, m/s)", fontsize=10)
    for i in range(len(families_with_data)):
        for j in range(len(pert_labels)):
            val = delta_mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="ΔMAE (m/s)")
    style_axes(ax, grid=False)
    fig.tight_layout()
    save_png_pdf(fig, out_path, dpi=300)
    plt.close(fig)
    print(f"Saved sensitivity heatmap: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified PEMs inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # cv
    p_cv = sub.add_parser("cv", help="IND 5-fold CV inference + parity plots + sensitivity")
    p_cv.add_argument("--series", nargs="+", default=None,
                      help="Experiment families to run (default: all). E.g. exp7a exp8a exp9a")
    p_cv.add_argument("--no-plot", action="store_true", help="Skip plotting")
    p_cv.add_argument("--no-sensitivity", action="store_true", help="Skip sensitivity analysis")
    p_cv.add_argument("--no-save", action="store_true",
                      help="Skip writing pems_inference_summary.json / pems_predictions.json"
                      " (use for monitoring runs against in-progress checkpoints)")

    # ood
    p_ood = sub.add_parser("ood", help="OOD 5-fold ensemble from cleaned CIFs")
    p_ood.add_argument("--series", nargs="+", default=None,
                       help="Experiment families to run (default: all)")

    # uq
    p_uq = sub.add_parser("uq", help="Model deviation UQ (IND + OOD)")
    p_uq.add_argument("--series", nargs="+", default=["exp7a_lr1e4"],
                      help="Experiment families (default: exp7a_lr1e4)")

    # single
    p_single = sub.add_parser("single", help="Single-material inference from pre-built npy")
    p_single.add_argument("--material", required=True, help="Material name (e.g. DAC-4)")
    p_single.add_argument("--series", nargs="+", default=None,
                          help="Series to evaluate (default: all)")

    args = parser.parse_args()

    if args.subcommand == "cv":
        cmd_cv(args)
    elif args.subcommand == "ood":
        cmd_ood(args)
    elif args.subcommand == "uq":
        cmd_uq(args)
    elif args.subcommand == "single":
        cmd_single(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
