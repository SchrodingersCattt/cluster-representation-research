#!/usr/bin/env python3
"""
Predict detonation velocity for the SY series: SY, PEP, HPEP, MPEP.

This script is INDEPENDENT of the EAP-4 pipeline (predict_eap4_paph6.py).

Materials:
  SY   = (H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄  — real CIF (ABX₄, P1, Z=4)
  PEP  = (H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄     — real CIF (ABX₄, P1)
  MPEP = (MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄    — real CIF (ABX₄, P1, Z=4)
  HPEP = (H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄   — no real CIF; use SY template

Prediction modes:
  1. Direct from CIF (SY, PEP, MPEP): 9 cluster variants each
  2. SY template (A-site substitution): HPEP, PEP, MPEP (9 clusters each)

Cluster CIFs are saved to:
  experiments/00_data_prep/pems_cluster_cifs_ood/

Usage:
    python -u \\
        experiments/predict_sy_series.py [--exp EXP_NAME] [--use-exp7a-folds]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.operations.defects import VacancyGenerator
from molcrys_kit.operations.molecule_manipulation import MoleculeManipulator, MoleculeClashError
from molcrys_kit.structures.crystal import MolecularCrystal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments"
CLEANED_CIF_DIR = EXP_DIR / "00_data_prep" / "pems_cleaned_cifs"
CLUSTER_CIF_OOD_DIR = EXP_DIR / "00_data_prep" / "pems_cluster_cifs_ood"
CLUSTER_N1_DIR = EXP_DIR / "00_data_prep" / "pems_cluster_n1_systems"
DEFAULT_EXP = "exp6v2_allpems"
EXP7A_FOLDS = [f"ablation/exp7a_lr1e4_fold{i}" for i in range(5)]  # ablation/
EXP7A_BASE_FOLDS = [f"exp7a_fold{i}" for i in range(5)]
HEAD = "pems_vdet_kj"

# ---------------------------------------------------------------------------
# Bond thresholds
# ---------------------------------------------------------------------------
_HEAVY = ["I", "Na", "K", "Rb", "Ba", "Ag"]
_ORGANIC = ["C", "H", "N", "O", "Cl"]

PEM_BOND_THRESHOLDS: dict[tuple[str, str], float] = {}
_HEAVY_NONO_LIMITS: dict[str, float] = {
    "I": 2.10, "Na": 2.30, "K": 2.50, "Rb": 2.60, "Ba": 2.60, "Ag": 2.30,
}
_HEAVY_O_LIMITS: dict[str, float] = {
    "I": 2.05, "Na": 2.20, "K": 2.30, "Rb": 2.40, "Ba": 2.40, "Ag": 2.20,
}
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

# ---------------------------------------------------------------------------
# Cluster build parameters
# ---------------------------------------------------------------------------
CLUSTER_BOX = np.eye(3, dtype=np.float64) * 100.0
CLUSTER_RANDOM_SEEDS = {
    "cluster_n1": 101, "cluster_n2": 202, "cluster_n3": 303,
}
CLUSTER_VARIANT_OFFSETS = {
    "cluster_n1": 0, "cluster_n2": 1, "cluster_n3": 2,
}
SUPERCELL_SCHEDULE = [(2, 2, 2), (3, 3, 3), (2, 3, 3), (3, 3, 4), (4, 4, 4)]
MIN_DISTINCT_SEEDS = 3

TYPE_MAP = [
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce',
    'Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir',
    'Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm',
    'Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc',
    'Lv','Ts','Og',
]
TYPE_TO_ID = {sym: i for i, sym in enumerate(TYPE_MAP)}

METAL_ELEMENTS = {
    'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
    'Sb','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
    'Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
}


# ---------------------------------------------------------------------------
# Cluster building utilities
# ---------------------------------------------------------------------------

def _get_seed_com(supercell: MolecularCrystal, seed_species_id: str,
                  mol_index: int) -> np.ndarray:
    mol = supercell.molecules[mol_index]
    return np.array(mol.get_center_of_mass(), dtype=np.float64)


def _select_spread_seeds(supercell: MolecularCrystal, analyzer: StoichiometryAnalyzer,
                         seed_species_id: str, n: int = 3) -> list[int]:
    candidates = list(analyzer.species_map.get(seed_species_id, []))
    if len(candidates) <= n:
        return candidates

    coms = np.array([_get_seed_com(supercell, seed_species_id, idx)
                     for idx in candidates])
    centroid = coms.mean(axis=0)

    dists_to_centroid = np.linalg.norm(coms - centroid, axis=1)
    first = int(np.argmin(dists_to_centroid))
    selected = [first]

    for _ in range(n - 1):
        best_idx = -1
        best_min_dist = -1.0
        for ci, cand_idx in enumerate(candidates):
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


def build_seeded_stoichiometric_cluster(
    crystal: MolecularCrystal,
    dataset_name: str,
    seed: int,
) -> tuple[MolecularCrystal, int, tuple[int, ...]]:
    offset = CLUSTER_VARIANT_OFFSETS[dataset_name]

    for sc_dims in SUPERCELL_SCHEDULE:
        supercell = crystal.get_supercell(*sc_dims)
        analyzer = StoichiometryAnalyzer(supercell)
        simplest_unit = analyzer.get_simplest_unit()

        seed_species_id = next(iter(simplest_unit.keys()))
        seed_candidates = analyzer.species_map.get(seed_species_id, [])
        if not seed_candidates:
            continue

        spread_seeds = _select_spread_seeds(supercell, analyzer, seed_species_id,
                                            n=MIN_DISTINCT_SEEDS)
        if len(spread_seeds) >= MIN_DISTINCT_SEEDS:
            seed_index = spread_seeds[offset % len(spread_seeds)]
            generator = VacancyGenerator(supercell)
            _, removed_cluster = generator.generate_vacancy(
                target_spec=simplest_unit,
                seed_index=seed_index,
                return_removed_cluster=True,
                random_seed=seed,
            )
            return removed_cluster, seed_index, sc_dims

    # Fallback
    supercell = crystal.get_supercell(*SUPERCELL_SCHEDULE[-1])
    analyzer = StoichiometryAnalyzer(supercell)
    simplest_unit = analyzer.get_simplest_unit()
    seed_species_id = next(iter(simplest_unit.keys()))
    seed_candidates = analyzer.species_map.get(seed_species_id, [])
    if not seed_candidates:
        raise RuntimeError(f"No seed candidates for species {seed_species_id}")
    seed_index = seed_candidates[offset % len(seed_candidates)]
    generator = VacancyGenerator(supercell)
    _, removed_cluster = generator.generate_vacancy(
        target_spec=simplest_unit,
        seed_index=seed_index,
        return_removed_cluster=True,
        random_seed=seed,
    )
    return removed_cluster, seed_index, SUPERCELL_SCHEDULE[-1]


def crystal_to_minimum_image_atoms(crystal: MolecularCrystal) -> Atoms:
    unwrapped = crystal.get_unwrapped_molecules()
    molecules = [mol.copy() for mol in unwrapped]
    if not molecules:
        raise RuntimeError("Crystal contains no molecules")

    ref = molecules[0].get_center_of_mass()
    inv_lattice = np.linalg.inv(np.array(crystal.lattice, dtype=np.float64))
    for mol in molecules[1:]:
        shift_cart = mol.get_center_of_mass() - ref
        shift_frac = np.dot(shift_cart, inv_lattice)
        shift_frac -= np.round(shift_frac)
        wrapped_cart = np.dot(shift_frac, crystal.lattice)
        mol.positions += wrapped_cart - shift_cart

    combined = MolecularCrystal(
        lattice=np.array(crystal.lattice, dtype=np.float64).copy(),
        molecules=molecules,
        pbc=crystal.pbc,
    )
    return combined.to_ase()


def cluster_atoms_centered(atoms: Atoms) -> Atoms:
    """Center cluster at (50,50,50) in 100 Å box, set nopbc."""
    a = atoms.copy()
    a.positions = (
        a.get_positions()
        - a.get_positions().mean(axis=0, keepdims=True)
        + np.array([[50.0, 50.0, 50.0]])
    )
    a.set_cell(CLUSTER_BOX)
    a.set_pbc(False)
    return a


# ---------------------------------------------------------------------------
# Molecule classification
# ---------------------------------------------------------------------------

def is_x_site_candidate(sym_set: set[str]) -> bool:
    has_oxygen = "O" in sym_set
    has_halogen = bool(sym_set & {"Cl", "I"})
    nitrate_like = "N" in sym_set and "H" not in sym_set
    return has_oxygen and (has_halogen or nitrate_like)


def classify_mol(mol) -> str:
    syms = mol.get_chemical_symbols()
    sym_set = set(syms)
    has_C = 'C' in sym_set
    has_halogen_or_N = bool(sym_set & {'Cl', 'I', 'N'})
    has_metal = bool(sym_set & METAL_ELEMENTS)
    if has_C:
        return 'A'
    if has_metal and "O" not in sym_set and not has_halogen_or_N:
        return 'B'
    if is_x_site_candidate(sym_set) and not has_C:
        return 'X'
    if len(syms) <= 6 and not has_C:
        return 'B'
    return 'B'


def get_abx_indices(mc: MolecularCrystal) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {'A': [], 'B': [], 'X': []}
    for i, mol in enumerate(mc.molecules):
        mapping[classify_mol(mol)].append(i)
    return mapping


def canonical_formula(mol) -> str:
    from collections import Counter
    counts = Counter(mol.get_chemical_symbols())
    parts = []
    for elem in ['C', 'H']:
        if elem in counts:
            n = counts.pop(elem)
            parts.append(f"{elem}{n if n > 1 else ''}")
    for elem in sorted(counts):
        n = counts[elem]
        parts.append(f"{elem}{n if n > 1 else ''}")
    return "".join(parts)


def mc_to_atoms(mc: MolecularCrystal) -> Atoms:
    """Convert MolecularCrystal to centered ASE Atoms (nopbc cluster)."""
    all_symbols = []
    all_positions = []
    for mol in mc.molecules:
        all_symbols.extend(mol.get_chemical_symbols())
        all_positions.extend(mol.get_positions().tolist())
    positions = np.array(all_positions)
    positions = (
        positions
        - positions.mean(axis=0, keepdims=True)
        + np.array([[50.0, 50.0, 50.0]])
    )
    return Atoms(symbols=all_symbols, positions=positions, cell=CLUSTER_BOX, pbc=False)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, model_type_map: list[str], atoms: Atoms) -> float:
    symbols = atoms.get_chemical_symbols()
    unique_types = sorted(set(symbols), key=lambda s: TYPE_TO_ID.get(s, 999))
    atom_types_local = np.array([unique_types.index(s) for s in symbols], dtype=np.int32)
    atom_types_model = np.array([
        np.where(np.array(model_type_map) == t)[0][0]
        for t in unique_types
    ], dtype=np.int32)
    atom_types_for_model = np.array([
        atom_types_model[at] for at in atom_types_local
    ], dtype=np.int32)

    coords = atoms.get_positions().reshape(1, -1, 3)
    pred = model.eval(coords=coords, atom_types=atom_types_for_model, cells=None)
    return float(pred[0].reshape(-1)[0])


# ---------------------------------------------------------------------------
# Predict from CIF (direct, no substitution)
# ---------------------------------------------------------------------------

def predict_from_cif(
    cif_path: Path,
    material_name: str,
    model,
    model_type_map: list[str],
    save_cif: bool = True,
) -> dict:
    """Build 3 clusters from a CIF and predict Vdet. Saves cluster CIFs."""
    print(f"Loading crystal from {cif_path}...")
    crystal = MolecularCrystal.from_ase(
        ase_read(str(cif_path)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    print(f"  Crystal: {len(crystal.molecules)} molecules")
    for i, mol in enumerate(crystal.molecules):
        formula = canonical_formula(mol)
        site = classify_mol(mol)
        print(f"  Mol {i}: {formula} → {site}")

    preds: dict[str, float] = {}
    xyz_dir = EXP_DIR / "ood_cluster_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    # Per-material subdirectory: pems_cluster_cifs_ood/{material_name}/cluster_n1.cif ...
    mat_cif_dir = CLUSTER_CIF_OOD_DIR / material_name
    if save_cif:
        mat_cif_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms = cluster_atoms_centered(cluster_atoms)

            n_atoms = len(cluster_atoms)
            pred = run_inference(model, model_type_map, cluster_atoms)
            preds[dataset_name] = pred
            print(f"  [{dataset_name}] {n_atoms} atoms, seed_idx={seed_idx}, "
                  f"sc={sc_dims} → pred={pred:.1f} m/s")

            # Save XYZ
            ase_write(str(xyz_dir / f"{material_name}_{dataset_name}.xyz"), cluster_atoms)
            # Save CIF into per-material subdir as cluster_n1.cif / cluster_n2.cif / cluster_n3.cif
            if save_cif:
                ase_write(str(mat_cif_dir / f"{dataset_name}.cif"), cluster_atoms)
        except Exception as e:
            print(f"  [{dataset_name}] ERROR: {e}")
            import traceback; traceback.print_exc()

    if preds:
        vals = list(preds.values())
        mean_pred = float(np.mean(vals))
        std_pred = float(np.std(vals))
        print(f"\n  {material_name} ({len(vals)} clusters) "
              f"→ mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        result = {f"pred_{k}_m_s": v for k, v in preds.items()}
        result.update({
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "n_clusters": len(vals),
        })
        return result
    return {}


def predict_from_cif_ensemble(
    cif_path: Path,
    material_name: str,
    fold_models: list[tuple],
) -> dict:
    """5-fold × 9-cluster ensemble prediction from a CIF file."""
    crystal = MolecularCrystal.from_ase(
        ase_read(str(cif_path)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    cluster_atoms_list: list[tuple[str, Atoms]] = []
    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms = cluster_atoms_centered(cluster_atoms)
            cluster_atoms_list.append((dataset_name, cluster_atoms))
            print(f"  [{dataset_name}] built: {len(cluster_atoms)} atoms, "
                  f"seed_idx={seed_idx}, sc={sc_dims}")
        except Exception as e:
            print(f"  [{dataset_name}] ERROR building cluster: {e}")

    if not cluster_atoms_list:
        return {}

    all_preds: list[float] = []
    fold_means: list[float] = []

    for fi, (model, type_map) in enumerate(fold_models):
        fold_preds = []
        for dataset_name, cluster_atoms in cluster_atoms_list:
            try:
                pred = run_inference(model, type_map, cluster_atoms)
                fold_preds.append(pred)
                all_preds.append(pred)
            except Exception as e:
                print(f"  [fold{fi}/{dataset_name}] ERROR: {e}")
        if fold_preds:
            fold_means.append(float(np.mean(fold_preds)))
            print(f"  [fold{fi}] mean={fold_means[-1]:.1f} m/s "
                  f"({len(fold_preds)} clusters)")

    if all_preds:
        mean_pred = float(np.mean(all_preds))
        std_pred = float(np.std(all_preds))
        print(f"\n  {material_name} (5-fold ensemble, {len(all_preds)} preds) "
              f"→ mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "fold_means_m_s": fold_means,
            "n_preds": len(all_preds),
        }
    return {}


# ---------------------------------------------------------------------------
# Predict via SY template (A-site substitution only)
# ---------------------------------------------------------------------------

def _build_sy_substituted_clusters(
    sy_crystal: MolecularCrystal,
    a_mol,
    material_name: str,
    save_cif: bool = True,
) -> list[tuple[str, Atoms]]:
    """Build 9 SY-template clusters with A-site substituted. Returns list of (name, atoms)."""
    cluster_atoms_list: list[tuple[str, Atoms]] = []
    xyz_dir = EXP_DIR / "ood_cluster_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    # Per-material subdirectory: pems_cluster_cifs_ood/{material_name}_sytpl/cluster_n1.cif ...
    mat_cif_dir = CLUSTER_CIF_OOD_DIR / f"{material_name}_sytpl"
    if save_cif:
        mat_cif_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                sy_crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms_raw = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms_raw = cluster_atoms_centered(cluster_atoms_raw)

            template_mc = MolecularCrystal.from_ase(cluster_atoms_raw)
            template_abx = get_abx_indices(template_mc)

            np.random.seed(42)
            try:
                working_mc = template_mc
                for idx in template_abx['A']:
                    working_mc = MoleculeManipulator(working_mc).replace_molecule(
                        idx, a_mol, clash_threshold=0.8, max_rotation_attempts=200
                    )
                sub_atoms = mc_to_atoms(working_mc)
            except MoleculeClashError as e:
                print(f"  [{dataset_name}] Substitution FAILED (clash): {e}")
                continue
            except Exception as e:
                print(f"  [{dataset_name}] Build error: {e}")
                continue

            cluster_atoms_list.append((dataset_name, sub_atoms))
            print(f"  [{dataset_name}] {len(sub_atoms)} atoms, "
                  f"seed_idx={seed_idx}, sc={sc_dims}")

            # Save XYZ
            ase_write(str(xyz_dir / f"{material_name}_sytpl_{dataset_name}.xyz"), sub_atoms)
            # Save CIF into per-material subdir as cluster_n1.cif / cluster_n2.cif / cluster_n3.cif
            if save_cif:
                ase_write(str(mat_cif_dir / f"{dataset_name}.cif"), sub_atoms)
        except Exception as e:
            print(f"  [{dataset_name}] ERROR: {e}")
            import traceback; traceback.print_exc()

    return cluster_atoms_list


def predict_via_sy_template(
    material_name: str,
    a_mol,
    model,
    model_type_map: list[str],
    sy_crystal: MolecularCrystal,
) -> dict:
    """Predict using SY (ABX₄) template with A-site substitution only (9 clusters)."""
    cluster_atoms_list = _build_sy_substituted_clusters(sy_crystal, a_mol, material_name)

    preds: dict[str, float] = {}
    for dataset_name, sub_atoms in cluster_atoms_list:
        try:
            pred = run_inference(model, model_type_map, sub_atoms)
            preds[dataset_name] = pred
            print(f"  [{dataset_name}] → pred={pred:.1f} m/s")
        except Exception as e:
            print(f"  [{dataset_name}] inference ERROR: {e}")

    if preds:
        vals = list(preds.values())
        mean_pred = float(np.mean(vals))
        std_pred = float(np.std(vals))
        print(f"\n  {material_name} (SY tpl, {len(vals)} clusters) "
              f"→ mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        result = {f"pred_{k}_m_s": v for k, v in preds.items()}
        result.update({
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "n_clusters": len(vals),
        })
        return result
    return {}


def predict_via_sy_template_ensemble(
    material_name: str,
    a_mol,
    fold_models: list[tuple],
    sy_crystal: MolecularCrystal,
) -> dict:
    """5-fold × 9-cluster ensemble prediction via SY template."""
    # Build clusters once (reuse across folds); CIFs already saved by single-model run
    cluster_atoms_list = _build_sy_substituted_clusters(
        sy_crystal, a_mol, material_name, save_cif=False
    )

    if not cluster_atoms_list:
        return {}

    all_preds: list[float] = []
    fold_means: list[float] = []

    for fi, (model, type_map) in enumerate(fold_models):
        fold_preds = []
        for dataset_name, sub_atoms in cluster_atoms_list:
            try:
                pred = run_inference(model, type_map, sub_atoms)
                fold_preds.append(pred)
                all_preds.append(pred)
            except Exception as e:
                print(f"  [fold{fi}/{dataset_name}] ERROR: {e}")
        if fold_preds:
            fold_means.append(float(np.mean(fold_preds)))
            print(f"  [fold{fi}] mean={fold_means[-1]:.1f} m/s "
                  f"({len(fold_preds)} clusters)")

    if all_preds:
        mean_pred = float(np.mean(all_preds))
        std_pred = float(np.std(all_preds))
        print(f"\n  {material_name} (SY tpl, 5-fold ensemble, "
              f"{len(all_preds)} preds) → mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "fold_means_m_s": fold_means,
            "n_preds": len(all_preds),
        }
    return {}


# ---------------------------------------------------------------------------
# Load fold models
# ---------------------------------------------------------------------------

def _load_fold_models(exp_dir: Path, fold_names: list[str]) -> list[tuple]:
    from deepmd.pt.infer.deep_eval import DeepProperty
    fold_models = []
    for fold_name in fold_names:
        fold_path = exp_dir / fold_name
        ckpts = sorted(fold_path.glob("model.ckpt-*.pt"),
                       key=lambda p: int(p.stem.split("-")[1]))
        if not ckpts:
            print(f"  WARNING: no checkpoint in {fold_path}, skipping")
            continue
        ckpt = ckpts[-1]
        print(f"  Loading fold {fold_name}: {ckpt.name}")
        m = DeepProperty(str(ckpt), head=HEAD)
        fold_models.append((m, m.get_type_map()))
    return fold_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Vdet for SY series (SY, PEP, MPEP, HPEP) — standalone script")
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP,
                        help=f"Experiment directory name (default: {DEFAULT_EXP})")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Explicit checkpoint path")
    parser.add_argument("--use-exp7a-folds", action="store_true",
                        help="Also run 5-fold exp7a_lr1e4 and exp7a ensemble predictions")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Output JSON filename (default: sy_series_predictions.json)")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        exp_dir = EXP_DIR / args.exp
        ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
        if not ckpts:
            print(f"ERROR: no checkpoint found in {exp_dir}")
            sys.exit(1)
        ckpt_path = ckpts[-1]

    print(f"Checkpoint: {ckpt_path}")
    print(f"Head: {HEAD}")
    print()

    # Load model
    print("Loading model...")
    from deepmd.pt.infer.deep_eval import DeepProperty
    model = DeepProperty(str(ckpt_path), head=HEAD)
    model_type_map = model.get_type_map()
    print(f"Model type map: {model_type_map[:20]}...")
    print()

    # Load SY crystal
    sy_cif = CLEANED_CIF_DIR / "SY.cif"
    print(f"Loading SY crystal from {sy_cif}...")
    sy_crystal = MolecularCrystal.from_ase(
        ase_read(str(sy_cif)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    print(f"  SY crystal: {len(sy_crystal.molecules)} molecules")
    for i, mol in enumerate(sy_crystal.molecules):
        print(f"  Mol {i}: {canonical_formula(mol)} → {classify_mol(mol)}")
    print()

    # Extract A-site molecule from SY cluster (H₂dabco²⁺)
    # Build one cluster to get the A-site molecule
    _tmp_cluster, _, _ = build_seeded_stoichiometric_cluster(
        sy_crystal, dataset_name="cluster_n1", seed=101,
    )
    _tmp_atoms = crystal_to_minimum_image_atoms(_tmp_cluster)
    _tmp_atoms = cluster_atoms_centered(_tmp_atoms)
    _tmp_mc = MolecularCrystal.from_ase(_tmp_atoms)
    _tmp_abx = get_abx_indices(_tmp_mc)
    h2dabco_mol = _tmp_mc.molecules[_tmp_abx['A'][0]].copy()
    print(f"  H₂dabco²⁺ from SY cluster: {canonical_formula(h2dabco_mol)} "
          f"({len(h2dabco_mol.get_chemical_symbols())} atoms)")

    # Load A-site molecules for HPEP, PEP, MPEP from training cluster data
    def _get_a_mol_from_training(material: str) -> object:
        sys_dir = CLUSTER_N1_DIR / material
        type_map = (sys_dir / "type_map.raw").read_text(encoding="utf-8").strip().split()
        types = np.loadtxt(sys_dir / "type.raw", dtype=int)
        coord = np.load(sys_dir / "set.000" / "coord.npy")[0].reshape(-1, 3)
        symbols = [type_map[t] for t in types]
        atoms = Atoms(symbols=symbols, positions=coord, cell=CLUSTER_BOX, pbc=False)
        mc = MolecularCrystal.from_ase(atoms)
        abx = get_abx_indices(mc)
        if not abx['A']:
            raise RuntimeError(f"No A-site molecules found in {material}")
        mol = mc.molecules[abx['A'][0]]
        print(f"  A-site from {material}: {canonical_formula(mol)} "
              f"({len(mol.get_chemical_symbols())} atoms)")
        return mol.copy()

    print("\nLoading A-site molecules for substitution...")
    h2hpz_mol = _get_a_mol_from_training("PAP-H4")   # H₂hpz²⁺ (for HPEP)
    h2pz_mol = _get_a_mol_from_training("PAP-4")      # H₂pz²⁺  (for PEP)
    mehpz_mol = _get_a_mol_from_training("PAP-M4")    # MeHpz²⁺ (for MPEP)
    print()

    results: dict[str, dict] = {}

    # ===================================================================
    # 1. SY — direct from CIF
    # ===================================================================
    print("=" * 70)
    print("1. SY = (H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄ — direct from CIF")
    print("=" * 70)
    r = predict_from_cif(sy_cif, "SY", model, model_type_map)
    if r:
        results["SY (direct CIF, ABX₄)"] = {
            "composition": "(H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄",
            "template": "own CIF",
            "stoichiometry": "ABX₄",
            **r,
        }

    # ===================================================================
    # 2. PEP — direct from CIF
    # ===================================================================
    pep_cif = CLEANED_CIF_DIR / "PEP.cif"
    print("=" * 70)
    print("2. PEP = (H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄ — direct from CIF")
    print("=" * 70)
    r = predict_from_cif(pep_cif, "PEP", model, model_type_map)
    if r:
        results["PEP (direct CIF, ABX₄)"] = {
            "composition": "(H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄",
            "template": "own CIF",
            "stoichiometry": "ABX₄",
            **r,
        }

    # ===================================================================
    # 3. MPEP — direct from CIF
    # ===================================================================
    mpep_cif = CLEANED_CIF_DIR / "MPEP.cif"
    print("=" * 70)
    print("3. MPEP = (MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄ — direct from CIF")
    print("=" * 70)
    r = predict_from_cif(mpep_cif, "MPEP", model, model_type_map)
    if r:
        results["MPEP (direct CIF, ABX₄)"] = {
            "composition": "(MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄",
            "template": "own CIF",
            "stoichiometry": "ABX₄",
            **r,
        }

    # ===================================================================
    # 4. HPEP — direct from CIF
    # ===================================================================
    hpep_cif = CLEANED_CIF_DIR / "HPEP.cif"
    print("=" * 70)
    print("4. HPEP = (H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄ — direct from CIF")
    print("=" * 70)
    r = predict_from_cif(hpep_cif, "HPEP", model, model_type_map)
    if r:
        results["HPEP (direct CIF, ABX₄)"] = {
            "composition": "(H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄",
            "template": "own CIF",
            "stoichiometry": "ABX₄",
            **r,
        }

    # ===================================================================
    # 5-7. HPEP / PEP / MPEP — via SY template (A-site substitution)
    # ===================================================================
    print("=" * 70)
    print("5-7. HPEP / PEP / MPEP — via SY template (A-site substitution)")
    print("   B=H₂en²⁺ and X=ClO₄⁻ kept from SY template")
    print("=" * 70)

    sy_tpl_materials = [
        ("HPEP", "H₂hpz²⁺",  h2hpz_mol,  "(H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("PEP",  "H₂pz²⁺",   h2pz_mol,   "(H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("MPEP", "MeHpz²⁺",  mehpz_mol,  "(MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
    ]

    for mat_name, a_label, a_mol_sub, composition in sy_tpl_materials:
        print(f"\n--- {mat_name} via SY template: A={a_label} ---")
        r = predict_via_sy_template(mat_name, a_mol_sub, model, model_type_map, sy_crystal)
        if r:
            results[f"{mat_name} (SY tpl, ABX₄)"] = {
                "composition": composition,
                "template": "SY",
                "stoichiometry": "ABX₄",
                "a_site": a_label,
                "b_site": "H₂en²⁺",
                "x_site": "ClO₄⁻",
                **r,
            }

    # ===================================================================
    # 5-fold ensemble predictions (optional)
    # ===================================================================
    ensemble_results: dict[str, dict] = {}

    def _run_ensemble_block(fold_models: list[tuple], tag: str) -> None:
        # SY direct CIF
        print(f"\n--- SY (direct CIF) — {tag} ---")
        r = predict_from_cif_ensemble(sy_cif, "SY", fold_models)
        if r:
            ensemble_results[f"SY (direct CIF, ABX₄) [{tag}]"] = {
                "composition": "(H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄",
                "template": "own CIF", "stoichiometry": "ABX₄",
                "ensemble": tag, **r}

        # PEP direct CIF
        print(f"\n--- PEP (direct CIF) — {tag} ---")
        r = predict_from_cif_ensemble(pep_cif, "PEP", fold_models)
        if r:
            ensemble_results[f"PEP (direct CIF, ABX₄) [{tag}]"] = {
                "composition": "(H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄",
                "template": "own CIF", "stoichiometry": "ABX₄",
                "ensemble": tag, **r}

        # MPEP direct CIF
        print(f"\n--- MPEP (direct CIF) — {tag} ---")
        r = predict_from_cif_ensemble(mpep_cif, "MPEP", fold_models)
        if r:
            ensemble_results[f"MPEP (direct CIF, ABX₄) [{tag}]"] = {
                "composition": "(MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄",
                "template": "own CIF", "stoichiometry": "ABX₄",
                "ensemble": tag, **r}

        # HPEP direct CIF
        print(f"\n--- HPEP (direct CIF) — {tag} ---")
        r = predict_from_cif_ensemble(hpep_cif, "HPEP", fold_models)
        if r:
            ensemble_results[f"HPEP (direct CIF, ABX₄) [{tag}]"] = {
                "composition": "(H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄",
                "template": "own CIF", "stoichiometry": "ABX₄",
                "ensemble": tag, **r}

        # HPEP/PEP/MPEP via SY template
        for mat_name, a_label, a_mol_sub, composition in sy_tpl_materials:
            print(f"\n--- {mat_name} (SY tpl) — {tag} ---")
            r = predict_via_sy_template_ensemble(mat_name, a_mol_sub, fold_models, sy_crystal)
            if r:
                ensemble_results[f"{mat_name} (SY tpl, ABX₄) [{tag}]"] = {
                    "composition": composition,
                    "template": "SY", "stoichiometry": "ABX₄",
                    "a_site": a_label, "b_site": "H₂en²⁺", "x_site": "ClO₄⁻",
                    "ensemble": tag, **r}

    if args.use_exp7a_folds:
        print("\n" + "=" * 70)
        print("5-FOLD ENSEMBLE (exp7a_lr1e4, fold0–fold4, lr=1e-4)")
        print("=" * 70)
        fold_models_lr1e4 = _load_fold_models(EXP_DIR, EXP7A_FOLDS)
        if fold_models_lr1e4:
            _run_ensemble_block(fold_models_lr1e4, "exp7a_lr1e4")

        print("\n" + "=" * 70)
        print("5-FOLD ENSEMBLE (exp7a, fold0–fold4, base lr)")
        print("=" * 70)
        fold_models_base = _load_fold_models(EXP_DIR, EXP7A_BASE_FOLDS)
        if fold_models_base:
            _run_ensemble_block(fold_models_base, "exp7a")

    # ===================================================================
    # Summary table
    # ===================================================================
    print("\n" + "=" * 100)
    print("SUMMARY TABLE — single model")
    print("=" * 100)
    print(f"{'Label':<42} {'Template':<12} {'Stoich':<8} {'N':>4} {'Pred (m/s)':>12} {'Std':>8}")
    print("-" * 100)
    for label, r in results.items():
        pred_str = f"{r['pred_mean_m_s']:.1f}" if 'pred_mean_m_s' in r else "FAIL"
        std_str = f"{r['pred_std_m_s']:.1f}" if 'pred_std_m_s' in r else "-"
        n_str = str(r.get('n_clusters', '?'))
        print(f"{label:<42} {r.get('template','?'):<12} {r.get('stoichiometry','?'):<8} "
              f"{n_str:>4} {pred_str:>12} {std_str:>8}")
    print("=" * 100)

    if ensemble_results:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE — 5-fold ensemble")
        print("=" * 100)
        print(f"{'Label':<48} {'Template':<12} {'Stoich':<8} {'N':>5} {'Pred (m/s)':>12} {'Std':>8}")
        print("-" * 100)
        for label, r in ensemble_results.items():
            pred_str = f"{r['pred_mean_m_s']:.1f}" if 'pred_mean_m_s' in r else "FAIL"
            std_str = f"{r['pred_std_m_s']:.1f}" if 'pred_std_m_s' in r else "-"
            n_str = str(r.get('n_preds', '?'))
            print(f"{label:<48} {r.get('template','?'):<12} {r.get('stoichiometry','?'):<8} "
                  f"{n_str:>5} {pred_str:>12} {std_str:>8}")
        print("=" * 100)

    # Save JSON
    all_results = {**results, **ensemble_results}
    out_name = args.out_json or "sy_series_predictions.json"
    out_path = EXP_DIR / out_name
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")
    print(f"Cluster CIFs saved to: {CLUSTER_CIF_OOD_DIR}")


if __name__ == "__main__":
    main()
