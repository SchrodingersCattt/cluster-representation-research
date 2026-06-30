#!/usr/bin/env python3
"""
Predict detonation velocity for EAP-4, EAP-8, PAP-H6, MPEP, HPEP, PEP, SY
using the EAP-4 (A₂BX₅), DAP-4 (ABX₃/ABX₄), and SY (ABX₄) templates.

Materials:
  EAP-4  = (H₂en²⁺)₂(NH₄⁺)(ClO₄⁻)₅       — real CIF
  EAP-8  = (H₂en²⁺)₂(CH₃NH₃⁺)(ClO₄⁻)₅    — real CIF
  PAP-H6 = (H₂hpz²⁺)₂(NH₃OH⁺)(ClO₄⁻)₅    — hypothetical (ion substitution)
  MPEP   = (MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄      — hypothetical ABX₄
  HPEP   = (H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄     — hypothetical ABX₄
  PEP    = (H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄      — hypothetical ABX₄
  SY     = (H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄   — real CIF (ABX₄)

Templates:
  EAP-4 template (A₂BX₅) — correct stoichiometry for EAP/PAP-H6 series
  DAP-4 template (ABX₃) — for comparison (wrong stoichiometry for MPEP/HPEP/PEP/SY)
  DAP-4 template + extra ClO₄⁻ (ABX₄) — correct stoichiometry for MPEP/HPEP/PEP/SY
    (A and B are both divalent 2+, so 4× ClO₄⁻ needed for charge balance)
  SY template (ABX₄) — correct stoichiometry for MPEP/HPEP/PEP; substitute A-site only

Usage:
    python -u \
        experiments/predict_eap4_paph6.py [--exp EXP_NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
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
CLUSTER_N1_DIR = EXP_DIR / "00_data_prep" / "pems_cluster_n1_systems"
CLUSTER_N2_DIR = EXP_DIR / "00_data_prep" / "pems_cluster_n2_systems"
CLUSTER_N3_DIR = EXP_DIR / "00_data_prep" / "pems_cluster_n3_systems"
DEFAULT_EXP = "exp6v1_allpems"  # canonical single-model for production / OOD
# Two 5-fold ensemble variants
EXP7A_FOLDS = [f"ablation/exp7a_lr1e4_fold{i}" for i in range(5)]  # lr=1e-4 (ablation/)
EXP7A_BASE_FOLDS = [f"exp7a_fold{i}" for i in range(5)]         # base lr (exp7a)
HEAD = "pems_vdet_kj"

# ---------------------------------------------------------------------------
# Bond thresholds (from infer_ood_from_cleaned_cifs.py)
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
# 9 cluster variants: seeds and offsets for spread-seed selection
CLUSTER_RANDOM_SEEDS = {
    "cluster_n1": 101, "cluster_n2": 202, "cluster_n3": 303,
    "cluster_n4": 404, "cluster_n5": 505, "cluster_n6": 606,
    "cluster_n7": 707, "cluster_n8": 808, "cluster_n9": 909,
}
CLUSTER_VARIANT_OFFSETS = {
    "cluster_n1": 0, "cluster_n2": 1, "cluster_n3": 2,
    "cluster_n4": 0, "cluster_n5": 1, "cluster_n6": 2,
    "cluster_n7": 0, "cluster_n8": 1, "cluster_n9": 2,
}
# DAP-4 pre-built cluster dirs (3 variants: n1/n2/n3 from separate seed dirs)
# These are resolved at runtime after path constants are defined.
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
# Cluster building (from infer_ood_from_cleaned_cifs.py)
# ---------------------------------------------------------------------------

def _get_seed_com(supercell: MolecularCrystal, analyzer: StoichiometryAnalyzer,
                  seed_species_id: str, mol_index: int) -> np.ndarray:
    mol = supercell.molecules[mol_index]
    return np.array(mol.get_center_of_mass(), dtype=np.float64)


def _select_spread_seeds(supercell: MolecularCrystal, analyzer: StoichiometryAnalyzer,
                         seed_species_id: str, n: int = 3) -> list[int]:
    candidates = list(analyzer.species_map.get(seed_species_id, []))
    if len(candidates) <= n:
        return candidates

    coms = np.array([_get_seed_com(supercell, analyzer, seed_species_id, idx)
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
# Molecule classification (from predict_abx_grid.py)
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
            parts.append(f"{elem}{counts.pop(elem) if counts[elem] > 1 else ''}")
    for elem in sorted(counts):
        parts.append(f"{elem}{counts[elem] if counts[elem] > 1 else ''}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Species loading from existing cluster training data
# ---------------------------------------------------------------------------

def read_cluster_system(sys_dir: Path) -> tuple[np.ndarray, list[str], float]:
    type_map = (sys_dir / "type_map.raw").read_text(encoding="utf-8").strip().split()
    types = np.loadtxt(sys_dir / "type.raw", dtype=int)
    coord = np.load(sys_dir / "set.000" / "coord.npy")[0].reshape(-1, 3)
    prop = float(np.load(sys_dir / "set.000" / "property.npy")[0])
    symbols = [type_map[t] for t in types]
    return coord, symbols, prop


def build_molecular_crystal_from_npy(coord: np.ndarray, symbols: list[str]) -> MolecularCrystal:
    atoms = Atoms(symbols=symbols, positions=coord, cell=CLUSTER_BOX, pbc=False)
    return MolecularCrystal.from_ase(atoms)


def get_representative_mol(material: str, site: str):
    """Load a representative A/B/X molecule from an existing PEM cluster."""
    sys_dir = CLUSTER_N1_DIR / material
    coord, symbols, _ = read_cluster_system(sys_dir)
    mc = build_molecular_crystal_from_npy(coord, symbols)
    abx = get_abx_indices(mc)
    if not abx[site]:
        raise RuntimeError(f"No {site}-site molecules found in {material}")
    mol = mc.molecules[abx[site][0]]
    formula = canonical_formula(mol)
    print(f"  Representative {site}-site from {material}: {formula} "
          f"({len(mol.get_chemical_symbols())} atoms)")
    return mol.copy()


# ---------------------------------------------------------------------------
# ABX structure building with substitution
# ---------------------------------------------------------------------------

def build_substituted_structure(
    template_mc: MolecularCrystal,
    template_abx: dict[str, list[int]],
    a_mol=None,
    b_mol=None,
    x_mol=None,
) -> MolecularCrystal | None:
    """Replace A/B/X molecules in template. Only replaces if mol is not None."""
    np.random.seed(42)
    try:
        working_mc = template_mc
        if a_mol is not None:
            for idx in template_abx['A']:
                working_mc = MoleculeManipulator(working_mc).replace_molecule(
                    idx, a_mol, clash_threshold=0.8, max_rotation_attempts=200
                )
        if b_mol is not None:
            for idx in template_abx['B']:
                working_mc = MoleculeManipulator(working_mc).replace_molecule(
                    idx, b_mol, clash_threshold=0.8, max_rotation_attempts=200
                )
        if x_mol is not None:
            for idx in template_abx['X']:
                working_mc = MoleculeManipulator(working_mc).replace_molecule(
                    idx, x_mol, clash_threshold=0.8, max_rotation_attempts=200
                )
        return working_mc
    except MoleculeClashError as e:
        print(f"    Clash error during substitution: {e}")
        return None
    except Exception as e:
        print(f"    Build error: {e}")
        return None


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
    """Run DeepProperty inference on a cluster, return predicted Vdet (m/s)."""
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
# Predict a material from CIF (direct cluster, no substitution)
# ---------------------------------------------------------------------------

def predict_from_cif(
    cif_path: Path,
    material_name: str,
    model,
    model_type_map: list[str],
) -> dict:
    """Build clusters from a CIF and predict Vdet. Returns result dict."""
    print(f"Loading crystal from {cif_path}...")
    crystal = MolecularCrystal.from_ase(
        ase_read(str(cif_path)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    print(f"  Crystal: {len(crystal.molecules)} molecules")

    # Show molecule classification
    for i, mol in enumerate(crystal.molecules):
        formula = canonical_formula(mol)
        site = classify_mol(mol)
        print(f"  Mol {i}: {formula} → {site}")

    preds: dict[str, float] = {}
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
            xyz_dir = EXP_DIR / "ood_cluster_xyz"
            xyz_dir.mkdir(parents=True, exist_ok=True)
            ase_write(str(xyz_dir / f"{material_name}_{dataset_name}.xyz"), cluster_atoms)
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
    # Pre-build all 9 clusters once
    crystal = MolecularCrystal.from_ase(
        ase_read(str(cif_path)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    cluster_atoms_list: list[tuple[str, "Atoms"]] = []
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
# Predict via DAP-4 template (ABX₃ substitution)
# ---------------------------------------------------------------------------

# Map variant name → pre-built cluster directory (3 distinct DAP-4 clusters)
_DAP4_VARIANT_DIRS = {
    "cluster_n1": CLUSTER_N1_DIR,
    "cluster_n2": CLUSTER_N2_DIR,
    "cluster_n3": CLUSTER_N3_DIR,
}


def _build_dap4_substituted(
    variant: str,
    a_mol,
    b_mol,
) -> "Atoms | None":
    """Load the DAP-4 cluster for `variant` and substitute A/B sites."""
    cluster_dir = _DAP4_VARIANT_DIRS[variant] / "DAP-4"
    coord, symbols, _ = read_cluster_system(cluster_dir)
    template_mc = build_molecular_crystal_from_npy(coord, symbols)
    template_abx = get_abx_indices(template_mc)
    substituted_mc = build_substituted_structure(
        template_mc, template_abx,
        a_mol=a_mol,
        b_mol=b_mol,
        x_mol=None,
    )
    if substituted_mc is None:
        return None
    return mc_to_atoms(substituted_mc)


def _add_extra_clo4(atoms: Atoms, variant_seed: int = 0) -> Atoms:
    """Append one extra ClO₄⁻ molecule to an existing cluster (Atoms).

    The ClO₄⁻ is placed at the cluster periphery: the Cl is offset from the
    cluster centroid by ~8 Å in a direction determined by `variant_seed`, and
    the 4 O atoms are placed at tetrahedral positions ~1.45 Å from Cl.

    This converts an ABX₃ cluster into an ABX₄ cluster for charge-balanced
    materials where both A and B sites are divalent (2+).
    """
    # Tetrahedral unit vectors (normalised)
    _tet = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    _tet /= np.linalg.norm(_tet[0])

    # Cluster centroid (already centred near [50,50,50])
    centroid = atoms.get_positions().mean(axis=0)

    # Pick a placement direction based on variant_seed (reproducible)
    rng = np.random.default_rng(variant_seed + 12345)
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)

    # Place Cl 8 Å from centroid
    cl_pos = centroid + direction * 8.0

    # Place 4 O atoms at tetrahedral positions 1.45 Å from Cl
    # Rotate the canonical tet frame to align with `direction`
    # (simple rotation: find rotation from [1,1,1]/sqrt(3) to direction)
    ref = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    axis = np.cross(ref, direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # direction is parallel or anti-parallel to ref
        rot_tet = _tet if np.dot(ref, direction) > 0 else -_tet
    else:
        axis /= axis_norm
        cos_a = float(np.clip(np.dot(ref, direction), -1.0, 1.0))
        sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a ** 2)))
        # Rodrigues rotation matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
        rot_tet = (_tet @ R.T)

    o_positions = cl_pos + rot_tet * 1.45

    new_symbols = list(atoms.get_chemical_symbols()) + ["Cl"] + ["O"] * 4
    new_positions = np.vstack([atoms.get_positions(), cl_pos.reshape(1, 3), o_positions])
    return Atoms(symbols=new_symbols, positions=new_positions, cell=CLUSTER_BOX, pbc=False)


def _build_dap4_abx4_substituted(
    variant: str,
    a_mol,
    b_mol,
) -> "Atoms | None":
    """Build ABX₄ cluster: DAP-4 ABX₃ template + substitute A/B + add one ClO₄⁻.

    Used for MPEP/HPEP/PEP/SY where both A and B are divalent (2+), requiring
    4× ClO₄⁻ for charge balance (ABX₄ stoichiometry).
    """
    abx3_atoms = _build_dap4_substituted(variant, a_mol, b_mol)
    if abx3_atoms is None:
        return None
    # Use variant index as seed for reproducible ClO₄⁻ placement
    variant_idx = list(_DAP4_VARIANT_DIRS.keys()).index(variant)
    return _add_extra_clo4(abx3_atoms, variant_seed=variant_idx)


def predict_via_dap4_template(
    material_name: str,
    a_mol,
    b_mol,
    model,
    model_type_map: list[str],
) -> dict:
    """Predict using DAP-4 (ABX₃) template with ion substitution.

    Uses 3 distinct pre-built DAP-4 cluster variants (n1/n2/n3 from separate
    seed directories) so that structural diversity is captured and std > 0.
    """
    preds: dict[str, float] = {}
    xyz_dir = EXP_DIR / "ood_cluster_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    # Print template info once
    _tmp_coord, _tmp_sym, _ = read_cluster_system(CLUSTER_N1_DIR / "DAP-4")
    _tmp_mc = build_molecular_crystal_from_npy(_tmp_coord, _tmp_sym)
    _tmp_abx = get_abx_indices(_tmp_mc)
    print(f"  DAP-4 template (n1): {len(_tmp_abx['A'])} A, "
          f"{len(_tmp_abx['B'])} B, {len(_tmp_abx['X'])} X")

    for variant in ("cluster_n1", "cluster_n2", "cluster_n3"):
        try:
            sub_atoms = _build_dap4_substituted(variant, a_mol, b_mol)
            if sub_atoms is None:
                print(f"  [{variant}] Substitution FAILED (clash)")
                continue
            pred = run_inference(model, model_type_map, sub_atoms)
            preds[variant] = pred
            print(f"  [{variant}] {len(sub_atoms)} atoms → pred={pred:.1f} m/s")
            ase_write(str(xyz_dir / f"{material_name}_dap4tpl_{variant}.xyz"), sub_atoms)
        except Exception as e:
            print(f"  [{variant}] ERROR: {e}")
            import traceback; traceback.print_exc()

    if preds:
        vals = list(preds.values())
        mean_pred = float(np.mean(vals))
        std_pred = float(np.std(vals))
        print(f"\n  {material_name} (DAP-4 tpl) → mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_n1_m_s": preds.get("cluster_n1"),
            "pred_n2_m_s": preds.get("cluster_n2"),
            "pred_n3_m_s": preds.get("cluster_n3"),
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "n_clusters": len(vals),
        }
    return {}


def predict_via_dap4_abx4_template(
    material_name: str,
    a_mol,
    b_mol,
    model,
    model_type_map: list[str],
) -> dict:
    """Predict using DAP-4 + extra ClO₄⁻ (ABX₄) template with ion substitution.

    For materials where both A and B are divalent (2+), charge balance requires
    4× ClO₄⁻ (ABX₄). Builds ABX₃ cluster then appends one extra ClO₄⁻.
    Uses 3 distinct pre-built DAP-4 cluster variants (n1/n2/n3).
    """
    preds: dict[str, float] = {}
    xyz_dir = EXP_DIR / "ood_cluster_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    for variant in ("cluster_n1", "cluster_n2", "cluster_n3"):
        try:
            sub_atoms = _build_dap4_abx4_substituted(variant, a_mol, b_mol)
            if sub_atoms is None:
                print(f"  [{variant}] Substitution FAILED (clash)")
                continue
            pred = run_inference(model, model_type_map, sub_atoms)
            preds[variant] = pred
            print(f"  [{variant}] {len(sub_atoms)} atoms (ABX₄) → pred={pred:.1f} m/s")
            ase_write(str(xyz_dir / f"{material_name}_dap4abx4tpl_{variant}.xyz"), sub_atoms)
        except Exception as e:
            print(f"  [{variant}] ERROR: {e}")
            import traceback; traceback.print_exc()

    if preds:
        vals = list(preds.values())
        mean_pred = float(np.mean(vals))
        std_pred = float(np.std(vals))
        print(f"\n  {material_name} (DAP-4 ABX₄ tpl) → mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_n1_m_s": preds.get("cluster_n1"),
            "pred_n2_m_s": preds.get("cluster_n2"),
            "pred_n3_m_s": preds.get("cluster_n3"),
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "n_clusters": len(vals),
        }
    return {}


# ---------------------------------------------------------------------------
# Predict via EAP-4 template (A₂BX₅ substitution)
# ---------------------------------------------------------------------------

def predict_via_eap4_template(
    material_name: str,
    a_mol,
    b_mol,
    model,
    model_type_map: list[str],
    eap4_crystal: MolecularCrystal,
) -> dict:
    """Predict using EAP-4 (A₂BX₅) template with ion substitution.

    Builds 9 cluster variants (n1–n9) from the EAP-4 CIF using different
    random seeds and spread-seed offsets for structural diversity.
    """
    preds: dict[str, float] = {}
    xyz_dir = EXP_DIR / "ood_cluster_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                eap4_crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms_raw = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms_raw = cluster_atoms_centered(cluster_atoms_raw)

            template_mc = MolecularCrystal.from_ase(cluster_atoms_raw)
            template_abx = get_abx_indices(template_mc)

            substituted_mc = build_substituted_structure(
                template_mc, template_abx,
                a_mol=a_mol,
                b_mol=b_mol,
                x_mol=None,
            )

            if substituted_mc is None:
                print(f"  [{dataset_name}] Substitution FAILED (clash)")
                continue

            sub_atoms = mc_to_atoms(substituted_mc)
            pred = run_inference(model, model_type_map, sub_atoms)
            preds[dataset_name] = pred
            print(f"  [{dataset_name}] {len(sub_atoms)} atoms, "
                  f"seed_idx={seed_idx}, sc={sc_dims} → pred={pred:.1f} m/s")
            ase_write(str(xyz_dir / f"{material_name}_{dataset_name}.xyz"), sub_atoms)
        except Exception as e:
            print(f"  [{dataset_name}] ERROR: {e}")
            import traceback; traceback.print_exc()

    if preds:
        vals = list(preds.values())
        mean_pred = float(np.mean(vals))
        std_pred = float(np.std(vals))
        print(f"\n  {material_name} (EAP-4 tpl, {len(vals)} clusters) "
              f"→ mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        result = {f"pred_{k}_m_s": v for k, v in preds.items()}
        result.update({
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "n_clusters": len(vals),
        })
        return result
    return {}


# ---------------------------------------------------------------------------
# Predict via SY template (ABX₄ substitution — A-site only)
# ---------------------------------------------------------------------------

def predict_via_sy_template(
    material_name: str,
    a_mol,
    model,
    model_type_map: list[str],
    sy_crystal: MolecularCrystal,
) -> dict:
    """Predict using SY (ABX₄) template with A-site substitution only.

    SY = (H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄, P1, Z=4.
    Builds 9 cluster variants (n1–n9) from the SY CIF using different
    random seeds and spread-seed offsets for structural diversity.
    B-site (H₂en²⁺) and X-site (ClO₄⁻) are kept from the template.
    """
    preds: dict[str, float] = {}
    xyz_dir = EXP_DIR / "ood_cluster_xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                sy_crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms_raw = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms_raw = cluster_atoms_centered(cluster_atoms_raw)

            template_mc = MolecularCrystal.from_ase(cluster_atoms_raw)
            template_abx = get_abx_indices(template_mc)

            substituted_mc = build_substituted_structure(
                template_mc, template_abx,
                a_mol=a_mol,
                b_mol=None,   # keep H₂en²⁺ from SY template
                x_mol=None,   # keep ClO₄⁻ from SY template
            )

            if substituted_mc is None:
                print(f"  [{dataset_name}] Substitution FAILED (clash)")
                continue

            sub_atoms = mc_to_atoms(substituted_mc)
            pred = run_inference(model, model_type_map, sub_atoms)
            preds[dataset_name] = pred
            print(f"  [{dataset_name}] {len(sub_atoms)} atoms, "
                  f"seed_idx={seed_idx}, sc={sc_dims} → pred={pred:.1f} m/s")
            ase_write(str(xyz_dir / f"{material_name}_sytpl_{dataset_name}.xyz"), sub_atoms)
        except Exception as e:
            print(f"  [{dataset_name}] ERROR: {e}")
            import traceback; traceback.print_exc()

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
    """5-fold × 9-cluster ensemble prediction via SY template (A-site substitution only)."""
    # Pre-build all 9 substituted clusters once (reuse across folds)
    cluster_atoms_list: list[tuple[str, "Atoms"]] = []
    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                sy_crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms_raw = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms_raw = cluster_atoms_centered(cluster_atoms_raw)
            template_mc = MolecularCrystal.from_ase(cluster_atoms_raw)
            template_abx = get_abx_indices(template_mc)
            substituted_mc = build_substituted_structure(
                template_mc, template_abx,
                a_mol=a_mol, b_mol=None, x_mol=None,
            )
            if substituted_mc is None:
                print(f"  [{dataset_name}] Substitution FAILED (clash)")
                continue
            sub_atoms = mc_to_atoms(substituted_mc)
            cluster_atoms_list.append((dataset_name, sub_atoms))
            print(f"  [{dataset_name}] built: {len(sub_atoms)} atoms, "
                  f"seed_idx={seed_idx}, sc={sc_dims}")
        except Exception as e:
            print(f"  [{dataset_name}] ERROR building cluster: {e}")

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
# 5-fold ensemble prediction helpers
# ---------------------------------------------------------------------------

def _load_fold_models(exp_dir: Path, fold_names: list[str] | None = None) -> list[tuple]:
    """Load 5-fold checkpoints. Returns list of (model, type_map).

    Args:
        exp_dir: Root experiments directory.
        fold_names: List of fold directory names to load. Defaults to EXP7A_FOLDS (lr=1e-4).
    """
    from deepmd.pt.infer.deep_eval import DeepProperty
    if fold_names is None:
        fold_names = EXP7A_FOLDS
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


def predict_via_dap4_template_ensemble(
    material_name: str,
    a_mol,
    b_mol,
    fold_models: list[tuple],
) -> dict:
    """5-fold × 3-cluster ensemble prediction via DAP-4 template."""
    all_preds: list[float] = []
    fold_means: list[float] = []

    for fi, (model, type_map) in enumerate(fold_models):
        fold_preds = []
        for variant in ("cluster_n1", "cluster_n2", "cluster_n3"):
            try:
                sub_atoms = _build_dap4_substituted(variant, a_mol, b_mol)
                if sub_atoms is None:
                    continue
                pred = run_inference(model, type_map, sub_atoms)
                fold_preds.append(pred)
                all_preds.append(pred)
            except Exception as e:
                print(f"  [fold{fi}/{variant}] ERROR: {e}")
        if fold_preds:
            fold_means.append(float(np.mean(fold_preds)))
            print(f"  [fold{fi}] mean={fold_means[-1]:.1f} m/s "
                  f"({len(fold_preds)} clusters)")

    if all_preds:
        mean_pred = float(np.mean(all_preds))
        std_pred = float(np.std(all_preds))
        print(f"\n  {material_name} (DAP-4 tpl, 5-fold ensemble, "
              f"{len(all_preds)} preds) → mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "fold_means_m_s": fold_means,
            "n_preds": len(all_preds),
        }
    return {}


def predict_via_dap4_abx4_template_ensemble(
    material_name: str,
    a_mol,
    b_mol,
    fold_models: list[tuple],
) -> dict:
    """5-fold × 3-cluster ensemble prediction via DAP-4 ABX₄ template.

    For materials where both A and B are divalent (2+): builds ABX₃ cluster
    then appends one extra ClO₄⁻ to give ABX₄ stoichiometry.
    """
    all_preds: list[float] = []
    fold_means: list[float] = []

    for fi, (model, type_map) in enumerate(fold_models):
        fold_preds = []
        for variant in ("cluster_n1", "cluster_n2", "cluster_n3"):
            try:
                sub_atoms = _build_dap4_abx4_substituted(variant, a_mol, b_mol)
                if sub_atoms is None:
                    continue
                pred = run_inference(model, type_map, sub_atoms)
                fold_preds.append(pred)
                all_preds.append(pred)
            except Exception as e:
                print(f"  [fold{fi}/{variant}] ERROR: {e}")
        if fold_preds:
            fold_means.append(float(np.mean(fold_preds)))
            print(f"  [fold{fi}] mean={fold_means[-1]:.1f} m/s "
                  f"({len(fold_preds)} clusters)")

    if all_preds:
        mean_pred = float(np.mean(all_preds))
        std_pred = float(np.std(all_preds))
        print(f"\n  {material_name} (DAP-4 ABX₄ tpl, 5-fold ensemble, "
              f"{len(all_preds)} preds) → mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "fold_means_m_s": fold_means,
            "n_preds": len(all_preds),
        }
    return {}


def predict_via_eap4_template_ensemble(
    material_name: str,
    a_mol,
    b_mol,
    fold_models: list[tuple],
    eap4_crystal: MolecularCrystal,
) -> dict:
    """5-fold × 9-cluster ensemble prediction via EAP-4 template."""
    # Pre-build all 9 substituted clusters once (reuse across folds)
    cluster_atoms_list: list[tuple[str, "Atoms"]] = []
    for dataset_name, seed in CLUSTER_RANDOM_SEEDS.items():
        try:
            cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
                eap4_crystal, dataset_name=dataset_name, seed=seed,
            )
            cluster_atoms_raw = crystal_to_minimum_image_atoms(cluster_crystal)
            cluster_atoms_raw = cluster_atoms_centered(cluster_atoms_raw)
            template_mc = MolecularCrystal.from_ase(cluster_atoms_raw)
            template_abx = get_abx_indices(template_mc)
            substituted_mc = build_substituted_structure(
                template_mc, template_abx,
                a_mol=a_mol, b_mol=b_mol, x_mol=None,
            )
            if substituted_mc is None:
                print(f"  [{dataset_name}] Substitution FAILED (clash)")
                continue
            sub_atoms = mc_to_atoms(substituted_mc)
            cluster_atoms_list.append((dataset_name, sub_atoms))
            print(f"  [{dataset_name}] built: {len(sub_atoms)} atoms")
        except Exception as e:
            print(f"  [{dataset_name}] ERROR building cluster: {e}")

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
        print(f"\n  {material_name} (EAP-4 tpl, 5-fold ensemble, "
              f"{len(all_preds)} preds) → mean={mean_pred:.1f} ± {std_pred:.1f} m/s\n")
        return {
            "pred_mean_m_s": mean_pred,
            "pred_std_m_s": std_pred,
            "fold_means_m_s": fold_means,
            "n_preds": len(all_preds),
        }
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Vdet for EAP-4/EAP-8/PAP-H6 via both EAP-4 and DAP-4 templates")
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP,
                        help=f"Experiment directory name (default: {DEFAULT_EXP})")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Explicit checkpoint path")
    parser.add_argument("--use-exp7a-folds", action="store_true",
                        help="Also run 5-fold exp7a_lr1e4 ensemble predictions")
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

    # Load representative molecules for substitution
    print("Loading representative molecules for substitution...")
    h2en_mol = get_representative_mol("DAP-4", "A")      # H₂en²⁺ = H₂dabco²⁺ (C₆H₁₄N₂)
    # Actually DAP-4 A-site is H₂dabco²⁺, we need H₂en²⁺ from EAP-4 itself
    # H₂hpz²⁺ from PAP-H4
    h2hpz_mol = get_representative_mol("PAP-H4", "A")    # H₂hpz²⁺ (C₅H₁₄N₂)
    nh4_mol = get_representative_mol("DAP-4", "B")        # NH₄⁺ (H₄N)
    nh3oh_mol = get_representative_mol("DAP-6", "B")      # NH₃OH⁺ (H₄NO)
    ch3nh3_mol = get_representative_mol("DAP-M4", "B")    # CH₃NH₃⁺... wait, let me check
    print()

    # Load EAP-4 crystal for template building
    eap4_cif = CLEANED_CIF_DIR / "EAP-4.cif"
    eap4_crystal = MolecularCrystal.from_ase(
        ase_read(str(eap4_cif)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )

    # Load SY crystal for template building
    sy_cif = CLEANED_CIF_DIR / "SY.cif"
    sy_crystal = MolecularCrystal.from_ase(
        ase_read(str(sy_cif)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )

    results: dict[str, dict] = {}

    # ===================================================================
    # 1. EAP-4 — direct from CIF (A₂BX₅ stoichiometry)
    # ===================================================================
    print("=" * 70)
    print("1. EAP-4 = (H₂en²⁺)₂(NH₄⁺)(ClO₄⁻)₅ — direct from CIF")
    print("=" * 70)
    r = predict_from_cif(eap4_cif, "EAP-4", model, model_type_map)
    if r:
        results["EAP-4 (direct CIF, A₂BX₅)"] = {
            "composition": "(H₂en²⁺)₂(NH₄⁺)(ClO₄⁻)₅",
            "template": "own CIF",
            "stoichiometry": "A₂BX₅",
            **r,
        }

    # ===================================================================
    # 2. EAP-4 — via DAP-4 template (ABX₃, wrong stoichiometry)
    # ===================================================================
    print("=" * 70)
    print("2. EAP-4 via DAP-4 template (ABX₃) — WRONG stoichiometry")
    print("   A: H₂en²⁺ (from EAP-4 cluster), B: NH₄⁺ (from DAP-4)")
    print("=" * 70)
    # Need H₂en²⁺ from EAP-4 cluster
    # Build one EAP-4 cluster and get the A-site molecule
    cluster_crystal_tmp, _, _ = build_seeded_stoichiometric_cluster(
        eap4_crystal, dataset_name="cluster_n1", seed=101,
    )
    cluster_atoms_tmp = crystal_to_minimum_image_atoms(cluster_crystal_tmp)
    cluster_atoms_tmp = cluster_atoms_centered(cluster_atoms_tmp)
    eap4_mc_tmp = MolecularCrystal.from_ase(cluster_atoms_tmp)
    eap4_abx_tmp = get_abx_indices(eap4_mc_tmp)
    h2en_mol = eap4_mc_tmp.molecules[eap4_abx_tmp['A'][0]].copy()
    print(f"  H₂en²⁺ from EAP-4 cluster: {canonical_formula(h2en_mol)}")

    r = predict_via_dap4_template("EAP-4", h2en_mol, nh4_mol, model, model_type_map)
    if r:
        results["EAP-4 (DAP-4 tpl, ABX₃)"] = {
            "composition": "(H₂en²⁺)(NH₄⁺)(ClO₄⁻)₃",
            "template": "DAP-4",
            "stoichiometry": "ABX₃",
            **r,
        }

    # ===================================================================
    # 3. EAP-8 — direct from CIF (A₂BX₅ stoichiometry)
    # ===================================================================
    print("=" * 70)
    print("3. EAP-8 = (H₂en²⁺)₂(CH₃NH₃⁺)(ClO₄⁻)₅ — direct from CIF")
    print("=" * 70)
    eap8_cif = CLEANED_CIF_DIR / "EAP-8.cif"
    r = predict_from_cif(eap8_cif, "EAP-8", model, model_type_map)
    if r:
        results["EAP-8 (direct CIF, A₂BX₅)"] = {
            "composition": "(H₂en²⁺)₂(CH₃NH₃⁺)(ClO₄⁻)₅",
            "template": "own CIF",
            "stoichiometry": "A₂BX₅",
            **r,
        }

    # ===================================================================
    # 4. EAP-8 — via EAP-4 template (A₂BX₅, substitute B only)
    # ===================================================================
    print("=" * 70)
    print("4. EAP-8 via EAP-4 template (A₂BX₅) — substitute B-site only")
    print("   B: CH₃NH₃⁺")
    print("=" * 70)
    # Get CH₃NH₃⁺ from EAP-8 cluster
    # Note: classify_mol puts CH₃NH₃⁺ in 'A' because it has C.
    # Identify it as the smallest C-containing molecule (7 atoms vs 14 for H₂en²⁺).
    eap8_crystal = MolecularCrystal.from_ase(
        ase_read(str(eap8_cif)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    cluster_crystal_tmp8, _, _ = build_seeded_stoichiometric_cluster(
        eap8_crystal, dataset_name="cluster_n1", seed=101,
    )
    cluster_atoms_tmp8 = crystal_to_minimum_image_atoms(cluster_crystal_tmp8)
    cluster_atoms_tmp8 = cluster_atoms_centered(cluster_atoms_tmp8)
    eap8_mc_tmp = MolecularCrystal.from_ase(cluster_atoms_tmp8)
    # Find CH₃NH₃⁺ by smallest atom count among C-containing molecules
    c_mols = [(i, mol) for i, mol in enumerate(eap8_mc_tmp.molecules)
              if 'C' in set(mol.get_chemical_symbols())]
    c_mols.sort(key=lambda x: len(x[1].get_chemical_symbols()))
    ch3nh3_mol = c_mols[0][1].copy()  # smallest C-containing mol = CH₃NH₃⁺
    print(f"  CH₃NH₃⁺ from EAP-8 cluster: {canonical_formula(ch3nh3_mol)} "
          f"({len(ch3nh3_mol.get_chemical_symbols())} atoms)")

    r = predict_via_eap4_template("EAP-8_eap4tpl", None, ch3nh3_mol, model, model_type_map, eap4_crystal)
    if r:
        results["EAP-8 (EAP-4 tpl, A₂BX₅)"] = {
            "composition": "(H₂en²⁺)₂(CH₃NH₃⁺)(ClO₄⁻)₅",
            "template": "EAP-4",
            "stoichiometry": "A₂BX₅",
            **r,
        }

    # ===================================================================
    # 5. EAP-8 — via DAP-4 template (ABX₃, wrong stoichiometry)
    # ===================================================================
    print("=" * 70)
    print("5. EAP-8 via DAP-4 template (ABX₃) — WRONG stoichiometry")
    print("=" * 70)
    r = predict_via_dap4_template("EAP-8", h2en_mol, ch3nh3_mol, model, model_type_map)
    if r:
        results["EAP-8 (DAP-4 tpl, ABX₃)"] = {
            "composition": "(H₂en²⁺)(CH₃NH₃⁺)(ClO₄⁻)₃",
            "template": "DAP-4",
            "stoichiometry": "ABX₃",
            **r,
        }

    # ===================================================================
    # 6. PAP-H6 — via EAP-4 template (A₂BX₅, correct)
    # ===================================================================
    print("=" * 70)
    print("6. PAP-H6 = (H₂hpz²⁺)₂(NH₃OH⁺)(ClO₄⁻)₅ — via EAP-4 template")
    print("=" * 70)
    r = predict_via_eap4_template("PAP-H6", h2hpz_mol, nh3oh_mol, model, model_type_map, eap4_crystal)
    if r:
        results["PAP-H6 (EAP-4 tpl, A₂BX₅)"] = {
            "composition": "(H₂hpz²⁺)₂(NH₃OH⁺)(ClO₄⁻)₅",
            "template": "EAP-4",
            "stoichiometry": "A₂BX₅",
            **r,
        }

    # ===================================================================
    # 7. PAP-H6 — via DAP-4 template (ABX₃, wrong stoichiometry)
    # ===================================================================
    print("=" * 70)
    print("7. PAP-H6 via DAP-4 template (ABX₃) — WRONG stoichiometry")
    print("=" * 70)
    r = predict_via_dap4_template("PAP-H6", h2hpz_mol, nh3oh_mol, model, model_type_map)
    if r:
        results["PAP-H6 (DAP-4 tpl, ABX₃)"] = {
            "composition": "(H₂hpz²⁺)(NH₃OH⁺)(ClO₄⁻)₃",
            "template": "DAP-4",
            "stoichiometry": "ABX₃",
            **r,
        }

    # ===================================================================
    # 8-11. MPEP / HPEP / PEP / SY — ABX₄ with en²⁺ B-site, DAP-4+ClO₄⁻ template
    # A position: from {PAP-M4, PAP-H4, PAP-4, DAP-4}  (all divalent 2+)
    # B position: H₂en²⁺ (ethylenediammonium, from EAP-4's A-site)  (divalent 2+)
    # X position: ClO₄⁻ × 4  (charge balance: 2+ + 2+ = 4+)
    # ===================================================================
    print("=" * 70)
    print("8-11. MPEP / HPEP / PEP / SY — ABX₄ with en²⁺ B-site (DAP-4+ClO₄⁻ template)")
    print("   Both A and B are divalent (2+) → 4× ClO₄⁻ needed for charge balance")
    print("=" * 70)

    # Load A-site molecules
    mehpz_mol = get_representative_mol("PAP-M4", "A")   # MeHpz²⁺
    h2pz_mol = get_representative_mol("PAP-4", "A")      # H₂pz²⁺
    h2dabco_mol = get_representative_mol("DAP-4", "A")   # H₂dabco²⁺
    # h2hpz_mol already loaded above
    # h2en_mol already extracted from EAP-4 cluster above

    bx4_materials = [
        ("MPEP", "MeHpz²⁺",    mehpz_mol,   "(MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("HPEP", "H₂hpz²⁺",    h2hpz_mol,   "(H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("PEP",  "H₂pz²⁺",     h2pz_mol,    "(H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("SY",   "H₂dabco²⁺",  h2dabco_mol, "(H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
    ]

    for mat_name, a_label, a_mol_sub, composition in bx4_materials:
        print(f"\n--- {mat_name}: A={a_label} (2+), B=H₂en²⁺ (2+), X=ClO₄⁻ ×4 ---")
        r = predict_via_dap4_abx4_template(mat_name, a_mol_sub, h2en_mol, model, model_type_map)
        if r:
            results[f"{mat_name} (DAP-4 tpl, ABX₄)"] = {
                "composition": composition,
                "template": "DAP-4",
                "stoichiometry": "ABX₄",
                "a_site": a_label,
                "b_site": "H₂en²⁺",
                "x_site": "ClO₄⁻",
                **r,
            }

    # ===================================================================
    # 12-14. MPEP / HPEP / PEP — via SY template (ABX₄, correct stoichiometry)
    # SY = (H₂dabco²⁺)(H₂en²⁺)(ClO₄⁻)₄, real CIF, P1, Z=4
    # Substitute A-site only; B=H₂en²⁺ and X=ClO₄⁻ kept from SY template
    # ===================================================================
    print("=" * 70)
    print("12-14. MPEP / HPEP / PEP — via SY template (ABX₄, real CIF)")
    print("   A-site substitution only; B=H₂en²⁺ and X=ClO₄⁻ from SY template")
    print("=" * 70)

    # bx4_materials_sy: only MPEP/HPEP/PEP (SY itself is the template, no need to predict SY via SY)
    bx4_sy_materials = [
        ("MPEP", "MeHpz²⁺",   mehpz_mol,  "(MeHpz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("HPEP", "H₂hpz²⁺",   h2hpz_mol,  "(H₂hpz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
        ("PEP",  "H₂pz²⁺",    h2pz_mol,   "(H₂pz²⁺)(H₂en²⁺)(ClO₄⁻)₄"),
    ]

    for mat_name, a_label, a_mol_sub, composition in bx4_sy_materials:
        print(f"\n--- {mat_name} via SY template: A={a_label} (2+), B=H₂en²⁺, X=ClO₄⁻ ×4 ---")
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
    # 5-fold ensemble predictions (exp7a_lr1e4, optional)
    # ===================================================================
    ensemble_results: dict[str, dict] = {}

    def _run_ensemble_block(fold_models: list[tuple], tag: str) -> None:
        """Run all ensemble predictions for a given set of fold models.

        Results are stored in ensemble_results with keys suffixed by `tag`
        (e.g. '[exp7a_lr1e4]' or '[exp7a]').
        """
        # EAP-4 direct CIF — 5-fold × 9 clusters
        print(f"\n--- EAP-4 (direct CIF, A₂BX₅) — {tag} ---")
        r = predict_from_cif_ensemble(eap4_cif, "EAP-4", fold_models)
        if r:
            ensemble_results[f"EAP-4 (direct CIF, A₂BX₅) [{tag}]"] = {
                "composition": "(H₂en²⁺)₂(NH₄⁺)(ClO₄⁻)₅",
                "template": "own CIF", "stoichiometry": "A₂BX₅",
                "ensemble": tag, **r}

        # EAP-8 direct CIF — 5-fold × 9 clusters
        print(f"\n--- EAP-8 (direct CIF, A₂BX₅) — {tag} ---")
        r = predict_from_cif_ensemble(eap8_cif, "EAP-8", fold_models)
        if r:
            ensemble_results[f"EAP-8 (direct CIF, A₂BX₅) [{tag}]"] = {
                "composition": "(H₂en²⁺)₂(CH₃NH₃⁺)(ClO₄⁻)₅",
                "template": "own CIF", "stoichiometry": "A₂BX₅",
                "ensemble": tag, **r}

        # EAP-8 via EAP-4 template — 5-fold × 9 clusters
        print(f"\n--- EAP-8 (EAP-4 tpl, A₂BX₅) — {tag} ---")
        r = predict_via_eap4_template_ensemble(
            "EAP-8_eap4tpl", None, ch3nh3_mol, fold_models, eap4_crystal)
        if r:
            ensemble_results[f"EAP-8 (EAP-4 tpl, A₂BX₅) [{tag}]"] = {
                "composition": "(H₂en²⁺)₂(CH₃NH₃⁺)(ClO₄⁻)₅",
                "template": "EAP-4", "stoichiometry": "A₂BX₅",
                "ensemble": tag, **r}

        # PAP-H6 via EAP-4 template — 5-fold × 9 clusters
        print(f"\n--- PAP-H6 (EAP-4 tpl, A₂BX₅) — {tag} ---")
        r = predict_via_eap4_template_ensemble(
            "PAP-H6", h2hpz_mol, nh3oh_mol, fold_models, eap4_crystal)
        if r:
            ensemble_results[f"PAP-H6 (EAP-4 tpl, A₂BX₅) [{tag}]"] = {
                "composition": "(H₂hpz²⁺)₂(NH₃OH⁺)(ClO₄⁻)₅",
                "template": "EAP-4", "stoichiometry": "A₂BX₅",
                "ensemble": tag, **r}

        # PAP-H6 via DAP-4 template — 5-fold × 3 clusters
        print(f"\n--- PAP-H6 (DAP-4 tpl, ABX₃) — {tag} ---")
        r = predict_via_dap4_template_ensemble(
            "PAP-H6", h2hpz_mol, nh3oh_mol, fold_models)
        if r:
            ensemble_results[f"PAP-H6 (DAP-4 tpl, ABX₃) [{tag}]"] = {
                "composition": "(H₂hpz²⁺)(NH₃OH⁺)(ClO₄⁻)₃",
                "template": "DAP-4", "stoichiometry": "ABX₃",
                "ensemble": tag, **r}

        # MPEP/HPEP/PEP/SY via DAP-4 ABX₄ template — 5-fold × 3 clusters
        for mat_name, a_label, a_mol_sub, composition in bx4_materials:
            print(f"\n--- {mat_name} (DAP-4 tpl, ABX₄) — {tag} ---")
            r = predict_via_dap4_abx4_template_ensemble(
                mat_name, a_mol_sub, h2en_mol, fold_models)
            if r:
                ensemble_results[f"{mat_name} (DAP-4 tpl, ABX₄) [{tag}]"] = {
                    "composition": composition,
                    "template": "DAP-4", "stoichiometry": "ABX₄",
                    "a_site": a_label, "b_site": "H₂en²⁺", "x_site": "ClO₄⁻",
                    "ensemble": tag, **r}

        # MPEP/HPEP/PEP via SY template (ABX₄) — 5-fold × 9 clusters
        for mat_name, a_label, a_mol_sub, composition in bx4_sy_materials:
            print(f"\n--- {mat_name} (SY tpl, ABX₄) — {tag} ---")
            r = predict_via_sy_template_ensemble(
                mat_name, a_mol_sub, fold_models, sy_crystal)
            if r:
                ensemble_results[f"{mat_name} (SY tpl, ABX₄) [{tag}]"] = {
                    "composition": composition,
                    "template": "SY", "stoichiometry": "ABX₄",
                    "a_site": a_label, "b_site": "H₂en²⁺", "x_site": "ClO₄⁻",
                    "ensemble": tag, **r}

    if args.use_exp7a_folds:
        # --- exp7a_lr1e4 (lr=1e-4) ---
        print("\n" + "=" * 70)
        print("5-FOLD ENSEMBLE (exp7a_lr1e4, fold0–fold4, lr=1e-4)")
        print("=" * 70)
        fold_models_lr1e4 = _load_fold_models(EXP_DIR, EXP7A_FOLDS)
        if fold_models_lr1e4:
            _run_ensemble_block(fold_models_lr1e4, "exp7a_lr1e4")

        # --- exp7a base (default lr) ---
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
    print(f"{'Label':<40} {'Template':<12} {'Stoich':<8} {'N':>4} {'Pred (m/s)':>12} {'Std':>8}")
    print("-" * 100)
    for label, r in results.items():
        pred_str = f"{r['pred_mean_m_s']:.1f}" if 'pred_mean_m_s' in r else "FAIL"
        std_str = f"{r['pred_std_m_s']:.1f}" if 'pred_std_m_s' in r else "-"
        n_str = str(r.get('n_clusters', '?'))
        print(f"{label:<40} {r.get('template','?'):<12} {r.get('stoichiometry','?'):<8} "
              f"{n_str:>4} {pred_str:>12} {std_str:>8}")
    print("=" * 100)

    if ensemble_results:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE — 5-fold ensemble (exp7a_lr1e4)")
        print("=" * 100)
        print(f"{'Label':<45} {'Template':<12} {'Stoich':<8} {'N':>5} {'Pred (m/s)':>12} {'Std':>8}")
        print("-" * 100)
        for label, r in ensemble_results.items():
            pred_str = f"{r['pred_mean_m_s']:.1f}" if 'pred_mean_m_s' in r else "FAIL"
            std_str = f"{r['pred_std_m_s']:.1f}" if 'pred_std_m_s' in r else "-"
            n_str = str(r.get('n_preds', '?'))
            print(f"{label:<45} {r.get('template','?'):<12} {r.get('stoichiometry','?'):<8} "
                  f"{n_str:>5} {pred_str:>12} {std_str:>8}")
        print("=" * 100)

    # Save JSON
    all_results = {**results, **ensemble_results}
    out_path = EXP_DIR / "eap4_paph6_predictions.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
