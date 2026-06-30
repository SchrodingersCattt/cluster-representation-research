#!/usr/bin/env python3
"""
ABX combinatorial grid prediction pipeline for PEMs.

Uses the DAP-4 cluster as a geometric template. For each (A, B, X) combination
from the set of known PEM species, replaces the template's A/B/X molecules with
the target species and runs DeepProperty inference to predict detonation velocity.

Results are plotted as three 2D heatmaps (rows = A, columns = B, panels = X).

Usage:
    # Default: use exp6v1_allpems latest checkpoint
    python predict_abx_grid.py

    # Use a specific exp5 fold checkpoint (v1 or v2)
    python predict_abx_grid.py --exp exp5v1_fold0

    # Use a specific checkpoint file
    python predict_abx_grid.py --ckpt experiments_davis2024/exp5v1_fold0/model.ckpt-100000.pt

    # Skip inference, only re-plot from cached results
    python predict_abx_grid.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from paper_plot_style import save_png_pdf, setup_nature_style, style_axes

sys.path.insert(0, "/path/to/MolCrysKit")
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.structures.molecule import CrystalMolecule
from molcrys_kit.operations.molecule_manipulation import MoleculeManipulator, MoleculeClashError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "00_data_prep"
CLUSTER_N1_DIR = DATA_ROOT / "pems_cluster_n1_systems"
PEMS_CSV = ROOT.parent / "data" / "pems" / "pems.csv"

DEFAULT_EXP = "exp6v1_allpems"
HEAD = "pems_vdet_kj"
RESULTS_PATH = ROOT / "abx_grid_predictions.json"
PLOT_PATH = ROOT / "abx_grid_2d.png"
PLOT_PDF_PATH = ROOT / "abx_grid_2d.pdf"

CLUSTER_BOX = np.eye(3, dtype=np.float64) * 100.0
SEED = 42

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

# Mandatory for PEMs: without these shorter heavy-atom cutoffs MolCrysKit can
# merge ionic metal···O contacts into a single molecule, which drops K+/Rb+ and
# some oxyanions from the species registry.
_HEAVY = ["I", "Na", "K", "Rb", "Ba", "Ag"]
_HEAVY_NONO_LIMITS = {"I": 2.10, "Na": 2.30, "K": 2.50, "Rb": 2.60, "Ba": 2.60, "Ag": 2.30}
_HEAVY_O_LIMITS = {"I": 2.05, "Na": 2.20, "K": 2.30, "Rb": 2.40, "Ba": 2.40, "Ag": 2.20}
PEM_BOND_THRESHOLDS: dict[tuple[str, str], float] = {}
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
del _metal, _limit, _org, _i, _m1, _m2

# Materials to exclude (same as prep_pems_npy.py)
SKIP_MATERIALS: set[str] = {"DAC-4"}

# ---------------------------------------------------------------------------
# PEM nomenclature and site labels
# ---------------------------------------------------------------------------

# Preferred axis order in the 2D panel plot.
_A_SITE_ORDER = [
    "H2dabco2+",
    "MeHdabco2+",
    "H2odabco2+",
    "H2pz2+",
    "H2hpz2+",
    "MeHpz2+",
    "Huru+",
    "HQ+",
]
_B_SITE_ORDER = [
    "Na+",
    "K+",
    "Rb+",
    "NH4+",
    "Ag+",
    "H3O+",
    "NH3OH+",
    "NH2NH3+",
    "CH3NH3+",
    "Ba2+",
]
_X_SITE_ORDER = ["NO3-", "ClO4-", "IO4-", "H4IO6-"]

_A_SITE_LABELS: dict[str, str] = {
    "H2dabco2+": "H₂dabco²⁺",
    "MeHdabco2+": "MeHdabco²⁺",
    "H2odabco2+": "H₂odabco²⁺",
    "H2pz2+": "H₂pz²⁺",
    "H2hpz2+": "H₂hpz²⁺",
    "MeHpz2+": "MeHpz²⁺",
    "Huru+": "Huru⁺",
    "HQ+": "HQ⁺",
}
_B_SITE_LABELS: dict[str, str] = {
    "Na+": "Na⁺",
    "K+": "K⁺",
    "Rb+": "Rb⁺",
    "NH4+": "NH₄⁺",
    "Ag+": "Ag⁺",
    "H3O+": "H₃O⁺",
    "NH3OH+": "NH₃OH⁺",
    "NH2NH3+": "NH₂NH₃⁺",
    "CH3NH3+": "CH₃NH₃⁺",
    "Ba2+": "Ba²⁺",
    "Na+/NH4+ (ordered)": "Na⁺/NH₄⁺ (ordered)",
}
_X_SITE_LABELS: dict[str, str] = {
    "ClO4-": "ClO₄⁻",
    "NO3-": "NO₃⁻",
    "IO4-": "IO₄⁻",
    "H4IO6-": "[H₄IO₆]⁻",
    "ClO3-": "ClO₃⁻",
}
_A_SITE_TO_CODE: dict[str, tuple[str, str]] = {
    "H2dabco2+": ("D", ""),
    "MeHdabco2+": ("D", "M"),
    "H2odabco2+": ("D", "O"),
    "H2pz2+": ("P", ""),
    "H2hpz2+": ("P", "H"),
    "MeHpz2+": ("P", "M"),
    "Huru+": ("T", ""),
    "HQ+": ("Q", ""),
}
_B_SITE_TO_CODE: dict[str, tuple[str, str]] = {
    "Na+": ("A", "1"),
    "K+": ("A", "2"),
    "Rb+": ("A", "3"),
    "NH4+": ("A", "4"),
    "Ag+": ("A", "5"),
    "NH3OH+": ("A", "6"),
    "NH2NH3+": ("A", "7"),
    "CH3NH3+": ("A", "8"),
    "H3O+": ("A", "H3O"),
    "Ba2+": ("B", ""),
}
_X_SITE_TO_CODE: dict[str, str] = {
    "ClO4-": "P",
    "NO3-": "N",
    "IO4-": "I",
    "H4IO6-": "X",  # X for orthoperiodate (DAI-X1)
    "ClO3-": "C",
}

# Older cached JSON files store extracted molecular formulas instead of PEM site labels.
_B_FORMULA_TO_SITE = {
    "Na": "Na+",
    "K": "K+",
    "Rb": "Rb+",
    "Ag": "Ag+",
    "H4N": "NH4+",
    "HHN": "NH4+",
    "H3O": "H3O+",
    "H4NO": "NH3OH+",
    "H6N2": "NH2NH3+",
}
_X_FORMULA_TO_SITE = {
    "NO3": "NO3-",
    "ClO4": "ClO4-",
    "IO4": "IO4-",
    "H4IO6": "H4IO6-",  # CIF X-site composition (4 H, 1 I, 6 O) -> [H4IO6]- orthoperiodate anion
}


def load_pem_table() -> dict[str, dict[str, str]]:
    with PEMS_CSV.open(newline="", encoding="utf-8") as handle:
        return {row["material"]: row for row in csv.DictReader(handle) if row.get("material")}


def pretty_site_label(site_kind: str, site_key: str) -> str:
    mapping = {"A": _A_SITE_LABELS, "B": _B_SITE_LABELS, "X": _X_SITE_LABELS}[site_kind]
    return mapping.get(site_key, site_key)


def order_site_keys(keys: list[str], preferred_order: list[str]) -> list[str]:
    present = list(dict.fromkeys(keys))
    ordered = [key for key in preferred_order if key in present]
    ordered.extend(sorted(key for key in present if key not in preferred_order))
    return ordered


def decode_pem_name(material: str) -> tuple[str, str, str]:
    """Fallback decoder when a material is missing from `pems.csv`."""
    a_from_code = {
        "D": "H2dabco2+",
        "DM": "MeHdabco2+",
        "DO": "H2odabco2+",
        "P": "H2pz2+",
        "PM": "MeHpz2+",
        "PH": "H2hpz2+",
        "T": "Huru+",
        "Q": "HQ+",
    }
    b_from_code = {
        "1": "Na+",
        "2": "K+",
        "3": "Rb+",
        "4": "NH4+",
        "5": "Ag+",
        "6": "NH3OH+",
        "7": "NH2NH3+",
        "8": "CH3NH3+",
        "B": "Ba2+",
    }
    x_from_code = {"P": "ClO4-", "N": "NO3-", "I": "IO4-", "C": "ClO3-"}

    if material == "QBP":
        return ("HQ+", "Ba2+", "ClO4-")
    if material == "DAI-1_0.5 4_0.5":
        return ("H2dabco2+", "Na+/NH4+ (ordered)", "IO4-")
    if material == "DPE-1":
        return ("H2dabco2+", "Na+/NH4+ (ordered)", "ClO4-")

    if len(material) < 3 or material[1] not in ("A", "B"):
        return (material, "?", "?")

    a_family = material[0]
    x_letter = material[2]
    parts = material.split("-", 1)
    suffix = parts[1] if len(parts) > 1 else ""

    b_num = ""
    a_suffix = ""
    for i, ch in enumerate(suffix):
        if ch.isdigit():
            b_num = suffix[i:]
            a_suffix = suffix[:i]
            break
    if not b_num:
        b_num = suffix

    # DAI-X1 changes the X moiety, but the A-site chemistry stays H2dabco2+.
    if a_suffix.upper() == "X":
        a_suffix = ""

    a_site = a_from_code.get(a_family + a_suffix.upper(), material)
    b_site = b_from_code.get(b_num, f"B{b_num}" if b_num else "?")
    x_site = x_from_code.get(x_letter, f"X={x_letter}")
    return (a_site, b_site, x_site)


def get_material_site_keys(material: str, pem_table: dict[str, dict[str, str]]) -> tuple[str, str, str]:
    row = pem_table.get(material)
    if row:
        fallback = decode_pem_name(material)
        return (
            row["A_site"].strip() or fallback[0],
            row["B_site"].strip() or fallback[1],
            row["X_site"].strip() or fallback[2],
        )
    return decode_pem_name(material)


def normalize_b_site_key(value: str) -> str:
    if value in _B_SITE_LABELS:
        return value
    return _B_FORMULA_TO_SITE.get(value, value)


def normalize_x_site_key(value: str) -> str:
    if value in _X_SITE_LABELS:
        return value
    return _X_FORMULA_TO_SITE.get(value, value)


def load_known_combo_map(
    pem_table: dict[str, dict[str, str]] | None = None,
) -> OrderedDict[tuple[str, str, str], list[str]]:
    pem_table = pem_table or load_pem_table()
    known_combos: OrderedDict[tuple[str, str, str], list[str]] = OrderedDict()
    for material, row in pem_table.items():
        a_site = (row.get("A_site") or "").strip()
        b_site = normalize_b_site_key((row.get("B_site") or "").strip())
        x_site = normalize_x_site_key((row.get("X_site") or "").strip())
        if not a_site or not b_site or not x_site:
            continue
        known_combos.setdefault((a_site, b_site, x_site), []).append(material)
    return known_combos


def infer_pem_material_code(a_site: str, b_site: str, x_site: str) -> str:
    a_code = _A_SITE_TO_CODE.get(a_site)
    b_code = _B_SITE_TO_CODE.get(b_site)
    x_code = _X_SITE_TO_CODE.get(x_site)
    if not a_code or not b_code or not x_code:
        return "?"

    family_code, a_suffix = a_code
    b_letter, b_suffix = b_code
    if b_letter == "B":
        return f"{family_code}B{x_code}"
    return f"{family_code}A{x_code}-{a_suffix}{b_suffix}"


def format_cell_label(label: str) -> str:
    if len(label) <= 7:
        return label
    if "-" in label:
        head, tail = label.split("-", 1)
        return f"{head}-\n{tail}"
    return label


def normalize_result_record(record: dict, pem_table: dict[str, dict[str, str]]) -> dict:
    a_site = record.get("a_site") or record.get("a_key") or ""
    if a_site in pem_table:
        a_site = get_material_site_keys(a_site, pem_table)[0]

    b_site = normalize_b_site_key(record.get("b_site") or record.get("b_key") or record.get("b_formula") or "")
    x_site = normalize_x_site_key(record.get("x_site") or record.get("x_key") or record.get("x_formula") or "")

    return {
        **record,
        "a_site": a_site,
        "b_site": b_site,
        "x_site": x_site,
        "a_site_label": pretty_site_label("A", a_site),
        "b_site_label": pretty_site_label("B", b_site),
        "x_site_label": pretty_site_label("X", x_site),
    }


def get_a_site_key(material: str, pem_table: dict[str, dict[str, str]] | None = None) -> str:
    pem_table = pem_table or load_pem_table()
    return get_material_site_keys(material, pem_table)[0]

# ---------------------------------------------------------------------------
# I/O and molecule helpers (reused from build_pems_dap4_template.py)
# ---------------------------------------------------------------------------

def read_cluster_system(sys_dir: Path) -> tuple[np.ndarray, list[str], float]:
    type_map = (sys_dir / "type_map.raw").read_text(encoding="utf-8").strip().split()
    types = np.loadtxt(sys_dir / "type.raw", dtype=int)
    coord = np.load(sys_dir / "set.000" / "coord.npy")[0].reshape(-1, 3)
    prop = float(np.load(sys_dir / "set.000" / "property.npy")[0])
    symbols = [type_map[t] for t in types]
    return coord, symbols, prop


def build_molecular_crystal(coord: np.ndarray, symbols: list[str]) -> MolecularCrystal:
    from ase import Atoms
    atoms = Atoms(symbols=symbols, positions=coord, cell=CLUSTER_BOX, pbc=False)
    return MolecularCrystal.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS)


def is_x_site_candidate(sym_set: set[str]) -> bool:
    """Return True for oxidizing anions, but not O-containing B-site cations."""
    has_oxygen = "O" in sym_set
    has_halogen = bool(sym_set & {"Cl", "I"})
    nitrate_like = "N" in sym_set and "H" not in sym_set
    return has_oxygen and (has_halogen or nitrate_like)


def classify_mol(mol: CrystalMolecule) -> str:
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


def canonical_formula(mol: CrystalMolecule) -> str:
    """Compact formula like 'C2H6N2' from a CrystalMolecule."""
    from collections import Counter
    counts = Counter(mol.get_chemical_symbols())
    # Hill order: C first, H second, then alphabetical
    parts = []
    for elem in ['C', 'H']:
        if elem in counts:
            parts.append(f"{elem}{counts.pop(elem) if counts[elem] > 1 else ''}")
    for elem in sorted(counts):
        parts.append(f"{elem}{counts[elem] if counts[elem] > 1 else ''}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Build species registry from actual PEM clusters
# ---------------------------------------------------------------------------

def choose_site_representatives(
    candidates: dict[str, list[tuple[str, CrystalMolecule]]],
    site_kind: str,
    preferred_order: list[str],
) -> OrderedDict[str, tuple[str, CrystalMolecule]]:
    """Pick one representative molecule per PEM site label."""
    selected: OrderedDict[str, tuple[str, CrystalMolecule]] = OrderedDict()
    for site_key in order_site_keys(list(candidates), preferred_order):
        items = candidates[site_key]
        formula_counts: OrderedDict[str, int] = OrderedDict()
        for formula, _ in items:
            formula_counts[formula] = formula_counts.get(formula, 0) + 1
        chosen_formula = max(formula_counts, key=formula_counts.get)
        chosen_formula, chosen_mol = next((formula, mol) for formula, mol in items if formula == chosen_formula)
        selected[site_key] = (chosen_formula, chosen_mol)
        if len(formula_counts) > 1:
            print(
                f"  {site_kind}-site {pretty_site_label(site_kind, site_key)}: "
                f"picked representative {chosen_formula} from {dict(formula_counts)}"
            )
    return selected


def build_species_registry() -> tuple[
    dict[str, tuple[str, CrystalMolecule]],  # A_species: {A-site label: (formula, mol)}
    dict[str, tuple[str, CrystalMolecule]],  # B_species: {B-site label: (formula, mol)}
    dict[str, tuple[str, CrystalMolecule]],  # X_species: {X-site label: (formula, mol)}
    MolecularCrystal,  # DAP-4 template MC
    dict[str, list[int]],  # template ABX indices
]:
    """Scan PEM n1 clusters and deduplicate species by A/B/X site identity."""
    pem_table = load_pem_table()

    # Load DAP-4 template
    dap4_coord, dap4_syms, _ = read_cluster_system(CLUSTER_N1_DIR / "DAP-4")
    template_mc = build_molecular_crystal(dap4_coord, dap4_syms)
    template_abx = get_abx_indices(template_mc)

    a_candidates: dict[str, list[tuple[str, CrystalMolecule]]] = OrderedDict()
    b_candidates: dict[str, list[tuple[str, CrystalMolecule]]] = OrderedDict()
    x_candidates: dict[str, list[tuple[str, CrystalMolecule]]] = OrderedDict()

    materials = sorted(p.name for p in CLUSTER_N1_DIR.iterdir() if p.is_dir())

    for material in materials:
        if material in SKIP_MATERIALS:
            continue
        try:
            coord, symbols, _ = read_cluster_system(CLUSTER_N1_DIR / material)
            mc = build_molecular_crystal(coord, symbols)
            abx = get_abx_indices(mc)
        except Exception as e:
            print(f"  SKIP {material}: {e}")
            continue

        if not abx['A'] or not abx['B'] or not abx['X']:
            continue

        a_mol = mc.molecules[abx['A'][0]]
        b_mol = mc.molecules[abx['B'][0]]
        x_mol = mc.molecules[abx['X'][0]]

        a_formula = canonical_formula(a_mol)
        b_formula = canonical_formula(b_mol)
        x_formula = canonical_formula(x_mol)
        a_site, b_site, x_site = get_material_site_keys(material, pem_table)

        a_candidates.setdefault(a_site, []).append((a_formula, a_mol.copy()))
        b_candidates.setdefault(b_site, []).append((b_formula, b_mol.copy()))
        x_candidates.setdefault(x_site, []).append((x_formula, x_mol.copy()))

    A_species = choose_site_representatives(a_candidates, "A", _A_SITE_ORDER)
    B_species = choose_site_representatives(b_candidates, "B", _B_SITE_ORDER)
    X_species = choose_site_representatives(x_candidates, "X", _X_SITE_ORDER)

    print(f"Species registry: {len(A_species)} A, {len(B_species)} B, {len(X_species)} X")
    print(f"  A (site labels): {[pretty_site_label('A', key) for key in A_species]}")
    print(f"  B (site labels): {[pretty_site_label('B', key) for key in B_species]}")
    print(f"  X (site labels): {[pretty_site_label('X', key) for key in X_species]}")
    print(f"  Grid size: {len(A_species)} × {len(B_species)} × {len(X_species)} = "
          f"{len(A_species) * len(B_species) * len(X_species)} combinations")

    return A_species, B_species, X_species, template_mc, template_abx


# ---------------------------------------------------------------------------
# Build combinatorial structures
# ---------------------------------------------------------------------------

def build_abx_structure(
    template_mc: MolecularCrystal,
    template_abx: dict[str, list[int]],
    a_mol: CrystalMolecule,
    b_mol: CrystalMolecule,
    x_mol: CrystalMolecule,
) -> MolecularCrystal | None:
    """Replace A/B/X in template with given molecules. Returns None on clash."""
    np.random.seed(SEED)
    try:
        working_mc = template_mc
        for idx in template_abx['A']:
            working_mc = MoleculeManipulator(working_mc).replace_molecule(
                idx, a_mol, clash_threshold=0.8, max_rotation_attempts=200
            )
        for idx in template_abx['B']:
            working_mc = MoleculeManipulator(working_mc).replace_molecule(
                idx, b_mol, clash_threshold=0.8, max_rotation_attempts=200
            )
        for idx in template_abx['X']:
            working_mc = MoleculeManipulator(working_mc).replace_molecule(
                idx, x_mol, clash_threshold=0.8, max_rotation_attempts=200
            )
        return working_mc
    except MoleculeClashError:
        return None
    except Exception as e:
        print(f"    build error: {e}")
        return None


def mc_to_npy_arrays(mc: MolecularCrystal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert MolecularCrystal to coord/type/box arrays for DeepMD inference."""
    atoms = mc.to_ase()
    coord = atoms.get_positions()
    coord_c = coord - coord.mean(axis=0, keepdims=True) + 50.0
    symbols = atoms.get_chemical_symbols()
    atom_types = np.array([TYPE_TO_ID[s] for s in symbols], dtype=np.int32)
    return coord_c, atom_types, CLUSTER_BOX


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, head: str):
    """Load DeepProperty model for inference.

    Uses deepmd.pt.infer.deep_eval.DeepProperty (NOT DeepPot) because we need
    the property-head output rather than energy/force/virial.
    """
    from deepmd.pt.infer.deep_eval import DeepProperty
    kwargs: dict = {}
    if head:
        kwargs["head"] = head
    model = DeepProperty(str(ckpt_path), **kwargs)
    return model


def predict_single(model, coord: np.ndarray, atom_types: np.ndarray, box: np.ndarray) -> float | None:
    """Run inference on a single structure, return predicted property value.

    For cluster (non-periodic) systems we pass cells=None so DeepProperty
    does not apply PBC wrapping.
    """
    try:
        model_type_map = model.get_type_map()
        local_to_model = []
        for t in atom_types:
            sym = TYPE_MAP[t]
            if sym in model_type_map:
                local_to_model.append(model_type_map.index(sym))
            else:
                return None  # Element not in model

        mapped_types = np.array(local_to_model, dtype=np.int32)
        # coords shape: (nframes, natoms, 3);  cells=None for clusters
        result = model.eval(
            coords=coord.reshape(1, -1, 3),
            atom_types=mapped_types,
            cells=None,
        )
        # result[0] is the property prediction array, shape (nframes, 1) or (nframes,)
        return float(result[0].reshape(-1)[0])
    except Exception as e:
        print(f"    inference error: {e}")
        return None


def get_latest_ckpt(exp_dir: Path) -> Path | None:
    ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def aggregate_results_for_plot(results: list[dict]) -> list[dict]:
    pem_table = load_pem_table()
    known_combo_map = load_known_combo_map(pem_table)
    grouped: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict()

    for record in results:
        if record.get("pred_vdet") is None:
            continue
        norm = normalize_result_record(record, pem_table)
        key = (norm["a_site"], norm["b_site"], norm["x_site"])
        grouped.setdefault(key, []).append(float(norm["pred_vdet"]))

    aggregated: list[dict] = []
    for (a_site, b_site, x_site), values in grouped.items():
        if x_site not in _X_SITE_ORDER:
            continue
        known_materials = known_combo_map.get((a_site, b_site, x_site), [])
        aggregated.append(
            {
                "a_site": a_site,
                "b_site": b_site,
                "x_site": x_site,
                "a_site_label": pretty_site_label("A", a_site),
                "b_site_label": pretty_site_label("B", b_site),
                "x_site_label": pretty_site_label("X", x_site),
                "pred_vdet": float(np.mean(values)),
                "n_merged": len(values),
                "is_known": bool(known_materials),
                "known_materials": known_materials,
                "material_code": infer_pem_material_code(a_site, b_site, x_site),
            }
        )
    return aggregated


def plot_abx_2d_panels(results: list[dict], out_png: Path, out_pdf: Path) -> None:
    """2D heatmaps: rows = A site, columns = B site, panels = X site."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import StrMethodFormatter

    setup_nature_style()
    ok = aggregate_results_for_plot(results)
    if not ok:
        print("No successful predictions to plot.")
        return

    raw_count = sum(1 for record in results if record.get("pred_vdet") is not None)
    if raw_count != len(ok):
        print(f"Aggregated {raw_count} successful predictions into {len(ok)} unique A/B/X cells.")

    a_keys = order_site_keys([record["a_site"] for record in ok], _A_SITE_ORDER)
    b_keys = order_site_keys([record["b_site"] for record in ok], _B_SITE_ORDER)
    x_keys = [key for key in _X_SITE_ORDER if any(record["x_site"] == key for record in ok)]

    a_idx = {key: i for i, key in enumerate(a_keys)}
    b_idx = {key: i for i, key in enumerate(b_keys)}
    values = [record["pred_vdet"] for record in ok]
    vmin = float(min(values))
    vmax = float(max(values))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#F5F5F5")

    fig, all_axes = plt.subplots(
        1,
        len(x_keys) + 1,
        figsize=(4.2 * len(x_keys) + 1.55, 0.64 * len(a_keys) + 4.65),
        dpi=180,
        gridspec_kw={"width_ratios": [1.0] * len(x_keys) + [0.06]},
    )
    axes = list(all_axes[:-1])
    cax = all_axes[-1]

    image = None
    for panel_idx, (ax, x_key) in enumerate(zip(axes, x_keys)):
        matrix = np.full((len(a_keys), len(b_keys)), np.nan, dtype=float)
        known_mask = np.zeros((len(a_keys), len(b_keys)), dtype=bool)
        label_matrix = np.full((len(a_keys), len(b_keys)), "", dtype=object)
        for record in ok:
            if record["x_site"] != x_key:
                continue
            i = a_idx[record["a_site"]]
            j = b_idx[record["b_site"]]
            matrix[i, j] = record["pred_vdet"]
            known_mask[i, j] = bool(record["is_known"])
            label_matrix[i, j] = record["material_code"]

        masked = np.ma.masked_invalid(matrix)
        ax.set_facecolor("#F5F5F5")
        image = ax.imshow(
            masked,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(pretty_site_label("X", x_key), fontsize=10, pad=8)
        ax.set_xticks(range(len(b_keys)))
        ax.set_xticklabels([pretty_site_label("B", key) for key in b_keys], rotation=42, ha="right", fontsize=10)

        ax.set_yticks(range(len(a_keys)))
        if panel_idx == 0:
            ax.set_ylabel("A-site organic cation", fontsize=9)
            ax.set_yticklabels([pretty_site_label("A", key) for key in a_keys], fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_xticks(np.arange(-0.5, len(b_keys), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(a_keys), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        style_axes(ax, grid=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(len(a_keys)):
            for j in range(len(b_keys)):
                if np.isnan(matrix[i, j]):
                    ax.text(j, i, "×", ha="center", va="center", color="0.55", fontsize=10)
                    continue
                value = matrix[i, j]
                norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                rgba = cmap(np.clip(norm_value, 0.0, 1.0))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "black" if luminance > 0.6 else "white"
                stroke_color = "white" if text_color == "black" else "black"

                if not known_mask[i, j]:
                    hatch_color = (0.0, 0.0, 0.0, 0.28) if luminance > 0.58 else (1.0, 1.0, 1.0, 0.35)
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            hatch="////",
                            edgecolor=hatch_color,
                            linewidth=0.0,
                            zorder=3,
                        )
                    )

                txt = ax.text(
                    j,
                    i,
                    format_cell_label(label_matrix[i, j]),
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    fontweight="semibold",
                    linespacing=0.9,
                    color=text_color,
                    zorder=4,
                )
                txt.set_path_effects([pe.withStroke(linewidth=1.1, foreground=stroke_color, alpha=0.55)])

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Predicted Vdet (m/s)", fontsize=9, labelpad=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, length=3, color="0.35")
    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    legend_handles = [
        Patch(facecolor="white", edgecolor="0.45", linewidth=0.8, label="Known PEM"),
        Patch(facecolor="white", edgecolor="0.45", linewidth=0.8, hatch="////", label="Extrapolated"),
        Line2D(
            [0],
            [0],
            marker="$×$",
            linestyle="None",
            markersize=9,
            color="0.55",
            label="Unavailable",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.49, 0.055),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.8,
        handletextpad=0.6,
    )

    fig.suptitle(
        f"ABX detonation velocity map\n{len(ok)} valid A/B/X combinations",
        y=0.985,
    )
    fig.supxlabel("B-site cation", fontsize=9, y=0.13)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.2, wspace=0.12)
    save_png_pdf(fig, out_png, dpi=300)
    print(f"Saved 2D panel plot to {out_png} and {out_pdf}")
    plt.close(fig)

    ok_sorted = sorted(ok, key=lambda record: record["pred_vdet"], reverse=True)
    print(f"\nTop 10 predicted Vdet:")
    for record in ok_sorted[:10]:
        print(
            f"  {record['a_site_label']} + {record['b_site_label']} + {record['x_site_label']}  "
            f"→  {record['pred_vdet']:.0f} m/s"
        )
    print(f"\nBottom 10 predicted Vdet:")
    for record in ok_sorted[-10:]:
        print(
            f"  {record['a_site_label']} + {record['b_site_label']} + {record['x_site_label']}  "
            f"→  {record['pred_vdet']:.0f} m/s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ABX combinatorial grid prediction for PEMs")
    parser.add_argument("--exp", type=str, default=DEFAULT_EXP,
                        help=f"Experiment directory name (default: {DEFAULT_EXP})")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Specific checkpoint path (overrides --exp)")
    parser.add_argument("--head", type=str, default=HEAD,
                        help=f"Model head for inference (default: {HEAD})")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference, re-plot from cached results")
    parser.add_argument("--out-json", type=str, default=str(RESULTS_PATH),
                        help="Output JSON path for predictions")
    parser.add_argument("--out-png", type=str, default=str(PLOT_PATH),
                        help="Output PNG path for 2D panel plot")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_png = Path(args.out_png)
    out_pdf = out_png.with_suffix(".pdf")

    if args.plot_only:
        if not out_json.exists():
            print(f"ERROR: no cached results at {out_json}")
            return
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        # Support both old (plain list) and new (dict with metadata) formats
        results = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
        if isinstance(payload, dict) and "metadata" in payload:
            meta = payload["metadata"]
            print(f"Cached run metadata: ckpt={meta.get('checkpoint')}, "
                  f"head={meta.get('head')}, wall_time={meta.get('wall_time_seconds', '?')}s, "
                  f"timestamp={meta.get('timestamp')}")
        plot_abx_2d_panels(results, out_png, out_pdf)
        return

    # Resolve checkpoint
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        exp_dir = ROOT / args.exp
        ckpt_path = get_latest_ckpt(exp_dir)
        if ckpt_path is None:
            print(f"ERROR: no checkpoint found in {exp_dir}")
            print(f"  Available experiments: exp6v1_allpems, exp6_allpems, exp5v1_fold0..4, exp5v2_fold0..4")
            return

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Head: {args.head}")

    # Build species registry
    print("\nBuilding species registry from PEM clusters...")
    A_species, B_species, X_species, template_mc, template_abx = build_species_registry()
    known_combo_map = load_known_combo_map()

    # Load model
    print(f"\nLoading model from {ckpt_path}...")
    model = load_model(ckpt_path, args.head)
    print("Model loaded.")

    # Enumerate all A × B × X combinations
    combos = list(product(A_species.keys(), B_species.keys(), X_species.keys()))
    print(f"\nRunning inference on {len(combos)} combinations...")

    t_start = time.monotonic()
    results: list[dict] = []
    n_ok = 0
    n_clash = 0
    n_err = 0

    for i, (a_key, b_key, x_key) in enumerate(combos):
        a_formula, a_mol = A_species[a_key]
        b_formula, b_mol = B_species[b_key]
        x_formula, x_mol = X_species[x_key]
        known_materials = known_combo_map.get((a_key, b_key, x_key), [])

        # Build structure
        mc = build_abx_structure(template_mc, template_abx, a_mol, b_mol, x_mol)

        rec = {
            "a_key": a_key,
            "a_site": a_key,
            "a_site_label": pretty_site_label("A", a_key),
            "a_formula": a_formula,
            "b_key": b_key,
            "b_site": b_key,
            "b_site_label": pretty_site_label("B", b_key),
            "b_formula": b_formula,
            "x_key": x_key,
            "x_site": x_key,
            "x_site_label": pretty_site_label("X", x_key),
            "x_formula": x_formula,
            "is_known": bool(known_materials),
            "known_materials": known_materials,
            "pred_vdet": None,
            "status": "ok",
        }

        if mc is None:
            rec["status"] = "clash"
            n_clash += 1
        else:
            coord, atom_types, box = mc_to_npy_arrays(mc)
            pred = predict_single(model, coord, atom_types, box)
            if pred is not None:
                rec["pred_vdet"] = pred
                n_ok += 1
            else:
                rec["status"] = "inference_error"
                n_err += 1

        results.append(rec)

        if (i + 1) % 100 == 0 or (i + 1) == len(combos):
            print(f"  [{i+1}/{len(combos)}] ok={n_ok} clash={n_clash} err={n_err}")

    wall_time = time.monotonic() - t_start

    # Build output payload with metadata
    metadata = {
        "checkpoint": str(ckpt_path),
        "head": args.head,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": round(wall_time, 1),
        "grid_size": {"A": len(A_species), "B": len(B_species), "X": len(X_species)},
        "n_combinations": len(combos),
        "n_ok": n_ok,
        "n_clash": n_clash,
        "n_err": n_err,
        "n_known_combinations": sum(1 for combo in combos if combo in known_combo_map),
    }
    payload = {"metadata": metadata, "results": results}

    # Save results
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} results to {out_json}")
    print(f"  ok={n_ok}, clash={n_clash}, err={n_err}")
    print(f"  wall time: {wall_time:.1f}s ({wall_time/len(combos):.3f}s/combo)")

    # Plot
    plot_abx_2d_panels(results, out_png, out_pdf)


if __name__ == "__main__":
    main()
