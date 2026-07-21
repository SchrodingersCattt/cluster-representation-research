#!/usr/bin/env python3
"""Figure 4 -- Model-derived chemical insights.

Usage:
  python -u plot_fig4.py --compute-cache
  python -u plot_fig4.py --plot-only
  python -u plot_fig4.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_samples, silhouette_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import OneHotEncoder

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments"
DATA_DIR = ROOT / "data" / "pems"
PEMS_CSV = DATA_DIR / "pems.csv"
CLUSTER_N1_DIR = EXP_DIR / "00_data_prep" / "pems_cluster_n1_systems"
CACHE_PATH = THIS_DIR / "fig4_cache.npz"
COUNTERFACTUAL_CACHE = THIS_DIR / "fig4_counterfactual_abx3_abx4.json"
HYPERGRAPH_PDF = THIS_DIR / "figure4-hypergraph.pdf"
HYPERGRAPH_PNG = THIS_DIR / "figure4-hypergraph.png"
M4B_PATH = EXP_DIR / "mechanism_results" / "mechanism_m4b_results.json"
M4A_PATH = EXP_DIR / "mechanism_results" / "mechanism_m4a_results.json"
M5A_PATH = EXP_DIR / "mechanism_results" / "mechanism_m5a_results.json"
M1_PATH = EXP_DIR / "mechanism_results" / "mechanism_m1_results.json"
M3_PATH = EXP_DIR / "mechanism_results" / "mechanism_m3_results.json"
GRID_PATH = EXP_DIR / "abx_grid_predictions_exp6v1_allpems_400k.json"
BOOTSTRAP_PATH = EXP_DIR / "_stats_bootstrap" / "bootstrap_results.json"

sys.path.insert(0, str(EXP_DIR))
from paper_plot_style import add_panel_label, save_png_pdf, setup_nature_style, style_axes  # noqa: E402

# Programmatic figure-QA helpers (see FIGURE_QA.md). Imported via the figures
# directory so that running `plot_fig4.py` directly picks up the same module
# the rest of the figures use.
sys.path.insert(0, str(THIS_DIR))
import _qa_check as _qa  # noqa: E402
from figure_style import display_material  # noqa: E402

A_COLOR = "#8A5A67"
B_COLOR = "#70754A"
X_COLOR = "#5A6D7B"
BASELINE_COLOR = "#C8B8A8"
CHARCOAL = "#2F2F2F"
MID_GRAY = "#D6D6D6"
MT_COLOR = "#4A6274"
ST_COLOR = "#7A4B58"
SCRATCH_COLOR = "#6B6B6B"
SITE_TYPES = ("A", "B", "X")
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
ELLIPSE_CHI2_95 = 5.991464547107979
FIG4_FONT = 8.0
FIG4_SMALL_FONT = 8.0

A_SITE_ORDER = [
    "H2dabco2+",
    "MeHdabco2+",
    "H2odabco2+",
    "H2pz2+",
    "H2hpz2+",
    "MeHpz2+",
]
B_SITE_ORDER_GRID = ["Na+", "K+", "Rb+", "NH4+", "Ag+", "NH3OH+", "NH2NH3+"]

SITE_LABELS = {
    "H2dabco2+": r"H$_2$dabco$^{2+}$",
    "MeHdabco2+": r"MeHdabco$^{2+}$",
    "H2odabco2+": r"H$_2$odabco$^{2+}$",
    "H2pz2+": r"H$_2$pz$^{2+}$",
    "H2hpz2+": r"H$_2$hpz$^{2+}$",
    "MeHpz2+": r"MeHpz$^{2+}$",
    "Na+": r"Na$^+$",
    "K+": r"K$^+$",
    "Rb+": r"Rb$^+$",
    "NH4+": r"NH$_4^+$",
    "Ag+": r"Ag$^+$",
    "NH3OH+": r"NH$_3$OH$^+$",
    "NH2NH3+": r"NH$_2$NH$_3^+$",
    "ClO4-": r"ClO$_4^-$",
    "NO3-": r"NO$_3^-$",
    "IO4-": r"IO$_4^-$",
    "H4IO6-": r"H$_4$IO$_6^-$",
    "ClO3-": r"ClO$_3^-$",
}

X_SITE_COLORS = {
    "ClO4-": X_COLOR,
    "NO3-": "#9E8B5E",
    "IO4-": "#7A6B8A",
    "H4IO6-": "#7A6B8A",
    "ClO3-": "#8A7A63",
}

RIDGE_ALPHAS = np.logspace(-3, 3, 25)
FOLD_IDS = [0, 1, 2, 3, 4]
CKPT_STEP = 400000

_HEAVY = ["I", "Na", "K", "Rb", "Ba", "Ag"]
_HEAVY_NONO_LIMITS = {
    "I": 2.10,
    "Na": 2.30,
    "K": 2.50,
    "Rb": 2.60,
    "Ba": 2.60,
    "Ag": 2.30,
}
_HEAVY_O_LIMITS = {
    "I": 2.05,
    "Na": 2.20,
    "K": 2.30,
    "Rb": 2.40,
    "Ba": 2.40,
    "Ag": 2.20,
}
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

_METAL_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_pem_rows() -> dict[str, dict[str, str]]:
    with PEMS_CSV.open(newline="", encoding="utf-8") as handle:
        return {row["material"]: row for row in csv.DictReader(handle) if row.get("material")}


def load_training_materials() -> list[str]:
    rows = load_pem_rows()
    m4b = _load_json(M4B_PATH)
    return [m for m in m4b["materials"] if m in rows and m != "DAC-4"]


def load_metadata(materials: list[str]) -> dict[str, dict[str, object]]:
    rows = load_pem_rows()
    meta = {}
    for material in materials:
        row = rows[material]
        meta[material] = {
            "A": row["A_site"].strip(),
            "B": row["B_site"].strip(),
            "X": row["X_site"].strip(),
            "Vdet": float(row["D_km_s"]) * 1000.0,
        }
    return meta


def compute_crystal_density(cif_path: str | Path) -> float | None:
    import re
    from ase.data import atomic_masses, atomic_numbers
    from ase.io import read as ase_read

    cif_path = Path(cif_path)
    if not cif_path.exists():
        return None
    text = cif_path.read_text(errors="replace")
    avogadro = 6.02214076e23

    def _get_float(tag: str) -> float | None:
        match = re.search(rf"{tag}\s+([\d.]+)", text)
        return float(match.group(1)) if match else None

    reported_density = _get_float("_exptl_crystal_density_diffrn") or _get_float("_exptl_crystal_density_meas")
    if reported_density and 0.5 < reported_density < 6.0:
        return reported_density

    a = _get_float("_cell_length_a")
    b = _get_float("_cell_length_b")
    c = _get_float("_cell_length_c")
    alpha = _get_float("_cell_angle_alpha") or 90.0
    beta = _get_float("_cell_angle_beta") or 90.0
    gamma = _get_float("_cell_angle_gamma") or 90.0

    if a and b and c:
        ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
        vol = a * b * c * np.sqrt(
            1 - np.cos(ar) ** 2 - np.cos(br) ** 2 - np.cos(gr) ** 2
            + 2 * np.cos(ar) * np.cos(br) * np.cos(gr)
        )
    else:
        vol = None

    z_units = _get_float("_cell_formula_units_Z")
    match = re.search(r"_chemical_formula_sum\s+'([^']+)'", text)
    if not match:
        match = re.search(r"_chemical_formula_sum\s+\"([^\"]+)\"", text)

    mw = 0.0
    if match:
        for elem, cnt in re.findall(r"([A-Z][a-z]?)([\d.]*)", match.group(1)):
            count = float(cnt) if cnt else 1.0
            if elem in atomic_numbers:
                mw += atomic_masses[atomic_numbers[elem]] * count

    if z_units and z_units > 0 and mw > 0 and vol and vol > 0:
        density = z_units * mw / (vol * avogadro) * 1e24
        if 0.5 < density < 6.0:
            return density

    try:
        atoms = ase_read(str(cif_path))
        density = atoms.get_masses().sum() / atoms.get_volume() * 1.6605
        if 0.5 < density < 6.0:
            return density
    except Exception:
        return None
    return None


def load_densities(materials: list[str]) -> dict[str, float]:
    conf_dir = DATA_DIR / "confs"
    densities = {}
    for material in materials:
        density = compute_crystal_density(conf_dir / f"{material}.cif")
        if density is not None:
            densities[material] = density
    return densities


def _resolve_ckpt(exp_subdir: str) -> Path:
    exp_dir = EXP_DIR / exp_subdir
    pinned = exp_dir / f"model.ckpt-{CKPT_STEP}.pt"
    if pinned.exists():
        return pinned
    ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {exp_dir}")


def _load_descriptor_model(exp_name: str, fold_id: int):
    from deepmd.infer import DeepPot

    ckpt = _resolve_ckpt(f"{exp_name}_fold{fold_id}")
    # Single-task models (exp7d) have no heads
    _MULTI_HEAD_EXPS = {"exp7a", "exp7b", "exp7c", "exp8a", "exp9a"}
    kwargs = {"head": "deepems_vanilla"} if exp_name in _MULTI_HEAD_EXPS else {}
    return DeepPot(str(ckpt), **kwargs)


def _read_cluster_system(sys_dir: Path) -> tuple[np.ndarray, list[str]]:
    type_map = (sys_dir / "type_map.raw").read_text(encoding="utf-8").strip().split()
    types = np.loadtxt(sys_dir / "type.raw", dtype=int)
    coord = np.load(sys_dir / "set.000" / "coord.npy")[0].reshape(-1, 3)
    return coord, [type_map[t] for t in types]


def _extract_per_atom_embedding(dp, coord: np.ndarray, symbols: list[str]) -> np.ndarray:
    type_map = dp.get_type_map()
    atom_types = np.array([type_map.index(symbol) for symbol in symbols], dtype=np.int32)
    desc = dp.eval_descriptor(np.asarray(coord, dtype=np.float64).reshape(1, -1, 3), None, atom_types)
    if isinstance(desc, (list, tuple)):
        desc = desc[0]
    arr = np.asarray(desc)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.reshape(-1, arr.shape[-1])


def _build_molecular_crystal(coord: np.ndarray, symbols: list[str]):
    if os.environ.get("MOLCRYSKIT_ROOT"):
        sys.path.insert(0, os.environ["MOLCRYSKIT_ROOT"])
    from ase import Atoms
    from molcrys_kit.structures.crystal import MolecularCrystal

    atoms = Atoms(symbols=symbols, positions=coord, cell=np.eye(3) * 100.0, pbc=False)
    return MolecularCrystal.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS)


def _classify_mol(mol) -> str:
    syms = mol.get_chemical_symbols()
    sym_set = set(syms)
    has_c = "C" in sym_set
    has_halogen_or_n = bool(sym_set & {"Cl", "I", "N"})
    has_metal = bool(sym_set & _METAL_ELEMENTS)
    if has_c:
        return "A"
    if has_metal and "O" not in sym_set and not has_halogen_or_n:
        return "B"
    if "O" in sym_set and (bool(sym_set & {"Cl", "I"}) or ("N" in sym_set and "H" not in sym_set)):
        return "X"
    if len(syms) <= 6 and not has_c:
        return "B"
    return "B"


def _get_abx_atom_indices(mc, coord: np.ndarray, symbols: list[str]) -> dict[str, list[int]]:
    site_atoms = {"A": [], "B": [], "X": []}
    used = set()
    for mol in mc.molecules:
        mol_pos = mol.get_positions()
        mol_syms = mol.get_chemical_symbols()
        mol_sym_set = set(mol_syms)
        metal_atoms = [i for i, sym in enumerate(mol_syms) if sym in _METAL_ELEMENTS]
        has_anion_signature = "O" in mol_sym_set and (
            bool(mol_sym_set & {"Cl", "I"}) or ("N" in mol_sym_set and "H" not in mol_sym_set)
        )

        if metal_atoms and has_anion_signature and "C" not in mol_sym_set:
            for atom_idx, atom_pos in enumerate(mol_pos):
                dists = np.linalg.norm(coord - atom_pos, axis=1)
                best = int(np.argmin(dists))
                if dists[best] < 0.05 and symbols[best] == mol_syms[atom_idx] and best not in used:
                    site = "B" if atom_idx in metal_atoms else "X"
                    site_atoms[site].append(best)
                    used.add(best)
            continue

        site = _classify_mol(mol)
        for atom_idx, atom_pos in enumerate(mol_pos):
            dists = np.linalg.norm(coord - atom_pos, axis=1)
            best = int(np.argmin(dists))
            if dists[best] < 0.05 and symbols[best] == mol_syms[atom_idx] and best not in used:
                site_atoms[site].append(best)
                used.add(best)
    return site_atoms


def _prepare_material_systems(materials: list[str]) -> dict[str, dict[str, object]]:
    systems: dict[str, dict[str, object]] = {}
    for material in materials:
        coord, symbols = _read_cluster_system(CLUSTER_N1_DIR / material)
        mc = _build_molecular_crystal(coord, symbols)
        systems[material] = {
            "coord": coord,
            "symbols": symbols,
            "abx_idx": _get_abx_atom_indices(mc, coord, symbols),
        }
    return systems


def _compute_model_embeddings(
    materials: list[str],
    material_systems: dict[str, dict[str, object]],
    exp_name: str,
) -> dict[str, object]:
    site_acc = {site: {material: [] for material in materials} for site in SITE_TYPES}
    material_acc = {material: [] for material in materials}
    atomic_acc: dict[str, list[np.ndarray]] = {material: [] for material in materials}

    for fold_id in FOLD_IDS:
        print(f"{exp_name} descriptor fold {fold_id}")
        model = _load_descriptor_model(exp_name, fold_id)
        for material in materials:
            system = material_systems[material]
            coord = system["coord"]
            symbols = system["symbols"]
            abx_idx = system["abx_idx"]
            desc = _extract_per_atom_embedding(model, coord, symbols)
            material_acc[material].append(desc.mean(axis=0))
            atomic_acc[material].append(desc)
            for site in SITE_TYPES:
                idx = abx_idx[site]
                pooled = desc[idx].mean(axis=0) if idx else np.zeros(desc.shape[1])
                site_acc[site][material].append(pooled)

    material_emb = np.stack([np.mean(material_acc[m], axis=0) for m in materials], axis=0)
    site_emb = {
        site: np.stack([np.mean(site_acc[site][m], axis=0) for m in materials], axis=0)
        for site in SITE_TYPES
    }
    atomic_emb = {m: np.mean(np.stack(atomic_acc[m], axis=0), axis=0) for m in materials}

    atom_blocks: list[np.ndarray] = []
    atom_sites: list[str] = []
    atom_elements: list[str] = []
    atom_materials: list[str] = []
    for material in materials:
        system = material_systems[material]
        symbols = system["symbols"]
        abx_idx = system["abx_idx"]
        site_lookup = ["?"] * len(symbols)
        for site in SITE_TYPES:
            for idx in abx_idx[site]:
                site_lookup[idx] = site
        atom_blocks.append(atomic_emb[material])
        atom_sites.extend(site_lookup)
        atom_elements.extend(symbols)
        atom_materials.extend([material] * len(symbols))

    return {
        "material_emb": material_emb,
        "site_emb": site_emb,
        "atomic_emb": np.concatenate(atom_blocks, axis=0),
        "atom_sites": np.array(atom_sites),
        "atom_elements": np.array(atom_elements),
        "atom_materials": np.array(atom_materials),
    }


def _flatten_site_embeddings(
    materials: list[str],
    site_emb: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = []
    point_sites = []
    point_materials = []
    for material_idx, material in enumerate(materials):
        for site in SITE_TYPES:
            stacked.append(site_emb[site][material_idx])
            point_sites.append(site)
            point_materials.append(material)
    return (
        np.stack(stacked, axis=0),
        np.array(point_sites),
        np.array(point_materials),
    )


def _joint_umap_2d(arrays: list[np.ndarray]) -> list[np.ndarray]:
    from umap import UMAP

    combined = np.concatenate(arrays, axis=0)
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(UMAP_NEIGHBORS, len(combined) - 1),
        min_dist=UMAP_MIN_DIST,
        random_state=42,
    )
    reducer.fit(combined)
    return [reducer.transform(array) for array in arrays]


def _atomic_umap_2d(matrix: np.ndarray) -> np.ndarray:
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=min(UMAP_NEIGHBORS, len(matrix) - 1),
        min_dist=UMAP_MIN_DIST,
        random_state=42,
    )
    return reducer.fit_transform(matrix)


def compute_cache(materials: list[str]) -> None:
    material_systems = _prepare_material_systems(materials)
    mt = _compute_model_embeddings(materials, material_systems, "exp7a")
    st = _compute_model_embeddings(materials, material_systems, "exp7c")
    scratch = _compute_model_embeddings(materials, material_systems, "exp7d")

    mt_points, point_sites, point_materials = _flatten_site_embeddings(materials, mt["site_emb"])
    st_points, _, _ = _flatten_site_embeddings(materials, st["site_emb"])
    mt_umap_coords, st_umap_coords = _joint_umap_2d([mt_points, st_points])

    print(f"Atomic UMAP: MT ({len(mt['atomic_emb'])} atoms)")
    mt_atomic_umap = _atomic_umap_2d(mt["atomic_emb"])
    print(f"Atomic UMAP: ST ({len(st['atomic_emb'])} atoms)")
    st_atomic_umap = _atomic_umap_2d(st["atomic_emb"])
    print(f"Atomic UMAP: scratch ({len(scratch['atomic_emb'])} atoms)")
    scratch_atomic_umap = _atomic_umap_2d(scratch["atomic_emb"])

    np.savez_compressed(
        str(CACHE_PATH),
        materials=np.array(materials),
        mt_material_emb=mt["material_emb"],
        mt_site_emb_A=mt["site_emb"]["A"],
        mt_site_emb_B=mt["site_emb"]["B"],
        mt_site_emb_X=mt["site_emb"]["X"],
        st_material_emb=st["material_emb"],
        st_site_emb_A=st["site_emb"]["A"],
        st_site_emb_B=st["site_emb"]["B"],
        st_site_emb_X=st["site_emb"]["X"],
        scratch_material_emb=scratch["material_emb"],
        scratch_site_emb_A=scratch["site_emb"]["A"],
        scratch_site_emb_B=scratch["site_emb"]["B"],
        scratch_site_emb_X=scratch["site_emb"]["X"],
        mt_umap_coords=mt_umap_coords,
        st_umap_coords=st_umap_coords,
        point_sites=point_sites,
        point_materials=point_materials,
        mt_atomic_umap=mt_atomic_umap,
        st_atomic_umap=st_atomic_umap,
        scratch_atomic_umap=scratch_atomic_umap,
        mt_atomic_emb=mt["atomic_emb"],
        st_atomic_emb=st["atomic_emb"],
        scratch_atomic_emb=scratch["atomic_emb"],
        mt_atom_sites=mt["atom_sites"],
        mt_atom_elements=mt["atom_elements"],
        mt_atom_materials=mt["atom_materials"],
        st_atom_sites=st["atom_sites"],
        st_atom_elements=st["atom_elements"],
        st_atom_materials=st["atom_materials"],
        scratch_atom_sites=scratch["atom_sites"],
        scratch_atom_elements=scratch["atom_elements"],
        scratch_atom_materials=scratch["atom_materials"],
    )
    print(f"Saved cache to {CACHE_PATH}")


def refit_umap_in_cache(materials: list[str]) -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing {CACHE_PATH}. Run with --compute-cache first.")
    with np.load(str(CACHE_PATH), allow_pickle=False) as cache:
        stored = {key: cache[key] for key in cache.files}

    if stored["materials"].tolist() != materials:
        raise RuntimeError("fig4_cache.npz material order does not match current figure inputs.")

    mt_site_emb = {site: stored[f"mt_site_emb_{site}"] for site in SITE_TYPES}
    st_site_emb = {site: stored[f"st_site_emb_{site}"] for site in SITE_TYPES}
    mt_points, point_sites, point_materials = _flatten_site_embeddings(materials, mt_site_emb)
    st_points, _, _ = _flatten_site_embeddings(materials, st_site_emb)

    mt_umap_coords, st_umap_coords = _joint_umap_2d([mt_points, st_points])
    stored["mt_umap_coords"] = mt_umap_coords
    stored["st_umap_coords"] = st_umap_coords
    stored["point_sites"] = point_sites
    stored["point_materials"] = point_materials

    np.savez_compressed(str(CACHE_PATH), **stored)
    print(
        f"Refit joint UMAP (min_dist={UMAP_MIN_DIST}, n_neighbors={UMAP_NEIGHBORS}); "
        f"updated {CACHE_PATH}"
    )


def ensure_atomic_descriptor_cache(materials: list[str]) -> None:
    """Add raw per-atom descriptors to an existing cache without refitting UMAP."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing {CACHE_PATH}. Run with --compute-cache first.")
    with np.load(str(CACHE_PATH), allow_pickle=False) as cache:
        stored = {key: cache[key] for key in cache.files}

    if stored["materials"].tolist() != materials:
        raise RuntimeError("fig4_cache.npz material order does not match current figure inputs.")
    if "mt_atomic_emb" in stored and "st_atomic_emb" in stored:
        return

    material_systems = _prepare_material_systems(materials)
    mt = _compute_model_embeddings(materials, material_systems, "exp7a")
    st = _compute_model_embeddings(materials, material_systems, "exp7c")
    stored["mt_atomic_emb"] = mt["atomic_emb"]
    stored["st_atomic_emb"] = st["atomic_emb"]
    np.savez_compressed(str(CACHE_PATH), **stored)
    print(f"Backfilled raw atomic descriptors in {CACHE_PATH} without refitting UMAP")


def _load_cache(materials: list[str]) -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing {CACHE_PATH}. Run with --compute-cache first.")
    cache = np.load(str(CACHE_PATH), allow_pickle=False)
    if cache["materials"].tolist() != materials:
        raise RuntimeError("fig4_cache.npz material order does not match current figure inputs.")
    required = [
        "mt_material_emb",
        "mt_site_emb_A",
        "mt_site_emb_B",
        "mt_site_emb_X",
        "mt_umap_coords",
        "point_sites",
        "point_materials",
        "mt_atomic_umap",
        "st_atomic_umap",
        "mt_atomic_emb",
        "st_atomic_emb",
        "mt_atom_sites",
        "mt_atom_elements",
        "st_atom_sites",
        "st_atom_elements",
        "scratch_atomic_umap",
        "scratch_atomic_emb",
        "scratch_atom_sites",
        "scratch_atom_elements",
        "scratch_material_emb",
        "scratch_site_emb_A",
        "scratch_site_emb_B",
        "scratch_site_emb_X",
    ]
    missing = [key for key in required if key not in cache]
    if missing:
        raise RuntimeError(f"fig4_cache.npz is outdated; missing keys: {missing}. Run with --compute-cache first.")
    return {
        "materials": cache["materials"],
        "material_emb": cache["mt_material_emb"],
        "site_emb_A": cache["mt_site_emb_A"],
        "site_emb_B": cache["mt_site_emb_B"],
        "site_emb_X": cache["mt_site_emb_X"],
        "mt_umap_coords": cache["mt_umap_coords"],
        "st_umap_coords": cache["st_umap_coords"],
        "point_sites": cache["point_sites"],
        "point_materials": cache["point_materials"],
        "mt_atomic_umap": cache["mt_atomic_umap"],
        "st_atomic_umap": cache["st_atomic_umap"],
        "mt_atomic_emb": cache["mt_atomic_emb"],
        "st_atomic_emb": cache["st_atomic_emb"],
        "mt_atom_sites": cache["mt_atom_sites"],
        "mt_atom_elements": cache["mt_atom_elements"],
        "st_atom_sites": cache["st_atom_sites"],
        "st_atom_elements": cache["st_atom_elements"],
        "scratch_atomic_umap": cache["scratch_atomic_umap"],
        "scratch_atomic_emb": cache["scratch_atomic_emb"],
        "scratch_atom_sites": cache["scratch_atom_sites"],
        "scratch_atom_elements": cache["scratch_atom_elements"],
        "scratch_material_emb": cache["scratch_material_emb"],
        "scratch_site_emb_A": cache["scratch_site_emb_A"],
        "scratch_site_emb_B": cache["scratch_site_emb_B"],
        "scratch_site_emb_X": cache["scratch_site_emb_X"],
    }


def _loo_ridge_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(5, len(y)))
    model.fit(x, y)
    return cross_val_predict(Ridge(alpha=float(model.alpha_)), x, y, cv=LeaveOneOut())


def _loo_composition_predict(site_rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    encoder = _make_onehot_encoder()
    x = encoder.fit_transform(site_rows)
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=min(5, len(y)))
    model.fit(x, y)
    return cross_val_predict(Ridge(alpha=float(model.alpha_)), x, y, cv=LeaveOneOut())


def compute_probe_data(cache: dict[str, np.ndarray], meta: dict[str, dict[str, object]]) -> dict[str, object]:
    materials = cache["materials"].tolist()
    y_vdet = np.array([meta[m]["Vdet"] for m in materials], dtype=float)
    site_rows = np.array([[meta[m]["A"], meta[m]["B"], meta[m]["X"]] for m in materials], dtype=object)

    m4b = _load_json(M4B_PATH)
    m4b_site_r2 = m4b["site_r2"]["exp7a"]
    m4b_site_r2_std = m4b.get("site_r2_std_across_folds", {}).get("exp7a", {})
    site_results = {
        "X-site": {
            "r2": float(m4b_site_r2["z_X"]),
            "r2_std": float(m4b_site_r2_std.get("z_X", 0.0)),
        },
        "B-site": {
            "r2": float(m4b_site_r2["z_B"]),
            "r2_std": float(m4b_site_r2_std.get("z_B", 0.0)),
        },
        "A-site": {
            "r2": float(m4b_site_r2["z_A"]),
            "r2_std": float(m4b_site_r2_std.get("z_A", 0.0)),
        },
        }

    baseline_preds = _loo_composition_predict(site_rows, y_vdet)
    site_results["Composition"] = {
        "r2": float(r2_score(y_vdet, baseline_preds)),
        "r2_std": 0.0,
    }

    m4a = _load_json(M4A_PATH)
    density_block = m4a["aggregated"]["exp7a"]
    densities = load_densities(materials)
    density_materials = [m for m in materials if m in densities]
    density_idx = np.array([materials.index(m) for m in density_materials], dtype=int)
    y_density = np.array([densities[m] for m in density_materials], dtype=float)
    y_density_pred = _loo_ridge_predict(cache["material_emb"][density_idx], y_density)

    return {
        "site_probe": site_results,
        "density": {
            "materials": density_materials,
            "y_true": y_density,
            "y_pred": y_density_pred,
            "r2": float(density_block.get("r2_density", r2_score(y_density, y_density_pred))),
            "r2_std": float(density_block.get("r2_density_std_across_folds", 0.0)),
            "mae": float(mean_absolute_error(y_density, y_density_pred)),
        },
    }


def compute_ab_cosines(cache: dict[str, np.ndarray], meta: dict[str, dict[str, object]]) -> np.ndarray:
    materials = cache["materials"].tolist()
    emb = {material: cache["material_emb"][i] for i, material in enumerate(materials)}
    cosines = []

    for anchor in materials:
        a_anchor = str(meta[anchor]["A"])
        b_anchor = str(meta[anchor]["B"])
        x_anchor = str(meta[anchor]["X"])
        a_neighbors = [
            mat for mat in materials
            if str(meta[mat]["B"]) == b_anchor and str(meta[mat]["X"]) == x_anchor and str(meta[mat]["A"]) != a_anchor
        ]
        b_neighbors = [
            mat for mat in materials
            if str(meta[mat]["A"]) == a_anchor and str(meta[mat]["X"]) == x_anchor and str(meta[mat]["B"]) != b_anchor
        ]
        for a_mat in a_neighbors:
            d_a = emb[a_mat] - emb[anchor]
            norm_a = np.linalg.norm(d_a)
            if norm_a < 1e-10:
                continue
            for b_mat in b_neighbors:
                d_b = emb[b_mat] - emb[anchor]
                norm_b = np.linalg.norm(d_b)
                if norm_b < 1e-10:
                    continue
                cosines.append(float(np.dot(d_a, d_b) / (norm_a * norm_b)))
    return np.array(cosines, dtype=float)


def load_grid_subset(x_key: str) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(GRID_PATH)
    matrix = np.full((len(A_SITE_ORDER), len(B_SITE_ORDER_GRID)), np.nan)
    known = np.zeros_like(matrix, dtype=bool)
    a_index = {a: i for i, a in enumerate(A_SITE_ORDER)}
    b_index = {b: i for i, b in enumerate(B_SITE_ORDER_GRID)}
    for rec in payload["results"]:
        if rec["status"] != "ok" or rec["x_key"] != x_key:
            continue
        if rec["a_key"] not in a_index or rec["b_key"] not in b_index:
            continue
        i = a_index[rec["a_key"]]
        j = b_index[rec["b_key"]]
        matrix[i, j] = float(rec["pred_vdet"])
        known[i, j] = bool(rec["is_known"])
    if np.isnan(matrix).any():
        raise RuntimeError(f"Incomplete grid for X = {x_key}")
    return matrix, known


def compute_additive_residual(matrix: np.ndarray) -> np.ndarray:
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    return matrix - (row_mean + col_mean - float(matrix.mean()))


def compute_abx4_counterfactual(refresh: bool = False) -> dict[str, dict[str, float]]:
    if COUNTERFACTUAL_CACHE.exists() and not refresh:
        return _load_json(COUNTERFACTUAL_CACHE)

    from ase.io import read as ase_read
    from deepmd.pt.infer.deep_eval import DeepProperty
    from molcrys_kit.structures.crystal import MolecularCrystal
    from predict_eap4_paph6 import (
        CLEANED_CIF_DIR,
        HEAD,
        _build_dap4_abx4_substituted,
        _build_dap4_substituted,
        build_seeded_stoichiometric_cluster,
        cluster_atoms_centered,
        crystal_to_minimum_image_atoms,
        get_abx_indices as get_template_abx_indices,
        get_representative_mol,
        run_inference,
    )

    model = DeepProperty(str(_resolve_ckpt("exp6v1_allpems")), head=HEAD)
    model_type_map = model.get_type_map()

    eap4_crystal = MolecularCrystal.from_ase(
        ase_read(str(CLEANED_CIF_DIR / "EAP-4.cif")),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    cluster_crystal, _, _ = build_seeded_stoichiometric_cluster(eap4_crystal, dataset_name="cluster_n1", seed=101)
    cluster_atoms = cluster_atoms_centered(crystal_to_minimum_image_atoms(cluster_crystal))
    eap4_mc = MolecularCrystal.from_ase(cluster_atoms)
    eap4_abx = get_template_abx_indices(eap4_mc)
    h2en_mol = eap4_mc.molecules[eap4_abx["A"][0]].copy()

    a_mols = {
        "PEP": get_representative_mol("PAP-4", "A"),
        "MPEP": get_representative_mol("PAP-M4", "A"),
        "HPEP": get_representative_mol("PAP-H4", "A"),
        "SY": get_representative_mol("DAP-4", "A"),
    }

    results = {}
    for material, a_mol in a_mols.items():
        abx3_preds = []
        abx4_preds = []
        for variant in ("cluster_n1", "cluster_n2", "cluster_n3"):
            atoms3 = _build_dap4_substituted(variant, a_mol, h2en_mol)
            atoms4 = _build_dap4_abx4_substituted(variant, a_mol, h2en_mol)
            if atoms3 is not None:
                abx3_preds.append(run_inference(model, model_type_map, atoms3))
            if atoms4 is not None:
                abx4_preds.append(run_inference(model, model_type_map, atoms4))
        results[material] = {
            "abx3_mean_m_s": float(np.mean(abx3_preds)),
            "abx3_std_m_s": float(np.std(abx3_preds)),
            "abx4_mean_m_s": float(np.mean(abx4_preds)),
            "abx4_std_m_s": float(np.std(abx4_preds)),
            "delta_m_s": float(np.mean(abx4_preds) - np.mean(abx3_preds)),
        }

    COUNTERFACTUAL_CACHE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _value_text_color(value: float, norm, threshold: float = 0.63) -> str:
    try:
        rel = float(norm(value))
    except Exception:
        rel = 0.5
    return "white" if abs(rel - 0.5) > threshold - 0.5 else CHARCOAL


def plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    known: np.ndarray,
    title: str,
    cmap,
    norm,
    annotate_signed: bool = False,
    show_y_axis: bool = True,
    title_loc: str = "center",
) -> None:
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, pad=4, loc=title_loc)
    ax.set_xticks(range(len(B_SITE_ORDER_GRID)))
    ax.set_xticklabels([SITE_LABELS[b] for b in B_SITE_ORDER_GRID], rotation=40, ha="right")
    ax.set_yticks(range(len(A_SITE_ORDER)))
    ax.set_yticklabels([SITE_LABELS[a] for a in A_SITE_ORDER] if show_y_axis else [])
    ax.set_xlabel("B-site")
    ax.set_ylabel("A-site" if show_y_axis else "")
    style_axes(ax, grid=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:+.0f}" if annotate_signed else f"{value:.0f}"
            text_color = _value_text_color(value, norm)
            stroke_color = CHARCOAL if text_color == "white" else "white"
            txt = ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=FIG4_FONT, color=text_color, zorder=6,
            )
            if not known[i, j]:
                txt.set_path_effects([
                    mpe.Stroke(linewidth=0.5, foreground=stroke_color),
                    mpe.Normal(),
                ])
            if known[i, j]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=CHARCOAL,
                        linewidth=0.5,
                        zorder=4,
                    )
                )
            else:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=text_color,
                        linewidth=0.0,
                        hatch="////",
        zorder=3,
    )
                )
                ax.add_patch(
                    mpatches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor=CHARCOAL,
                        linewidth=0.5,
                        zorder=4,
                    )
                )


def _compute_site_silhouette(coords: np.ndarray, point_sites: np.ndarray) -> dict[str, float]:
    label_map = {site: idx for idx, site in enumerate(SITE_TYPES)}
    labels = np.array([label_map[site] for site in point_sites], dtype=int)
    sil = silhouette_samples(coords, labels)
    return {
        site: float(np.mean(sil[labels == idx]))
        for site, idx in label_map.items()
    }


def _plot_umap_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    point_sites: np.ndarray,
    point_materials: np.ndarray,
    meta: dict[str, dict[str, object]],
    title: str,
    title_color: str,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    for site, color, label in [
        ("A", A_COLOR, "A-site (fuel)"),
        ("B", B_COLOR, "B-site (modulator)"),
        ("X", X_COLOR, "X-site (oxidizer)"),
    ]:
        mask = np.array(point_sites == site, dtype=bool)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=25,
            color=color,
            alpha=0.90,
            edgecolors="none",
            linewidths=0.3,
            label=label,
            zorder=3,
        )

    idx = [
        i
        for i, (site, mat) in enumerate(zip(point_sites, point_materials))
        if site == "B" and str(meta[str(mat)]["B"]) == "NH3OH+"
    ]
    if idx:
        point = coords[idx[0]]
        ax.annotate(
            r"NH$_3$OH$^+$",
            tuple(point),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=FIG4_FONT,
        ha="left",
        va="bottom",
            arrowprops={"arrowstyle": "-", "color": CHARCOAL, "lw": 0.5},
        )

    ax.text(0.03, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=FIG4_FONT, color=title_color)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    if show_legend:
        ax.legend(frameon=False, loc="lower left", handletextpad=0.3, labelspacing=0.25)
    style_axes(ax, grid=False)


def _load_compactness() -> dict[str, dict[str, float]]:
    payload = _load_json(M4A_PATH)
    agg = payload["aggregated"]
    return {
        "MT": {
            "mean": float(agg["exp7a"]["compactness"]),
            "std": float(agg["exp7a"]["compactness_std_across_folds"]),
        },
        "ST": {
            "mean": float(agg["exp7c"]["compactness"]),
            "std": float(agg["exp7c"]["compactness_std_across_folds"]),
        },
    }


def _plot_silhouette_panel(
    ax: plt.Axes,
    silhouette_mt: dict[str, float],
    silhouette_st: dict[str, float],
) -> None:
    x = np.arange(len(SITE_TYPES), dtype=float)
    width = 0.36
    mt_vals = [silhouette_mt[site] for site in SITE_TYPES]
    st_vals = [silhouette_st[site] for site in SITE_TYPES]
    ax.bar(x - width / 2.0, mt_vals, width=width, color=MT_COLOR, label="MT")
    ax.bar(x + width / 2.0, st_vals, width=width, color=ST_COLOR, label="ST")
    ax.set_xticks(x)
    ax.set_xticklabels(["A", "B", "X"])
    ax.set_ylabel("Silhouette score", labelpad=1)
    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.0,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    y_lo = min(0.0, min(mt_vals), min(st_vals)) - 0.05
    y_hi = max(max(mt_vals), max(st_vals)) + 0.25
    ax.axhline(0.0, color=CHARCOAL, lw=0.5, ls="--", zorder=1)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    style_axes(ax, grid=True)
    ax.grid(axis="x", visible=False)


def _plot_compactness_panel(
    ax: plt.Axes,
    compactness: dict[str, dict[str, float]],
) -> None:
    labels = ["MT", "ST"]
    means = [compactness["MT"]["mean"], compactness["ST"]["mean"]]
    stds = [compactness["MT"]["std"], compactness["ST"]["std"]]
    colors = [MT_COLOR, ST_COLOR]
    x = np.arange(len(labels), dtype=float)
    ax.bar(
        x,
            means,
        yerr=stds,
        color=colors,
        width=0.58,
        error_kw={"lw": 0.25, "capsize": 1.8, "capthick": 0.25},
    )
    for xi, val in zip(x, means):
        ax.text(xi, val * 1.20, f"{val:.2f}", va="bottom", ha="center", fontsize=FIG4_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 70.0)
    ax.set_ylabel(r"Mean $\|\mathbf{x}-\bar{\mathbf{x}}\|_2$ (a.u., log)", labelpad=1)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    style_axes(ax, grid=True)
    ax.grid(axis="x", visible=False)



def _load_stability() -> dict:
    """Load m5a cross-fold descriptor stability results.

    Returns {model_key: {mat: {cos_mean, cos_std, ...}}} and the aggregated summary.
    """
    payload = _load_json(M5A_PATH)
    return payload


def _load_template_invariance() -> dict:
    """Load DAP-4 template-invariance results from the merged M1 JSON.

    M5b was folded into M1 in 2026-04 as the ``template_dap4`` perturbation,
    using held-out-fold semantics (each material predicted by the single
    fold model that did not see it during training; M5b previously averaged
    across all 5 folds, so absolute deltas can shift but ranking is stable).

    Returns the legacy shape this script's panel plotter expects:
        {"per_material": {model_key: {mat: {v_orig_mean, v_tmpl_mean, delta}}}}
    """
    payload = _load_json(M1_PATH)
    aggregated = payload.get("aggregated", {})
    per_material: dict[str, dict[str, dict]] = {}
    for model_key, mr in aggregated.items():
        tmpl = mr.get("template_dap4")
        if tmpl is None:
            continue
        orig_block = mr.get("original", {})
        template_pm = tmpl.get("per_material", {})
        per_material[model_key] = {}
        for mat, preds in template_pm.items():
            if mat not in orig_block:
                continue
            v_orig_mean = float(orig_block[mat])
            v_tmpl_mean = float(np.mean(preds))
            per_material[model_key][mat] = {
                "v_orig_mean": v_orig_mean,
                "v_tmpl_mean": v_tmpl_mean,
                "delta": v_tmpl_mean - v_orig_mean,
            }
    return {
        "per_material": per_material,
        "evaluation": payload.get("evaluation", "held_out_5fold_pooled"),
        "fold_ids": payload.get("fold_ids", [0, 1, 2, 3, 4]),
        "models": sorted(per_material.keys()),
    }


def _plot_stability_panel(ax: plt.Axes, stab: dict) -> None:
    """Panel b: cross-fold descriptor stability strip plot.

    y = mean pairwise cosine similarity across 5 folds, per material.
    x-groups = MT-DFT / ST-pretrained / ST-scratch.
    Annotate median per group.
    """
    model_keys = ["exp7a", "exp7c", "exp7d"]
    model_labels = ["MT-DFT", "ST-\npretrained", "ST-\nscratch"]
    model_colors = [MT_COLOR, ST_COLOR, SCRATCH_COLOR]

    rng = np.random.default_rng(42)
    for i, (mk, lbl, color) in enumerate(zip(model_keys, model_labels, model_colors)):
        pm = stab.get("per_material", {}).get(mk, {})
        vals = [v["cos_mean"] for v in pm.values() if not np.isnan(v["cos_mean"])]
        if not vals:
            continue
        vals_arr = np.array(vals)
        # Jitter
        xj = i + rng.uniform(-0.18, 0.18, len(vals_arr))
        ax.scatter(xj, vals_arr, s=11, color=color, alpha=0.65, edgecolors="none", zorder=4)
        # IQR box
        q25, q50, q75 = np.percentile(vals_arr, [25, 50, 75])
        ax.fill_betweenx([q25, q75], i - 0.16, i + 0.16, color=color, alpha=0.18, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [q50, q50], color=color, lw=0.5, zorder=5)
        ax.text(i, q50 + 0.0015, f"{q50:.3f}", ha="center", va="bottom", fontsize=FIG4_FONT, color=color)

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=FIG4_FONT)
    ax.set_ylabel("Mean pairwise cosine (5 folds)", labelpad=1)
    ax.set_title("Cross-fold descriptor stability", pad=3, loc="left")
    lo = min(
        v["cos_mean"]
        for mk in model_keys
        for v in stab.get("per_material", {}).get(mk, {}).values()
        if not np.isnan(v["cos_mean"])
    )
    ax.set_ylim(max(0.0, lo - 0.02), 1.01)
    style_axes(ax, grid=True)
    ax.grid(axis="x", visible=False)


def _plot_template_invariance_panel(
    ax: plt.Axes,
    tmpl: dict,
    meta: dict[str, dict[str, object]],
) -> None:
    """Panel e: template invariance scatter.

    x = V_orig (model prediction on material's own cluster, m·s⁻¹)
    y = V_template (model prediction on DAP-4 template cluster, m·s⁻¹)
    Points are coloured and shaped by X-site family.
    MT shown as filled markers; ST-pretrained as open markers (lower alpha).
    """
    x_site_markers = {"IO4-": "o", "H4IO6-": "s", "NO3-": "^", "ClO4-": "D"}
    pm_mt = tmpl.get("per_material", {}).get("exp7a", {})
    pm_st = tmpl.get("per_material", {}).get("exp7c", {})

    all_orig = [pm_mt[m]["v_orig_mean"] for m in pm_mt]
    all_tmpl = [pm_mt[m]["v_tmpl_mean"] for m in pm_mt] + [pm_st[m]["v_tmpl_mean"] for m in pm_st]
    lo = min(all_orig + all_tmpl) - 60
    hi = max(all_orig + all_tmpl) + 60

    seen_x: set[str] = set()
    for mat in sorted(pm_mt.keys()):
        x_site = str(meta[mat]["X"])
        color = X_SITE_COLORS.get(x_site, MID_GRAY)
        marker = x_site_markers.get(x_site, "o")
        legend_label = SITE_LABELS.get(x_site, x_site) if x_site not in seen_x else "_nolegend_"
        seen_x.add(x_site)
        v_orig_mt = pm_mt[mat]["v_orig_mean"]
        v_tmpl_mt = pm_mt[mat]["v_tmpl_mean"]
        # MT: filled
        ax.scatter(v_orig_mt, v_tmpl_mt, s=20, marker=marker, color=color, edgecolors="none",
                   zorder=5, label=legend_label)
        # ST-pretrained: open
        if mat in pm_st:
            v_orig_st = pm_st[mat]["v_orig_mean"]
            v_tmpl_st = pm_st[mat]["v_tmpl_mean"]
            ax.scatter(v_orig_st, v_tmpl_st, s=20, marker=marker, facecolors="none",
                       edgecolors=color, linewidths=0.5, alpha=0.70, zorder=4, label="_nolegend_")

    ax.plot([lo, hi], [lo, hi], color=CHARCOAL, lw=0.5, ls="--", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$V_{\mathrm{det}}$ — own cluster (m$\cdot$s$^{-1}$)", labelpad=1)
    ax.set_ylabel(r"$V_{\mathrm{det}}$ — DAP-4 template (m$\cdot$s$^{-1}$)", labelpad=1)
    ax.set_title("Template invariance (DAP-4 universal cluster)", pad=3, loc="left")

    # Compute MAE for MT and ST-pretrained
    mae_mt = np.mean([abs(pm_mt[m]["delta"]) for m in pm_mt])
    mae_st = np.mean([abs(pm_st[m]["delta"]) for m in pm_st])
    ann_lines = [
        f"MT MAE = {mae_mt:.0f} m$\\cdot$s$^{{-1}}$",
        f"ST MAE = {mae_st:.0f} m$\\cdot$s$^{{-1}}$",
    ]
    ax.text(0.03, 0.97, "\n".join(ann_lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=FIG4_FONT)

    # Legend: X-site families
    ax.legend(frameon=False, loc="lower right", ncol=2,
              handletextpad=0.3, labelspacing=0.25, columnspacing=0.7, borderaxespad=0.2)
    # Model encoding legend (filled=MT, open=ST)
    mt_proxy = mpatches.Patch(facecolor=CHARCOAL, edgecolor="none", label="MT-DFT (filled)")
    st_proxy = mpatches.Patch(facecolor="none", edgecolor=CHARCOAL, linewidth=0.5, label="ST-pretrained (open)")
    ax.legend(
        handles=list(ax.get_legend_handles_labels()[0]) + [mt_proxy, st_proxy],
        labels=list(ax.get_legend_handles_labels()[1]) + ["MT-DFT", "ST-pretrained"],
        frameon=False,
        loc="lower right",
        ncol=2,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.7,
        borderaxespad=0.2,
        fontsize=FIG4_FONT,
    )
    style_axes(ax, grid=False)


ELEMENT_PALETTE = {
    "H":  "#DCDCDC",
    "C":  "#5E5E5E",
    "N":  "#2C61AF",
    "O":  "#B85060",
    "Cl": "#218E6A",
    "I":  "#7A2F8F",
    "Na": "#D9A91E",
    "K":  "#7E5BBF",
    "Rb": "#C44A1E",
    "Ag": "#7B8C99",
    "Ba": "#3F7A57",
    "?":  "#CCCCCC",
}
# Hydrogen atoms make up roughly 40 % of the atoms in this set. Their pale CPK
# colour (#DCDCDC) is too close to any neutral silver/grey to be readable as a
# scatter point in a print figure, and rendering them at full opacity overwhelms
# the heavy-atom chemistry (C, N, O, Cl) that the panels are meant to convey.
# Instead of dropping them entirely, we draw the H positions as a soft
# transparent-to-grey hexbin density underlay (zorder 1) and overlay the
# heavy-atom scatter on top (zorder 3). The underlying UMAP is still fitted on
# *all* atoms, so the layout itself is unchanged; the H layer just appears as a
# pale "shadow" of where the dabco / [H4IO6]- protons sit.
ATOMIC_UMAP_BACKGROUND_ELEMENTS: set[str] = {"H"}
H_BACKGROUND_COLOR: str = "#9A9A9A"
H_BACKGROUND_GRIDSIZE: int = 42

ATOMIC_SITE_COLORS = {
    "A": A_COLOR,
    "B": B_COLOR,
    "X": X_COLOR,
    "?": "#CCCCCC",
}


PER_XSITE_FAMILIES = (
    ("ClO4-", ("ClO4-",)),
    ("NO3-", ("NO3-",)),
    ("IO4-/H4IO6-", ("IO4-", "H4IO6-")),
)
PER_XSITE_LABELS = {
    "ClO4-": r"ClO$_4^-$",
    "NO3-": r"NO$_3^-$",
    "IO4-/H4IO6-": r"IO$_4^-$ / H$_4$IO$_6^-$",
}
PER_XSITE_FAMILY_COLOR = {
    "ClO4-": X_SITE_COLORS["ClO4-"],
    "NO3-": X_SITE_COLORS["NO3-"],
    "IO4-/H4IO6-": X_SITE_COLORS["IO4-"],
}


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 1-D array."""
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def compute_per_xsite_mae(
    materials: list[str],
    meta: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Per-X-site MAE breakdown from cluster-pooled per-material AE values.

    Returns one record per X-site family with mean MAE and 95 % bootstrap CI for
    three predictors:
      * MT-DFT (exp7a) -- multi-task DFT-baseline finetune
      * ST-pretrained (exp7c) -- single-task DPA3 finetune
      * X-site mean baseline -- predict each material as the mean Vdet of its
        X-site family (rotates each material out of its family mean to keep the
        comparison out-of-sample).
    """
    boot = _load_json(BOOTSTRAP_PATH)
    train_set = set(boot["config"]["materials"])
    materials_in = [m for m in materials if m in train_set]
    if len(materials_in) != len(materials):
        missing = sorted(set(materials) - train_set)
        raise RuntimeError(
            f"bootstrap_results.json is missing per-material AE for: {missing}"
        )

    per_mat_mt = boot["per_material_ae"]["exp7a"]
    per_mat_st = boot["per_material_ae"]["exp7c"]

    family_results: list[dict[str, object]] = []
    for family_key, x_keys in PER_XSITE_FAMILIES:
        x_set = set(x_keys)
        mats = [m for m in materials_in if str(meta[m]["X"]) in x_set]
        if not mats:
            continue
        v_obs = np.array([float(meta[m]["Vdet"]) for m in mats], dtype=float)
        ae_mt = np.array([per_mat_mt[m] for m in mats], dtype=float)
        ae_st = np.array([per_mat_st[m] for m in mats], dtype=float)
        if v_obs.size > 1:
            ae_xm = np.array(
                [
                    abs(v_obs[i] - np.mean(np.delete(v_obs, i)))
                    for i in range(v_obs.size)
                ],
                dtype=float,
            )
        else:
            ae_xm = np.array([0.0], dtype=float)

        family_results.append(
            {
                "family": family_key,
                "label": PER_XSITE_LABELS[family_key],
                "color": PER_XSITE_FAMILY_COLOR[family_key],
                "n_materials": len(mats),
                "MT": {
                    "mae": float(np.mean(ae_mt)),
                    "ci": _bootstrap_mean_ci(ae_mt),
                    "values": ae_mt,
                },
                "ST": {
                    "mae": float(np.mean(ae_st)),
                    "ci": _bootstrap_mean_ci(ae_st),
                    "values": ae_st,
                },
                "Xmean": {
                    "mae": float(np.mean(ae_xm)),
                    "ci": _bootstrap_mean_ci(ae_xm),
                    "values": ae_xm,
                },
            }
        )
    return {"families": family_results}


def _plot_per_xsite_mae(
    ax: plt.Axes,
    per_xsite: dict[str, object],
) -> None:
    """Grouped bar chart of per-X-site held-out MAE for MT, ST and family-mean."""
    families: list[dict[str, object]] = per_xsite["families"]
    n_fam = len(families)
    series = [
        ("MT-DFT", "MT", MT_COLOR, "filled"),
        ("ST-pretrained", "ST", ST_COLOR, "filled"),
        ("X-site mean", "Xmean", BASELINE_COLOR, "hatched"),
    ]
    bar_w = 0.26
    offsets = np.linspace(-bar_w, bar_w, len(series))

    rng = np.random.default_rng(42)
    handles: list[mpatches.Patch] = []
    for j, (label, key, color, style) in enumerate(series):
        means = np.array([f[key]["mae"] for f in families], dtype=float)
        ci = np.array([f[key]["ci"] for f in families], dtype=float)
        err_lo = np.clip(means - ci[:, 0], 0, None)
        err_hi = np.clip(ci[:, 1] - means, 0, None)
        xs = np.arange(n_fam) + offsets[j]
        bar_kwargs = dict(
            color=color if style == "filled" else "white",
            edgecolor=color if style == "hatched" else "none",
            hatch="////" if style == "hatched" else None,
            linewidth=0.5 if style == "hatched" else 0.0,
        )
        ax.bar(
            xs,
            means,
            width=bar_w,
            yerr=[err_lo, err_hi],
            error_kw={"lw": 0.6, "capsize": 1.8, "capthick": 0.6, "ecolor": CHARCOAL},
            zorder=3,
            **bar_kwargs,
        )
        for x_center, fam in zip(xs, families):
            vals = np.asarray(fam[key]["values"], dtype=float)
            jitter = rng.uniform(-0.07, 0.07, size=vals.size)
            ax.scatter(
                np.full_like(vals, x_center) + jitter,
                vals,
                s=4.5,
                color=CHARCOAL,
                alpha=0.55,
                edgecolors="none",
                zorder=4,
            )
        handles.append(
            mpatches.Patch(
                facecolor=color if style == "filled" else "white",
                edgecolor=color,
                hatch="////" if style == "hatched" else None,
                linewidth=0.5,
                label=label,
            )
        )

    ax.set_xticks(range(n_fam))
    ax.set_xticklabels(
        [
            f"{f['label']}\n(n={f['n_materials']})"
            for f in families
        ],
        fontsize=FIG4_FONT,
    )
    ax.set_ylabel(r"Held-out $V_{\mathrm{det}}$ MAE (m$\cdot$s$^{-1}$)", labelpad=1)
    ax.set_title(r"Per-X-site error breakdown", pad=3, loc="left")
    ax.set_ylim(bottom=0.0)
    style_axes(ax, grid=True)
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        ncol=1,
        handletextpad=0.4,
        labelspacing=0.25,
        fontsize=FIG4_FONT,
        borderaxespad=0.2,
    )


def _plot_site_probe_bars(ax: plt.Axes, probe: dict[str, object]) -> None:
    """Bar chart of site-resolved LOO Ridge probe R²."""
    probe_order = ["X-site", "B-site", "A-site", "Composition"]
    probe_colors = [X_COLOR, B_COLOR, A_COLOR, BASELINE_COLOR]
    probe_vals = [probe["site_probe"][key]["r2"] for key in probe_order]
    probe_errs = [probe["site_probe"][key].get("r2_std", 0.0) for key in probe_order]
    bars = ax.bar(
        range(len(probe_order)),
        probe_vals,
        color=probe_colors,
        width=0.72,
        yerr=probe_errs,
        error_kw={"lw": 0.6, "capsize": 2.0, "capthick": 0.6, "ecolor": CHARCOAL},
        zorder=3,
    )
    ax.axhline(0.0, color=CHARCOAL, lw=0.5, ls="--")
    ax.set_xticks(range(len(probe_order)))
    ax.set_xticklabels(["X-site", "B-site", "A-site", "Baseline"], rotation=20, ha="right")
    ax.set_ylabel("LOO $R^2$", labelpad=1)
    ax.set_ylim(-0.55, 1.10)
    ax.set_title(r"Site probe ($V_{\mathrm{det}}$, multi-task)", pad=3, loc="left")
    style_axes(ax, grid=True)
    for bar, value, err in zip(bars, probe_vals, probe_errs):
        offset = 0.04 if value >= 0 else -0.10
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + err + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=FIG4_FONT,
        )


def _annotate_element_subclusters(
    ax: plt.Axes,
    coords: np.ndarray,
    elements: np.ndarray,
    sites: np.ndarray,
    target_element: str = "N",
    *,
    label_offsets: dict[str, tuple[float, float]] | None = None,
    label_format: str = "{site}-N",
    annotate_compactness: bool = False,
    annotate_in_panel: bool = True,
) -> tuple[dict[str, dict[str, float]], float | None]:
    """Draw 1-σ covariance ellipses around (element, site) sub-clusters.

    The ellipse is the 1-σ contour of the 2-D Gaussian fit to the sub-cluster
    (so its size is proportional to the within-site spread of that element).
    Used on both Panel d (multi-task baseline) and Panel f (single-task
    pretrained); side-by-side comparison of the ellipse sizes makes the
    quantitative contrast in within-site compactness directly readable.

    Returns:
      - info dict mapping site → {within_std, n}
      - mean within-site spread across the available A/B/X sub-clusters
    """
    from matplotlib.patches import Ellipse

    if label_offsets is None:
        label_offsets = {
            "A": (0.5, -1.0),
            "B": (1.0, 0.8),
            "X": (-1.2, 0.8),
        }

    info: dict[str, dict[str, float]] = {}
    for site in ("A", "B", "X"):
        mask = (elements == target_element) & (sites == site)
        n = int(mask.sum())
        if n < 2:
            continue
        pts = coords[mask]
        mu = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)
        if not np.isfinite(cov).all():
            continue
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        # 2.4 ≈ 95 % of an isotropic 2-D Gaussian; visually compact yet generous
        scale = 2.4
        width = scale * float(np.sqrt(vals[0]))
        height = scale * float(np.sqrt(vals[1]))
        ax.add_patch(
            Ellipse(
                xy=tuple(mu),
                width=max(width, 0.5),
                height=max(height, 0.5),
                angle=angle,
                facecolor="none",
                edgecolor=ATOMIC_SITE_COLORS[site],
                linewidth=0.5,
                linestyle="-",
                alpha=0.95,
                zorder=5,
            )
        )
        dx, dy = label_offsets.get(site, (0.6, 0.6))
        ax.annotate(
            label_format.format(site=site),
            xy=tuple(mu),
            xytext=(mu[0] + dx, mu[1] + dy),
            fontsize=FIG4_FONT,
            color=ATOMIC_SITE_COLORS[site],
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=ATOMIC_SITE_COLORS[site],
                lw=0.45,
                alpha=0.8,
            ),
            zorder=6,
        )
        within_std = float(np.linalg.norm(pts.std(axis=0)))
        info[site] = {"within_std": within_std, "n": n, "mu_x": float(mu[0]), "mu_y": float(mu[1])}

    within_mean: float | None = None
    if annotate_compactness and info:
        within_mean = float(np.mean([v["within_std"] for v in info.values()]))
    if annotate_compactness and annotate_in_panel and within_mean is not None:
        ax.text(
            0.97,
            0.97,
            f"N within-site\nspread = {within_mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FIG4_FONT,
            color=CHARCOAL,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                boxstyle="round,pad=0.20",
                alpha=0.85,
            ),
        )
    return info, within_mean


def _annotate_silhouette_index(
    ax: plt.Axes,
    descriptors: np.ndarray,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    *,
    label_kind: str,
    mask: np.ndarray | None = None,
    location: tuple[float, float] = (0.97, 0.04),
) -> tuple[float | None, float | None]:
    """Annotate full-descriptor and 2D-projection silhouette coefficients.

    Silhouette compares each point's same-label cohesion with its nearest
    other-label separation. Higher is better, with an upper bound of 1.
    The primary value is computed in the full descriptor space. The 2D value
    is computed on the rendered UMAP coordinates as a visual consistency check.
    """
    x_full = descriptors
    x_2d = coords_2d
    y = labels
    if mask is not None:
        x_full = x_full[mask]
        x_2d = x_2d[mask]
        y = y[mask]
    if x_full.shape[0] < 3:
        return None, None

    # Keep only labels with at least two points so singleton labels do not
    # dominate the nearest-cluster term.
    uniq, counts = np.unique(y, return_counts=True)
    keep_labels = set(uniq[counts >= 2].tolist())
    if len(keep_labels) < 2:
        return None, None
    keep = np.isin(y, list(keep_labels))
    x_full = x_full[keep]
    x_2d = x_2d[keep]
    y = y[keep]
    if np.unique(y).size < 2 or x_full.shape[0] < 4:
        return None, None

    silhouette_256d = float(silhouette_score(x_full, y))
    silhouette_2d = float(silhouette_score(x_2d, y))
    ax.text(
        location[0],
        location[1],
        f"Sil = {silhouette_256d:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FIG4_FONT,
        color=CHARCOAL,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            boxstyle="round,pad=0.20",
            alpha=0.85,
        ),
        zorder=7,
    )
    return silhouette_256d, silhouette_2d


def _draw_background_density(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    color: str = H_BACKGROUND_COLOR,
    gridsize: int = H_BACKGROUND_GRIDSIZE,  # kept for backward compat, unused
    max_alpha: float = 0.55,
) -> None:
    """Render the supplied 2D coords as a muted scatter layer.

    Used to depict the H atoms (about 40 % of the dataset) without letting
    their pale CPK colour swamp the heavy-atom scatter. The previous
    hexbin density approach made sparse H atoms vanish entirely, so the
    reader could not tell whether the gap was a real absence of H or just
    a low-density region. Switching to a constant-alpha scatter makes
    every H atom visible as a small, low-contrast dot while keeping the
    overall layer obviously de-emphasised relative to the heavy atoms.
    """
    if coords.size == 0:
        return
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=8.0,
        facecolors="none",
        edgecolors=color,
        linewidths=0.5,
        alpha=max_alpha,
        zorder=1,
        rasterized=True,
    )


def _plot_atomic_umap_by_site(
    ax: plt.Axes,
    coords: np.ndarray,
    sites: np.ndarray,
    title: str,
    title_color: str,
    show_ylabel: bool,
    show_legend: bool,
    *,
    elements: np.ndarray | None = None,
    background_elements: set[str] | None = None,
) -> None:
    site_order = ["A", "B", "X", "?"]
    site_labels = {"A": "A-site (fuel)", "B": "B-site (modulator)", "X": "X-site (oxidizer)", "?": "Unassigned"}
    if background_elements and elements is not None:
        bg_mask = np.isin(elements, list(background_elements))
        _draw_background_density(ax, coords[bg_mask], max_alpha=0.50)
        keep = ~bg_mask
        coords = coords[keep]
        sites = sites[keep]
    for site in site_order:
        mask = sites == site
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8.0,
            color=ATOMIC_SITE_COLORS[site],
            alpha=0.65,
            edgecolors="none",
            label=site_labels[site],
            rasterized=True,
            zorder=3,
        )
    # Axis labels tell the reader the embedding is UMAP without needing
    # an explicit title; the panel-level "by site" tag is added by the
    # caller as corner text inside the axes.
    ax.set_xlabel("UMAP1", labelpad=1)
    ax.set_ylabel("UMAP2" if show_ylabel else "", labelpad=1)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            frameon=False,
            loc="lower right",
            handletextpad=0.3,
            labelspacing=0.25,
            markerscale=2.5,
            fontsize=FIG4_FONT,
        )
    style_axes(ax, grid=False)


def _annotate_site_clusters(ax: plt.Axes, coords: np.ndarray, sites: np.ndarray, elements: np.ndarray) -> None:
    """Add A/B/X labels to the heavy-atom clusters in by-site UMAP panels."""
    heavy = ~np.isin(elements, list(ATOMIC_UMAP_BACKGROUND_ELEMENTS))
    coords = coords[heavy]
    sites = sites[heavy]
    for site in ("A", "B", "X"):
        mask = sites == site
        if not mask.any():
            continue
        x_val, y_val = np.median(coords[mask], axis=0)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if y_val > y_min + 0.78 * (y_max - y_min):
            y_val -= 0.12 * (y_max - y_min)
        if x_val < x_min + 0.25 * (x_max - x_min):
            x_val += 0.10 * (x_max - x_min)
        ax.text(
            x_val,
            y_val,
            f"{site}-site",
            ha="center",
            va="center",
            fontsize=FIG4_FONT,
            color=ATOMIC_SITE_COLORS[site],
            weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.18", alpha=0.78),
            zorder=8,
        )


def _plot_atomic_umap_by_element(
    ax: plt.Axes,
    coords: np.ndarray,
    elements: np.ndarray,
    title: str,
    title_color: str,
    show_ylabel: bool,
    show_legend: bool,
    *,
    background_elements: set[str] | None = None,
) -> None:
    if background_elements:
        bg_mask = np.isin(elements, list(background_elements))
        if bg_mask.any():
            _draw_background_density(ax, coords[bg_mask])
        keep = ~bg_mask
        coords = coords[keep]
        elements = elements[keep]
    unique_elems = sorted(set(elements.tolist()), key=lambda e: (ELEMENT_PALETTE.get(e) is None, e))
    for elem in unique_elems:
        mask = elements == elem
        if not mask.any():
            continue
        color = ELEMENT_PALETTE.get(elem, "#888888")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8.0,
            color=color,
            alpha=0.65,
            edgecolors="none",
            label=elem,
            rasterized=True,
            zorder=3,
        )
    ax.set_xlabel("UMAP1", labelpad=1)
    ax.set_ylabel("UMAP2" if show_ylabel else "", labelpad=1)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if background_elements:
            handles.append(
                mlines.Line2D(
                    [], [], marker="o", linestyle="None", markersize=5.0,
                    markerfacecolor="none", markeredgecolor=H_BACKGROUND_COLOR,
                    markeredgewidth=0.5, alpha=0.85,
                )
            )
            labels.append("H density")
        ax.legend(
            handles, labels,
            frameon=False,
            loc="lower right",
            ncol=3,
            handletextpad=0.2,
            labelspacing=0.18,
            columnspacing=0.5,
            markerscale=2.5,
            fontsize=FIG4_FONT,
    )
    style_axes(ax, grid=False)


def _plot_site_dotplot(
    ax: plt.Axes,
    records: list[tuple[str, float, bool]],
    color: str,
    title: str,
) -> None:
    records = sorted(records, key=lambda item: item[1], reverse=True)
    y_pos = np.arange(len(records), dtype=float)
    values = np.array([value for _, value, _ in records], dtype=float)
    labels = [SITE_LABELS[key] for key, _, _ in records]

    for y_val, (_, x_val, is_known) in zip(y_pos, records):
        ax.scatter(
            [x_val],
            [y_val],
            s=40,
            facecolors=color if is_known else "white",
            edgecolors=color,
            linewidths=0.5,
        zorder=3,
    )
        ax.text(x_val + 18.0, y_val, f"{x_val:.0f}", va="center", ha="left", fontsize=FIG4_FONT)

    x_lo = float(np.floor((values.min() - 200.0) / 50.0) * 50.0)
    x_hi = float(np.ceil((values.max() + 130.0) / 50.0) * 50.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.5, len(records) - 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Predicted $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.set_title(title, pad=4)
    style_axes(ax, grid=True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _plot_descriptor_diagnostics_schematic(ax: plt.Axes) -> None:
    """Compact two-row probe schematic.

    Layout (in axes-relative coords, ax is in the top-left of the figure):
      [Cluster] -> [Descriptor] ----+--> [Ridge] -> rho     (Material row)
                                    |
                                    +--> [Ridge] -> Vdet    (Site row)

    All content is constrained to the left ~85% of ``ax`` so that the right
    edge of the schematic stays clear of panel d's UMAP y-axis label.
    """
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        1.06,
        "Probe protocol",
        ha="left",
        va="bottom",
        fontsize=FIG4_FONT,
        color=CHARCOAL,
    )

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        fc: str = "white",
        ec: str = "#C8C8C8",
        ls: str = "-",
        text_color: str = CHARCOAL,
    ) -> None:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.010",
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.5,
                linestyle=ls,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2.0,
            y + h / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=FIG4_FONT,
            color=text_color,
            zorder=3,
        )

    arrow_kw = dict(
        arrowstyle="-|>",
        lw=0.5,
        color=CHARCOAL,
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=2,
    )

    # Geometry.
    cluster_x, cluster_w = 0.00, 0.18
    desc_x, desc_w = 0.22, 0.24
    ridge_x, ridge_w = 0.54, 0.18
    out_x = ridge_x + ridge_w + 0.04  # 0.76
    y_top, y_bot = 0.78, 0.22
    box_h = 0.18  # vertical extent of each row's Ridge/Cluster/Descriptor box

    # Shared input pipeline (vertically centred between the two rows).
    mid_y = 0.50
    box(cluster_x, mid_y - box_h / 2, cluster_w, box_h, "Cluster", fc="#F4F4F4")
    ax.annotate(
        "",
        xy=(desc_x, mid_y),
        xytext=(cluster_x + cluster_w, mid_y),
        arrowprops=arrow_kw,
    )
    box(
        desc_x,
        mid_y - box_h / 2,
        desc_w,
        box_h,
        "Descriptor",
        fc=mcolors.to_rgba(MT_COLOR, alpha=0.12),
        ec=MT_COLOR,
    )

    # Branch arrows fan out from the descriptor's right edge.
    branch_x = desc_x + desc_w  # 0.46
    ax.annotate("", xy=(ridge_x, y_top), xytext=(branch_x, mid_y + 0.04), arrowprops=arrow_kw)
    ax.annotate("", xy=(ridge_x, y_bot), xytext=(branch_x, mid_y - 0.04), arrowprops=arrow_kw)

    # Row labels above / below each Ridge box (gray, no math symbols).
    ax.text(
        ridge_x + ridge_w / 2.0,
        y_top + box_h / 2.0 + 0.02,
        "Material",
        ha="center",
        va="bottom",
        fontsize=FIG4_FONT,
        color="#6A6A6A",
    )
    ax.text(
        ridge_x + ridge_w / 2.0,
        y_bot - box_h / 2.0 - 0.02,
        "Site",
        ha="center",
        va="top",
        fontsize=FIG4_FONT,
        color="#6A6A6A",
    )

    # Material probe (top): Ridge -> rho.
    box(ridge_x, y_top - box_h / 2, ridge_w, box_h, "Ridge", ls="--")
    ax.annotate("", xy=(out_x, y_top), xytext=(ridge_x + ridge_w, y_top), arrowprops=arrow_kw)
    ax.text(
        out_x + 0.015,
        y_top,
        r"$\rho$",
        ha="left",
        va="center",
        fontsize=FIG4_FONT,
        color=CHARCOAL,
    )

    # Site probe (bottom): Ridge -> V_det.
    box(ridge_x, y_bot - box_h / 2, ridge_w, box_h, "Ridge", ls="--")
    ax.annotate("", xy=(out_x, y_bot), xytext=(ridge_x + ridge_w, y_bot), arrowprops=arrow_kw)
    ax.text(
        out_x + 0.015,
        y_bot,
        r"$V_{\mathrm{det}}$",
        ha="left",
        va="center",
        fontsize=FIG4_FONT,
        color=CHARCOAL,
    )


# ===========================================================================
# Revised Figure 4 (2026-04-30): three-panel layout
#   Panel a -- protocol schematic + material/site linear probes
#   Panel b -- multi-task atomic UMAP (by site, by element)
#   Panel c -- perturbation schematics + DAP-4 template parity + |delta| bars
# Single-task atomic UMAP variants and the site-pooled UMAP move to SI.
# ===========================================================================

PROBE_TARGETS_A1 = ("Vdet", "density", "OB")
PROBE_TARGET_LABELS = {
    "Vdet": r"$V_{\mathrm{det}}$",
    "density": r"$\rho$",
    "OB": r"OB%",
}
SITE_PROBE_KEYS = ("z_X", "z_B", "z_A", "z_all")  # ordered by one-way eta-squared (X > B; A added)
SITE_PROBE_LABELS = {
    "z_X": r"$z_{X}$",
    "z_B": r"$z_{B}$",
    "z_A": r"$z_{A}$",
    "z_all": r"$z_{\mathrm{all}}$",
}
PERT_ORDER = (
    "template_dap4",
    "rotation", "translation",
    "stretch_bx", "stretch_ax", "stretch_ab",
    "swap_a_b", "swap_b_x", "swap_a_x",
    "scrambled_swap", "scrambled_random", "random_sphere", "sorted_line",
)
PERT_LABELS = {
    "template_dap4": "template",
    "rotation": "rotation",
    "translation": "translation",
    "stretch_bx": "B-X stretch",
    "stretch_ax": "A-X stretch",
    "stretch_ab": "A-B stretch",
    "swap_a_b": "A-B swap",
    "swap_b_x": "B-X swap",
    "swap_a_x": "A-X swap",
    "scrambled_swap": "swap",
    "scrambled_random": "random",
    "random_sphere": "sphere",
    "sorted_line": "line",
}
COMP_BASELINE_COLOR = "#A09487"

# Per-material nested-range plot (panel c2) ----------------------------
# Order from outer (most disruptive, lightest) to inner (least disruptive,
# darkest); when drawn in this order with diminishing line width, the inner
# segments visually "nest" inside the outer ones for each material.
# The local-polyhedron tests inserted in 2026-05 fill the gap between
# rigid transforms (geometry preserved) and atom-level scrambles.
PERT_NESTED_ORDER: tuple[tuple[str, str, float, float], ...] = (
    # (perturbation_id, display label, line width, alpha)
    ("sorted_line",      "Line",        8.0, 0.28),
    ("random_sphere",    "Sphere",      6.8, 0.36),
    ("scrambled_random", "Random",      5.6, 0.46),
    ("scrambled_swap",   "Atom-pair",   4.7, 0.56),
    ("swap_a_x",         "A-X swap",    3.8, 0.66),
    ("swap_b_x",         "B-X swap",    3.2, 0.72),
    ("swap_a_b",         "A-B swap",    2.7, 0.78),
    ("stretch_ab",       "A-B stretch", 2.2, 0.84),
    ("stretch_ax",       "A-X stretch", 1.9, 0.88),
    ("stretch_bx",       "B-X stretch", 1.6, 0.91),
    ("translation",      "Translation", 1.35, 0.94),
    ("rotation",         "Rotation",    1.15, 0.97),
    ("template_dap4",    "Template",    1.0, 1.00),
)
NESTED_CATEGORY_COLORS = {
    "invariant": MT_COLOR,
    "poly_break": "#B07A3C",
    "scramble": ST_COLOR,
}


def _nested_category(pid: str) -> str:
    if pid in {"template_dap4", "rotation", "translation", "stretch_bx", "stretch_ax", "stretch_ab"}:
        return "invariant"
    if pid in {"swap_a_b", "swap_b_x", "swap_a_x"}:
        return "poly_break"
    return "scramble"


def _nested_color(pid: str) -> str:
    return NESTED_CATEGORY_COLORS[_nested_category(pid)]
# X-site grouping for the nested-range plot.  Materials are reordered so
# rows belonging to the same X-site family sit together with thin separator
# lines between groups.
X_FAMILY_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ClO4-", ("ClO4-",)),
    ("NO3-",  ("NO3-",)),
    ("IO4-",  ("IO4-", "H4IO6-")),
)
X_FAMILY_DISPLAY: dict[str, str] = {
    "ClO4-": r"X = ClO$_{4}^{-}$",
    "NO3-":  r"X = NO$_{3}^{-}$",
    "IO4-":  r"X = IO$_{4}^{-}$ / H$_{4}$IO$_{6}^{-}$",
}


def load_material_probe_metrics(
    cache: dict[str, np.ndarray] | None = None,
    meta: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load m3 aggregated probe LOO R^2 for V_det, density and OB%.

    Returns
    -------
    dict
        ``{target: {"mt": {"r2", "r2_std"}, "st": {...}, "scratch": {...}, "comp": {...}}}``.
        The composition baseline is target-only and identical regardless of model.
        exp7d (scratch) is computed from cached embeddings on-the-fly.
    """
    m3 = _load_json(M3_PATH)
    ag = m3["aggregated"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for target in PROBE_TARGETS_A1:
        out[target] = {}
        for src_key, model_key in (("exp7a", "mt"), ("exp7c", "st")):
            pr = ag[src_key]["probe_results"][target]
            out[target][model_key] = {
                "r2": float(pr["r2_embedding"]),
                "r2_std": float(pr.get("r2_embedding_std_across_folds", 0.0)),
            }
        pr_comp = ag["exp7a"]["probe_results"][target]
        out[target]["comp"] = {
            "r2": float(pr_comp["r2_composition"]),
            "r2_std": float(pr_comp.get("r2_composition_std_across_folds", 0.0)),
        }

    # Compute exp7d (scratch) probes from cached embeddings if available
    if cache is not None and meta is not None and "scratch_material_emb" in cache:
        materials = cache["materials"].tolist()
        scratch_emb = cache["scratch_material_emb"]
        densities_dict = load_densities(materials)

        # Vdet probe
        y_vdet = np.array([meta[m]["Vdet"] for m in materials], dtype=float)
        y_pred = _loo_ridge_predict(scratch_emb, y_vdet)
        out["Vdet"]["scratch"] = {"r2": float(r2_score(y_vdet, y_pred)), "r2_std": 0.0}

        # density probe
        density_mats = [m for m in materials if m in densities_dict]
        if len(density_mats) >= 5:
            d_idx = np.array([materials.index(m) for m in density_mats])
            y_d = np.array([densities_dict[m] for m in density_mats])
            y_dp = _loo_ridge_predict(scratch_emb[d_idx], y_d)
            out["density"]["scratch"] = {"r2": float(r2_score(y_d, y_dp)), "r2_std": 0.0}

        # OB probe: extract ground-truth from M3 per-fold data (fold 0)
        try:
            pf0 = m3["per_fold"]["0"]["exp7a"]["probe_results"]["OB"]
            y_ob = np.array(pf0["y_true"], dtype=float)
            if len(y_ob) == len(materials):
                y_ob_pred = _loo_ridge_predict(scratch_emb, y_ob)
                out["OB"]["scratch"] = {"r2": float(r2_score(y_ob, y_ob_pred)), "r2_std": 0.0}
        except (KeyError, TypeError):
            pass

    return out


def load_site_probe_metrics() -> dict[str, dict[str, float]]:
    m4b = _load_json(M4B_PATH)
    return {
        "mt": {k: float(v) for k, v in m4b["site_r2"]["exp7a"].items()},
        "st": {k: float(v) for k, v in m4b["site_r2"]["exp7c"].items()},
    }


def load_perturbation_metrics() -> dict[str, dict[str, dict[str, float]]]:
    m1 = _load_json(M1_PATH)
    ag = m1["aggregated"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for pid in PERT_ORDER:
        out[pid] = {}
        for exp_key, model_key in (("exp7a", "mt"), ("exp7c", "st")):
            d = ag.get(exp_key, {}).get(pid)
            if d is None:
                continue
            out[pid][model_key] = {
                "delta_mean": float(d["delta_mean"]),
                "delta_std": float(d.get("delta_std", 0.0)),
            }
    return out


def load_perturb_per_material_deltas(
    exp_key: str = "exp7a",
) -> tuple[dict[str, dict[str, tuple[float, float, int]]], dict[str, float]]:
    """Per-material per-perturbation signed-delta ranges for the nested
    range plot.

    Returns
    -------
    deltas : dict[material -> dict[pert_id -> (delta_min, delta_max, n_samples)]]
        ``delta_*`` is in m·s⁻¹, signed so positive = perturbation predicted a
        higher V_det than the unperturbed cluster.
    raw : dict[material -> raw_pred] of the model's prediction on the
        unperturbed cluster (the "anchor" that all deltas are relative to).
    """
    m1 = _load_json(M1_PATH)
    ag = m1["aggregated"][exp_key]
    raw_preds = {mat: float(v) for mat, v in ag["original"].items()}
    deltas: dict[str, dict[str, tuple[float, float, int]]] = {}
    for mat, raw in raw_preds.items():
        deltas[mat] = {}
        for pid, _label, _lw, _alpha in PERT_NESTED_ORDER:
            samples = ag.get(pid, {}).get("per_material", {}).get(mat)
            if not samples:
                continue
            ds = [float(s) - raw for s in samples]
            deltas[mat][pid] = (min(ds), max(ds), len(ds))
    return deltas, raw_preds


def load_perturb_per_sample_deltas(
    exp_key: str = "exp7a",
) -> dict[str, dict[str, list[float]]]:
    """Per-material per-perturbation list of signed sample deltas.

    Returns ``deltas[material][pert_id]`` = list of
    ``V_pred(perturbed_sample) - V_pred(raw)`` values in m·s⁻¹.  Used by
    the per-perturbation distribution plot (panel c2 in Fig 4) where we
    need every individual sample, not just min/max.
    """
    m1 = _load_json(M1_PATH)
    ag = m1["aggregated"][exp_key]
    out: dict[str, dict[str, list[float]]] = {}
    for mat, raw in ag["original"].items():
        out[mat] = {}
        raw = float(raw)
        for pid, *_ in PERT_NESTED_ORDER:
            samples = ag.get(pid, {}).get("per_material", {}).get(mat)
            if not samples:
                continue
            out[mat][pid] = [float(s) - raw for s in samples]
    return out


def load_template_dap4_data() -> dict[str, dict[str, object]]:
    m1 = _load_json(M1_PATH)
    ag = m1["aggregated"]
    out: dict[str, dict[str, object]] = {}
    for exp_key, model_key in (("exp7a", "mt"), ("exp7c", "st"), ("exp7d", "scratch")):
        if exp_key not in ag:
            continue
        original = ag[exp_key]["original"]
        per_mat = ag[exp_key]["template_dap4"]["per_material"]
        materials = sorted(set(original) & set(per_mat))
        own = np.array([original[m] for m in materials], dtype=float)
        tmpl = np.array([float(np.mean(per_mat[m])) for m in materials], dtype=float)
        out[model_key] = {
            "materials": materials,
            "own": own,
            "tmpl": tmpl,
            "delta_mean": float(ag[exp_key]["template_dap4"]["delta_mean"]),
            "spearman": float(ag[exp_key]["template_dap4"]["spearman"]),
        }
    return out


# ---------------------------------------------------------------------------
# Panel a -- schematic + probe sub-panels
# ---------------------------------------------------------------------------

def _mix_color(color: str, target: str = "#FFFFFF", amount: float = 0.45) -> tuple[float, float, float]:
    """Blend ``color`` toward ``target`` for the light facets used in icons."""
    c = np.array(mcolors.to_rgb(color), dtype=float)
    t = np.array(mcolors.to_rgb(target), dtype=float)
    return tuple((1.0 - amount) * c + amount * t)


def _site_edge_color(color: str) -> tuple[float, float, float]:
    """Slightly darkened site colour for thin icon outlines."""
    return _mix_color(color, "#000000", 0.32)


def _load_hypergraph_strip() -> np.ndarray | None:
    """Rasterize and trim the standalone hypergraph PDF for use inside panel a."""
    if not HYPERGRAPH_PDF.exists():
        return None

    try:
        if (not HYPERGRAPH_PNG.exists()) or HYPERGRAPH_PNG.stat().st_mtime < HYPERGRAPH_PDF.stat().st_mtime:
            subprocess.run(
                [
                    "pdftoppm", "-png", "-singlefile", "-r", "240",
                    str(HYPERGRAPH_PDF), str(HYPERGRAPH_PNG.with_suffix("")),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        arr = np.asarray(Image.open(HYPERGRAPH_PNG).convert("RGB"))
    except Exception:
        return None

    # The source PDF carries a large white lower margin.  Trim to the inked
    # motif strip so it can be legible in the compact Figure 4 panel.
    ink = np.any(arr < 245, axis=2)
    if not np.any(ink):
        return arr
    ys, xs = np.where(ink)
    pad = 18
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def _draw_tetrahedron_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    size: float,
    *,
    facecolor: str = "#D9D9D9",
    edgecolor: str = "#111111",
    zorder: int = 5,
) -> None:
    """Draw a compact faceted tetrahedron in axes-fraction coordinates."""
    pts = np.array([
        [x, y + 0.65 * size],
        [x - 0.62 * size, y - 0.52 * size],
        [x + 0.58 * size, y - 0.42 * size],
        [x + 0.04 * size, y - 0.05 * size],
    ])
    facets = [
        (pts[[0, 1, 3]], _mix_color(facecolor, "#FFFFFF", 0.30)),
        (pts[[0, 3, 2]], _mix_color(facecolor, "#FFFFFF", 0.05)),
        (pts[[1, 2, 3]], _mix_color(facecolor, "#000000", 0.08)),
    ]
    for poly, fc in facets:
        ax.add_patch(mpatches.Polygon(
            poly, closed=True, facecolor=fc, edgecolor=edgecolor, lw=0.55,
            transform=ax.transAxes, zorder=zorder, clip_on=False,
        ))
    for i, j in ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)):
        ax.plot(
            [pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
            color=edgecolor, lw=0.55, transform=ax.transAxes,
            zorder=zorder + 0.1, clip_on=False,
        )


def _draw_ellipsoid_icon(
    ax: plt.Axes,
    x: float,
    y: float,
    size: float,
    *,
    facecolor: str = "#BFBFBF",
    edgecolor: str = "#111111",
    zorder: int = 5,
    xscale: float = 1.0,
) -> None:
    """Draw the A-site organic fragment as a tilted, unified gray ellipsoid."""
    ax.add_patch(mpatches.Ellipse(
        (x, y), width=1.45 * size * xscale, height=0.76 * size, angle=-24,
        facecolor=_mix_color(facecolor, "#FFFFFF", 0.12),
        edgecolor=edgecolor, lw=0.6,
        transform=ax.transAxes, zorder=zorder, clip_on=False,
    ))
    ax.add_patch(mpatches.Ellipse(
        (x + 0.14 * size * xscale, y + 0.08 * size), width=0.55 * size * xscale, height=0.20 * size, angle=-24,
        facecolor="white", edgecolor="none", alpha=0.45,
        transform=ax.transAxes, zorder=zorder + 0.1, clip_on=False,
    ))


def _draw_x_cuboid(
    ax: plt.Axes,
    x: float,
    y: float,
    size: float,
    color: str,
    *,
    zorder: int = 7,
    xscale: float = 1.0,
) -> None:
    """Draw the X-site cuboid motif used in the hypergraph schematic."""
    w = 1.05 * size * xscale
    h = 0.78 * size
    sx = 0.22 * size * xscale
    sy = 0.16 * size
    front = np.array([
        [x - w / 2.0, y - h / 2.0],
        [x + w / 2.0, y - h / 2.0],
        [x + w / 2.0, y + h / 2.0],
        [x - w / 2.0, y + h / 2.0],
    ])
    top = np.array([front[3], front[2], front[2] + [sx, sy], front[3] + [sx, sy]])
    side = np.array([front[2], front[1], front[1] + [sx, sy], front[2] + [sx, sy]])
    edge = _site_edge_color(color)
    for pts, fc in (
        (top, _mix_color(color, "#FFFFFF", 0.34)),
        (side, _mix_color(color, "#000000", 0.05)),
        (front, _mix_color(color, "#FFFFFF", 0.18)),
    ):
        ax.add_patch(mpatches.Polygon(
            pts, closed=True, facecolor=fc, edgecolor=edge, lw=0.25,
            transform=ax.transAxes, zorder=zorder, clip_on=False,
        ))


def _draw_atom_ball(
    ax: plt.Axes,
    x: float,
    y: float,
    radius: float,
    color: str,
    *,
    zorder: int = 8,
    xscale: float = 1.0,
) -> None:
    if abs(xscale - 1.0) < 1e-6:
        ax.add_patch(mpatches.Circle(
            (x, y), radius=radius, facecolor=_mix_color(color, "#FFFFFF", 0.18),
            edgecolor=_site_edge_color(color), lw=0.25, transform=ax.transAxes,
            zorder=zorder, clip_on=False,
        ))
        ax.add_patch(mpatches.Circle(
            (x - 0.28 * radius, y + 0.28 * radius), radius=0.32 * radius,
            facecolor="white", edgecolor="none", alpha=0.38,
            transform=ax.transAxes, zorder=zorder + 0.1, clip_on=False,
        ))
        return
    ax.add_patch(mpatches.Ellipse(
        (x, y), width=2.0 * radius * xscale, height=2.0 * radius,
        facecolor=_mix_color(color, "#FFFFFF", 0.18),
        edgecolor=_site_edge_color(color), lw=0.25, transform=ax.transAxes,
        zorder=zorder, clip_on=False,
    ))
    ax.add_patch(mpatches.Ellipse(
        (x - 0.28 * radius * xscale, y + 0.28 * radius),
        width=0.64 * radius * xscale, height=0.64 * radius,
        facecolor="white", edgecolor="none", alpha=0.38,
        transform=ax.transAxes, zorder=zorder + 0.1, clip_on=False,
    ))


CHAIR_RING_OFFSETS = np.array([
    [-0.052, 0.003],
    [-0.030, 0.030],
    [0.008, 0.025],
    [0.045, 0.000],
    [0.022, -0.030],
    [-0.020, -0.026],
])
TETRA_VERTEX_OFFSETS = np.array([
    [0.000, 0.034],
    [-0.040, -0.022],
    [0.040, -0.022],
    [0.012, 0.000],
])
TETRA_CENTER_OFFSET = np.array([0.0, -0.002])


def _draw_chair_ring(
    ax: plt.Axes,
    x: float,
    y: float,
    scale: float,
    color: str | list[str],
) -> None:
    """Draw a 6-atom chair ring.

    If ``color`` is a single string the whole ring uses one site colour
    (the original block-cluster behaviour).  If ``color`` is a length-6
    iterable, each ring atom uses the corresponding colour and the edge
    colour falls back to the per-vertex colour as well; this is what
    Atom-pair RIGHT renders to show ``scrambled_swap`` recolouring the
    27-atom cluster while leaving the chair scaffold intact.
    """
    pts = CHAIR_RING_OFFSETS * scale + np.array([x, y])
    if isinstance(color, str):
        ring_colors = [color] * len(pts)
    else:
        ring_colors = list(color)
    assert len(ring_colors) == len(pts), (
        f"chair ring expects {len(pts)} colours, got {len(ring_colors)}"
    )
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        ax.plot(
            [pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
            color=_site_edge_color(ring_colors[i]),
            lw=0.25, transform=ax.transAxes,
            zorder=7, clip_on=False,
        )
    for (px, py), c in zip(pts, ring_colors):
        _draw_atom_ball(ax, float(px), float(py), 0.015 * scale, c)


def _draw_tetrahedral_atoms(
    ax: plt.Axes,
    x: float,
    y: float,
    scale: float,
    color: str | list[str],
) -> None:
    """Draw a 5-atom tetrahedron (1 centre, 4 vertices).

    Per-atom colours follow the chair ring convention: a single string
    colours the whole motif, or a length-5 iterable colours
    [center, vert0, vert1, vert2, vert3] independently.  Used to render
    Atom-pair's atom-level colour scramble without dismantling the
    tetrahedral scaffold.
    """
    pts = TETRA_VERTEX_OFFSETS * scale + np.array([x, y])
    center = np.array([x, y]) + TETRA_CENTER_OFFSET
    if isinstance(color, str):
        atom_colors = [color] * 5
    else:
        atom_colors = list(color)
    assert len(atom_colors) == 5, (
        f"tetrahedron expects 5 colours [center, v0..v3], got {len(atom_colors)}"
    )
    center_color = atom_colors[0]
    vertex_colors = atom_colors[1:]
    for (px, py), c in zip(pts, vertex_colors):
        ax.plot(
            [center[0], px], [center[1], py],
            color=_site_edge_color(c),
            lw=0.25, transform=ax.transAxes,
            zorder=7, clip_on=False,
        )
    _draw_atom_ball(ax, float(center[0]), float(center[1]),
                    0.017 * scale, center_color, zorder=8)
    for (px, py), c in zip(pts, vertex_colors):
        _draw_atom_ball(ax, float(px), float(py), 0.014 * scale, c, zorder=9)


def _plot_panel_a_protocol_schematic(ax: plt.Axes) -> None:
    """Full-width hypergraph schematic for the top panel."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    text_color = CHARCOAL

    strip = _load_hypergraph_strip()
    if strip is not None:
        ax.imshow(
            strip, extent=(0.00, 1.00, 0.00, 1.00), transform=ax.transAxes,
            aspect="auto", interpolation="lanczos", zorder=1,
        )
    else:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.08), 0.96, 0.84, boxstyle="round,pad=0.010",
            facecolor="#F7F7F7", edgecolor="#C7C7C7", lw=0.5,
            transform=ax.transAxes, zorder=1,
        ))
        ax.text(
            0.50, 0.50, "hypergraph density representation",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FIG4_FONT, color=text_color,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def _overlay_panel_a_vector_pdf(
    out_pdf: Path,
    *,
    fig_w_mm: float,
    fig_h_mm: float,
    panel_x_mm: float,
    panel_y_mm: float,
    panel_w_mm: float,
    panel_h_mm: float,
) -> None:
    """Replace raster panel a in ``out_pdf`` with the source vector PDF.

    Pure-Python in-memory overlay via :mod:`pikepdf`.  Matplotlib only
    knows how to embed the imported hypergraph PDF as a raster preview;
    we therefore post-process the saved figure PDF: cover the panel-a
    rectangle with a white fill (hides the raster placeholder
    underneath), then place the original vector
    ``figure4-hypergraph.pdf`` as a Form XObject in the same rectangle.

    No temporary files, no LaTeX dependency.
    """
    if not HYPERGRAPH_PDF.exists() or not out_pdf.exists():
        return

    import pikepdf

    # PDF user-space unit is the "big point" (bp): 72 bp per inch,
    # 25.4 mm per inch -> 72/25.4 bp per mm.
    mm_to_bp = 72.0 / 25.4
    panel_x_bp = panel_x_mm * mm_to_bp
    panel_y_bp = panel_y_mm * mm_to_bp
    panel_w_bp = panel_w_mm * mm_to_bp
    panel_h_bp = panel_h_mm * mm_to_bp
    overlay_rect = pikepdf.Rectangle(
        panel_x_bp,
        panel_y_bp,
        panel_x_bp + panel_w_bp,
        panel_y_bp + panel_h_bp,
    )

    # Trim values match the inked bbox of the AI export at 240 dpi.  The
    # original Illustrator page leaves a large blank area beneath the
    # schematic, so we crop the source page to the inked region before
    # using it as an overlay form xobject (the form's bbox follows the
    # page's MediaBox after this crop).
    trim_left, trim_bottom, trim_right, trim_top = 0.0, 665.8339, 4.1984, 26.3934

    with pikepdf.open(HYPERGRAPH_PDF) as hyper_doc:
        hyper_page = hyper_doc.pages[0]
        media = hyper_page.mediabox
        llx, lly, urx, ury = (
            float(media[0]),
            float(media[1]),
            float(media[2]),
            float(media[3]),
        )
        cropped_box = pikepdf.Array([
            llx + trim_left,
            lly + trim_bottom,
            urx - trim_right,
            ury - trim_top,
        ])
        hyper_page.MediaBox = cropped_box
        hyper_page.CropBox = cropped_box

        with pikepdf.open(out_pdf, allow_overwriting_input=True) as base:
            base_page = base.pages[0]
            form = base.copy_foreign(hyper_page.as_form_xobject())
            # ``as_form_xobject`` keeps the original page MediaBox as the
            # Form BBox in pikepdf 10.x even after mutating the page crop.
            # If left unchanged, the actual inked strip (the upper ~150 bp)
            # is scaled as if it were an entire A4 page, so panel a becomes
            # a thin flattened line in the manuscript PDF.
            form.BBox = pikepdf.Array(cropped_box)
            form_name = pikepdf.Name("/FmPanelA")
            resources = base_page.obj.get("/Resources", pikepdf.Dictionary())
            if "/XObject" not in resources:
                resources["/XObject"] = pikepdf.Dictionary()
            resources["/XObject"][form_name] = form
            base_page.obj["/Resources"] = resources

            # Explicit matrix placement avoids pikepdf.add_overlay's default
            # aspect/box handling, which can leave the vector schematic tiny
            # when the source crop has a non-zero lower-left origin.
            bbox = [float(v) for v in form.BBox]
            src_x0, src_y0, src_x1, src_y1 = bbox
            src_w = src_x1 - src_x0
            src_h = src_y1 - src_y0
            if src_w <= 0 or src_h <= 0:
                return
            sx = panel_w_bp / src_w
            sy = panel_h_bp / src_h
            tx = panel_x_bp - sx * src_x0
            ty = panel_y_bp - sy * src_y0

            # White fill over panel a so the raster preview underneath
            # cannot bleed through anti-aliased edges of the vector art,
            # followed by exact Form XObject placement.
            panel_a_overlay = (
                f"q\n"
                f"1 g\n"
                f"{panel_x_bp:.4f} {panel_y_bp:.4f} "
                f"{panel_w_bp:.4f} {panel_h_bp:.4f} re\n"
                f"f\n"
                f"Q\n"
                f"q\n"
                f"{sx:.8f} 0 0 {sy:.8f} {tx:.4f} {ty:.4f} cm\n"
                f"{form_name} Do\n"
                f"Q\n"
            ).encode("ascii")
            base_page.contents_add(panel_a_overlay, prepend=False)
            base.save()


def _plot_panel_a1_material_probes(
    ax: plt.Axes,
    metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Horizontal bar chart for material-level LOO R² across V_det / density / OB%.

    Transposed layout: targets on y-axis, R² on x-axis.  Four series:
    MT-FT, ST-FT, ST-TFS baseline, Composition.
    """
    targets = list(PROBE_TARGETS_A1)
    y = np.arange(len(targets), dtype=float)
    height = 0.22
    series = (
        ("mt",      "MT-FT",              MT_COLOR),
        ("st",      "ST-FT",  ST_COLOR),
        ("scratch", "ST-TFS",   SCRATCH_COLOR),
        ("comp",    "Composition",             COMP_BASELINE_COLOR),
    )
    n_series = len(series)
    offsets = np.linspace(-(n_series - 1) / 2.0 * height, (n_series - 1) / 2.0 * height, n_series)
    for (model_key, label, color), offset in zip(series, offsets):
        vals = np.array([metrics[t].get(model_key, {}).get("r2", np.nan) for t in targets], dtype=float)
        errs = np.array([metrics[t].get(model_key, {}).get("r2_std", 0.0) for t in targets], dtype=float)
        valid = ~np.isnan(vals)
        bars = ax.barh(
            y[valid] + offset, vals[valid], height,
            color=color, edgecolor="none", linewidth=0.0, label=label,
            xerr=errs[valid], error_kw={"elinewidth": 0.5, "capthick": 0.5, "ecolor": CHARCOAL},
            capsize=1.5, zorder=3,
        )
        for bar, val, err in zip(bars, vals[valid], errs[valid]):
            x_pos = val + (err if np.isfinite(err) else 0.0) + 0.03
            ha = "left"
            if val < 0:
                x_pos = val - 0.03
                ha = "right"
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2f}",
                ha=ha, va="center",
                fontsize=FIG4_FONT - 1.0, color=CHARCOAL,
            )
    ax.axvline(0.0, color=CHARCOAL, lw=0.5, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([PROBE_TARGET_LABELS[t] for t in targets])
    ax.set_xlabel(r"LOO $R^{2}$", labelpad=1)
    ax.set_xlim(-0.45, 1.15)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.invert_yaxis()
    style_axes(ax, grid=False)


def _plot_panel_a2_site_probes(
    ax: plt.Axes,
    metrics: dict[str, dict[str, float]],
) -> None:
    """Bar chart for site-level LOO R^2 (V_det target).  Same colour scheme
    as a1 so it inherits the panel-a-level legend."""
    keys = list(SITE_PROBE_KEYS)
    x = np.arange(len(keys), dtype=float)
    width = 0.36
    for offset, model_key, label, color in (
        (-width / 2.0, "mt", "Multi-task", MT_COLOR),
        (+width / 2.0, "st", "Single-task", ST_COLOR),
    ):
        vals = np.array([metrics[model_key].get(k, np.nan) for k in keys], dtype=float)
        bars = ax.bar(
            x + offset, vals, width,
            color=color, edgecolor="none", linewidth=0.0, zorder=3,
        )
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            if val >= 0:
                y_pos = val + 0.04
                va = "bottom"
            else:
                y_pos = val - 0.04
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{val:.2f}",
                ha="center", va=va, rotation=90,
                fontsize=FIG4_FONT - 0.5, color=CHARCOAL,
            )
    ax.axhline(0.0, color=CHARCOAL, lw=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([SITE_PROBE_LABELS[k] for k in keys])
    ax.set_ylabel(r"site-level $V_{\mathrm{det}}$ LOO $R^{2}$", labelpad=1)
    ax.set_ylim(-0.45, 1.45)
    style_axes(ax, grid=False)


# ---------------------------------------------------------------------------
# Panel c -- perturbation schematics + template parity + |delta| bars
# ---------------------------------------------------------------------------

def _draw_mini_cluster(ax: plt.Axes, x_center: float, y_center: float, *, scale: float = 1.0,
                       jitter: np.ndarray | None = None, swap_b: bool = False) -> None:
    """Draw a compact ABX cartoon (5 markers) anchored at (x_center, y_center).

    ``jitter`` (5,2) optionally adds per-marker offsets in axes-fraction units; used
    for the ``geometry scrambled'' panel.  ``swap_b`` toggles a curved double arrow
    between the B-site marker and an X-site marker for the B-swap mini.
    """
    site_marks = [
        ("A", "o", -0.06 * scale, +0.04 * scale),
        ("A", "o", +0.06 * scale, +0.04 * scale),
        ("B", "s", 0.0, +0.01 * scale),
        ("X", "v", -0.04 * scale, -0.05 * scale),
        ("X", "v", +0.04 * scale, -0.05 * scale),
    ]
    for idx, (site, marker, dx, dy) in enumerate(site_marks):
        ox = jitter[idx, 0] if jitter is not None else 0.0
        oy = jitter[idx, 1] if jitter is not None else 0.0
        ax.scatter(
            x_center + dx + ox, y_center + dy + oy,
            s=22.0, marker=marker,
            color=ATOMIC_SITE_COLORS[site],
            edgecolors="none", linewidths=0.0,
            transform=ax.transAxes, clip_on=False, zorder=4,
        )
    if swap_b:
        ax.annotate(
            "", xy=(x_center - 0.04 * scale, y_center - 0.05 * scale),
            xytext=(x_center + 0.04 * scale, y_center - 0.05 * scale),
            xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="<->", color=CHARCOAL, lw=0.5, mutation_scale=6.0,
                connectionstyle="arc3,rad=0.45",
            ),
        )


def _draw_arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=CHARCOAL, lw=0.5, mutation_scale=6.5),
    )


def _cluster_xy(
    atoms: list[tuple[str, str, float, float, float]],
    x_center: float,
    y_center: float,
    *,
    site: str,
    side: str | None = None,
) -> tuple[float, float]:
    candidates = []
    for s, _motif, dx, dy, _msize in atoms:
        if s != site:
            continue
        if side == "left" and dx >= 0:
            continue
        if side == "right" and dx < 0:
            continue
        candidates.append((dx, dy))
    if not candidates:
        return x_center, y_center
    dx, dy = candidates[0]
    return x_center + dx, y_center + dy


def _highlight_moiety(
    ax: plt.Axes,
    xy: tuple[float, float],
    color: str,
    *,
    radius: float = 0.034,
    xscale: float = 1.0,
) -> None:
    ax.add_patch(mpatches.Ellipse(
        xy, width=2.0 * radius * xscale, height=2.0 * radius,
        fill=False, edgecolor=_site_edge_color(color),
        lw=0.35, transform=ax.transAxes, zorder=11, clip_on=False,
    ))


def _draw_exchange_arrow(
    ax: plt.Axes,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    rad: float,
    color: str,
) -> None:
    ax.annotate(
        "", xy=p1, xytext=p0, xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="<->", color=color, lw=0.45, mutation_scale=5.5,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=12,
    )


def _draw_swap_summary(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[tuple[str, str, float, float, float]],
) -> None:
    _draw_atomic_cluster(ax, x_center, y_center, atoms)
    a_xy = _cluster_xy(atoms, x_center, y_center, site="A", side="left")
    b_xy = _cluster_xy(atoms, x_center, y_center, site="B")
    x_xy = _cluster_xy(atoms, x_center, y_center, site="X", side="left")
    for xy, site in ((a_xy, "A"), (b_xy, "B"), (x_xy, "X")):
        _highlight_moiety(ax, xy, ATOMIC_SITE_COLORS[site])
    color = _nested_color("swap_a_x")
    _draw_exchange_arrow(ax, a_xy, b_xy, rad=-0.30, color=color)
    _draw_exchange_arrow(ax, b_xy, x_xy, rad=-0.35, color=color)
    _draw_exchange_arrow(ax, a_xy, x_xy, rad=0.55, color=color)


def _draw_stretch_summary(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[tuple[str, str, float, float, float]],
) -> None:
    _draw_atomic_cluster(ax, x_center, y_center, atoms)
    pairs = [
        ("A", "left", "X", "left", "stretch_ax", -0.32),
        ("B", None, "X", "right", "stretch_bx", 0.28),
        ("A", "right", "B", None, "stretch_ab", 0.35),
    ]
    for s0, side0, s1, side1, pid, rad in pairs:
        p0 = _cluster_xy(atoms, x_center, y_center, site=s0, side=side0)
        p1 = _cluster_xy(atoms, x_center, y_center, site=s1, side=side1)
        _highlight_moiety(ax, p0, ATOMIC_SITE_COLORS[s0], radius=0.030)
        _highlight_moiety(ax, p1, ATOMIC_SITE_COLORS[s1], radius=0.030)
        _draw_exchange_arrow(ax, p0, p1, rad=rad, color=_nested_color(pid))


def _draw_rigid_summary(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[tuple[str, str, float, float, float]],
) -> None:
    _draw_atomic_cluster(ax, x_center, y_center, atoms)
    a_xy = _cluster_xy(atoms, x_center, y_center, site="A", side="left")
    x_xy = _cluster_xy(atoms, x_center, y_center, site="X", side="right")
    rot_color = _nested_color("rotation")
    trans_color = _nested_color("translation")
    ax.annotate(
        "", xy=(a_xy[0] + 0.018, a_xy[1] + 0.018),
        xytext=(a_xy[0] - 0.022, a_xy[1] + 0.004),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=rot_color, lw=0.45, mutation_scale=5.5,
            connectionstyle="arc3,rad=0.75",
        ),
        zorder=12,
    )
    ax.annotate(
        "", xy=(x_xy[0] + 0.050, x_xy[1] + 0.010),
        xytext=(x_xy[0] + 0.010, x_xy[1] + 0.010),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=trans_color, lw=0.45, mutation_scale=5.5),
        zorder=12,
    )


def _block_cluster_positions(
    scale: float = 1.0, *, y_squash: float = 0.45
) -> list[tuple[str, str, float, float, float]]:
    """Block-scale DAP-4-like cluster: 2 A fragments + 1 B + 2 X fragments.

    Returns (site, motif, dx, dy, marker_size) tuples in axes-fraction units
    relative to (0, 0).  The motif names are rendered with the same A chair /
    B-X tetrahedral visual grammar as panel a.

    ``y_squash`` compresses the cluster vertically so multiple stacked rows
    fit inside the panel-c schematic axes without overlapping each other or
    the row titles.
    """
    s = scale
    q = y_squash
    atoms: list[tuple[str, str, float, float, float]] = []
    atoms.append(("B", "tetra", 0.0, 0.0, 14.0))
    for sign_x in (-1.0, +1.0):
        atoms.append(("A", "chair", sign_x * 0.075 * s, +0.080 * s * q, 14.0))
    for sign_x in (-1.0, +1.0):
        atoms.append(("X", "tetra", sign_x * 0.090 * s, -0.080 * s * q, 14.0))
    return atoms


def _atom_cluster_positions(
    scale: float = 1.0, *, y_squash: float = 0.45
) -> list[tuple[str, str, float, float, float]]:
    """Atom-level DAP-4-like cluster used for destructive scramble controls.

    The same fragment topology as ``_block_cluster_positions`` -- 1 B at the
    centre, 2 A heavy + 4 A surrogates, and 2 X heavy + 8 X surrogates -- but
    every atom is rendered as an individual sphere so the cluster looks
    "exploded" relative to the chair / tetrahedron motifs on the left side
    of each row.  The dimensions are intentionally chosen so the full
    cluster fits inside one quantitative row of panel d's right strip.
    """
    s = scale
    q = y_squash
    atoms: list[tuple[str, str, float, float, float]] = []
    atoms.append(("B", "s", 0.0, 0.0, 16.0))
    for sign_x in (-1.0, +1.0):
        cx, cy = sign_x * 0.075 * s, +0.060 * s * q
        atoms.append(("A", "o", cx, cy, 11.0))
        for theta in (0.6, 2.4):
            atoms.append((
                "A", "o",
                cx + 0.022 * s * np.cos(theta),
                cy + 0.018 * s * np.sin(theta) * q,
                4.0,
            ))
    for sign_x in (-1.0, +1.0):
        cx, cy = sign_x * 0.090 * s, -0.060 * s * q
        atoms.append(("X", "v", cx, cy, 11.0))
        for theta in (0.4, 1.7, 3.1, 4.6):
            atoms.append((
                "X", "v",
                cx + 0.026 * s * np.cos(theta),
                cy + 0.020 * s * np.sin(theta) * q,
                4.0,
            ))
    return atoms


def _draw_atomic_cluster(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[tuple[str, str, float, float, float]],
) -> None:
    for site, motif, dx, dy, msize in atoms:
        x = x_center + dx
        y = y_center + dy
        if motif == "chair":
            icon_scale = 0.34 * (msize / 14.0)
            _draw_chair_ring(ax, x, y, icon_scale, ATOMIC_SITE_COLORS[site])
        elif motif == "tetra":
            icon_scale = 0.34 * (msize / 14.0)
            _draw_tetrahedral_atoms(ax, x, y, icon_scale, ATOMIC_SITE_COLORS[site])
        else:
            # Atom-level perturbation schematics use a single ball size so
            # colour/site shuffling never creates misleading hybrid symbols
            # such as an A-coloured B-sized centre atom.
            radius = 0.0062
            _draw_atom_ball(ax, x, y, radius, ATOMIC_SITE_COLORS[site], zorder=7)


PANEL_D_ATOM = tuple[str, str, float, float, float]
PANEL_D_SOURCE_Y = 0.55
PANEL_D_SOURCE_SCALE = 2.15
PANEL_D_SOURCE_RING_DIAM_MM = 12.9
PANEL_D_ICON_SCALE = 2.15
PANEL_D_DECORATIONS = {
    "template_dap4": "template",
    "rotation": "rotation",
    "translation": "translation",
    "stretch_bx": "stretch_bx",
    "stretch_ax": "stretch_ax",
    "stretch_ab": "stretch_ab",
    "swap_a_b": "swap_a_b",
    "swap_b_x": "swap_b_x",
    "swap_a_x": "swap_a_x",
    "scrambled_swap": "recolor_perm",
    "scrambled_random": "exploded_random",
    "random_sphere": "exploded_sphere",
    "sorted_line": "exploded_line",
}
PANEL_D_ICON_YFRAC = 0.80
PANEL_D_VALUE_LABEL_YFRAC = 0.13


def _panel_axes_size_mm(ax: plt.Axes) -> tuple[float, float]:
    """Physical size of an axes in mm, used to draw undistorted atom icons."""
    fig = ax.figure
    bbox = ax.get_position()
    fig_w_mm = fig.get_figwidth() * 25.4
    fig_h_mm = fig.get_figheight() * 25.4
    return float(bbox.width * fig_w_mm), float(bbox.height * fig_h_mm)


def _mm_to_axes_delta(ax: plt.Axes, dx_mm: float, dy_mm: float) -> tuple[float, float]:
    ax_w_mm, ax_h_mm = _panel_axes_size_mm(ax)
    return float(dx_mm / ax_w_mm), float(dy_mm / ax_h_mm)


def _axes_from_mm(ax: plt.Axes, x_center: float, y_center: float, dx_mm: float, dy_mm: float) -> tuple[float, float]:
    dx, dy = _mm_to_axes_delta(ax, dx_mm, dy_mm)
    return x_center + dx, y_center + dy


def _draw_atom_mm(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    dx_mm: float,
    dy_mm: float,
    radius_mm: float,
    color: str,
    *,
    zorder: int = 8,
    alpha: float = 1.0,
) -> None:
    ax_w_mm, ax_h_mm = _panel_axes_size_mm(ax)
    x, y = _axes_from_mm(ax, x_center, y_center, dx_mm, dy_mm)
    ax.add_patch(mpatches.Ellipse(
        (x, y),
        width=2.0 * radius_mm / ax_w_mm,
        height=2.0 * radius_mm / ax_h_mm,
        facecolor=_mix_color(color, "#FFFFFF", 0.18),
        edgecolor=_site_edge_color(color),
        lw=0.25,
        alpha=alpha,
        transform=ax.transAxes,
        zorder=zorder,
        clip_on=False,
    ))
    ax.add_patch(mpatches.Ellipse(
        (x - 0.28 * radius_mm / ax_w_mm, y + 0.28 * radius_mm / ax_h_mm),
        width=0.64 * radius_mm / ax_w_mm,
        height=0.64 * radius_mm / ax_h_mm,
        facecolor="white",
        edgecolor="none",
        alpha=0.38 * alpha,
        transform=ax.transAxes,
        zorder=zorder + 0.1,
        clip_on=False,
    ))


def _abx3_atom_cluster(scale: float = 1.0) -> list[PANEL_D_ATOM]:
    """Atom-scale ABX3 schematic: 1 A fragment, 1 B fragment, 3 X fragments.

    Coordinates and radii are in local millimetres.  Fragment IDs let the
    schematic perturbations move group COMs while preserving each fragment's
    internal atom-scale geometry.
    """
    atoms: list[PANEL_D_ATOM] = []
    atom_radius_boost = 1.60

    def add(site: str, group: str, x: float, y: float, r: float) -> None:
        atoms.append((site, group, x * scale, y * scale, r * scale * atom_radius_boost))

    # A fragment: bottom chair/ring-like molecular cation.
    a_center = np.array([0.0, -1.35])
    for i, theta in enumerate(np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)):
        x = a_center[0] + 0.78 * np.cos(theta)
        y = a_center[1] + 0.42 * np.sin(theta + 0.35)
        add("A", "A", float(x), float(y), 0.16 if i % 2 == 0 else 0.13)
    add("A", "A", -0.35, -0.82, 0.11)
    add("A", "A", 0.38, -1.86, 0.11)

    # B fragment: top compact atom-scale cation cluster.
    for x, y, r in ((0.0, 2.15, 0.17), (-0.32, 1.88, 0.12), (0.34, 1.88, 0.12), (0.0, 1.62, 0.10)):
        add("B", "B", x, y, r)

    # Three X fragments: middle triangle of repeated tetra/anion-like groups.
    x_centers = {
        "X0": np.array([-1.35, 0.15]),
        "X1": np.array([1.35, 0.15]),
        "X2": np.array([0.0, 0.95]),
    }
    tetra_offsets = (
        (0.0, 0.0, 0.15),
        (-0.42, -0.24, 0.12),
        (0.42, -0.24, 0.12),
        (0.0, 0.46, 0.12),
        (0.20, 0.14, 0.10),
    )
    for group, center in x_centers.items():
        for ox, oy, r in tetra_offsets:
            add("X", group, float(center[0] + ox), float(center[1] + oy), r)
    return atoms


def _fragment_com(atoms: list[PANEL_D_ATOM], group: str) -> np.ndarray:
    coords = np.array([[dx, dy] for _site, gid, dx, dy, _r in atoms if gid == group], dtype=float)
    if coords.size == 0:
        return np.zeros(2, dtype=float)
    return coords.mean(axis=0)


def _translate_fragment(atoms: list[PANEL_D_ATOM], groups: set[str], vec: np.ndarray) -> list[PANEL_D_ATOM]:
    out: list[PANEL_D_ATOM] = []
    for site, gid, dx, dy, r in atoms:
        if gid in groups:
            out.append((site, gid, float(dx + vec[0]), float(dy + vec[1]), r))
        else:
            out.append((site, gid, dx, dy, r))
    return out


def _rotate_fragment(atoms: list[PANEL_D_ATOM], group: str, theta: float) -> list[PANEL_D_ATOM]:
    com = _fragment_com(atoms, group)
    ct, st_ = float(np.cos(theta)), float(np.sin(theta))
    out: list[PANEL_D_ATOM] = []
    for site, gid, dx, dy, r in atoms:
        if gid == group:
            rel = np.array([dx, dy], dtype=float) - com
            rot = np.array([ct * rel[0] - st_ * rel[1], st_ * rel[0] + ct * rel[1]])
            new = com + rot
            out.append((site, gid, float(new[0]), float(new[1]), r))
        else:
            out.append((site, gid, dx, dy, r))
    return out


def _swap_fragment_coms(atoms: list[PANEL_D_ATOM], group_a: str, group_b: str) -> list[PANEL_D_ATOM]:
    com_a = _fragment_com(atoms, group_a)
    com_b = _fragment_com(atoms, group_b)
    out = _translate_fragment(atoms, {group_a}, com_b - com_a)
    return _translate_fragment(out, {group_b}, com_a - com_b)


def _stretch_fragment_pair(atoms: list[PANEL_D_ATOM], anchor_group: str, target_groups: tuple[str, ...], factor: float) -> list[PANEL_D_ATOM]:
    anchor = _fragment_com(atoms, anchor_group)
    out = list(atoms)
    for group in target_groups:
        target = _fragment_com(out, group)
        out = _translate_fragment(out, {group}, (factor - 1.0) * (target - anchor))
    return out


def _perturb_abx3_atom_cluster(
    atoms: list[PANEL_D_ATOM],
    decoration: str | None,
    *,
    rng: np.random.Generator,
) -> list[PANEL_D_ATOM]:
    """Mechanism schematic perturbations on atom-scale ABX3 atoms."""
    if decoration is None:
        return list(atoms)
    if decoration == "template":
        tmpl = list(atoms)
        return _translate_fragment(
            _translate_fragment(tmpl, {"X0"}, np.array([0.45, 0.25])),
            {"X2"}, np.array([-0.30, -0.35]),
        )
    if decoration == "recolor_perm":
        sites = [a[0] for a in atoms]
        perm = rng.permutation(len(sites))
        return [
            (sites[int(perm[i])], gid, dx, dy, r)
            for i, (_site, gid, dx, dy, r) in enumerate(atoms)
        ]
    if decoration == "exploded_line":
        order = sorted(range(len(atoms)), key=lambda i: SITE_TO_Z.get(atoms[i][0], 99))
        xs = np.linspace(-2.55, 2.55, len(atoms))
        out: list[PANEL_D_ATOM] = []
        for slot, idx in enumerate(order):
            site, gid, _dx, _dy, r = atoms[idx]
            out.append((site, gid, float(xs[slot]), 0.0, r))
        return out
    if decoration == "exploded_sphere":
        theta = np.linspace(0.0, 2.0 * np.pi, len(atoms), endpoint=False)
        rng.shuffle(theta)
        return [
            (site, gid, float(2.20 * np.cos(theta[i])), float(1.20 * np.sin(theta[i])), r)
            for i, (site, gid, _dx, _dy, r) in enumerate(atoms)
        ]
    if decoration == "exploded_random":
        return [
            (site, gid, float(rng.uniform(-2.25, 2.25)), float(rng.uniform(-1.35, 1.35)), r)
            for site, gid, _dx, _dy, r in atoms
        ]
    if decoration == "rotation":
        out = list(atoms)
        for group in ("A", "B", "X0", "X1", "X2"):
            out = _rotate_fragment(out, group, float(rng.uniform(-np.pi, np.pi)))
        return out
    if decoration == "translation":
        out = list(atoms)
        for group in ("A", "B", "X0", "X1", "X2"):
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            out = _translate_fragment(out, {group}, np.array([0.45 * np.cos(phi), 0.35 * np.sin(phi)]))
        return out
    if decoration == "stretch_bx":
        return _stretch_fragment_pair(atoms, "B", ("X0", "X1", "X2"), 1.24)
    if decoration == "stretch_ax":
        return _stretch_fragment_pair(atoms, "A", ("X0", "X1", "X2"), 1.24)
    if decoration == "stretch_ab":
        return _stretch_fragment_pair(atoms, "A", ("B",), 1.34)
    if decoration == "swap_a_b":
        return _swap_fragment_coms(atoms, "A", "B")
    if decoration == "swap_b_x":
        return _swap_fragment_coms(atoms, "B", "X0")
    if decoration == "swap_a_x":
        return _swap_fragment_coms(atoms, "A", "X0")
    return list(atoms)


def _draw_group_ring_mm(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[PANEL_D_ATOM],
    groups: set[str],
    color: str,
    *,
    pad_mm: float = 0.42,
) -> None:
    coords = np.array([[dx, dy] for _site, gid, dx, dy, _r in atoms if gid in groups], dtype=float)
    if coords.size == 0:
        return
    ax_w_mm, ax_h_mm = _panel_axes_size_mm(ax)
    xmin, ymin = coords.min(axis=0) - pad_mm
    xmax, ymax = coords.max(axis=0) + pad_mm
    cx, cy = _axes_from_mm(ax, x_center, y_center, float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0))
    ax.add_patch(mpatches.Ellipse(
        (cx, cy),
        width=float((xmax - xmin) / ax_w_mm),
        height=float((ymax - ymin) / ax_h_mm),
        fill=False,
        edgecolor=_site_edge_color(color),
        lw=0.35,
        transform=ax.transAxes,
        zorder=13,
        clip_on=False,
    ))


def _draw_mm_exchange_arrow(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    p0_mm: np.ndarray,
    p1_mm: np.ndarray,
    *,
    color: str,
    rad: float,
) -> None:
    p0 = _axes_from_mm(ax, x_center, y_center, float(p0_mm[0]), float(p0_mm[1]))
    p1 = _axes_from_mm(ax, x_center, y_center, float(p1_mm[0]), float(p1_mm[1]))
    _draw_exchange_arrow(ax, p0, p1, rad=rad, color=color)


def _draw_abx3_atom_cluster(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[PANEL_D_ATOM],
    *,
    zorder: int = 8,
) -> None:
    for site, _gid, dx, dy, radius in atoms:
        _draw_atom_mm(
            ax, x_center, y_center, dx, dy, radius,
            ATOMIC_SITE_COLORS[site], zorder=zorder,
        )


def _draw_abx3_atom_decorations(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[PANEL_D_ATOM],
    decoration: str | None,
) -> None:
    if decoration in {"swap_a_x", "swap_b_x", "swap_a_b", "stretch_ax", "stretch_bx", "stretch_ab"}:
        pair_groups = {
            "swap_a_x": ("A", "X0", "swap_a_x", 0.38),
            "swap_b_x": ("B", "X0", "swap_b_x", -0.40),
            "swap_a_b": ("A", "B", "swap_a_b", 0.58),
            "stretch_ax": ("A", "X0", "stretch_ax", -0.28),
            "stretch_bx": ("B", "X0", "stretch_bx", 0.32),
            "stretch_ab": ("A", "B", "stretch_ab", -0.55),
        }
        g0, g1, pid, rad = pair_groups[decoration]
        site0 = "X" if g0.startswith("X") else g0
        site1 = "X" if g1.startswith("X") else g1
        _draw_group_ring_mm(ax, x_center, y_center, atoms, {g0}, ATOMIC_SITE_COLORS[site0])
        _draw_group_ring_mm(ax, x_center, y_center, atoms, {g1}, ATOMIC_SITE_COLORS[site1])
        _draw_mm_exchange_arrow(
            ax, x_center, y_center,
            _fragment_com(atoms, g0), _fragment_com(atoms, g1),
            color=_nested_color(pid), rad=rad,
        )
    elif decoration == "translation":
        com = _fragment_com(atoms, "X2")
        _draw_mm_exchange_arrow(
            ax, x_center, y_center,
            com + np.array([-0.45, 0.10]), com + np.array([0.35, 0.10]),
            color=_nested_color("translation"), rad=0.0,
        )
    elif decoration == "rotation":
        com = _fragment_com(atoms, "B")
        p0 = _axes_from_mm(ax, x_center, y_center, float(com[0] - 0.55), float(com[1] - 0.05))
        p1 = _axes_from_mm(ax, x_center, y_center, float(com[0] + 0.45), float(com[1] + 0.35))
        ax.annotate(
            "", xy=p1, xytext=p0, xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>", color=_nested_color("rotation"),
                lw=0.45, mutation_scale=5.2,
                connectionstyle="arc3,rad=0.75",
            ),
            zorder=13,
        )


def _draw_abx3_cluster(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    scale: float,
    *,
    decoration: str | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw one atom-scale ABX3 cluster or mechanism-decorated perturbation."""
    rng = rng or np.random.default_rng(12)
    source = _abx3_atom_cluster(scale)
    atoms = _perturb_abx3_atom_cluster(source, decoration, rng=rng)
    _draw_abx3_atom_cluster(ax, x_center, y_center, atoms)
    _draw_abx3_atom_decorations(ax, x_center, y_center, atoms, decoration)


# ---------------------------------------------------------------------------
# Atom-level explosion of a block cluster
# ---------------------------------------------------------------------------
#
# ``_block_cluster_positions`` returns 5 fragment icons (1 B tetra + 2 A
# chairs + 2 X tetras).  The atom-level perturbations (Line / Sphere /
# Random / Atom-pair) come from ``build_mechanism_perturbations.py`` and
# operate on the full set of constituent atoms; reproducing them faithfully
# in the cartoon requires expanding the 5 fragments into their atom-level
# constituents so we can shuffle / relocate them per-atom and stay true to
# the mechanism.
#
# Each entry in the exploded list is
# ``(site, role, dx, dy, radius)`` in axes-fraction units relative to the
# cluster centre; ``role`` distinguishes ring atoms / tetra centres /
# tetra vertices so the perturbed rendering can keep using the matching
# ball radius (the "different diameter balls" requirement).


def _explode_block_to_atoms(
    block_cluster: list[tuple[str, str, float, float, float]],
    icon_scale: float = 0.34,
) -> list[tuple[str, str, float, float, float]]:
    """Expand a 5-fragment block cluster into its ~27 constituent atoms.

    Mirrors what ``_draw_chair_ring`` / ``_draw_tetrahedral_atoms`` would
    render for the unperturbed cluster, but returns the data instead of
    drawing.  ``icon_scale`` matches the scaling those primitives apply.
    """
    atoms: list[tuple[str, str, float, float, float]] = []
    for site, motif, dx, dy, msize in block_cluster:
        scale = icon_scale * (msize / 14.0)
        if motif == "chair":
            for ox, oy in CHAIR_RING_OFFSETS:
                atoms.append((
                    site, "chair_ring",
                    float(dx + ox * scale),
                    float(dy + oy * scale),
                    float(0.015 * scale),
                ))
        elif motif == "tetra":
            cx = dx + TETRA_CENTER_OFFSET[0]
            cy = dy + TETRA_CENTER_OFFSET[1]
            atoms.append((
                site, "tetra_center",
                float(cx), float(cy), float(0.017 * scale),
            ))
            for ox, oy in TETRA_VERTEX_OFFSETS:
                atoms.append((
                    site, "tetra_vertex",
                    float(dx + ox * scale),
                    float(dy + oy * scale),
                    float(0.014 * scale),
                ))
        else:
            atoms.append((site, "ball", float(dx), float(dy), float(0.0062)))
    return atoms


SITE_TO_Z = {"A": 6, "B": 5, "X": 17}


def _perturb_block_atoms(
    atoms: list[tuple[str, str, float, float, float]],
    kind: str,
    *,
    rng: np.random.Generator,
    cluster_scale: float = 1.0,
) -> list[tuple[str, str, float, float, float]]:
    """Apply mechanism-faithful atom-level perturbations to an exploded cluster.

    ``kind`` is one of:

    * ``"swap"``   - mirrors ``make_scrambled_swap``: same atomic
      coordinates, site labels permuted across all atoms.  The role /
      radius is left attached to the spatial point so the chair / tetra
      scaffold stays intact while colours are scrambled.
    * ``"line"``   - mirrors ``make_sorted_line``: all atoms placed on a
      horizontal line sorted by ``Z(site)``; chair / tetra scaffold is
      lost (this is the destructive control).
    * ``"sphere"`` - mirrors ``make_random_sphere``: atoms scattered onto
      a small ellipse around the cluster centre, original site labels
      kept on each atom.
    * ``"random"`` - mirrors ``make_scrambled_random``: atoms uniformly
      jittered inside a small box, original site labels kept on each
      atom.
    """
    s = cluster_scale
    n = len(atoms)
    if kind == "swap":
        sites = [a[0] for a in atoms]
        perm = rng.permutation(n)
        return [
            (sites[perm[i]], a[1], a[2], a[3], a[4])
            for i, a in enumerate(atoms)
        ]
    if kind == "line":
        order = sorted(range(n), key=lambda i: SITE_TO_Z.get(atoms[i][0], 99))
        xs = np.linspace(-0.118 * s, 0.118 * s, n)
        out: list[tuple[str, str, float, float, float]] = [
            ("", "ball", 0.0, 0.0, 0.0)
        ] * n
        for slot, idx in enumerate(order):
            site, _role, _dx, _dy, _r = atoms[idx]
            out[slot] = (
                site, "ball",
                float(xs[slot]), 0.0,
                float(0.005),
            )
        return out
    if kind == "sphere":
        radius_x = 0.112 * s
        radius_y = 0.018 * s
        thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        rng.shuffle(thetas)
        return [
            (
                a[0], "ball",
                float(radius_x * np.cos(thetas[i])),
                float(radius_y * np.sin(thetas[i])),
                float(0.0052),
            )
            for i, a in enumerate(atoms)
        ]
    if kind == "random":
        cap_x = 0.105 * s
        cap_y = 0.020 * s
        return [
            (
                a[0], "ball",
                float(rng.uniform(-cap_x, cap_x)),
                float(rng.uniform(-cap_y, cap_y)),
                float(0.0052),
            )
            for a in atoms
        ]
    return list(atoms)


def _draw_exploded_atoms(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    atoms: list[tuple[str, str, float, float, float]],
) -> None:
    """Render a flat list of atoms as individual coloured balls.

    Used for the destructive Line / Sphere / Random rows where the
    chair / tetra scaffold has been dismantled by the perturbation; each
    atom keeps its (perturbed) site colour and explicit radius.
    """
    for site, _role, dx, dy, radius in atoms:
        _draw_atom_ball(
            ax, x_center + dx, y_center + dy,
            float(radius), ATOMIC_SITE_COLORS[site], zorder=7,
        )


def _draw_block_with_atom_colors(
    ax: plt.Axes,
    x_center: float,
    y_center: float,
    block_cluster: list[tuple[str, str, float, float, float]],
    exploded_atoms_with_perm_colors: list[tuple[str, str, float, float, float]],
    icon_scale: float = 0.34,
) -> None:
    """Render the original chair / tetra scaffold, colouring each atom
    individually from a parallel list of permuted-colour atoms.

    The two lists must share atom order with ``_explode_block_to_atoms``:
    each fragment contributes a known number of consecutive atoms
    (6 for ``chair``, 5 for ``tetra``).  Used by the Atom-pair row to
    show ``scrambled_swap``'s atom-level recolouring while keeping the
    cluster's geometric scaffold intact.
    """
    idx = 0
    for site, motif, dx, dy, msize in block_cluster:
        scale = icon_scale * (msize / 14.0)
        if motif == "chair":
            colors = [
                ATOMIC_SITE_COLORS[exploded_atoms_with_perm_colors[idx + k][0]]
                for k in range(6)
            ]
            _draw_chair_ring(ax, x_center + dx, y_center + dy, scale, colors)
            idx += 6
        elif motif == "tetra":
            colors = [
                ATOMIC_SITE_COLORS[exploded_atoms_with_perm_colors[idx + k][0]]
                for k in range(5)
            ]
            _draw_tetrahedral_atoms(ax, x_center + dx, y_center + dy, scale, colors)
            idx += 5
        else:
            site_p = exploded_atoms_with_perm_colors[idx][0]
            _draw_atom_ball(
                ax, x_center + dx, y_center + dy,
                0.0062, ATOMIC_SITE_COLORS[site_p], zorder=7,
            )
            idx += 1


def _perturb_atoms(
    atoms: list[tuple[str, str, float, float, float]],
    kind: str,
    *,
    rng: np.random.Generator,
    scale: float = 1.0,
) -> list[tuple[str, str, float, float, float]]:
    """Apply an atom-level perturbation matching the mechanism."""
    n = len(atoms)
    s = scale
    if kind == "swap":
        # Mirrors mechanism's ``make_scrambled_swap`` =
        # ``new_coord = coord[rng.permutation(N)]`` with ``symbols``
        # untouched, i.e. same 17 spatial points, site labels shuffled.
        n_atoms = len(atoms)
        sites = [a[0] for a in atoms]
        perm = rng.permutation(n_atoms)
        return [
            (sites[perm[i]], a[1], a[2], a[3], a[4])
            for i, a in enumerate(atoms)
        ]
    if kind == "random":
        # Tight sigma_y so the scrambled cloud never bleeds into the
        # adjacent quantitative row when this schematic is squeezed into
        # the 1/13-of-axes row spacing of panel d.
        sigma_x = 0.050 * s
        sigma_y = 0.012 * s
        cap_x = 0.090 * s
        cap_y = 0.020 * s
        out = []
        for site, marker, dx, dy, sz in atoms:
            ox = float(np.clip(rng.normal(scale=sigma_x), -cap_x, cap_x))
            oy = float(np.clip(rng.normal(scale=sigma_y), -cap_y, cap_y))
            out.append((site, marker, dx + ox, dy + oy, sz))
        return out
    if kind == "sphere":
        radius_x = 0.110 * s
        radius_y = 0.020 * s
        thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        rng.shuffle(thetas)
        return [
            (site, marker, float(radius_x * np.cos(thetas[i])),
             float(radius_y * np.sin(thetas[i])), sz)
            for i, (site, marker, dx, dy, sz) in enumerate(atoms)
        ]
    if kind == "line":
        xs = np.linspace(-0.115 * s, 0.115 * s, n)
        return [
            (site, marker, float(xs[i]), 0.0, sz)
            for i, (site, marker, dx, dy, sz) in enumerate(atoms)
        ]
    if kind == "rotation":
        # Rigid 2D rotation of every "molecule" (= every group of atoms
        # sharing the same site type and roughly co-located) about its own
        # COM.  In the 17-atom DAP-4 cartoon there are 5 such groups: the
        # 1-atom B centre and 2 A + 2 X local clusters.  The B atom
        # is left unchanged; each A / X group is rotated by an
        # independent random angle around its local mean (xc, yc).
        groups = [
            ("A_left",  [i for i, a in enumerate(atoms) if a[0] == "A" and a[2] < 0.0]),
            ("A_right", [i for i, a in enumerate(atoms) if a[0] == "A" and a[2] >= 0.0]),
            ("X_left",  [i for i, a in enumerate(atoms) if a[0] == "X" and a[2] < 0.0]),
            ("X_right", [i for i, a in enumerate(atoms) if a[0] == "X" and a[2] >= 0.0]),
        ]
        out = list(atoms)
        for _gname, idxs in groups:
            if not idxs:
                continue
            xc = float(np.mean([atoms[i][2] for i in idxs]))
            yc = float(np.mean([atoms[i][3] for i in idxs]))
            theta = float(rng.uniform(-np.pi, np.pi))
            ct, st_ = np.cos(theta), np.sin(theta)
            for i in idxs:
                site, marker, dx, dy, sz = atoms[i]
                rx, ry = dx - xc, dy - yc
                nx = xc + ct * rx - st_ * ry
                ny = yc + st_ * rx + ct * ry
                out[i] = (site, marker, float(nx), float(ny), sz)
        return out
    if kind == "translation":
        # Rigid translation of every group by an independent random offset
        # of magnitude ~0.025 (axes-fraction units).  Mirrors the 0.5 A
        # per-molecule translation applied in build_pems_perturbations.py.
        groups = [
            ("A_left",  [i for i, a in enumerate(atoms) if a[0] == "A" and a[2] < 0.0]),
            ("A_right", [i for i, a in enumerate(atoms) if a[0] == "A" and a[2] >= 0.0]),
            ("X_left",  [i for i, a in enumerate(atoms) if a[0] == "X" and a[2] < 0.0]),
            ("X_right", [i for i, a in enumerate(atoms) if a[0] == "X" and a[2] >= 0.0]),
            ("B",       [i for i, a in enumerate(atoms) if a[0] == "B"]),
        ]
        out = list(atoms)
        mag = 0.030 * s
        for _gname, idxs in groups:
            if not idxs:
                continue
            phi = float(rng.uniform(0, 2 * np.pi))
            tx = mag * np.cos(phi)
            ty = mag * np.sin(phi) * 0.6  # vertical squash matches cluster aspect
            for i in idxs:
                site, marker, dx, dy, sz = atoms[i]
                out[i] = (site, marker, float(dx + tx), float(dy + ty), sz)
        return out
    if kind in {"stretch_bx", "stretch_ax", "stretch_ab", "swap_a_b", "swap_b_x", "swap_a_x"}:
        groups = {
            "A_left":  [i for i, a in enumerate(atoms) if a[0] == "A" and a[2] < 0.0],
            "A_right": [i for i, a in enumerate(atoms) if a[0] == "A" and a[2] >= 0.0],
            "X_left":  [i for i, a in enumerate(atoms) if a[0] == "X" and a[2] < 0.0],
            "X_right": [i for i, a in enumerate(atoms) if a[0] == "X" and a[2] >= 0.0],
            "B":       [i for i, a in enumerate(atoms) if a[0] == "B"],
        }

        def _com(idxs: list[int]) -> np.ndarray:
            return np.array([
                float(np.mean([atoms[i][2] for i in idxs])),
                float(np.mean([atoms[i][3] for i in idxs])),
            ])

        def _translate(out_atoms, idxs: list[int], vec: np.ndarray) -> None:
            for ii in idxs:
                site, marker, dx, dy, sz = out_atoms[ii]
                out_atoms[ii] = (site, marker, float(dx + vec[0]), float(dy + vec[1]), sz)

        out = list(atoms)
        if kind.startswith("stretch_"):
            factor = 1.35
            if kind == "stretch_bx":
                anchor_pairs = [(groups["B"], groups["X_left"]), (groups["B"], groups["X_right"])]
            elif kind == "stretch_ax":
                anchor_pairs = [(groups["A_left"], groups["X_left"]), (groups["A_right"], groups["X_right"])]
            else:
                anchor_pairs = [(groups["B"], groups["A_left"]), (groups["B"], groups["A_right"])]
            for anchor_idxs, target_idxs in anchor_pairs:
                if not anchor_idxs or not target_idxs:
                    continue
                vec = (factor - 1.0) * (_com(target_idxs) - _com(anchor_idxs))
                _translate(out, target_idxs, vec)
            return out

        swap_pairs = {
            "swap_a_b": (groups["A_left"], groups["B"]),
            "swap_b_x": (groups["B"], groups["X_left"]),
            "swap_a_x": (groups["A_left"], groups["X_left"]),
        }
        idxs_a, idxs_b = swap_pairs[kind]
        if idxs_a and idxs_b:
            com_a, com_b = _com(idxs_a), _com(idxs_b)
            _translate(out, idxs_a, com_b - com_a)
            _translate(out, idxs_b, com_a - com_b)
        return out
    return list(atoms)


def _plot_panel_c_perturb_schematics(ax: plt.Axes) -> None:
    """Panel-d source schematic: a single ABX3 reference cluster."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    original_x = 0.5
    original_y = PANEL_D_SOURCE_Y

    _draw_abx3_cluster(ax, original_x, original_y, PANEL_D_SOURCE_SCALE)
    ax_w_mm, ax_h_mm = _panel_axes_size_mm(ax)
    ax.add_patch(mpatches.Ellipse(
        (original_x, original_y),
        width=PANEL_D_SOURCE_RING_DIAM_MM / ax_w_mm,
        height=PANEL_D_SOURCE_RING_DIAM_MM / ax_h_mm,
        angle=0.0, fill=False, edgecolor=_nested_color("translation"),
        lw=1.0, transform=ax.transAxes, zorder=14, clip_on=False,
    ))


def _plot_template_parity_single(
    ax: plt.Axes,
    data: dict[str, object],
    meta: dict[str, dict[str, object]],
    *,
    title: str,
    title_color: str,
    show_ylabel: bool = True,
) -> None:
    """One DAP-4 template parity panel for a single model variant."""
    lo, hi = 4500.0, 9500.0
    ticks = np.array([5000, 7000, 9000])
    ax.plot([lo, hi], [lo, hi], color=CHARCOAL, lw=0.5, ls="--", zorder=2)
    for material, x_val, y_val in zip(data["materials"], data["own"], data["tmpl"]):
        x_site = str(meta[material]["X"]) if material in meta else "ClO4-"
        mcolor = X_SITE_COLORS.get(x_site, "#6F6F6F")
        ax.scatter(
            x_val, y_val,
            s=18.0, color=mcolor,
            edgecolors="none", linewidths=0.0, zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=1.5, loc="left", color=title_color, fontsize=FIG4_FONT)
    ax.text(
        0.97, 0.04,
        f"MAE = {data['delta_mean']:.0f}\n" + r"$\rho$ = " + f"{data['spearman']:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FIG4_FONT, color=CHARCOAL,
    )
    if show_ylabel:
        ax.set_ylabel(r"DAP-4 template (m$\cdot$s$^{-1}$)", labelpad=1)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel(r"Own cluster $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)", labelpad=1)
    style_axes(ax, grid=False)


def _plot_panel_c1_template_parity_mt(
    ax: plt.Axes,
    template_data: dict[str, dict[str, object]],
    meta: dict[str, dict[str, object]],
) -> None:
    """Single MT-only DAP-4 parity plot (main figure 4c1)."""
    if "mt" not in template_data:
        ax.axis("off")
        return
    _plot_template_parity_single(
        ax, template_data["mt"], meta,
        title="Multi-task", title_color=MT_COLOR, show_ylabel=True,
    )


def _make_nested_pert_legend_handles(
    *, include_signed_error: bool = True,
) -> list:
    handles = [
        mlines.Line2D(
            [], [],
            color=_nested_color(pid), lw=lw, alpha=alpha, solid_capstyle="round",
            label=label,
        )
        for pid, label, lw, alpha in PERT_NESTED_ORDER
    ]
    if include_signed_error:
        handles.append(
            mlines.Line2D(
                [], [], linestyle="None", marker="$\\pm$",
                markersize=6, color=CHARCOAL,
                label=r"raw error (m$\cdot$s$^{-1}$)",
            )
        )
    return handles


def _plot_panel_c2_perturbation_strip(
    ax: plt.Axes,
    per_sample: dict[str, dict[str, list[float]]],
    materials: list[str],
    meta: dict[str, dict[str, object]],
) -> None:
    """Vertical KDE per perturbation with mechanism icons in unused space."""
    # PERT_NESTED_ORDER is most->least disruptive; the transposed panel reads
    # left-to-right from least to most disruptive.
    pert_left_to_right = list(reversed(PERT_NESTED_ORDER))
    n_cols = len(pert_left_to_right)
    pid_to_col = {pid: i for i, (pid, *_rest) in enumerate(pert_left_to_right)}

    # ---- collect every sample by perturbation -------------------------
    per_pert_values: dict[str, list[float]] = {pid: [] for pid in pid_to_col}
    for mat in materials:
        if mat not in per_sample:
            continue
        for pid, samples in per_sample[mat].items():
            if pid not in pid_to_col:
                continue
            per_pert_values[pid].extend(samples)

    # ---- KDE density per perturbation, mirrored left/right -------------
    # The density bodies are neutral: X-site-family color is no longer used
    # here, which keeps the empty bands available for mechanism icons.
    from scipy.stats import gaussian_kde

    half_width = 0.38
    kde_face = "#D9DEE1"
    kde_edge = "#9EA8AD"
    for col, (pid, *_rest) in enumerate(pert_left_to_right):
        vals = np.asarray(per_pert_values[pid], dtype=float)
        if vals.size < 2 or float(np.std(vals)) < 1e-6:
            y0 = float(vals.mean()) if vals.size else 0.0
            ax.hlines(y0, col - 0.25, col + 0.25,
                      colors=kde_edge, linewidth=0.5, zorder=2)
            continue
        try:
            kde = gaussian_kde(vals, bw_method=0.30)
        except Exception:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.06 * (hi - lo + 1e-9)
        ys = np.linspace(lo - pad, hi + pad, 256)
        dens = kde.evaluate(ys)
        peak = float(dens.max())
        if peak < 1e-12:
            continue
        h = dens / peak * half_width
        ax.fill_betweenx(
            ys, col - h, col + h,
            facecolor=kde_face, edgecolor=kde_edge,
            linewidth=0.55, alpha=0.72, zorder=2,
        )

    def _compact_delta_label(value: float) -> str:
        value = float(value)
        if value >= 1000.0:
            return f"{value / 1000.0:.1f}k"
        if value >= 100.0:
            return f"{value:.0f}"
        return f"{value:.1f}"

    # ---- median tick and mean-|delta| label on each KDE ----------------
    for col, (pid, *_rest) in enumerate(pert_left_to_right):
        vals = np.asarray(per_pert_values[pid], dtype=float)
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        ax.hlines(med, col - 0.30, col + 0.30,
                  colors=CHARCOAL, linewidth=0.5, zorder=4)
        mean_abs = float(np.mean(np.abs(vals)))
        ax.text(
            col, PANEL_D_VALUE_LABEL_YFRAC, _compact_delta_label(mean_abs),
            transform=ax.get_xaxis_transform(),
            ha="center", va="center",
            fontsize=FIG4_FONT - 2.2, color=CHARCOAL,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.25),
            zorder=9,
        )

    # ---- bold zero anchor ---------------------------------------------
    ax.axhline(0.0, color=CHARCOAL, lw=0.5, zorder=3)

    # ---- x-tick labels = perturbation labels ---------------------------
    xt_labels: list[str] = []
    for _pid, label, *_rest in pert_left_to_right:
        if label == "Template":
            xt_labels.append("Template\n(DAP-4)")
        else:
            xt_labels.append(label)
    ax.set_xticks(list(range(n_cols)))
    ax.set_xticklabels(xt_labels, rotation=45, ha="right", fontsize=FIG4_FONT - 1.0)
    for tick in ax.get_xticklabels():
        tick.set_color(CHARCOAL)
    ax.tick_params(axis="x", which="both", bottom=False, pad=0.5)
    ax.set_xlim(-0.6, n_cols - 0.4)

    # ---- y axis --------------------------------------------------------
    ax.set_yscale("symlog", linthresh=250.0, linscale=0.7)
    # Keep the KDE bodies in the middle band; the mechanism icons are drawn
    # in axes coordinates in the upper/lower blank bands.
    ax.set_ylim(-2.0e6, 2.0e6)
    symlog_ticks = [-8000, -2000, 0, 2000]
    ax.set_yticks(symlog_ticks)
    ax.set_yticklabels([r"$-8k$", r"$-2k$", "0", r"$2k$"])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel(r"$\Delta V_{\mathrm{det}}$ relative to unperturbed cluster (m$\cdot$s$^{-1}$)", labelpad=1)
    ax.yaxis.set_label_coords(-0.065, 0.5)
    style_axes(ax, grid=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- mechanism icons inside the data panel -------------------------
    # Place all mechanism icons in the upper blank band; mean-shift labels
    # sit in the lower blank band so neither overlaps the KDE bodies.
    rng = np.random.default_rng(11)
    x0, x1 = ax.get_xlim()
    for col, (pid, *_rest) in enumerate(pert_left_to_right):
        x_frac = (float(col) - x0) / (x1 - x0)
        if pid == "sorted_line":
            x_frac -= 0.012
        _draw_abx3_cluster(
            ax, x_frac, PANEL_D_ICON_YFRAC, PANEL_D_ICON_SCALE,
            decoration=PANEL_D_DECORATIONS[pid], rng=rng,
        )


def _plot_panel_c2_nested_range(
    ax: plt.Axes,
    deltas: dict[str, dict[str, tuple[float, float, int]]],
    raw_preds: dict[str, float],
    materials: list[str],
    meta: dict[str, dict[str, object]],
    *,
    show_signed_error: bool = True,
    draw_legend: bool = False,
) -> None:
    """Per-material nested range plot of signed Delta V_det.

    Each row is a material; the seven perturbation types (Template,
    Rotation, Translation, Atom-pair swap, Random, Sphere, Line) are
    drawn as horizontal segments centred on the same y, with line widths
    from thick (outer, lightest, most disruptive) to thin (inner,
    darkest, least disruptive).  A bold vertical line at x = 0 marks
    the unperturbed prediction anchor.  Materials are reordered by
    X-site family with thin separator lines between groups.
    """
    # ---- order materials by X-site family --------------------------------
    rows: list[tuple[str, str]] = []  # (material, family_key)
    for fam_key, fam_set in X_FAMILY_ORDER:
        for mat in materials:
            xs = str(meta.get(mat, {}).get("X", "")).strip()
            if xs in fam_set and mat in deltas:
                rows.append((mat, fam_key))
    n = len(rows)
    if n == 0:
        ax.axis("off")
        return
    # Top row = first material; matplotlib y increases upward, so reverse.
    y_pos = {mat: float(n - 1 - i) for i, (mat, _) in enumerate(rows)}

    # ---- determine x-extent ---------------------------------------------
    all_d = []
    for mat, _ in rows:
        for pid, _lab, _lw, _alpha in PERT_NESTED_ORDER:
            if pid in deltas[mat]:
                lo, hi, _ = deltas[mat][pid]
                all_d.extend([lo, hi])
    d_lo = min(all_d) if all_d else -1000.0
    d_hi = max(all_d) if all_d else 1000.0
    pad = max(150.0, 0.05 * (d_hi - d_lo))
    x_lo = d_lo - pad
    # Reserve ~12% of the right-side range for signed-error annotations so
    # large values (e.g., +777, +559) are not clipped by the axis spine.
    x_hi = d_hi + pad + 0.12 * (d_hi - d_lo + 2 * pad)
    if x_lo > -250.0:
        x_lo = -250.0

    # ---- background family bands & separators ---------------------------
    # Alternating shade across families gives every group equal visual
    # weight and matches the convention used for striped tables.  The
    # subtle warm tint is identical to the panel-a inter-bar gridlines so
    # it never competes with the data segments.
    fam_band_palette = ("#F4F1EC", "#FFFFFF")
    fam_y_min: dict[str, float] = {}
    fam_y_max: dict[str, float] = {}
    for mat, fam in rows:
        y = y_pos[mat]
        fam_y_min[fam] = min(fam_y_min.get(fam, y), y)
        fam_y_max[fam] = max(fam_y_max.get(fam, y), y)
    fam_index = 0
    fam_band_colors: dict[str, str] = {}
    for fam, _ in X_FAMILY_ORDER:
        if fam not in fam_y_min:
            continue
        fam_band_colors[fam] = fam_band_palette[fam_index % len(fam_band_palette)]
        fam_index += 1
    for fam, color in fam_band_colors.items():
        ax.axhspan(
            fam_y_min[fam] - 0.5, fam_y_max[fam] + 0.5,
            facecolor=color, edgecolor="none", zorder=0,
        )
    # Family separator lines
    fam_sequence = [fam for _, fam in rows]
    for i in range(1, n):
        if fam_sequence[i] != fam_sequence[i - 1]:
            sep_y = (y_pos[rows[i][0]] + y_pos[rows[i - 1][0]]) / 2.0
            ax.axhline(sep_y, color=MID_GRAY, lw=0.5, zorder=1)

    # ---- nested range segments ------------------------------------------
    # Draw outer perturbations first so darker inner segments lay on top.
    for pid, _label, lw, alpha in PERT_NESTED_ORDER:
        for mat, _ in rows:
            if pid not in deltas[mat]:
                continue
            d_min, d_max, n_samples = deltas[mat][pid]
            y = y_pos[mat]
            if d_max - d_min < 1.0:
                # Single-sample perturbations (template / line / rotation /
                # translation): draw a small
                # circle at the point so it stays visible against the
                # thicker outer segments.
                ax.plot(
                    [d_min], [y],
                    marker="o", markersize=max(2.0, lw * 0.8),
                    markerfacecolor=_nested_color(pid), markeredgecolor="none",
                    markeredgewidth=0.0, alpha=alpha, zorder=4 + lw / 10.0,
                )
            else:
                ax.plot(
                    [d_min, d_max], [y, y],
                    color=_nested_color(pid), lw=lw, alpha=alpha,
                    solid_capstyle="round", zorder=4 + lw / 10.0,
                )

    # ---- bold zero anchor ------------------------------------------------
    ax.axvline(0.0, color=CHARCOAL, lw=0.5, zorder=6)

    # ---- signed-error annotation on the right --------------------------
    if show_signed_error:
        x_text = x_hi - (x_hi - x_lo) * 0.012
        for mat, _ in rows:
            v_true = float(meta[mat]["Vdet"])
            err = raw_preds[mat] - v_true
            ax.text(
                x_text, y_pos[mat],
                f"{err:+.0f}",
                ha="right", va="center",
                fontsize=FIG4_FONT, color=CHARCOAL,
                zorder=7,
            )

    # ---- axes formatting -------------------------------------------------
    ax.set_xlim(x_lo, x_hi)
    # Leave 0.7 unit headroom above the first row so the family tag for
    # the topmost band can sit in a separator gap without clipping.
    ax.set_ylim(-0.7, n)
    ax.set_yticks([y_pos[m] for m, _ in rows])
    ax.set_yticklabels([display_material(m) for m, _ in rows])
    ax.tick_params(axis="y", which="both", left=False, pad=1.5)
    ax.set_xlabel(r"$\Delta V_{\mathrm{det}}$ relative to unperturbed cluster (m$\cdot$s$^{-1}$)", labelpad=1)
    style_axes(ax, grid=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- per-family X-site tag in the separator gap --------------------
    # Each tag sits in the gap above its family band at the left edge of
    # the data area on a transparent background; segments rarely reach
    # this short empty stretch above each band, so no bbox mask is needed.
    tag_x = x_lo + (x_hi - x_lo) * 0.007
    for fam, _ in X_FAMILY_ORDER:
        if fam not in fam_y_min:
            continue
        y_top = fam_y_max[fam]
        ax.text(
            tag_x, y_top + 0.55,
            X_FAMILY_DISPLAY[fam],
            ha="left", va="center",
            fontsize=FIG4_FONT, color=CHARCOAL,
            zorder=10,
        )

    # ---- optional in-axes legend (used by SI variant) -------------------
    if draw_legend:
        # Keep the perturbation legend compact; the 2026-05 ladder has
        # 13 perturbations plus the optional signed-error marker.
        ax.legend(
            handles=_make_nested_pert_legend_handles(include_signed_error=show_signed_error),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.005),
            ncol=7,
            frameon=False, fontsize=FIG4_FONT - 1.0,
            handlelength=1.4, handletextpad=0.30, columnspacing=0.8,
            borderaxespad=0.0,
        )


def plot(materials: list[str]) -> None:
    cache = _load_cache(materials)
    meta = load_metadata(materials)

    material_probes = load_material_probe_metrics(cache=cache, meta=meta)
    site_probes = load_site_probe_metrics()
    perturb_metrics = load_perturbation_metrics()
    template_data = load_template_dap4_data()
    perturb_deltas, perturb_raw = load_perturb_per_material_deltas("exp7a")
    perturb_per_sample = load_perturb_per_sample_deltas("exp7a")

    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    # Figure dimensions: 183 mm wide (Nature double-column).  The height keeps
    # panels a-c at their physical sizes while compressing panel d to roughly
    # half of its previous vertical footprint.  Panel-d icons are drawn in
    # physical mm units below, so reducing the axes height does not squash them.
    fig_w_mm, fig_h_mm = 183.0, 200.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))

    def rect_mm(x: float, y: float, w: float, h: float) -> list[float]:
        return [x / fig_w_mm, y / fig_h_mm, w / fig_w_mm, h / fig_h_mm]

    # ---- Layout (mm) -----------------------------------------------------
    # Row 0: panel a full-width hypergraph schematic.
    # Row 1: b (left, material probes) | c (right, 3×2 UMAP grid)
    # Row 2: panel d full width (perturbation schematics + strip).
    margin_x = 7.0
    margin_top = 5.0
    # Bottom margin must hold panel-d's right-strip xlabel
    # (``$\Delta V_{\rm det}$ relative to unperturbed cluster (m/s)``).  We
    # use 12 mm (was 7) to keep the label's descender ~1 mm above the page
    # edge and prevent the previous truncation.
    margin_bottom = 12.0
    top_legend_strip = 12.0     # 2-row legend (by site / by element) + headroom
    row_gap = 6.0               # compact safety band between row 1 and row 2
    panel_label_strip = 4.0

    top_panel_h = 38.0
    top_panel_y = fig_h_mm - margin_top - panel_label_strip - top_panel_h
    top_to_row1_gap = 7.0

    row1_h = 64.0  # fits 3 UMAP rows while leaving room for the full-width panel a
    row1_y = top_panel_y - top_to_row1_gap - top_legend_strip - row1_h
    legend_y_band = row1_y + row1_h  # bottom of legend band = top of row 1
    row2_y = margin_bottom
    row2_h = row1_y - row_gap - margin_bottom

    # ---- Panel a (full-width top row): hypergraph density representation ----
    ax_a_schematic = fig.add_axes(rect_mm(margin_x, top_panel_y, fig_w_mm - 2 * margin_x, top_panel_h))

    # ---- Panel b (left ~33% of row 1): single horizontal material-probe bar chart.
    #
    # Layout discipline: row 1's BOTTOM-MOST rendered ink (panel a's
    # LOO R^2 xlabel and panel b's bottom-row UMAP tick labels) must end
    # above ``row1_y`` so it never enters the inter-row safety band that
    # _qa.validate_panel_boundary verifies after rendering. We lift each
    # axes box by an explicit pad in mm to make this hold.
    panel_a_frac = 0.32
    panel_a_w = (fig_w_mm - 2 * margin_x) * panel_a_frac
    bc_gap = 7.0           # blank QA-enforced gap between panels b and c
    umap_label_gutter = 7.0  # model-row labels live here, not in the blank gap
    panel_b_x = margin_x + panel_a_w + bc_gap + umap_label_gutter
    panel_b_w = fig_w_mm - 2 * margin_x - panel_a_w - bc_gap - umap_label_gutter

    # Panel a x-axis label (LOO R^2) plus its tick labels sit ~7 mm below
    # the bar chart's bottom spine. We lift the bar chart by that amount
    # so the label ends right at ``row1_y``.
    xlabel_pad_a = 7.0
    probes_y = row1_y + xlabel_pad_a
    probes_h = row1_h - xlabel_pad_a

    # Panel b's left-aligned y-tick labels (``$V_{\rm det}$``, ``rho``,
    # ``OB%``) sit OUTSIDE the bar-chart spine.  At the page-edge margin of
    # 7 mm there is not enough room for ``OB%``, so we reserve a 6 mm
    # gutter inside the panel for the labels by shifting the bar chart
    # right.  The chart width loses the same 6 mm but remains wide enough
    # for the bars and the LOO R^2 ticks.
    panel_b_ylabel_gutter = 6.0
    ax_a1 = fig.add_axes(
        rect_mm(
            margin_x + panel_b_ylabel_gutter,
            probes_y,
            panel_a_w - panel_b_ylabel_gutter,
            probes_h,
        )
    )

    # ---- Panel c (right ~66% of row 1): 3 rows × 2 columns UMAP grid ----
    # Rows: MT-FT | ST-FT | ST-TFS baseline
    # Cols: by site | by element
    umap_row_gap = 3.0
    umap_col_gap = 4.0
    n_umap_rows = 3
    n_umap_cols = 2
    umap_tick_pad = 4.0  # bottom row's tick labels live in this strip
    effective_row1_h_b = row1_h - umap_tick_pad
    umap_h = (effective_row1_h_b - (n_umap_rows - 1) * umap_row_gap) / n_umap_rows
    umap_w = (panel_b_w - (n_umap_cols - 1) * umap_col_gap) / n_umap_cols

    umap_axes = []  # list of (row, col, ax)
    for row_i in range(n_umap_rows):
        for col_i in range(n_umap_cols):
            ux = panel_b_x + col_i * (umap_w + umap_col_gap)
            # rows from top to bottom; bottom row's spine lands at
            # ``row1_y + umap_tick_pad`` so its tick labels stay in row 1.
            uy = row1_y + row1_h - (row_i + 1) * umap_h - row_i * umap_row_gap
            ax = fig.add_axes(rect_mm(ux, uy, umap_w, umap_h))
            umap_axes.append((row_i, col_i, ax))

    # ---- Panel d (row 2 full width): transposed perturbation layout ---------
    d_ylabel_gutter_mm = 11.5
    panel_c_x = margin_x + d_ylabel_gutter_mm
    panel_c_w_full = fig_w_mm - margin_x - panel_c_x
    d_top_h = 17.0
    d_gap = 1.5
    c2_y = row2_y
    c2_h = row2_h - d_top_h - d_gap
    d_top_y = c2_y + c2_h + d_gap

    ax_c_schematic = fig.add_axes(rect_mm(panel_c_x, d_top_y, panel_c_w_full, d_top_h))
    ax_c2 = fig.add_axes(rect_mm(panel_c_x, c2_y, panel_c_w_full, c2_h))

    # ---- Render panel a ---------------------------------------------------
    _plot_panel_a_protocol_schematic(ax_a_schematic)
    _plot_panel_a1_material_probes(ax_a1, material_probes)

    # Panel a legend (model series) lives in the top legend strip together
    # with panel b's by-site / by-element rows, so we build it as part of
    # the multi-row legend block below.
    panel_a_legend_handles = [
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=5.0, color=MT_COLOR, label="Multi-task"),
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=5.0, color=ST_COLOR, label="ST-pretrained"),
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=5.0, color=SCRATCH_COLOR, label="From-scratch"),
        mlines.Line2D([], [], marker="s", linestyle="None", markersize=5.0, color=COMP_BASELINE_COLOR, label="Composition"),
    ]

    # ---- Render panel b (3×2 UMAP grid) -----------------------------------
    model_configs = [
        ("mt", "Multi-task", MT_COLOR, cache["mt_atomic_umap"], cache["mt_atomic_emb"],
         cache["mt_atom_sites"], cache["mt_atom_elements"]),
        ("st", "ST-pretrained", ST_COLOR, cache["st_atomic_umap"], cache["st_atomic_emb"],
         cache["st_atom_sites"], cache["st_atom_elements"]),
        ("scratch", "From-scratch", SCRATCH_COLOR, cache["scratch_atomic_umap"], cache["scratch_atomic_emb"],
         cache["scratch_atom_sites"], cache["scratch_atom_elements"]),
    ]

    sil_results = {}
    for row_i, (model_key, model_label, model_color, atomic_umap, atomic_emb,
                atom_sites, atom_elems) in enumerate(model_configs):
        heavy_mask = ~np.isin(atom_elems, list(ATOMIC_UMAP_BACKGROUND_ELEMENTS))

        # Column 0: by site
        ax_site = [ax for (r, c, ax) in umap_axes if r == row_i and c == 0][0]
        _plot_atomic_umap_by_site(
            ax_site, atomic_umap, atom_sites,
            title="", title_color=model_color,
            show_ylabel=(row_i == 1), show_legend=False,
            elements=atom_elems, background_elements=ATOMIC_UMAP_BACKGROUND_ELEMENTS,
        )
        # Row label on left side of by-site panel (use figure coords for stability)
        ax_site.text(
            -0.11, 0.5, model_label,
            transform=ax_site.transAxes, ha="right", va="center",
            fontsize=FIG4_FONT - 0.5, color=model_color, rotation=90,
            fontweight="bold",
        )
        ax_site.tick_params(axis="y", pad=1.0)
        sil_site, _ = _annotate_silhouette_index(
            ax_site, atomic_emb, atomic_umap, atom_sites, label_kind="site", mask=heavy_mask,
        )
        sil_results[f"{model_key}_site"] = sil_site

        # Column 1: by element
        ax_elem = [ax for (r, c, ax) in umap_axes if r == row_i and c == 1][0]
        _plot_atomic_umap_by_element(
            ax_elem, atomic_umap, atom_elems,
            title="", title_color=model_color,
            show_ylabel=False, show_legend=False,
            background_elements=ATOMIC_UMAP_BACKGROUND_ELEMENTS,
        )
        sil_elem, _ = _annotate_silhouette_index(
            ax_elem, atomic_emb, atomic_umap, atom_elems, label_kind="element", mask=heavy_mask,
        )
        sil_results[f"{model_key}_element"] = sil_elem

    # Spines tidy-up for UMAP panels; suppress axis labels to save space
    for _, _, ax in umap_axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=FIG4_FONT - 2.0)

    # ---- Two-row legend block above row 1 -----------------------------------
    # Row 0 (top):    by site:  ● A-site  ● B-site  ● X-site
    # Row 1 (bottom): by element:  ● Ag ● C ● Cl ◯ H ● I ● K ● N ● Na ● O ● Rb
    # The "by site" / "by element" headers are part of the legend block, which
    # frees panel b's first-row UMAPs from having an inline column title that
    # used to sit ~1 line below the legend (the user flagged that as "legend
    # glued to title"). The legend block now sits with a clear ~3 mm of
    # whitespace above the row-1 panel boundary.
    site_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=4.0,
                      color=A_COLOR, label="A-site"),
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=4.0,
                      color=B_COLOR, label="B-site"),
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=4.0,
                      color=X_COLOR, label="X-site"),
    ]
    element_handles = []
    for elem in ("Ag", "C", "Cl", "H", "I", "K", "N", "Na", "O", "Rb"):
        if elem == "H":
            element_handles.append(mlines.Line2D(
                [], [], marker="o", linestyle="None", markersize=4.0,
                markerfacecolor="none", markeredgecolor=H_BACKGROUND_COLOR,
                markeredgewidth=0.5, label="H",
            ))
        else:
            element_handles.append(mlines.Line2D(
                [], [], marker="o", linestyle="None", markersize=4.0,
                color=ELEMENT_PALETTE[elem], label=elem,
            ))

    panel_a_center_x_frac = (margin_x + panel_a_w / 2.0) / fig_w_mm
    panel_b_center_x_frac = (panel_b_x + panel_b_w / 2.0) / fig_w_mm

    # Vertical bands within the legend strip (mm above row 1 top):
    #     bottom row (by element):  legend_y_band + ~3 mm
    #     top row    (by site /     legend_y_band + ~9 mm
    #                models a-key):
    legend_row_top_y_frac = (legend_y_band + 13.0) / fig_h_mm
    legend_row_bot_y_frac = (legend_y_band + 7.2) / fig_h_mm

    # Panel-a model-key legend (top row, centred over panel a)
    fig.legend(
        handles=panel_a_legend_handles,
        loc="upper center",
        bbox_to_anchor=(panel_a_center_x_frac, legend_row_top_y_frac),
        frameon=False, fontsize=FIG4_FONT - 0.5,
        ncol=2, handlelength=1.0, handletextpad=0.20, columnspacing=0.8,
        borderaxespad=0.0,
    )

    # Panel-b "by site" row (top), with category header at the row's
    # left edge so the swatch row reads as a labelled legend line.
    site_header_x_frac = (panel_b_x + 8.0) / fig_w_mm
    fig.text(
        site_header_x_frac, legend_row_top_y_frac,
        "by site:",
        ha="left", va="top",
        fontsize=FIG4_FONT - 0.5, color=CHARCOAL, fontstyle="italic",
    )
    fig.legend(
        handles=site_handles,
        loc="upper left",
        bbox_to_anchor=(
            (panel_b_x + 22.0) / fig_w_mm,
            legend_row_top_y_frac,
        ),
        frameon=False, fontsize=FIG4_FONT - 0.5,
        ncol=3, handlelength=1.0, handletextpad=0.20, columnspacing=1.2,
        borderaxespad=0.0,
    )

    # Panel-b "by element" row (bottom), with category header at the
    # row's left edge.
    elem_header_x_frac = (panel_b_x + 8.0) / fig_w_mm
    fig.text(
        elem_header_x_frac, legend_row_bot_y_frac,
        "by element:",
        ha="left", va="top",
        fontsize=FIG4_FONT - 0.5, color=CHARCOAL, fontstyle="italic",
    )
    fig.legend(
        handles=element_handles,
        loc="upper left",
        bbox_to_anchor=(
            (panel_b_x + 26.0) / fig_w_mm,
            legend_row_bot_y_frac,
        ),
        frameon=False, fontsize=FIG4_FONT - 0.5,
        ncol=10, handlelength=1.0, handletextpad=0.20, columnspacing=0.6,
        borderaxespad=0.0,
    )

    # ---- Render panel c ---------------------------------------------------
    _plot_panel_c_perturb_schematics(ax_c_schematic)
    _plot_panel_c2_perturbation_strip(ax_c2, perturb_per_sample, materials, meta)

    # Panel-letter labels.  The hypergraph schematic is the new full-width
    # panel a; the former probe, UMAP, and perturbation panels become b/c/d.
    panel_label_kwargs = dict(fontsize=9.0, fontweight="bold", color=CHARCOAL,
                              ha="left", va="top")
    panel_label_x_left = 2.5  # mm from figure left edge
    label_y_a = (top_panel_y + top_panel_h + panel_label_strip - 0.5) / fig_h_mm
    label_y_row1 = (legend_y_band + top_legend_strip - 0.5) / fig_h_mm
    # Panel d letter sits LOW in the row gap (close to top of row 2), so
    # row-1 axis labels (e.g. "LOO R^2") cannot collide with it. The
    # _qa.validate_panel_boundary call below enforces this programmatically.
    panel_d_label_y_mm = row2_y + row2_h + 2.0
    panel_d_label_y_frac = panel_d_label_y_mm / fig_h_mm
    fig.text(panel_label_x_left / fig_w_mm, label_y_a,
             "a", **panel_label_kwargs)
    fig.text(panel_label_x_left / fig_w_mm, label_y_row1,
             "b", **panel_label_kwargs)
    fig.text(panel_b_x / fig_w_mm, label_y_row1,
             "c", **panel_label_kwargs)
    fig.text(panel_label_x_left / fig_w_mm, panel_d_label_y_frac,
             "d", **panel_label_kwargs)

    print(
        f"[Fig4 silhouette 256D] "
        + " | ".join(f"{k}={v:.2f}" for k, v in sil_results.items())
    )

    # ---- Self-check (FIGURE_QA.md rule 1.6) -------------------------------
    # Render the figure to a numpy buffer so we can inspect the inter-row
    # safety band. We intentionally do this BEFORE save_png_pdf so the
    # build fails loudly if any row-1 element has crept into the row gap
    # (the failure mode the user just flagged: panel a's xlabel crowding
    # the panel-c letter).
    fig.canvas.draw()
    fig_image = np.asarray(fig.canvas.buffer_rgba()).copy()[:, :, :3]
    fig_h_px, fig_w_px = fig_image.shape[:2]

    def _mm_to_px_x(x_mm: float) -> float:
        return float(x_mm * fig_w_px / fig_w_mm)

    def _mm_to_px_y_top(y_mm: float) -> float:
        # mm grows bottom-up; pixel y grows top-down.
        return float((fig_h_mm - y_mm) * fig_h_px / fig_h_mm)

    # First check the safety band between the full-width hypergraph panel a
    # and the legend strip that belongs to row 1.
    top_gap_top_mm = top_panel_y - 1.5
    top_gap_bot_mm = legend_y_band + top_legend_strip + 1.5

    rect_top_gap = (
        _mm_to_px_x(0.0),
        _mm_to_px_y_top(top_gap_top_mm),
        _mm_to_px_x(fig_w_mm),
        _mm_to_px_y_top(top_gap_bot_mm),
    )

    qa_errors: list = []
    qa_errors += _qa.validate_panel_boundary(
        panel="figure4/a-to-bc-gap",
        figure_image=fig_image,
        rect_px=rect_top_gap,
    )

    # Column-boundary QA for row 1.  This catches the specific failure mode
    # where panel-c row labels or panel-b tick labels visually occupy the
    # inter-panel gap between the material-probe bars and the UMAP grid.
    bc_gap_left_mm = margin_x + panel_a_w + 0.8
    bc_gap_right_mm = margin_x + panel_a_w + bc_gap - 0.8
    if bc_gap_right_mm > bc_gap_left_mm:
        rect_bc_column_gap = (
            _mm_to_px_x(bc_gap_left_mm),
            _mm_to_px_y_top(row1_y + row1_h - 0.5),
            _mm_to_px_x(bc_gap_right_mm),
            _mm_to_px_y_top(row1_y + 0.5),
        )
        qa_errors += _qa.validate_panel_boundary(
            panel="figure4/b-c-column-gap",
            figure_image=fig_image,
            rect_px=rect_bc_column_gap,
        )

    # The lower row-gap safety band runs from the TOP of row 2 (= row2_y + row2_h)
    # up to the BOTTOM of row 1 (= row1_y), excluding the narrow vertical
    # column reserved for the panel-d letter. Anything else inside the band
    # MUST be background -- if we see ink there, a row-1 axis label has
    # invaded the gap.
    band_top_mm = row1_y - 1.5            # 1.5 mm above row 1's bottom
    band_bot_mm = row2_y + row2_h + 1.0   # 1.0 mm above the top of row 2 axes
    d_letter_left_mm = panel_label_x_left - 0.5
    d_letter_right_mm = panel_label_x_left + 4.0  # 4 mm column hosts the "d" glyph

    # Two strips: one to the LEFT of the d letter (covers the remaining
    # left edge), one to the RIGHT (covers panels a and b widths).
    rect_left_strip = (
        _mm_to_px_x(0.0),
        _mm_to_px_y_top(band_top_mm),
        _mm_to_px_x(d_letter_left_mm),
        _mm_to_px_y_top(band_bot_mm),
    )
    rect_right_strip = (
        _mm_to_px_x(d_letter_right_mm),
        _mm_to_px_y_top(band_top_mm),
        _mm_to_px_x(fig_w_mm),
        _mm_to_px_y_top(band_bot_mm),
    )

    qa_errors += _qa.validate_panel_boundary(
        panel="figure4/row-gap-left",
        figure_image=fig_image,
        rect_px=rect_left_strip,
    )
    qa_errors += _qa.validate_panel_boundary(
        panel="figure4/row-gap-right",
        figure_image=fig_image,
        rect_px=rect_right_strip,
    )

    # ---- Panel-d transposed-layout QA ------------------------------------
    n_cols = len(PERT_NESTED_ORDER)

    def _top_xfrac_to_mm(xa: float) -> float:
        return panel_c_x + xa * panel_c_w_full

    def _top_yfrac_to_mm(ya: float) -> float:
        return d_top_y + ya * d_top_h

    def _rect_crop_mm(x0_mm: float, y0_mm: float, x1_mm: float, y1_mm: float) -> np.ndarray:
        x0 = int(round(_mm_to_px_x(max(min(x0_mm, x1_mm), 0.0))))
        x1 = int(round(_mm_to_px_x(min(max(x0_mm, x1_mm), fig_w_mm))))
        y_top = int(round(_mm_to_px_y_top(max(y0_mm, y1_mm))))
        y_bot = int(round(_mm_to_px_y_top(min(y0_mm, y1_mm))))
        return fig_image[y_top:y_bot, x0:x1, :3]

    def _ink_mask(img: np.ndarray, thresh: int = 200) -> np.ndarray:
        return (img < thresh).any(axis=2)

    def _dilate(mask: np.ndarray, k: int = 1) -> np.ndarray:
        if k <= 0:
            return mask
        m = mask.copy()
        for _ in range(k):
            up = np.zeros_like(m); up[:-1] = m[1:]
            dn = np.zeros_like(m); dn[1:] = m[:-1]
            lt = np.zeros_like(m); lt[:, :-1] = m[:, 1:]
            rt = np.zeros_like(m); rt[:, 1:] = m[:, :-1]
            m = m | up | dn | lt | rt
        return m

    def _iou(a: np.ndarray, b: np.ndarray, *, dilate: int = 1) -> float:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h == 0 or w == 0:
            return 0.0
        ad = _dilate(a[:h, :w], dilate)
        bd = _dilate(b[:h, :w], dilate)
        inter = int(np.logical_and(ad, bd).sum())
        union = int(np.logical_or(ad, bd).sum())
        return float(inter) / float(union) if union > 0 else 1.0

    # Layout QA for the latest panel-d design notes: the inter-row whitespace
    # should be a compact separator, and the umbrella source should sit close
    # to the top of its own schematic panel rather than leaving a large blank
    # cap above it.
    if row_gap > 6.5:
        qa_errors.append(_qa.QAError(
            rule="figure4-c-d-gap-compact",
            panel="figure4/c-d-gap",
            detail=f"c-to-d row gap = {row_gap:.1f} mm, expected <= 6.5 mm",
        ))
    source_top_gap_mm = d_top_h * (1.0 - PANEL_D_SOURCE_Y) - PANEL_D_SOURCE_RING_DIAM_MM / 2.0
    if source_top_gap_mm > 2.2 or source_top_gap_mm < -0.4:
        qa_errors.append(_qa.QAError(
            rule="panel-d-source-top-gap",
            panel="figure4/d-source-top-gap",
            detail=(
                f"source top whitespace = {source_top_gap_mm:.2f} mm, "
                "expected between -0.4 and 2.2 mm"
            ),
        ))

    # The transposed quantitative plot needs a physical left gutter for its
    # y-axis title.  Guard against the previous failure where the ylabel was
    # set but clipped outside the figure / hidden at the page edge.
    ylabel_crop = _rect_crop_mm(
        panel_c_x - d_ylabel_gutter_mm + 0.3,
        c2_y + 5.0,
        panel_c_x - 0.7,
        c2_y + c2_h - 5.0,
    )
    ylabel_ink = int(_ink_mask(ylabel_crop, thresh=225).sum())
    if ylabel_ink < 30:
        qa_errors.append(_qa.QAError(
            rule="panel-d-ylabel-visible",
            panel="figure4/d-ylabel-visible",
            detail=f"d-panel ylabel gutter contains only {ylabel_ink} ink pixels; expected visible ylabel",
        ))

    # Single-source check: the top schematic strip should contain only the
    # central ABX3 reference cluster. Perturbed icons now live in the data
    # panel below.
    source_crop = _rect_crop_mm(
        panel_c_x, _top_yfrac_to_mm(0.05),
        panel_c_x + panel_c_w_full, _top_yfrac_to_mm(1.0),
    )
    source_mask = _ink_mask(source_crop)
    if source_mask.any():
        from scipy import ndimage

        labels, n_lab = ndimage.label(source_mask)
        off_center_components = 0
        center_lo_px = int(round(source_mask.shape[1] * 0.38))
        center_hi_px = int(round(source_mask.shape[1] * 0.62))
        center_ink = int(source_mask[:, center_lo_px:center_hi_px].sum())
        width_min_px = int(round(_mm_to_px_x(3.5) - _mm_to_px_x(0.0)))
        for lab in range(1, n_lab + 1):
            ys, xs = np.where(labels == lab)
            if xs.size and int(xs.max() - xs.min()) >= width_min_px:
                x_mid = 0.5 * (float(xs.min()) + float(xs.max()))
                if x_mid < center_lo_px or x_mid > center_hi_px:
                    off_center_components += 1
        if center_ink < 40 or off_center_components:
            qa_errors.append(_qa.QAError(
                rule="panel-d-single-original",
                panel="figure4/d-single-original",
                detail=(
                    f"central source ink = {center_ink} px, "
                    f"with {off_center_components} outside the central source region"
                ),
            ))

    # The umbrella source and the in-panel perturbed icons must share one
    # atom scale.  This catches the failure mode where one row looks visually
    # smaller/larger even when the layout is correct.
    base_atoms = _abx3_atom_cluster(PANEL_D_ICON_SCALE)
    base_radii = sorted(round(atom[4], 5) for atom in base_atoms)
    source_radii = sorted(round(atom[4], 5) for atom in _abx3_atom_cluster(PANEL_D_SOURCE_SCALE))
    if PANEL_D_SOURCE_SCALE != PANEL_D_ICON_SCALE or source_radii != base_radii:
        qa_errors.append(_qa.QAError(
            rule="panel-d-source-icon-atom-scale-consistent",
            panel="figure4/d-source-icon-scale",
            detail=(
                f"source scale/radii differ from lower icons "
                f"({PANEL_D_SOURCE_SCALE} vs {PANEL_D_ICON_SCALE})"
            ),
        ))
    if PANEL_D_ICON_SCALE < 2.0:
        qa_errors.append(_qa.QAError(
            rule="panel-d-icon-scale-readable",
            panel="figure4/d-icon-readable-scale",
            detail=f"panel-d icon scale = {PANEL_D_ICON_SCALE:.2f}, expected >= 2.0",
        ))
    for pid, _label, _lw, _alpha in reversed(PERT_NESTED_ORDER):
        radii = sorted(
            round(atom[4], 5)
            for atom in _perturb_abx3_atom_cluster(
                base_atoms, PANEL_D_DECORATIONS.get(pid), rng=np.random.default_rng(23)
            )
        )
        if radii != base_radii:
            qa_errors.append(_qa.QAError(
                rule="panel-d-icon-atom-scale-consistent",
                panel=f"figure4/d-icon-scale-{pid}",
                detail="perturbed icon atom radii differ from the shared ABX3 icon scale",
            ))

    # KDE columns remain one per perturbation.
    for boundary_i in range(1, n_cols):
        x_boundary_mm = panel_c_x + panel_c_w_full * boundary_i / n_cols
        violin_strip = (
            _mm_to_px_x(x_boundary_mm - 0.12),
            _mm_to_px_y_top(c2_y + c2_h - 0.5),
            _mm_to_px_x(x_boundary_mm + 0.12),
            _mm_to_px_y_top(c2_y + 0.5),
        )
        qa_errors += _qa.validate_panel_boundary(
            panel=f"figure4/d-column-isolation-violin-{boundary_i}",
            figure_image=fig_image,
            rect_px=violin_strip,
            max_data_pixels=40,
        )

    # Line column (rightmost) should collapse to a nearly horizontal row of
    # balls rather than a full ABX3 footprint.
    line_atoms = _perturb_abx3_atom_cluster(base_atoms, "exploded_line", rng=np.random.default_rng(23))
    line_extent_mm = max(atom[3] + atom[4] for atom in line_atoms) - min(atom[3] - atom[4] for atom in line_atoms)
    if line_extent_mm > 3.4:
        qa_errors.append(_qa.QAError(
            rule="panel-d-line-collapse",
            panel="figure4/d-line-collapse",
            detail=f"Line icon vertical extent = {line_extent_mm:.2f} mm, expected <= 3.40 mm",
        ))

    # Atom-pair column must preserve the ABX3 footprint while permuting
    # colours.  Compare same-size offscreen renderings so the large source
    # ring and fan arrows cannot pollute the silhouette check.
    def _render_abx3_mask(decoration: str | None) -> np.ndarray:
        tmp = plt.figure(figsize=(1.0, 1.0), dpi=160)
        tmp_ax = tmp.add_axes([0, 0, 1, 1])
        tmp_ax.set_xlim(0.0, 1.0)
        tmp_ax.set_ylim(0.0, 1.0)
        tmp_ax.axis("off")
        _draw_abx3_cluster(tmp_ax, 0.5, 0.5, 1.0, decoration=decoration)
        tmp.canvas.draw()
        arr = np.asarray(tmp.canvas.buffer_rgba()).copy()[:, :, :3]
        plt.close(tmp)
        return _ink_mask(arr)

    ap_iou = _iou(_render_abx3_mask(None), _render_abx3_mask("recolor_perm"), dilate=1)
    if ap_iou < 0.55:
        qa_errors.append(_qa.QAError(
            rule="panel-d-atom-pair-scaffold",
            panel="figure4/d-atom-pair-scaffold",
            detail=f"Atom-pair ABX3 footprint IoU vs plain ABX3 = {ap_iou:.3f}, expected >= 0.55",
        ))

    out_base = THIS_DIR / "figure4"
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300)
    plt.close(fig)

    if qa_errors:
        for err in qa_errors:
            print(f"FIGURE-QA  {err}", file=sys.stderr)
        raise RuntimeError(
            "figure 4 QA failed -- panel content has invaded a reserved "
            "row gap; see stderr for the offending strips."
        )
    _overlay_panel_a_vector_pdf(
        out_base.with_suffix(".pdf"),
        fig_w_mm=fig_w_mm,
        fig_h_mm=fig_h_mm,
        panel_x_mm=margin_x,
        panel_y_mm=top_panel_y,
        panel_w_mm=fig_w_mm - 2 * margin_x,
        panel_h_mm=top_panel_h,
    )
    print(f"Saved {out_base}.png and {out_base}.pdf")

    # SI figures retained in the manuscript: parity expansions of the
    # material-level and site-pooled linear probes. The UMAP / perturbation
    # diagnostic exports overlap with main Fig. 4 and are no longer emitted.
    plot_si_material_probe_parity(materials, meta)
    plot_si_site_probe_parity(meta)


def plot_si_perturb_nested_range_mt(
    deltas: dict[str, dict[str, tuple[float, float, int]]],
    raw_preds: dict[str, float],
    materials: list[str],
    meta: dict[str, dict[str, object]],
) -> None:
    """Supplementary: per-material nested range plot for the multi-task model.

    Companion to main Fig 4c, which now reports the per-perturbation
    aggregate distribution.  This SI variant exposes the per-material
    breakdown so a reader who wants to know whether a single material
    behaves anomalously (e.g. resists all perturbations while still
    carrying a large standing error) can find the answer.
    """
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    fig_w_mm, fig_h_mm = 170.0, 130.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    ax = fig.add_axes([0.18, 0.10, 0.78, 0.78])
    _plot_panel_c2_nested_range(
        ax, deltas, raw_preds, materials, meta,
        show_signed_error=True, draw_legend=True,
    )
    ax.set_title("Multi-task: per-material perturbation spread",
                 loc="left", color=MT_COLOR, pad=18.0, fontsize=FIG4_FONT)

    out_base = THIS_DIR / "figure_si_perturb_nested_range_mt"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def plot_si_template_parity_st_scratch(
    template_data: dict[str, dict[str, object]],
    meta: dict[str, dict[str, object]],
) -> None:
    """Supplementary: DAP-4 template parity for single-task and from-scratch.

    Companion to the MT-only panel in main figure 4c1.  Demoting these two
    variants to SI keeps the main figure focused on MT, while letting the
    reader still inspect template invariance for the comparison models.
    """
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    fig_w_mm, fig_h_mm = 130.0, 65.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.10, right=0.985, top=0.85, bottom=0.18, wspace=0.18,
    )
    ax_st = fig.add_subplot(gs[0, 0])
    ax_sc = fig.add_subplot(gs[0, 1])

    if "st" in template_data:
        _plot_template_parity_single(
            ax_st, template_data["st"], meta,
            title="Single-task", title_color=ST_COLOR, show_ylabel=True,
        )
    else:
        ax_st.axis("off")
    if "scratch" in template_data:
        _plot_template_parity_single(
            ax_sc, template_data["scratch"], meta,
            title="Scratch", title_color=SCRATCH_COLOR, show_ylabel=False,
        )
    else:
        ax_sc.axis("off")

    out_base = THIS_DIR / "figure_si_template_parity_st_scratch"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def plot_si_perturb_nested_range_st(
    materials: list[str],
    meta: dict[str, dict[str, object]],
) -> None:
    """Supplementary: per-material nested range plot for the single-task model.

    The single-task variant should show systematically larger spreads than
    multi-task across the more aggressive perturbations.  Caption notes the
    absolute numbers so the reader can compare against the main figure 4c2
    without flipping back to the source data.
    """
    try:
        st_deltas, st_raw = load_perturb_per_material_deltas("exp7c")
    except (KeyError, FileNotFoundError):
        return
    if not st_deltas:
        return

    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    fig_w_mm, fig_h_mm = 170.0, 130.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    ax = fig.add_axes([0.18, 0.10, 0.78, 0.78])
    _plot_panel_c2_nested_range(
        ax, st_deltas, st_raw, materials, meta,
        show_signed_error=True, draw_legend=True,
    )
    ax.set_title("Single-task: per-material perturbation spread",
                 loc="left", color=ST_COLOR, pad=18.0, fontsize=FIG4_FONT)

    out_base = THIS_DIR / "figure_si_perturb_nested_range_st"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


# --- SI: probe parity plots (companions to main Fig. 4a1 / 4a2) ----------

def _aggregate_m3_predictions(
    m3: dict, model_key: str, target: str, kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred_mean) averaging y_pred across the 5 LOO folds.

    ``kind`` is ``"emb"`` (embedding probe) or ``"comp"`` (composition
    baseline).  ``y_true`` and ``y_pred`` have one entry per material in
    the same order as the fold-0 list.
    """
    folds = sorted(m3["per_fold"].keys(), key=int)
    fold0 = m3["per_fold"][folds[0]][model_key]["probe_results"][target]
    y_true = np.asarray(fold0["y_true"], dtype=float)
    field = "y_pred_emb" if kind == "emb" else "y_pred_comp"
    stacked = np.stack(
        [np.asarray(m3["per_fold"][fk][model_key]["probe_results"][target][field],
                    dtype=float)
         for fk in folds],
        axis=0,
    )
    y_pred = stacked.mean(axis=0)
    return y_true, y_pred


def _draw_parity_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    materials: list[str],
    meta: dict[str, dict[str, object]],
    *,
    title: str | None = None,
    title_color: str = CHARCOAL,
    xlabel: str | None = None,
    ylabel: str | None = None,
    r2_value: float | None = None,
    r2_label: str = r"$R^{2}$",
    units: str = "",
) -> None:
    """One parity scatter, coloured by X-site, with diagonal + R² annotation."""
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.any():
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=FIG4_FONT, color=CHARCOAL)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    yt = y_true[finite]
    yp = y_pred[finite]
    mats = [materials[i] for i in range(len(materials)) if finite[i]]

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    pad = (hi - lo) * 0.07 if hi > lo else 1.0
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color=CHARCOAL, lw=0.5, ls="--", zorder=2)

    for material, x_val, y_val in zip(mats, yt, yp):
        x_site = str(meta[material]["X"]) if material in meta else "ClO4-"
        mcolor = X_SITE_COLORS.get(x_site, "#6F6F6F")
        ax.scatter(
            x_val, y_val,
            s=14.0, color=mcolor,
            edgecolors="none", linewidths=0.0, zorder=3,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title, pad=2.0, loc="left", color=title_color, fontsize=FIG4_FONT)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=1)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=1)
    if r2_value is not None:
        ax.text(
            0.97, 0.04,
            r2_label + f" = {r2_value:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FIG4_FONT, color=CHARCOAL,
        )
    style_axes(ax, grid=False)


def _make_xsite_legend_handles() -> list:
    return [
        mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=6.0,
            markerfacecolor=X_SITE_COLORS["ClO4-"], markeredgecolor="none",
            markeredgewidth=0.4, label="Perchlorate",
        ),
        mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=6.0,
            markerfacecolor=X_SITE_COLORS["NO3-"], markeredgecolor="none",
            markeredgewidth=0.4, label="Nitrate",
        ),
        mlines.Line2D(
            [], [], marker="o", linestyle="None", markersize=6.0,
            markerfacecolor=X_SITE_COLORS["IO4-"], markeredgecolor="none",
            markeredgewidth=0.4, label="Periodate",
        ),
    ]


def plot_si_material_probe_parity(
    materials: list[str],
    meta: dict[str, dict[str, object]],
) -> None:
    """Supplementary: parity plots for the material-level linear probes.

    A 3 (target: V_det, density, OB%) by 3 (model: MT, ST, composition)
    grid expanding the bar summary in main Fig 4a (left).  Predictions
    are LOO ridge probes, averaged across the 5 fold checkpoints to give
    one point per material.
    """
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )
    try:
        m3 = _load_json(M3_PATH)
    except FileNotFoundError:
        return
    metrics = load_material_probe_metrics()

    targets = list(PROBE_TARGETS_A1)
    target_units = {"Vdet": r"(m$\cdot$s$^{-1}$)", "density": r"(g$\cdot$cm$^{-3}$)", "OB": "(%)"}
    cols = (
        ("mt",   "Multi-task",   MT_COLOR,           "exp7a", "emb"),
        ("st",   "Single-task",  ST_COLOR,           "exp7c", "emb"),
        ("comp", "Composition",  COMP_BASELINE_COLOR, "exp7a", "comp"),
    )

    n_rows = len(targets)
    n_cols = len(cols)
    fig_w_mm, fig_h_mm = 170.0, 185.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    legend_h_frac = 7.0 / fig_h_mm
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=n_cols,
        left=0.10, right=0.985,
        top=1.0 - legend_h_frac - 0.02, bottom=0.06,
        wspace=0.30, hspace=0.45,
    )

    for r, target in enumerate(targets):
        for c, (model_key, model_label, model_color, src_key, kind) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            y_true, y_pred = _aggregate_m3_predictions(m3, src_key, target, kind)
            mat_list = list(materials)
            r2_val = metrics[target][model_key]["r2"]
            target_label = PROBE_TARGET_LABELS[target]
            unit = target_units.get(target, "")
            # Each row has its own target scale, so always show xlabel / ticks.
            xlabel = f"True {target_label} {unit}".strip()
            ylabel = f"Predicted {target_label} {unit}".strip() if c == 0 else None
            title = model_label if r == 0 else None
            _draw_parity_panel(
                ax, y_true, y_pred, mat_list, meta,
                title=title, title_color=model_color,
                xlabel=xlabel, ylabel=ylabel,
                r2_value=r2_val,
                r2_label=r"$R^{2}_{\mathrm{LOO}}$",
            )
            if c != 0:
                ax.tick_params(labelleft=False)

    handles = _make_xsite_legend_handles()
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles), frameon=False,
        handletextpad=0.4, columnspacing=1.4,
        fontsize=FIG4_FONT,
    )

    out_base = THIS_DIR / "figure_si_material_probe_parity"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def plot_si_site_probe_parity(
    meta: dict[str, dict[str, object]],
) -> None:
    """Supplementary: parity plots for the site-pooled linear probes.

    A 2 (model: MT, ST) by 4 (site: z_X, z_B, z_A, z_all) grid expanding
    the bar summary in main Fig 4a (right).  Predictions come from the
    site-pooled LOO ridge probes (V_det target only).
    """
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )
    try:
        m4b = _load_json(M4B_PATH)
    except FileNotFoundError:
        return
    if "site_y_pred_mean" not in m4b or "site_y_true" not in m4b:
        print("[SI parity] m4b JSON missing site_y_pred_mean / site_y_true; "
              "re-run mechanism analysis m4b.")
        return

    site_keys = list(SITE_PROBE_KEYS)
    rows = (("exp7a", "Multi-task", MT_COLOR),
            ("exp7c", "Single-task", ST_COLOR))

    n_rows = len(rows)
    n_cols = len(site_keys)
    fig_w_mm, fig_h_mm = 170.0, 110.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    legend_h_frac = 7.0 / fig_h_mm
    left_margin = 0.13
    gs = fig.add_gridspec(
        nrows=n_rows, ncols=n_cols,
        left=left_margin, right=0.985,
        top=1.0 - legend_h_frac - 0.02, bottom=0.10,
        wspace=0.28, hspace=0.30,
    )

    row_axes: list[plt.Axes] = []
    for r, (src_key, model_label, model_color) in enumerate(rows):
        y_true = np.asarray(m4b["site_y_true"][src_key], dtype=float)
        mat_list = list(m4b["site_materials"][src_key])
        first_ax_in_row: plt.Axes | None = None
        for c, sk in enumerate(site_keys):
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                first_ax_in_row = ax
            y_pred = np.asarray(m4b["site_y_pred_mean"][src_key][sk], dtype=float)
            r2_val = float(m4b["site_r2"][src_key][sk])
            xlabel = (r"True $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)"
                      if r == n_rows - 1 else None)
            ylabel = (r"Predicted $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)"
                      if c == 0 else None)
            title = SITE_PROBE_LABELS[sk] if r == 0 else None
            _draw_parity_panel(
                ax, y_true, y_pred, mat_list, meta,
                title=title, title_color=CHARCOAL,
                xlabel=xlabel, ylabel=ylabel,
                r2_value=r2_val,
                r2_label=r"$R^{2}_{\mathrm{LOO}}$",
            )
            if r != n_rows - 1:
                ax.tick_params(labelbottom=False)
            if c != 0:
                ax.tick_params(labelleft=False)
        row_axes.append(first_ax_in_row)

    fig.canvas.draw()
    for (src_key, model_label, model_color), ax0 in zip(rows, row_axes):
        if ax0 is None:
            continue
        bbox = ax0.get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            0.012, y_center, model_label,
            ha="left", va="center", rotation=90,
            fontsize=FIG4_FONT + 1.0, color=model_color,
            fontweight="bold",
        )

    handles = _make_xsite_legend_handles()
    fig.legend(
        handles=handles, loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles), frameon=False,
        handletextpad=0.4, columnspacing=1.4,
        fontsize=FIG4_FONT,
    )

    out_base = THIS_DIR / "figure_si_site_probe_parity"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def plot_si_st_atomic_umap(cache: dict[str, np.ndarray]) -> None:
    """Supplementary: single-task atomic UMAP coloured by site and by element.

    Mirrors the multi-task panels in the main figure 4 (panel b) so the
    reader can inspect what the ST descriptor looks like in atomic space.
    """
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": FIG4_FONT,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    st_atomic = cache["st_atomic_umap"]
    st_emb = cache["st_atomic_emb"]
    st_atom_sites = cache["st_atom_sites"]
    st_atom_elems = cache["st_atom_elements"]
    st_heavy = ~np.isin(st_atom_elems, list(ATOMIC_UMAP_BACKGROUND_ELEMENTS))

    fig_w_mm, fig_h_mm = 170.0, 80.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        left=0.07, right=0.985, top=0.85, bottom=0.18, wspace=0.20,
    )
    ax_site = fig.add_subplot(gs[0, 0])
    ax_elem = fig.add_subplot(gs[0, 1])

    _plot_atomic_umap_by_site(
        ax_site, st_atomic, st_atom_sites,
        title="ST by site", title_color=ST_COLOR,
        show_ylabel=True, show_legend=False,
        elements=st_atom_elems,
        background_elements=ATOMIC_UMAP_BACKGROUND_ELEMENTS,
    )
    ax_site.text(
        0.50, 1.04, "ST atomic UMAP - by site",
        transform=ax_site.transAxes, ha="center", va="bottom",
        fontsize=FIG4_FONT, color=ST_COLOR,
    )
    _annotate_silhouette_index(
        ax_site, st_emb, st_atomic, st_atom_sites, label_kind="site", mask=st_heavy,
    )

    _plot_atomic_umap_by_element(
        ax_elem, st_atomic, st_atom_elems,
        title="ST by element", title_color=ST_COLOR,
        show_ylabel=False, show_legend=False,
        background_elements=ATOMIC_UMAP_BACKGROUND_ELEMENTS,
    )
    ax_elem.text(
        0.50, 1.04, "ST atomic UMAP - by element",
        transform=ax_elem.transAxes, ha="center", va="bottom",
        fontsize=FIG4_FONT, color=ST_COLOR,
    )
    _annotate_silhouette_index(
        ax_elem, st_emb, st_atomic, st_atom_elems, label_kind="element", mask=st_heavy,
    )

    # Shared element legend below the two panels.
    element_handles = []
    for elem in ("Ag", "C", "Cl", "H", "I", "K", "N", "Na", "O", "Rb"):
        face = ELEMENT_PALETTE.get(elem, MID_GRAY)
        if elem in ATOMIC_UMAP_BACKGROUND_ELEMENTS:
            element_handles.append(
                mlines.Line2D(
                    [], [], marker="o", linestyle="None", markersize=4.0,
                    markerfacecolor="none", markeredgecolor=face,
                    markeredgewidth=0.5, label=elem,
                )
            )
        else:
            element_handles.append(
                mlines.Line2D(
                    [], [], marker="o", linestyle="None", markersize=4.0,
                    color=face, label=elem,
                )
            )
    fig.legend(
        handles=element_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=10, frameon=False, fontsize=FIG4_FONT,
        handletextpad=0.30, columnspacing=1.0, borderaxespad=0.0,
    )

    for axis, label in zip((ax_site, ax_elem), ("a", "b")):
        add_panel_label(axis, label)

    out_base = THIS_DIR / "figure_si_st_atomic_umap"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def plot_si_site_pooled_umap(
    materials: list[str],
    cache: dict[str, np.ndarray],
    meta: dict[str, dict[str, object]],
) -> None:
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 8.0,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    mt_coords = cache["mt_umap_coords"]
    st_coords = cache["st_umap_coords"]
    point_sites = cache["point_sites"]
    point_materials = cache["point_materials"]

    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 88 / 25.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.13, wspace=0.18)

    for axis, coords, title, title_color, show_y, show_legend in (
        (axes[0], mt_coords, "MT-FT", MT_COLOR, True, True),
        (axes[1], st_coords, "ST-FT", ST_COLOR, False, False),
    ):
        _plot_umap_panel(
            axis,
            coords,
            point_sites,
            point_materials,
            meta,
            title,
            title_color,
            show_ylabel=show_y,
            show_legend=show_legend,
        )

    for axis, label in zip(axes, ("A", "B")):
        add_panel_label(axis, label)

    out_base = THIS_DIR / "figure_si_site_pooled_umap"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


def plot_si_site_probe(probe: dict[str, object]) -> None:
    """Supplementary figure: per-site LOO Ridge probe R² (z-scored, MT)."""
    setup_nature_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 8.0,
            "axes.labelsize": FIG4_FONT,
            "xtick.labelsize": FIG4_FONT,
            "ytick.labelsize": FIG4_FONT,
            "legend.fontsize": FIG4_FONT,
        }
    )

    fig, ax = plt.subplots(figsize=(86 / 25.4, 70 / 25.4))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.92, bottom=0.20)

    probe_order = ["X-site", "B-site", "A-site", "Composition"]
    probe_colors = [X_COLOR, B_COLOR, A_COLOR, BASELINE_COLOR]
    probe_vals = [probe["site_probe"][key]["r2"] for key in probe_order]
    probe_errs = [probe["site_probe"][key].get("r2_std", 0.0) for key in probe_order]
    bars = ax.bar(
        range(len(probe_order)),
        probe_vals,
        color=probe_colors,
        width=0.72,
        yerr=probe_errs,
        error_kw={"lw": 0.6, "capsize": 2.0, "capthick": 0.6, "ecolor": CHARCOAL},
    )
    ax.axhline(0.0, color=CHARCOAL, lw=0.5, ls="--")
    ax.set_xticks(range(len(probe_order)))
    ax.set_xticklabels(["X-site", "B-site", "A-site", "Baseline"], rotation=20, ha="right")
    ax.set_ylabel("LOO $R^2$", labelpad=1)
    ax.set_ylim(-0.55, 1.10)
    ax.set_title(r"Site probe ($V_{\mathrm{det}}$, multi-task)", pad=3, loc="left")
    style_axes(ax, grid=True)
    for bar, value, err in zip(bars, probe_vals, probe_errs):
        offset = 0.04 if value >= 0 else -0.10
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + err + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        fontsize=FIG4_FONT,
        )

    out_base = THIS_DIR / "figure_si_site_probe"
    save_png_pdf(fig, out_base)
    plt.close(fig)
    print(f"Saved {out_base}.png and {out_base}.pdf")


# === SI ST ATOMIC UMAP ===
# Implementation appended below via a separate function.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compute-cache", action="store_true", help="Compute exp7a + exp7c descriptor caches.")
    group.add_argument("--plot-only", action="store_true", help="Render figure from cached embeddings.")
    group.add_argument("--refit-umap", action="store_true", help="Recompute UMAP from cached site embeddings only.")
    args = parser.parse_args()

    materials = load_training_materials()
    if args.compute_cache:
        compute_cache(materials)
    elif args.refit_umap:
        ensure_atomic_descriptor_cache(materials)
        refit_umap_in_cache(materials)
        plot(materials)
    elif args.plot_only:
        ensure_atomic_descriptor_cache(materials)
        plot(materials)
    else:
        compute_cache(materials)
        plot(materials)


if __name__ == "__main__":
    main()
