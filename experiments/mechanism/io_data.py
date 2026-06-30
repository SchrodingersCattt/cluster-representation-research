"""Data I/O: cluster systems, materials, fold splits, ground truth, density, OB."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from . import constants, paths


def read_cluster_system(sys_dir: Path) -> tuple[np.ndarray, list[str], float]:
    type_map = (sys_dir / "type_map.raw").read_text(encoding="utf-8").strip().split()
    types = np.loadtxt(sys_dir / "type.raw", dtype=int)
    coord = np.load(sys_dir / "set.000" / "coord.npy")[0].reshape(-1, 3)
    prop = float(np.load(sys_dir / "set.000" / "property.npy")[0])
    return coord, [type_map[t] for t in types], prop


def get_materials() -> list[str]:
    return sorted(p.name for p in paths.CLUSTER_N1_DIR.iterdir() if p.is_dir())


_m1_fold_splits_cache: dict[int, list[str]] = {}


def get_m1_heldout_mats(fold_idx: int) -> list[str]:
    """Sorted held-out material names for a given fold (from pems_5fold_splits_v2.json)."""
    if fold_idx not in _m1_fold_splits_cache:
        data = json.loads(paths.M1_SPLITS_PATH.read_text(encoding="utf-8"))
        for k, v in data["folds"].items():
            _m1_fold_splits_cache[int(k)] = sorted(v)
    return _m1_fold_splits_cache[fold_idx]


def load_gt_vdet() -> dict[str, float]:
    """Ground-truth Vdet (m/s) for all PEMs materials.

    Primary source: pems_manifest.json crystal_results.
    Fallback: pems.csv D_km_s column (x 1000 to convert km/s -> m/s) when
    manifest has fewer records than expected (e.g. after a partial rebuild).
    """
    manifest = json.loads(paths.MANIFEST_PATH.read_text(encoding="utf-8"))
    gt = {rec["material"]: rec["target_m_s"] for rec in manifest["crystal_results"]}
    if paths.PEMS_CSV.exists():
        with paths.PEMS_CSV.open(newline="") as f:
            for row in csv.DictReader(f):
                mat = row.get("material", "").strip()
                d_km_s = row.get("D_km_s", "").strip()
                if mat and d_km_s and mat not in gt:
                    try:
                        gt[mat] = float(d_km_s) * 1000.0
                    except ValueError:
                        pass
    return gt


def get_family(material: str) -> str:
    for prefix in ["DAI", "DAN", "DAP", "PAN", "PAP"]:
        if material.startswith(prefix):
            return prefix
    return "other"


def compute_crystal_density(cif_path: str | Path) -> float | None:
    """Crystal density (g/cm^3) from CIF, robust to disordered structures.

    Strategy: prefer Z * MW / (V * N_A) from CIF header tags, which is
    immune to ASE over-expanding disordered/equivalent sites. Fall back to
    ASE mass/volume only when the header tags are missing.
    """
    from ase.data import atomic_masses as _am, atomic_numbers as _an

    cif_path = Path(cif_path)
    text = cif_path.read_text(errors="replace")
    AVOGADRO = 6.02214076e23

    def _get_float(tag: str) -> float | None:
        m = re.search(rf"{tag}\s+([\d.]+)", text)
        return float(m.group(1)) if m else None

    a, b, c = _get_float("_cell_length_a"), _get_float("_cell_length_b"), _get_float("_cell_length_c")
    alpha = _get_float("_cell_angle_alpha") or 90.0
    beta = _get_float("_cell_angle_beta") or 90.0
    gamma = _get_float("_cell_angle_gamma") or 90.0

    if a and b and c:
        ar_, br_, gr_ = np.radians(alpha), np.radians(beta), np.radians(gamma)
        vol = a * b * c * np.sqrt(
            1 - np.cos(ar_)**2 - np.cos(br_)**2 - np.cos(gr_)**2
            + 2 * np.cos(ar_) * np.cos(br_) * np.cos(gr_))
    else:
        vol = None

    Z = _get_float("_cell_formula_units_Z")

    m = re.search(r"_chemical_formula_sum\s+'([^']+)'", text)
    if not m:
        m = re.search(r'_chemical_formula_sum\s+"([^"]+)"', text)
    mw = 0.0
    if m:
        for tok in re.findall(r'([A-Z][a-z]?)([\d.]*)', m.group(1)):
            elem, cnt = tok
            cnt = float(cnt) if cnt else 1.0
            if elem in _an:
                mw += _am[_an[elem]] * cnt

    if Z and Z > 0 and mw > 0 and vol and vol > 0:
        density = Z * mw / (vol * AVOGADRO) * 1e24
        if 0.5 < density < 6.0:
            return density

    try:
        from ase.io import read as _ar
        atoms = _ar(str(cif_path))
        density = atoms.get_masses().sum() / atoms.get_volume() * 1.6605
        if 0.5 < density < 6.0:
            return density
    except Exception:
        pass

    return None


def load_densities() -> dict[str, float]:
    """Load crystal densities for all PEMs materials.

    Primary source: pems_manifest.json crystal_results (source_cif paths).
    Fallback: scan data/pems/confs/*.cif directly when manifest has fewer
    than expected records (e.g. after a partial rebuild).
    """
    CONFS_DIR = paths.PEMS_CSV.parent / "confs"
    manifest = json.loads(paths.MANIFEST_PATH.read_text(encoding="utf-8"))
    densities: dict[str, float] = {}
    for rec in manifest["crystal_results"]:
        cif = rec.get("source_cif", "")
        if cif and Path(cif).exists():
            d = compute_crystal_density(cif)
            if d is not None:
                densities[rec["material"]] = d
    if CONFS_DIR.is_dir():
        for cif_path in sorted(CONFS_DIR.glob("*.cif")):
            mat = cif_path.stem
            if mat not in densities:
                d = compute_crystal_density(cif_path)
                if d is not None:
                    densities[mat] = d
    return densities


def compute_composition_and_ob(
    materials: list[str],
) -> tuple[dict[str, dict], dict[str, float]]:
    """Element fractions and oxygen balance for all materials.

    Returns:
        comp: {material: {"C": frac, "H": frac, ..., "n_atoms": int}}
        ob_values: {material: OB%}

    OB formula (CO2 convention, Lothrop-Handrick 1949, ref 39 in Guo et al. EMF 2026):
      OB(%) = (1600/M) * (n_O - 2*n_C - (n_H - n_halogen)/2 - metal_oxide_O)
    where halogens (Cl, I) form HCl/HI (each saves 0.5 O by consuming one H),
    and metals form their lowest common oxides.
    """
    comp: dict[str, dict] = {}
    ob_values: dict[str, float] = {}
    for mat in materials:
        _, symbols, _ = read_cluster_system(paths.CLUSTER_N1_DIR / mat)
        n = len(symbols)
        comp[mat] = {e: symbols.count(e) / n for e in constants.COMPOSITION_ELEMENTS}
        comp[mat]["n_atoms"] = n
        counts = {e: symbols.count(e) for e in set(symbols)}
        mw = sum(counts.get(e, 0) * constants.ATOMIC_MASS.get(e, 0.0) for e in counts)
        if mw > 0:
            n_O = counts.get("O", 0)
            n_C = counts.get("C", 0)
            n_H = counts.get("H", 0)
            n_halogen = counts.get("Cl", 0) + counts.get("I", 0)
            metal_O = sum(counts.get(m, 0) * v for m, v in constants.METAL_O_DEMAND.items())
            ob_values[mat] = (1600.0 / mw) * (n_O - 2 * n_C - (n_H - n_halogen) / 2.0 - metal_O)
    return comp, ob_values


def build_probe_targets(
    materials: list[str],
    gt: dict[str, float],
    comp: dict[str, dict],
    ob_values: dict[str, float],
    densities: dict[str, float],
) -> dict[str, np.ndarray]:
    """Standard set of probe regression targets.

    Returns dict mapping target name -> array of values (NaN where missing).
    """
    tgts: dict[str, np.ndarray] = {
        "Vdet": np.array([gt.get(m, np.nan) for m in materials]),
        "frac_N": np.array([comp[m]["N"] for m in materials]),
        "frac_O": np.array([comp[m]["O"] for m in materials]),
        "n_atoms": np.array([float(comp[m]["n_atoms"]) for m in materials]),
    }
    if sum(1 for m in materials if m in densities) >= 10:
        tgts["density"] = np.array([densities.get(m, np.nan) for m in materials])
    if sum(1 for m in materials if m in ob_values) >= 10:
        tgts["OB"] = np.array([ob_values.get(m, np.nan) for m in materials])
    return tgts
