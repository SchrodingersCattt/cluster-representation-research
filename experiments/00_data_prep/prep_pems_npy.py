#!/usr/bin/env python3
"""
Prepare PEMs crystal and cluster DeepMD datasets.

Builds four datasets from [`data/pems/pems.csv`](data/pems/pems.csv):
- cleaned crystal structures
- three vacancy-cluster datasets derived from cleaned crystals

Disordered CIFs are resolved with MolCrysKit before export. Rows without a
usable detonation velocity target or missing CIF are skipped.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
from ase import Atoms
from ase.io import read, write as ase_write
from pymatgen.io.cif import CifParser
from molcrys_kit.analysis.disorder.process import generate_ordered_replicas_from_disordered_sites
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.io import write_cif
from molcrys_kit.operations.defects import VacancyGenerator
from molcrys_kit.structures.crystal import MolecularCrystal

# ---------------------------------------------------------------------------
# Bond thresholds for MolecularCrystal.from_ase() — prevents
# identify_molecules() from creating spurious bonds when heavy/metal atoms
# (I, Na, K, Rb, Ba, Ag) are present. DEFAULT_NEIGHBOR_CUTOFF=3.5 Å is far
# too large for these elements; we set explicit pairwise limits here.
# Format: {(elem1, elem2): max_bond_distance_angstrom}
# ---------------------------------------------------------------------------
_HEAVY = ["I", "Na", "K", "Rb", "Ba", "Ag"]
_ORGANIC = ["C", "H", "N", "O", "Cl"]

PEM_BOND_THRESHOLDS: dict[tuple[str, str], float] = {}

# Default heavy–organic thresholds for non-oxygen contacts.
# Oxygen is handled separately with stricter cutoffs to avoid merging ionic
# metal···O contacts into molecular clusters.
_HEAVY_NONO_LIMITS: dict[str, float] = {
    "I":  2.10,
    "Na": 2.30,
    "K":  2.50,
    "Rb": 2.60,
    "Ba": 2.60,
    "Ag": 2.30,
}
_HEAVY_O_LIMITS: dict[str, float] = {
    "I":  2.05,
    "Na": 2.20,
    "K":  2.30,
    "Rb": 2.40,
    "Ba": 2.40,
    "Ag": 2.20,
}
for metal, limit in _HEAVY_NONO_LIMITS.items():
    for org in ["C", "H", "N", "Cl"]:
        PEM_BOND_THRESHOLDS[(metal, org)] = limit
        PEM_BOND_THRESHOLDS[(org, metal)] = limit
for metal, limit in _HEAVY_O_LIMITS.items():
    PEM_BOND_THRESHOLDS[(metal, "O")] = limit
    PEM_BOND_THRESHOLDS[("O", metal)] = limit

# Metal–metal thresholds (should not bond)
for i, m1 in enumerate(_HEAVY):
    for m2 in _HEAVY[i:]:
        PEM_BOND_THRESHOLDS[(m1, m2)] = 3.2
        if m1 != m2:
            PEM_BOND_THRESHOLDS[(m2, m1)] = 3.2


ROOT = Path(__file__).resolve().parents[2]
PEMS_DIR = ROOT / "data" / "pems"
CSV_PATH = PEMS_DIR / "pems.csv"
CIF_DIR = PEMS_DIR / "confs"
OUT_ROOT = ROOT / "experiments" / "00_data_prep"

CRYSTAL_OUT_DIR = OUT_ROOT / "pems_crystal_systems"
CLUSTER_DIRS = {
    "cluster_n1": OUT_ROOT / "pems_cluster_n1_systems",
    "cluster_n2": OUT_ROOT / "pems_cluster_n2_systems",
    "cluster_n3": OUT_ROOT / "pems_cluster_n3_systems",
}
CLEANED_CIF_DIR = OUT_ROOT / "pems_cleaned_cifs"
CLUSTER_CIF_DIR = OUT_ROOT / "pems_cluster_cifs"
MANIFEST_PATH = OUT_ROOT / "pems_manifest.json"
CLEANING_REPORT_PATH = OUT_ROOT / "pems_cleaning_report.md"
CLUSTER_REPORT_PATH = OUT_ROOT / "pems_cluster_report.md"

TYPE_MAP = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
TYPE_TO_ID = {sym: i for i, sym in enumerate(TYPE_MAP)}
CLUSTER_BOX = np.eye(3, dtype=np.float64) * 100.0
CLUSTER_RANDOM_SEEDS = {
    "cluster_n1": 101,
    "cluster_n2": 202,
    "cluster_n3": 303,
}
CLUSTER_VARIANT_OFFSETS = {
    "cluster_n1": 0,
    "cluster_n2": 1,
    "cluster_n3": 2,
}

# Fingerprint tolerance: sorted pairwise-distance vectors that agree to within
# this threshold (Å) are considered geometrically identical (rotation+translation
# invariant). Same value as check_cluster_identity.py.
FINGERPRINT_IDENTICAL_TOL: float = 1e-4

# How many spread seeds to request when building the diversity-guard candidate
# pool. More seeds gives more room to find a distinct cluster without expanding
# the supercell.
MAX_SPREAD_SEEDS_FOR_DIVERSITY: int = 10


def cluster_fingerprint(coords: np.ndarray) -> np.ndarray:
    """Rotation/translation-invariant fingerprint: sorted upper-triangle pairwise distances."""
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    n = len(coords)
    upper = dists[np.triu_indices(n, k=1)]
    return np.sort(upper)


def fingerprint_distance(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Max-abs deviation between two fingerprints; returns inf if shapes differ."""
    if fp_a.shape != fp_b.shape:
        return float("inf")
    return float(np.max(np.abs(fp_a - fp_b)))

# Per-material overrides for the cluster target_spec. When a material appears
# in this map, build_seeded_stoichiometric_cluster uses the given
# ``{species_id: count}`` dict instead of the auto-computed simplest-unit.
#
# DAI-4: the ammonium counter-cation is disordered over 24 partial H sites
# per N in the raw CIF; after disorder resolution, four of the eight NH4+
# centres retain only three Hs and therefore get their own species_id
# ``H3N_1`` distinct from the fully-resolved ``H4N_1``. Because each ghost
# NH3 and real NH4 appears only once per unit cell after GCD reduction, the
# auto simplest-unit balloons to 2 x (H2-dabco)(NH4)(IO4)3 (two formula
# units). Pinning the target spec here restores the chemically-correct
# single formula unit: 1 H2-dabco + 3 IO4 + 1 NH4.
#
# DAI-4 was previously listed here as a workaround for MolCrysKit resolving
# its implicit-SHELX-riding-H N1 ammonium sites as NH3 instead of NH4+.
# That bug is fixed in MolCrysKit feat/sp-nh4-implicit-hardening (see
# DisorderSolver._select_motif_hydrogens one-per-asym_id guard).  After
# confirming NPY consistency with the override active vs removed, the entry
# was deleted.
CLUSTER_TARGET_SPEC_OVERRIDES: dict[str, dict[str, int]] = {}


# Materials to skip — excluded from the training set.
# DAC-4 now resolves correctly (C6H18Cl3N3O9) with the current MolCrysKit
# + PEM_BOND_THRESHOLDS rebuild, but it is kept out of the training set here
# because the manuscript is finalized on the DAC-4-free configuration.
# It may later be revisited as an OOD test material.
SKIP_MATERIALS: set[str] = {
    "DAC-4",
}


@dataclass
class PemRecord:
    material: str
    d_m_s: float
    cif_path: str
    raw_row: dict[str, str]


@dataclass
class CrystalBuildResult:
    material: str
    dataset_id: str
    target_m_s: float
    source_cif: str
    used_disorder_cleaning: bool
    disorder_atom_count: int
    cleaned_atom_count: int
    cleaned_molecule_count: int
    simplest_unit: dict[str, int]
    species_counts: dict[str, int]


@dataclass
class ClusterBuildResult:
    material: str
    dataset_name: str
    target_m_s: float
    random_seed: int
    seed_index: int
    cluster_atom_count: int
    cluster_molecule_count: int
    selected_species_counts: dict[str, int]
    supercell_dims: tuple[int, ...] = (2, 2, 2)
    seed_species: str = ""
    spread_seed_idx: int = 0


@dataclass
class SkipRecord:
    material: str
    reason: str
    cif_path: str | None = None


class PemPrepError(RuntimeError):
    pass


def load_candidate_records(
    only_materials: set[str] | None = None,
    bypass_skip: bool = False,
) -> tuple[list[PemRecord], list[SkipRecord]]:
    """Load PEM records from CSV.

    Args:
        only_materials: If provided, only process materials in this set.
        bypass_skip: If True, ignore the SKIP_MATERIALS blacklist (useful for
            processing previously-skipped OOD materials).
    """
    records: list[PemRecord] = []
    skips: list[SkipRecord] = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            material = (row.get("material") or "").strip()
            d_km_s_raw = (row.get("D_km_s") or "").strip()
            cif_path = CIF_DIR / f"{material}.cif"

            # Filter to requested materials if provided
            if only_materials is not None and material not in only_materials:
                continue

            if not d_km_s_raw:
                skips.append(SkipRecord(material=material, reason="missing_D_km_s", cif_path=str(cif_path)))
                continue
            try:
                d_m_s = float(d_km_s_raw) * 1000.0
            except ValueError:
                skips.append(SkipRecord(material=material, reason="invalid_D_km_s", cif_path=str(cif_path)))
                continue
            # Use actual file existence — don't trust has_cif CSV field
            if not cif_path.exists():
                skips.append(SkipRecord(material=material, reason="missing_cif_file", cif_path=str(cif_path)))
                continue
            if material in SKIP_MATERIALS and not bypass_skip:
                skips.append(SkipRecord(material=material, reason="manual_skip", cif_path=str(cif_path)))
                continue
            records.append(PemRecord(material=material, d_m_s=d_m_s, cif_path=str(cif_path), raw_row=row))
    return records, skips


def _parse_cif_occupancy_column(cif_path: Path) -> tuple[int | None, list[str], list[list[str]]]:
    """Parse a CIF file to find the _atom_site_occupancy column index.

    Returns (occ_col_index, column_names, data_rows).
    occ_col_index is None if no occupancy column found.
    """
    with cif_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    col_names: list[str] = []
    data_rows: list[list[str]] = []
    in_loop = False
    reading_header = False
    occ_idx: int | None = None

    for line in lines:
        s = line.strip()
        if s == "loop_":
            in_loop = True
            reading_header = True
            col_names = []
            data_rows = []
            occ_idx = None
        elif in_loop and reading_header and s.startswith("_atom_site"):
            col_names.append(s)
            if s.strip().lower() == "_atom_site_occupancy":
                occ_idx = len(col_names) - 1
        elif in_loop and reading_header and s.startswith("_"):
            # Different loop block — reset
            if col_names and any("_atom_site" in c for c in col_names):
                reading_header = False
                # This line belongs to a new loop, stop
                break
            else:
                col_names = []
                col_names.append(s)
        elif in_loop and col_names and s and not s.startswith("_") and not s.startswith("loop_"):
            reading_header = False
            data_rows.append(s.split())
        elif not reading_header and in_loop and (not s or s.startswith("loop_") or s.startswith("_")):
            if col_names and any("_atom_site" in c for c in col_names):
                break

    return occ_idx, col_names, data_rows


def contains_partial_occupancy(cif_path: Path) -> tuple[bool, int]:
    """Check if any atom site has occupancy < 1.0.

    Properly identifies the occupancy column by header name, not by position.
    Handles '?' placeholders gracefully.
    """
    occ_idx, col_names, data_rows = _parse_cif_occupancy_column(cif_path)

    if occ_idx is None:
        return False, 0

    partial_count = 0
    for row in data_rows:
        if len(row) <= occ_idx:
            continue
        token = row[occ_idx]
        if token in ("?", "."):
            continue
        try:
            # Handle CIF esd format like "0.500(2)"
            occ = float(re.sub(r"\(.+?\)", "", token))
        except (ValueError, IndexError):
            continue
        if occ < 0.999:
            partial_count += 1

    return partial_count > 0, partial_count


def _sanitize_cif_for_pymatgen(cif_path: Path) -> str:
    """Pre-clean a CIF file to work around common pymatgen/MolCrysKit parse errors.

    Returns path to a cleaned temporary CIF (or original if no cleaning needed).
    Fixes:
      - '?' in _atom_site_attached_hydrogens → '0'
      - '?' in occupancy columns → '1.0'
    """
    with cif_path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Replace ? in _atom_site_attached_hydrogens values
    # This is a heuristic — we replace ? only in data lines within _atom_site blocks
    modified = False
    lines = content.split("\n")
    out_lines = []

    in_atom_site_loop = False
    col_names: list[str] = []
    attached_h_idx: int | None = None

    for line in lines:
        s = line.strip()
        if s == "loop_":
            in_atom_site_loop = False
            col_names = []
            attached_h_idx = None
        if s.startswith("_atom_site"):
            in_atom_site_loop = True
            col_names.append(s)
            if "attached_hydrogens" in s:
                attached_h_idx = len(col_names) - 1
        elif in_atom_site_loop and col_names and s and not s.startswith("_") and not s.startswith("loop_"):
            if attached_h_idx is not None:
                parts = s.split()
                if len(parts) > attached_h_idx and parts[attached_h_idx] == "?":
                    parts[attached_h_idx] = "0"
                    line = "  ".join(parts)
                    modified = True
        out_lines.append(line)

    if modified:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".cif")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(out_lines))
        return tmp_path
    return str(cif_path)


def parse_ordered_atoms_from_cif(cif_path: Path) -> Atoms:
    """Parse ordered (no-disorder) CIF into ASE Atoms.

    Tries pymatgen CifParser first, falls back to ASE reader if that fails.
    """
    # Try pymatgen first (better symmetry handling)
    try:
        cleaned = _sanitize_cif_for_pymatgen(cif_path)
        parser = CifParser(cleaned, occupancy_tolerance=10, site_tolerance=1e-2)
        structure = parser.parse_structures()[0]
        atoms = Atoms(
            symbols=[str(site.species_string) for site in structure.sites],
            positions=np.array(structure.cart_coords, dtype=np.float64),
            cell=np.array(structure.lattice.matrix, dtype=np.float64),
            pbc=True,
        )
        if cleaned != str(cif_path):
            os.remove(cleaned)
        return atoms
    except Exception as e:
        print(f"    pymatgen CifParser failed ({e}), trying ASE fallback ...")

    # Fall back to ASE reader
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = read(str(cif_path))
    print(f"    ASE fallback: read {len(atoms)} atoms")
    return atoms


def resolve_implicit_sp_disorder_via_pymatgen(
    cif_path: Path,
    overlap_cutoff: float = 1.6,
    seed: int = 0,
) -> Atoms:
    """Explicit disorder resolver for CIFs with implicit partial-occupancy sites.

    Used for structures whose disorder is expressed solely through fractional
    occupancy without `_atom_site_disorder_group` / `_atom_site_disorder_assembly`
    markers (e.g. DAP-7's hydrazinium H1C at occ=0.5).  MolCrysKit's
    `DisorderGraphBuilder._add_implicit_sp_conflicts` skips such H atoms when
    they have a full-occupancy bonded non-H neighbour, which drops the required
    conflict edges and leaves the structure with an incorrect stoichiometry
    (N2H6 instead of N2H5 for DAP-7).

    Resolution strategy:
      1. Expand the CIF to P1 with pymatgen, keeping partial-occupancy sites.
      2. Group partial-occupancy sites whose pairwise PBC distance is below
         `overlap_cutoff` — these are symmetry-related copies competing for the
         same physical position.
      3. Keep exactly one representative per cluster, picking the copy with the
         smallest fractional-coordinate tuple (deterministic and
         seed-independent).
      4. Return the resulting ordered ASE Atoms.
    """
    from pymatgen.core import Structure

    cleaned = _sanitize_cif_for_pymatgen(cif_path)
    parser = CifParser(cleaned, occupancy_tolerance=10, site_tolerance=1e-2)
    structure: Structure = parser.parse_structures()[0]
    if cleaned != str(cif_path):
        os.remove(cleaned)

    lattice = np.array(structure.lattice.matrix, dtype=np.float64)
    symbols: list[str] = []
    fracs: list[np.ndarray] = []
    occs: list[float] = []
    for site in structure.sites:
        occ = float(site.species.num_atoms)
        if occ <= 0.0:
            continue
        sp_str = site.species.reduced_formula if len(site.species) > 1 else str(
            list(site.species.keys())[0].symbol
        )
        # In normal cases a partial-occ site is a single element.
        sp = list(site.species.keys())[0].symbol
        symbols.append(sp)
        fracs.append(np.array(site.frac_coords, dtype=np.float64))
        occs.append(occ)

    n = len(symbols)
    frac_arr = np.array(fracs)

    def pbc_dist(i: int, j: int) -> float:
        d = frac_arr[i] - frac_arr[j]
        d = d - np.round(d)
        cart = d @ lattice
        return float(np.linalg.norm(cart))

    partials = [i for i in range(n) if occs[i] < 0.999]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(partials)):
        for j in range(i + 1, len(partials)):
            a, b = partials[i], partials[j]
            if symbols[a] != symbols[b]:
                continue
            if pbc_dist(a, b) < overlap_cutoff:
                union(a, b)

    clusters: dict[int, list[int]] = {}
    for idx in partials:
        clusters.setdefault(find(idx), []).append(idx)

    keep = set(range(n)) - set(partials)
    for root, members in clusters.items():
        occ_sum = sum(occs[m] for m in members)
        n_sites = max(1, round(occ_sum))
        members_sorted = sorted(
            members, key=lambda i: tuple(np.round(frac_arr[i] % 1.0, 5))
        )
        for m in members_sorted[:n_sites]:
            keep.add(m)

    kept = sorted(keep)
    kept_symbols = [symbols[i] for i in kept]
    kept_frac = np.array([frac_arr[i] for i in kept])
    kept_cart = kept_frac @ lattice
    atoms = Atoms(
        symbols=kept_symbols,
        positions=kept_cart,
        cell=lattice,
        pbc=True,
    )
    return atoms


# Materials whose implicit-SP disorder cannot be resolved by MolCrysKit's
# automatic pipeline (partial-occ H atoms bonded to full-occ centers without
# explicit `_atom_site_disorder_group` markers). These are resolved via the
# generic pymatgen-based resolver above.
EXPLICIT_DISORDER_MATERIALS: set[str] = {
    "DAP-7",  # H1C at occ=0.5 without disorder_group → manual resolution
}


def load_clean_crystal(record: PemRecord) -> tuple[MolecularCrystal, bool, int]:
    cif_path = Path(record.cif_path)
    has_partial_occ, partial_atom_count = contains_partial_occupancy(cif_path)
    if record.material in EXPLICIT_DISORDER_MATERIALS and has_partial_occ:
        atoms = resolve_implicit_sp_disorder_via_pymatgen(cif_path)
        return (
            MolecularCrystal.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS),
            True,
            partial_atom_count,
        )
    if has_partial_occ:
        # Try sanitized CIF for disorder resolution
        cleaned = _sanitize_cif_for_pymatgen(cif_path)
        try:
            replicas = generate_ordered_replicas_from_disordered_sites(cleaned, generate_count=1, method="optimal")
        except Exception as e:
            if cleaned != str(cif_path):
                os.remove(cleaned)
            raise PemPrepError(f"MolCrysKit disorder resolution failed for {record.material}: {e}")
        finally:
            if cleaned != str(cif_path) and os.path.exists(cleaned):
                os.remove(cleaned)
        if not replicas:
            raise PemPrepError(f"MolCrysKit returned no ordered replicas for {record.material}")
        # MolCrysKit's disorder pipeline rebuilds the MolecularCrystal internally
        # with DEFAULT bond thresholds, which incorrectly merges heavy-atom
        # oxoanion frameworks (K/Na/I···O) into giant "molecules" and may split
        # PBC-crossing molecules. Re-identify molecules here with our PEM
        # thresholds to get correct, PBC-aware molecule grouping.
        resolved_atoms = replicas[0].to_ase()
        return (
            MolecularCrystal.from_ase(resolved_atoms, bond_thresholds=PEM_BOND_THRESHOLDS),
            True,
            partial_atom_count,
        )

    atoms = parse_ordered_atoms_from_cif(cif_path)
    return MolecularCrystal.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS), False, partial_atom_count


def get_species_counts(crystal: MolecularCrystal) -> dict[str, int]:
    analyzer = StoichiometryAnalyzer(crystal)
    return {species_id: len(indices) for species_id, indices in analyzer.species_map.items()}


# Supercell expansion schedule: try progressively larger supercells
# until we find at least 3 structurally distinct seed candidates.
SUPERCELL_SCHEDULE = [(2, 2, 2), (3, 3, 3), (2, 3, 3), (3, 3, 4), (4, 4, 4)]
MIN_DISTINCT_SEEDS = 3


def _get_seed_com(supercell: MolecularCrystal, analyzer: StoichiometryAnalyzer,
                  seed_species_id: str, mol_index: int) -> np.ndarray:
    """Return centre-of-mass of the molecule that contains *mol_index*."""
    mol = supercell.molecules[mol_index]
    return np.array(mol.get_center_of_mass(), dtype=np.float64)


def _select_spread_seeds(supercell: MolecularCrystal, analyzer: StoichiometryAnalyzer,
                         seed_species_id: str, n: int = 3) -> list[int]:
    """Pick *n* seed molecules of *seed_species_id* that are maximally spread.

    Strategy: sort candidates by distance to centroid, then greedily pick the
    one farthest from the already-selected set.  This avoids picking
    symmetry-equivalent neighbours.
    """
    candidates = list(analyzer.species_map.get(seed_species_id, []))
    if len(candidates) <= n:
        return candidates  # nothing we can do

    coms = np.array([_get_seed_com(supercell, analyzer, seed_species_id, idx) for idx in candidates])
    centroid = coms.mean(axis=0)

    # Start with the molecule closest to the centroid
    dists_to_centroid = np.linalg.norm(coms - centroid, axis=1)
    first = int(np.argmin(dists_to_centroid))
    selected = [first]

    for _ in range(n - 1):
        # Find candidate farthest from ALL already selected
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
    target_spec_override: dict[str, int] | None = None,
) -> tuple[MolecularCrystal, int, tuple[int, ...]]:
    """Build a stoichiometric cluster, expanding the supercell if needed.

    If ``target_spec_override`` is provided, use it verbatim instead of the
    auto-computed simplest unit (see ``CLUSTER_TARGET_SPEC_OVERRIDES``). The
    first key of the override dict is treated as the seed species.

    Returns (cluster_crystal, seed_index, supercell_dims).
    """
    offset = CLUSTER_VARIANT_OFFSETS[dataset_name]

    for sc_dims in SUPERCELL_SCHEDULE:
        supercell = crystal.get_supercell(*sc_dims)
        analyzer = StoichiometryAnalyzer(supercell)
        if target_spec_override is not None:
            target_spec = dict(target_spec_override)
            missing = [sid for sid in target_spec if sid not in analyzer.species_map]
            if missing:
                # Requested species not present in this supercell; try larger.
                continue
        else:
            target_spec = analyzer.get_simplest_unit()

        seed_species_id = next(iter(target_spec.keys()))
        seed_candidates = analyzer.species_map.get(seed_species_id, [])
        if not seed_candidates:
            continue

        # Pick maximally-spread seeds
        spread_seeds = _select_spread_seeds(supercell, analyzer, seed_species_id, n=MIN_DISTINCT_SEEDS)
        if len(spread_seeds) >= MIN_DISTINCT_SEEDS:
            seed_index = spread_seeds[offset % len(spread_seeds)]
            generator = VacancyGenerator(supercell)
            _, removed_cluster = generator.generate_vacancy(
                target_spec=target_spec,
                seed_index=seed_index,
                return_removed_cluster=True,
                random_seed=seed,
            )
            return removed_cluster, seed_index, sc_dims

    # Fallback: use largest supercell tried, pick by offset
    supercell = crystal.get_supercell(*SUPERCELL_SCHEDULE[-1])
    analyzer = StoichiometryAnalyzer(supercell)
    target_spec = (
        dict(target_spec_override)
        if target_spec_override is not None
        else analyzer.get_simplest_unit()
    )
    seed_species_id = next(iter(target_spec.keys()))
    seed_candidates = analyzer.species_map.get(seed_species_id, [])
    if not seed_candidates:
        raise PemPrepError(f"No seed candidates for species {seed_species_id} even in {SUPERCELL_SCHEDULE[-1]} supercell")
    seed_index = seed_candidates[offset % len(seed_candidates)]
    generator = VacancyGenerator(supercell)
    _, removed_cluster = generator.generate_vacancy(
        target_spec=target_spec,
        seed_index=seed_index,
        return_removed_cluster=True,
        random_seed=seed,
    )
    return removed_cluster, seed_index, SUPERCELL_SCHEDULE[-1]


def _centre_cluster_atoms(cluster_crystal: MolecularCrystal) -> Atoms:
    """Convert cluster crystal to ASE Atoms, centre at (50,50,50), apply CLUSTER_BOX."""
    atoms = crystal_to_minimum_image_atoms(cluster_crystal)
    atoms.positions = (
        atoms.get_positions()
        - atoms.get_positions().mean(axis=0, keepdims=True)
        + np.array([[50.0, 50.0, 50.0]], dtype=np.float64)
    )
    atoms.set_cell(CLUSTER_BOX)
    atoms.set_pbc(False)
    return atoms


def _build_cluster_attempt(
    crystal: MolecularCrystal,
    sc_dims: tuple[int, ...],
    target_spec_override: dict[str, int] | None,
    seed_species_id: str | None,
    spread_seed_idx: int,
    rng_seed: int,
    n_spread: int = MAX_SPREAD_SEEDS_FOR_DIVERSITY,
) -> tuple[MolecularCrystal, int, str] | None:
    """Single cluster build attempt at explicit (sc_dims, seed_species, spread_seed_idx).

    Returns ``(cluster_crystal, seed_mol_index, seed_species_id)`` or ``None`` if
    infeasible (species absent, or not enough spread seeds at this index).
    """
    supercell = crystal.get_supercell(*sc_dims)
    analyzer = StoichiometryAnalyzer(supercell)

    if target_spec_override is not None:
        target_spec = dict(target_spec_override)
        if any(sid not in analyzer.species_map for sid in target_spec):
            return None
    else:
        target_spec = analyzer.get_simplest_unit()

    if seed_species_id is None:
        seed_species_id = next(iter(target_spec.keys()))

    if seed_species_id not in analyzer.species_map:
        return None

    spread_seeds = _select_spread_seeds(supercell, analyzer, seed_species_id, n=n_spread)
    if spread_seed_idx >= len(spread_seeds):
        return None

    seed_mol_idx = spread_seeds[spread_seed_idx]
    generator = VacancyGenerator(supercell)
    _, cluster_crystal = generator.generate_vacancy(
        target_spec=target_spec,
        seed_index=seed_mol_idx,
        return_removed_cluster=True,
        random_seed=rng_seed,
    )
    return cluster_crystal, seed_mol_idx, seed_species_id


def build_diverse_cluster_variants(
    material: str,
    crystal: MolecularCrystal,
    target_spec_override: dict[str, int] | None = None,
) -> tuple[list[ClusterBuildResult], dict]:
    """Build n1/n2/n3 cluster variants with post-build geometric diversity checking.

    Escalation order (no coordinate perturbation ever; plan §1.2):
      1. Try next spread seed index (same supercell, same seed species).
      2. Advance to next supercell size in SUPERCELL_SCHEDULE; restart from
         default spread-seed offset for this variant.
      3. Try alternate seed species (next key in target_spec), same supercell
         schedule.
    If all combinations exhausted: accept the duplicate, append to
    ``degenerate_pairs``, and log a warning.

    Returns
    -------
    results       : list of ``(ClusterBuildResult, Atoms)`` pairs, one per dataset
                    (n1/n2/n3). ``ClusterBuildResult.target_m_s`` is set to 0.0 and
                    must be filled in by the caller.
    diversity_info: dict with fingerprint_distances, degenerate_pairs, escalations.
    """
    diversity_info: dict = {
        "fingerprint_distances": {},
        "degenerate_pairs": [],
        "escalations": [],
    }

    dataset_list = list(CLUSTER_DIRS.keys())  # [cluster_n1, cluster_n2, cluster_n3]
    tag_list = ["n1", "n2", "n3"]

    built_fps: dict[str, np.ndarray] = {}
    results: list[ClusterBuildResult] = []

    for dataset_name, tag in zip(dataset_list, tag_list):
        default_offset = CLUSTER_VARIANT_OFFSETS[dataset_name]  # 0, 1, 2
        rng_seed = CLUSTER_RANDOM_SEEDS[dataset_name]

        accepted: tuple[MolecularCrystal, int, tuple[int, ...], str, int] | None = None
        # accepted = (cluster_crystal, seed_mol_idx, sc_dims, seed_species, spread_idx)

        outer_break = False
        for sc_dims in SUPERCELL_SCHEDULE:
            if outer_break:
                break

            # Enumerate seed species for this supercell
            sc_tmp = crystal.get_supercell(*sc_dims)
            an_tmp = StoichiometryAnalyzer(sc_tmp)
            if target_spec_override is not None:
                ts_tmp = dict(target_spec_override)
                if any(sid not in an_tmp.species_map for sid in ts_tmp):
                    continue
                species_list = list(ts_tmp.keys())
            else:
                ts_tmp = an_tmp.get_simplest_unit()
                species_list = list(ts_tmp.keys())

            for sp_idx, seed_species in enumerate(species_list):
                if outer_break:
                    break

                # Compute available spread seeds
                spread = _select_spread_seeds(
                    sc_tmp, an_tmp, seed_species,
                    n=MAX_SPREAD_SEEDS_FOR_DIVERSITY,
                )
                n_avail = len(spread)
                if n_avail == 0:
                    continue

                # For the primary species start at default_offset, else start at 0
                start_idx = default_offset if sp_idx == 0 else 0

                for offset_step in range(n_avail):
                    ss_idx = (start_idx + offset_step) % n_avail
                    attempt = _build_cluster_attempt(
                        crystal, sc_dims, target_spec_override,
                        seed_species, ss_idx, rng_seed,
                    )
                    if attempt is None:
                        continue
                    cluster_crystal, seed_mol_idx, used_species = attempt
                    cluster_atoms = _centre_cluster_atoms(cluster_crystal)
                    fp = cluster_fingerprint(cluster_atoms.get_positions())

                    collides = [
                        pt for pt, pfp in built_fps.items()
                        if fingerprint_distance(fp, pfp) < FINGERPRINT_IDENTICAL_TOL
                    ]

                    if not collides:
                        accepted = (cluster_crystal, seed_mol_idx, sc_dims, used_species, ss_idx)
                        built_fps[tag] = fp
                        outer_break = True
                        break
                    else:
                        diversity_info["escalations"].append({
                            "variant": tag,
                            "attempted_sc_dims": list(sc_dims),
                            "attempted_species": seed_species,
                            "attempted_spread_seed_idx": ss_idx,
                            "collides_with": collides,
                        })

        # Fallback: symmetry-forced duplicate — accept with default parameters
        if accepted is None:
            logger.warning(
                "Material %s %s: could not find a distinct cluster after full "
                "escalation; accepting symmetry-forced duplicate.",
                material, tag,
            )
            fallback_sc = SUPERCELL_SCHEDULE[0]
            attempt = _build_cluster_attempt(
                crystal, fallback_sc, target_spec_override,
                None, default_offset, rng_seed,
            )
            if attempt is None:
                fallback_sc = SUPERCELL_SCHEDULE[-1]
                attempt = _build_cluster_attempt(
                    crystal, fallback_sc, target_spec_override,
                    None, 0, rng_seed,
                )
            if attempt is None:
                raise PemPrepError(
                    f"Material {material} {tag}: fallback cluster build failed."
                )
            cluster_crystal, seed_mol_idx, used_species = attempt
            cluster_atoms = _centre_cluster_atoms(cluster_crystal)
            fp = cluster_fingerprint(cluster_atoms.get_positions())
            built_fps[tag] = fp
            # Record which pairs are degenerate
            for prev_tag, prev_fp in built_fps.items():
                if prev_tag != tag and fingerprint_distance(fp, prev_fp) < FINGERPRINT_IDENTICAL_TOL:
                    pair_str = "_".join(sorted([prev_tag, tag]))
                    if pair_str not in diversity_info["degenerate_pairs"]:
                        diversity_info["degenerate_pairs"].append(pair_str)
            accepted = (cluster_crystal, seed_mol_idx, fallback_sc, used_species, default_offset)

        cluster_crystal, seed_mol_idx, sc_dims, seed_species, ss_idx = accepted
        cluster_atoms = _centre_cluster_atoms(cluster_crystal)
        results.append((
            ClusterBuildResult(
                material=material,
                dataset_name=dataset_name,
                target_m_s=0.0,  # filled in by caller
                random_seed=rng_seed,
                seed_index=seed_mol_idx,
                cluster_atom_count=len(cluster_atoms),
                cluster_molecule_count=len(cluster_crystal.molecules),
                selected_species_counts=get_species_counts(cluster_crystal),
                supercell_dims=sc_dims,
                seed_species=seed_species,
                spread_seed_idx=ss_idx,
            ),
            cluster_atoms,
        ))
        logger.debug(
            "  [%s] %s: %d atoms, seed_idx=%d, supercell=%s, species=%s, ss_idx=%d",
            dataset_name, material,
            results[-1][0].cluster_atom_count, seed_mol_idx,
            sc_dims, seed_species, ss_idx,
        )

    # Compute all pairwise fingerprint distances
    for a, b in [("n1", "n2"), ("n1", "n3"), ("n2", "n3")]:
        if a in built_fps and b in built_fps:
            d = fingerprint_distance(built_fps[a], built_fps[b])
            diversity_info["fingerprint_distances"][f"{a}_{b}"] = round(d, 8)
            if d < FINGERPRINT_IDENTICAL_TOL:
                pair_str = f"{a}_{b}"
                if pair_str not in diversity_info["degenerate_pairs"]:
                    diversity_info["degenerate_pairs"].append(pair_str)

    return results, diversity_info



def crystal_to_minimum_image_atoms(crystal: MolecularCrystal) -> Atoms:
    unwrapped = crystal.get_unwrapped_molecules()
    molecules = [mol.copy() for mol in unwrapped]
    if not molecules:
        raise PemPrepError("Cluster crystal contains no molecules")

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


def write_periodic_system(system_dir: Path, atoms: Atoms, prop: float) -> None:
    set_dir = system_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    coords_cart = np.array(atoms.get_positions(), dtype=np.float64)
    box = np.array(atoms.get_cell(), dtype=np.float64)
    atom_types = np.array([TYPE_TO_ID[s] for s in atoms.get_chemical_symbols()], dtype=np.int32)
    energy = np.array([prop], dtype=np.float64)
    force = np.zeros((1, coords_cart.shape[0], 3), dtype=np.float64)

    np.savetxt(system_dir / "type.raw", atom_types.astype(np.int32), fmt="%d")
    (system_dir / "type_map.raw").write_text("\n".join(TYPE_MAP) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", coords_cart.reshape(1, -1).astype(np.float64))
    np.save(set_dir / "box.npy", box.reshape(1, 9).astype(np.float64))
    np.save(set_dir / "energy.npy", energy)
    np.save(set_dir / "force.npy", force)
    np.save(set_dir / "property.npy", energy.copy())


def write_cluster_system(system_dir: Path, atoms: Atoms, prop: float) -> None:
    set_dir = system_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    coords_cart = np.array(atoms.get_positions(), dtype=np.float64)
    coords_cart = coords_cart - coords_cart.mean(axis=0, keepdims=True) + np.array([[50.0, 50.0, 50.0]], dtype=np.float64)
    atom_types = np.array([TYPE_TO_ID[s] for s in atoms.get_chemical_symbols()], dtype=np.int32)
    energy = np.array([prop], dtype=np.float64)
    force = np.zeros((1, coords_cart.shape[0], 3), dtype=np.float64)

    (system_dir / "nopbc").write_text("", encoding="utf-8")
    np.savetxt(system_dir / "type.raw", atom_types.astype(np.int32), fmt="%d")
    (system_dir / "type_map.raw").write_text("\n".join(TYPE_MAP) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", coords_cart.reshape(1, -1).astype(np.float64))
    np.save(set_dir / "box.npy", CLUSTER_BOX.reshape(1, 9).astype(np.float64))
    np.save(set_dir / "energy.npy", energy)
    np.save(set_dir / "force.npy", force)
    np.save(set_dir / "property.npy", energy.copy())


def write_cluster_check_cif(dataset_name: str, material: str, atoms: Atoms) -> None:
    cluster_cif_dir = CLUSTER_CIF_DIR / dataset_name
    cluster_cif_dir.mkdir(parents=True, exist_ok=True)

    centered = atoms.copy()
    centered.set_cell(CLUSTER_BOX)
    centered.set_pbc(False)
    cif_path = cluster_cif_dir / f"{material}.cif"
    xyz_path = cluster_cif_dir / f"{material}.xyz"
    ase_write(str(cif_path), centered, format="cif")
    ase_write(str(xyz_path), centered, format="xyz")



def build_cluster_variants(material: str, crystal: MolecularCrystal, target_m_s: float) -> list[ClusterBuildResult]:
    results: list[ClusterBuildResult] = []
    target_spec_override = CLUSTER_TARGET_SPEC_OVERRIDES.get(material)
    if target_spec_override is not None:
        print(f"  (using CLUSTER_TARGET_SPEC_OVERRIDES[{material}] = {target_spec_override})")

    for dataset_name, out_dir in CLUSTER_DIRS.items():
        seed = CLUSTER_RANDOM_SEEDS[dataset_name]
        cluster_crystal, seed_index, sc_dims = build_seeded_stoichiometric_cluster(
            crystal,
            dataset_name=dataset_name,
            seed=seed,
            target_spec_override=target_spec_override,
        )
        cluster_atoms = crystal_to_minimum_image_atoms(cluster_crystal)
        cluster_atoms.positions = (
            cluster_atoms.get_positions() - cluster_atoms.get_positions().mean(axis=0, keepdims=True) + np.array([[50.0, 50.0, 50.0]], dtype=np.float64)
        )
        cluster_atoms.set_cell(CLUSTER_BOX)
        cluster_atoms.set_pbc(False)
        write_cluster_system(out_dir / material, cluster_atoms, target_m_s)
        write_cluster_check_cif(dataset_name, material, cluster_atoms)
        print(f"  [{dataset_name}] {material}: {len(cluster_atoms)} atoms, seed_idx={seed_index}, supercell={sc_dims}")
        results.append(
            ClusterBuildResult(
                material=material,
                dataset_name=dataset_name,
                target_m_s=target_m_s,
                random_seed=seed,
                seed_index=seed_index,
                cluster_atom_count=len(cluster_atoms),
                cluster_molecule_count=len(cluster_crystal.molecules),
                selected_species_counts=get_species_counts(cluster_crystal),
                supercell_dims=sc_dims,
            )
        )
    return results


def build_reports(
    crystal_results: list[CrystalBuildResult],
    cluster_results: list[ClusterBuildResult],
    skips: list[SkipRecord],
    cluster_diversity: dict[str, dict] | None = None,
) -> None:
    lines = [
        "# PEMs Cleaning Report",
        "",
        f"Usable systems: **{len(crystal_results)}**",
        f"Skipped systems: **{len(skips)}**",
        "",
        "## Crystal dataset summary",
        "",
        "| Material | Target (m/s) | Disorder cleaned | Partial-occ atoms | Atoms | Molecules | Simplest unit |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for rec in crystal_results:
        lines.append(
            f"| {rec.material} | {rec.target_m_s:.1f} | {rec.used_disorder_cleaning} | {rec.disorder_atom_count} | {rec.cleaned_atom_count} | {rec.cleaned_molecule_count} | `{rec.simplest_unit}` |"
        )
    lines.extend(["", "## Skips", "", "| Material | Reason | CIF |", "|---|---|---|"])
    for skip in skips:
        lines.append(f"| {skip.material} | {skip.reason} | {skip.cif_path or ''} |")
    CLEANING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "# PEMs Cluster Report",
        "",
        "Mapping used: three different MolCrysKit-native stoichiometric ×1 clusters generated from the same simplest-unit target with maximally-spread seed molecules.",
        "Supercell is expanded (2×2×2 → 3×3×3 → ...) until ≥3 distinct seeds are available.",
        "",
        "| Material | Dataset | RNG seed | Seed mol idx | Supercell | Cluster atoms | Cluster mols | Selected species |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for rec in cluster_results:
        sc = "×".join(str(d) for d in rec.supercell_dims)
        lines.append(
            f"| {rec.material} | {rec.dataset_name} | {rec.random_seed} | {rec.seed_index} | {sc} | {rec.cluster_atom_count} | {rec.cluster_molecule_count} | `{rec.selected_species_counts}` |"
        )

    if cluster_diversity:
        lines.extend([
            "",
            "## Cluster diversity summary",
            "",
            "Sorted by minimum pairwise fingerprint distance (ascending). "
            "Materials with `degenerate_pairs` have symmetry-forced identical clusters.",
            "",
            "| Material | fp_n1_n2 | fp_n1_n3 | fp_n2_n3 | min | degenerate_pairs | escalations |",
            "|---|---:|---:|---:|---:|---|---:|",
        ])
        rows = []
        for mat, info in cluster_diversity.items():
            fp = info.get("fingerprint_distances", {})
            d12 = fp.get("n1_n2", float("nan"))
            d13 = fp.get("n1_n3", float("nan"))
            d23 = fp.get("n2_n3", float("nan"))
            valid = [v for v in [d12, d13, d23] if not (v != v)]  # exclude NaN
            min_d = min(valid) if valid else float("nan")
            deg = ", ".join(info.get("degenerate_pairs", [])) or "—"
            n_esc = len(info.get("escalations", []))
            rows.append((min_d, mat, d12, d13, d23, deg, n_esc))
        rows.sort(key=lambda r: r[0] if r[0] == r[0] else float("inf"))
        for min_d, mat, d12, d13, d23, deg, n_esc in rows:
            def _fmt(v):
                return f"{v:.6f}" if v == v else "n/a"
            lines.append(
                f"| {mat} | {_fmt(d12)} | {_fmt(d13)} | {_fmt(d23)} | {_fmt(min_d)} | {deg} | {n_esc} |"
            )

    CLUSTER_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare PEMs crystal and cluster DeepMD datasets."
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        metavar="MATERIAL",
        default=None,
        help="Only process these material names (e.g. --materials DAC-4 DAN-2). "
             "Default: process all eligible materials.",
    )
    parser.add_argument(
        "--bypass-skip",
        action="store_true",
        default=False,
        help="Bypass the SKIP_MATERIALS blacklist. Use this to re-attempt previously "
             "skipped materials (e.g. disorder-affected structures).",
    )
    parser.add_argument(
        "--ood-out-suffix",
        metavar="SUFFIX",
        default=None,
        help="If provided, write cluster systems to pems_cluster_n{1,2,3}_systems_SUFFIX/ "
             "instead of the default directories. Useful for OOD materials to avoid "
             "overwriting the training data.",
    )
    args = parser.parse_args()

    only_materials = set(args.materials) if args.materials else None

    # Optionally redirect output to OOD-specific directories
    crystal_out_dir = CRYSTAL_OUT_DIR
    cluster_dirs = CLUSTER_DIRS
    manifest_path = MANIFEST_PATH
    if args.ood_out_suffix:
        suffix = args.ood_out_suffix
        crystal_out_dir = OUT_ROOT / f"pems_crystal_systems_{suffix}"
        cluster_dirs = {
            k: OUT_ROOT / f"pems_cluster_n{i+1}_systems_{suffix}"
            for i, k in enumerate(CLUSTER_DIRS)
        }
        manifest_path = OUT_ROOT / f"pems_manifest_{suffix}.json"

    for path in [crystal_out_dir, CLEANED_CIF_DIR, CLUSTER_CIF_DIR, *cluster_dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)

    records, skips = load_candidate_records(
        only_materials=only_materials,
        bypass_skip=args.bypass_skip,
    )
    crystal_results: list[CrystalBuildResult] = []
    cluster_results: list[ClusterBuildResult] = []
    cluster_diversity: dict[str, dict] = {}
    runtime_errors: list[SkipRecord] = []

    for i, record in enumerate(records, 1):
        print(f"\n[{i}/{len(records)}] Processing {record.material} (D={record.d_m_s:.0f} m/s) ...")
        try:
            crystal, used_cleaning, disorder_atom_count = load_clean_crystal(record)
            print(f"  Crystal loaded: {len(crystal.molecules)} molecules, disorder_cleaned={used_cleaning}")
            clean_atoms = crystal_to_minimum_image_atoms(crystal)
            write_periodic_system(crystal_out_dir / record.material, clean_atoms, record.d_m_s)
            write_cif(crystal, filename=str(CLEANED_CIF_DIR / f"{record.material}.cif"))

            analyzer = StoichiometryAnalyzer(crystal)
            simplest_unit = analyzer.get_simplest_unit()
            species_counts = get_species_counts(crystal)
            crystal_results.append(
                CrystalBuildResult(
                    material=record.material,
                    dataset_id="pems_crystal",
                    target_m_s=record.d_m_s,
                    source_cif=record.cif_path,
                    used_disorder_cleaning=used_cleaning,
                    disorder_atom_count=disorder_atom_count,
                    cleaned_atom_count=len(clean_atoms),
                    cleaned_molecule_count=len(crystal.molecules),
                    simplest_unit=simplest_unit,
                    species_counts=species_counts,
                )
            )

            # Build cluster variants with diversity guard
            target_spec_override = CLUSTER_TARGET_SPEC_OVERRIDES.get(record.material)
            if target_spec_override is not None:
                print(
                    f"  (using CLUSTER_TARGET_SPEC_OVERRIDES[{record.material}] = "
                    f"{target_spec_override})"
                )
            var_results, div_info = build_diverse_cluster_variants(
                record.material, crystal,
                target_spec_override=target_spec_override,
            )
            cluster_diversity[record.material] = div_info
            if div_info["degenerate_pairs"]:
                print(
                    f"  WARNING: {record.material} has symmetry-forced identical "
                    f"cluster pairs: {div_info['degenerate_pairs']}"
                )
            if div_info["escalations"]:
                print(
                    f"  INFO: {record.material} needed {len(div_info['escalations'])} "
                    f"escalation step(s) to find distinct clusters."
                )

            for var_res, cluster_atoms in var_results:
                out_dir = cluster_dirs[var_res.dataset_name]
                out_dir.mkdir(parents=True, exist_ok=True)
                write_cluster_system(out_dir / record.material, cluster_atoms, record.d_m_s)
                write_cluster_check_cif(var_res.dataset_name, record.material, cluster_atoms)
                var_res.target_m_s = record.d_m_s
                print(
                    f"  [{var_res.dataset_name}] {record.material}: "
                    f"{var_res.cluster_atom_count} atoms, "
                    f"seed_idx={var_res.seed_index}, "
                    f"supercell={var_res.supercell_dims}, "
                    f"species={var_res.seed_species}, ss_idx={var_res.spread_seed_idx}"
                )
                cluster_results.append(var_res)

        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            runtime_errors.append(SkipRecord(material=record.material, reason=f"runtime_error: {exc}", cif_path=record.cif_path))

    all_skips = skips + runtime_errors
    manifest = {
        "source_csv": str(CSV_PATH.relative_to(ROOT)),
        "n_candidates": len(records),
        "n_built_crystal": len(crystal_results),
        "n_cluster_records": len(cluster_results),
        "crystal_results": [asdict(x) for x in crystal_results],
        "cluster_results": [asdict(x) for x in cluster_results],
        "cluster_diversity": cluster_diversity,
        "skips": [asdict(x) for x in all_skips],
        "cluster_mapping": {
            "description": "cluster_n1/n2/n3 use MolCrysKit vacancy generation with the same simplest-unit stoichiometric target and different seed molecules.",
            "variant_offsets": CLUSTER_VARIANT_OFFSETS,
            "random_seeds": CLUSTER_RANDOM_SEEDS,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    build_reports(crystal_results, cluster_results, all_skips, cluster_diversity=cluster_diversity)
    print(f"Built {len(crystal_results)} crystal PEMs systems")
    print(f"Built {len(cluster_results)} cluster PEMs systems")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
