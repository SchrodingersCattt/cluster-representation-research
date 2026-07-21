"""Topology-projection helpers extracted from the ABX4 working scripts."""

from __future__ import annotations

from itertools import product as iprod
from typing import Any

import numpy as np

from molcrys_kit import read_mol_crystal
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer


def classify(crys: Any, sa: StoichiometryAnalyzer) -> tuple[dict[int, str], dict[int, str]]:
    """Classify molecular species and molecule indices as A/B/X sites."""
    species_type: dict[int, str] = {}
    organic: list[int] = []
    for sid, idxs in sa.species_map.items():
        mol = crys.molecules[idxs[0]]
        if "Cl" in set(mol.get_chemical_symbols()):
            species_type[sid] = "X"
        else:
            organic.append(sid)
    if len(organic) >= 2:
        heavy_counts = sorted(
            organic,
            key=lambda sid: sum(
                1
                for symbol in crys.molecules[sa.species_map[sid][0]].get_chemical_symbols()
                if symbol != "H"
            ),
        )
        species_type[heavy_counts[0]] = "B"
        for sid in heavy_counts[1:]:
            species_type[sid] = "A"
    elif organic:
        species_type[organic[0]] = "B"

    mol_type: dict[int, str] = {}
    for sid, idxs in sa.species_map.items():
        for idx in idxs:
            mol_type[idx] = species_type.get(sid, "?")
    return species_type, mol_type


def get_centroids(crys: Any, mol_type: dict[int, str]) -> tuple[np.ndarray, list[str]]:
    """Return molecule centroids and their A/B/X labels."""
    centroids = np.array([mol.get_centroid() for mol in crys.molecules])
    types = [mol_type.get(idx, "?") for idx in range(len(crys.molecules))]
    return centroids, types


def supercell_cart(
    crys: Any,
    centroids: np.ndarray,
    types: list[str],
    rng: tuple[int, int] = (-1, 2),
) -> tuple[np.ndarray, list[str]]:
    """Replicate centroids over a small Cartesian supercell."""
    lat = crys.lattice
    sc, st = [], []
    for na, nb, nc in iprod(range(*rng), repeat=3):
        offset = na * lat[0] + nb * lat[1] + nc * lat[2]
        for pt, typ in zip(centroids, types):
            sc.append(pt + offset)
            st.append(typ)
    return np.array(sc), st


def find_b_shell(
    centroids: np.ndarray,
    types: list[str],
    sc_pts: np.ndarray,
    sc_types: list[str],
    b_index: int,
    n_x: int = 10,
) -> list[int]:
    """Return nearest X neighbours around a B-site centroid."""
    b_center = centroids[b_index]
    distances = [
        (np.linalg.norm(sc_pts[idx] - b_center), idx)
        for idx, typ in enumerate(sc_types)
        if typ == "X" and np.linalg.norm(sc_pts[idx] - b_center) > 0.1
    ]
    distances.sort()
    return [idx for _, idx in distances[:n_x]]


def pick_layer_proj(
    crys: Any,
    centroids: np.ndarray,
    types: list[str],
) -> tuple[int, list[int]]:
    """Pick stacking and in-plane axes for a 2D topology projection."""
    lat = crys.lattice
    inv_m = np.linalg.inv(lat.T)
    b_frac = np.array([inv_m @ centroids[idx] for idx, typ in enumerate(types) if typ == "B"])
    if len(b_frac) < 2:
        return 0, [1, 2]
    best_axis, best_gap = 0, -1.0
    for axis in range(3):
        vals = sorted(b_frac[:, axis] % 1.0)
        gaps = [vals[idx + 1] - vals[idx] for idx in range(len(vals) - 1)]
        gaps.append(1 - vals[-1] + vals[0])
        if max(gaps) > best_gap:
            best_gap = float(max(gaps))
            best_axis = axis
    in_plane = sorted([axis for axis in range(3) if axis != best_axis])
    return best_axis, in_plane


def rotate2d(points: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate 2D points by ``theta_deg`` degrees."""
    pts = np.atleast_2d(points)
    theta = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return pts @ rot.T


def _ang_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def best_proj_rotation(vh: np.ndarray, vv: np.ndarray) -> float:
    """Rotate projected lattice vectors to a stable upright orientation."""
    vh = np.asarray(vh, dtype=float)
    vv = np.asarray(vv, dtype=float)
    candidates: list[tuple[float, float]] = []
    for primary_name, primary in (("h", vh), ("v", vv)):
        angle = np.degrees(np.arctan2(primary[1], primary[0]))
        for target in (0.0, 90.0, 180.0, -90.0):
            rot = target - angle
            rvh = rotate2d(vh, rot)[0]
            rvv = rotate2d(vv, rot)[0]
            ah = np.degrees(np.arctan2(rvh[1], rvh[0]))
            av = np.degrees(np.arctan2(rvv[1], rvv[0]))

            h_axis_err = min(_ang_diff_deg(ah, t) for t in (0.0, 180.0))
            v_axis_err = min(_ang_diff_deg(av, t) for t in (90.0, -90.0))
            h_vert_err = min(_ang_diff_deg(ah, t) for t in (90.0, -90.0))
            v_horz_err = min(_ang_diff_deg(av, t) for t in (0.0, 180.0))
            ortho_err = min(h_axis_err + v_axis_err, h_vert_err + v_horz_err)

            primary_upright_bonus = 0.0
            if primary_name == "h" and abs(rvh[0]) < 1e-8:
                primary_upright_bonus += 1.0
            if primary_name == "v" and abs(rvv[1]) < 1e-8:
                primary_upright_bonus += 1.0

            upward_penalty = 0.0
            if abs(rvv[1]) >= abs(rvv[0]) and rvv[1] < 0:
                upward_penalty += 2.0
            if abs(rvh[0]) >= abs(rvh[1]) and rvh[0] < 0:
                upward_penalty += 1.0

            tilt_balance = abs(abs(ah) % 90.0) + abs(abs(av) % 90.0)
            score = 6.0 * ortho_err + 0.18 * tilt_balance + upward_penalty + primary_upright_bonus
            candidates.append((score, float(rot)))

    candidates.sort(key=lambda row: (row[0], abs(row[1])))
    return float(candidates[0][1]) if candidates else 0.0


__all__ = [
    "StoichiometryAnalyzer",
    "best_proj_rotation",
    "classify",
    "find_b_shell",
    "get_centroids",
    "pick_layer_proj",
    "read_mol_crystal",
    "rotate2d",
    "supercell_cart",
]
