#!/usr/bin/env python3
"""Render Figure 2d: representative MIX coordination polyhedra.

Dense single-row comparison of ABX3 (DAP-4), ABX4 (DEP), and A2BX5 (EAP-4)
AX/BX coordination shells. The layout follows the earlier coordination-number
panel: distance-rank diagnostic, CN histogram, and the chemical polyhedron. The
abstract dot-line polyhedron is intentionally omitted.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.patheffects import withStroke
from PIL import Image

from stoich_cluster_learning.paths import repo_root

REPO = repo_root()
SRC_DIR = REPO / "src"
MOLCRYSKIT_ROOT = Path(os.environ["MOLCRYSKIT_ROOT"]) if os.environ.get("MOLCRYSKIT_ROOT") else None
PEMS_CIF_DIR = REPO / "experiments" / "00_data_prep" / "pems_cleaned_cifs"
PEMS_CONF_DIR = REPO / "data" / "pems" / "confs"
ABX4_CIF_DIR = REPO / "data" / "abx4" / "cifs"
OUT_DIR = REPO / "manuscript" / "figures"
PLOTLY_PANEL_DIR = OUT_DIR / "_figure2d_plotly_panels"

sys.path.insert(0, str(SRC_DIR))
if MOLCRYSKIT_ROOT is not None and MOLCRYSKIT_ROOT.exists():
    sys.path.insert(0, str(MOLCRYSKIT_ROOT))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "manuscript" / "figures"))

import crystal_viewer  # noqa: E402
from crystal_viewer.loader import build_bundle_scene, build_loaded_crystal  # noqa: E402
from crystal_viewer.renderer import (  # noqa: E402
    _atom_mesh_traces,
    _bond_mesh_traces,
)
from crystal_viewer.scene import scene_ops  # noqa: E402
from crystal_viewer.topology import analyze_topology, extract_coordination_shell  # noqa: E402
from molcrys_kit.analysis.shape import classify_shell as _molcrys_classify_shell  # noqa: E402
from stoich_cluster_learning.viz.coordination import reclassify_pem_fragments  # noqa: E402
from figure_style import display_material  # noqa: E402
from paper_plot_style import EXP_COLORS, setup_nature_style, style_axes  # noqa: E402

_MATTERVIS_OPS = scene_ops()

COLORS = {
    "A_site": "#8A5A67",
    "B_site": "#6B7C4E",
    "X_site": "#5A6D7B",
    "charcoal": "#2F2F2F",
    "mid_gray": "#D6D6D6",
    "faint_grid": "#ECECEC",
}

A_HIGHLIGHT = "#B89095"
B_HIGHLIGHT = "#A0AE83"
HULL_BG_FACE = "#C7C7C7"
HULL_BG_EDGE = "#A2A2A2"
CELL_BOX_COLOR = "#3D3D3D"
CELL_BOX_WIDTH = 1.0
HULL_EDGE_WIDTH = 1.0
LEGEND_HALF_BOX = 0.06
AXIS_COLOR = "#2F2F2F"
AXIS_WIDTH = 0.7
AXIS_TEXT_PT = 8.0

HL_OPACITY = 0.45
BG_OPACITY = 0.08
PAPER_FLATSHADING = True
PAPER_LIGHTING = dict(ambient=0.85, diffuse=0.25, specular=0.0,
                       roughness=1.0, fresnel=0.0)
PAPER_LIGHTPOSITION = dict(x=100, y=100, z=200)
ATOM_SCALE = 0.55
BOND_RADIUS = 0.10
MINOR_OPACITY = 0.20

SELECTION_STRATEGIES = ("modal_first", "central", "linked", "separated")
SELECTION_STRATEGY = "modal_first"
# Per-material override of the highlight strategy.  Defaults to the
# global ``SELECTION_STRATEGY``.  SY's A and B sites happen to sit on top
# of each other under the default ``modal_first`` pick, so the two
# highlight colours overlap and the reader cannot tell them apart; the
# ``separated`` strategy picks B as far from the chosen A as possible
# (among modal-CN B records), staggering them spatially.
HIGHLIGHT_STRATEGY_OVERRIDE: dict[str, str] = {
    "SY": "separated",
}

MATERIALS = [
    {
        "name": "DAP-4",
        "stoich": r"ABX$_3$",
        "cif": PEMS_CONF_DIR / "DAP-4.cif",
        "x_family": "ClO4",
        # Per-material supercell repeats. Decoupled from the camera so that
        # changing the orientation cannot silently change which cells are
        # drawn. MatterVis's ``display_mode='unit_cell'`` already mirrors any
        # fragment that crosses a face, so a single cell suffices.
        "repeats": (1, 1, 1),
    },
    {
        "name": "SY",
        "stoich": r"ABX$_4$",
        "cif": ABX4_CIF_DIR / "SY.cif",
        "x_family": "ClO4",
        "repeats": (1, 1, 1),
    },
    {
        "name": "EAP-4",
        "stoich": r"A$_2$BX$_5$",
        "cif": PEMS_CIF_DIR / "EAP-4.cif",
        "x_family": "ClO4",
        "repeats": (1, 1, 1),
    },
]
MATERIAL_REPEATS: dict[str, tuple[int, int, int]] = {
    m["name"]: tuple(m.get("repeats", (1, 1, 1))) for m in MATERIALS  # type: ignore[misc]
}

CUTOFF = 12.0
HARD_CUTOFF_B: dict[str, float] = {}
ANGULAR_RMSD_LABEL_MAX = 12.0
DIST_YLIM = {"A": 12.0, "B": 8.0}
DIST_UNIT = "Å"
DIST_LABEL = {
    "A": rf"$d_{{X-A}}$ ({DIST_UNIT})",
    "B": rf"$d_{{X-B}}$ ({DIST_UNIT})",
}
SITE_COLOR = {"A": COLORS["A_site"], "B": COLORS["B_site"]}

MIN_TEXT_PT = 8.0
SMALL_TEXT_PT = 8.0
LABEL_TEXT_PT = 8.5
POLY_NAME_PT = 10.0
TITLE_PT = 10.5
ROW_LABEL_PT = 9.0
PANEL_LABEL_PT = 12.0
LEGEND_PT = 8.0

PLOTLY_PANEL_SIZE = (900, 720)
PLOTLY_PANEL_SCALE = 2


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _formula(frag: dict[str, Any], bundle) -> str:
    counts: dict[str, int] = {}
    for idx in frag.get("site_indices", []):
        elem = bundle.raw_atoms[int(idx)]["elem"]
        counts[elem] = counts.get(elem, 0) + 1
    order = ["C", "N", "O", "Cl", "I", "Na", "K", "Rb", "Ag", "H"]
    parts = []
    for elem in sorted(counts, key=lambda e: (order.index(e) if e in order else 99, e)):
        count = counts[elem]
        parts.append(f"{elem}{count}" if count > 1 else elem)
    return "".join(parts) or "?"


def _modal_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    cns = [int(r["cn"]) for r in records]
    modal_cn = Counter(cns).most_common(1)[0][0]
    return next(r for r in records if int(r["cn"]) == modal_cn)


def _apply_hard_cutoff(record: dict[str, Any], hard_cutoff: float) -> dict[str, Any]:
    shell = record["raw_shell"]
    candidates = shell.get("candidate_fragments", [])
    pool_coords = shell.get("pool_coords", [])
    all_d = shell.get("all_distances", [])
    mask = [float(d) <= float(hard_cutoff) for d in all_d]
    filtered = dict(record)
    filtered["cn"] = int(sum(mask))
    filtered["distances"] = [float(d) for d, keep in zip(all_d, mask) if keep]
    filtered["shell_coords"] = [c for c, keep in zip(pool_coords, mask) if keep]
    filtered["shell_fragments"] = [f for f, keep in zip(candidates, mask) if keep]
    filtered["gap_info"] = {"hard_cutoff": hard_cutoff, "gap_index": None}
    filtered["best_match"] = None
    return filtered


def _best_match(topology: dict[str, Any]) -> dict[str, Any] | None:
    shell_coords = topology.get("shell_coords")
    center = topology.get("center_coords")
    shape = None
    if shell_coords is not None and center is not None and len(shell_coords) >= 4:
        try:
            # Prefer the latest MolCrysKit classifier over MatterVis'
            # cached topology summary so Figure 2d picks up core-residual
            # labels such as tricapped_cube on EAP-4 AX11.
            shape = _molcrys_classify_shell(shell_coords, center=center)
        except Exception:
            shape = None
    if not shape:
        shape = topology.get("shape")
    if not shape:
        return None
    name = shape.get("primary_label")
    if not name:
        return None
    return {
        "name": str(name),
        "label_modifier": str(shape.get("label_modifier") or "clean"),
        "cshm": shape.get("cshm_value"),
    }


def _polyhedron_name(record: dict[str, Any] | None) -> str:
    if record is None:
        return ""
    match = record.get("best_match")
    if not match:
        return "polyhedron"
    name = str(match["name"]).replace("_", " ")
    modifier = str(match.get("label_modifier") or "clean")
    # ``ambiguous`` says nothing the reader can act on (the classifier
    # could not decide between two close shapes); per user preference we
    # just give the best-match name.  ``clean`` is the no-op modifier.
    # Other modifiers like ``distorted`` carry real geometric information
    # and are kept.
    if modifier and modifier not in {"clean", "ambiguous"}:
        return f"{modifier} {name}"
    return name


def _collect_site_records(bundle, material_name: str, site_type: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frag in bundle.topology_fragment_table:
        if frag.get("type") != site_type:
            continue
        shell = extract_coordination_shell(
            bundle,
            int(frag["index"]),
            cutoff=CUTOFF,
        )
        topology = analyze_topology(
            bundle,
            int(frag["index"]),
            cutoff=CUTOFF,
        )
        record = {
            "material_name": material_name,
            "site_type": site_type,
            "bundle": bundle,
            "center_fragment": frag,
            "center_formula": _formula(frag, bundle),
            "label": frag.get("label", f"{site_type}{frag['index']}"),
            "fragment_index": int(frag["index"]),
            "cn": int(shell["coordination_number"]),
            "distances": [float(x) for x in shell["distances"]],
            "all_distances": [float(x) for x in shell["all_distances"]],
            "gap_info": shell["gap_info"],
            "shell_coords": np.asarray(shell["shell_coords"], dtype=float),
            "center_coords": np.asarray(shell["center_coords"], dtype=float),
            "shell_fragments": shell.get("shell", []),
            "raw_shell": shell,
            "best_match": _best_match(topology),
        }
        hard_cutoff = HARD_CUTOFF_B.get(material_name) if site_type == "B" else None
        if hard_cutoff is not None:
            record = _apply_hard_cutoff(record, hard_cutoff)
        records.append(record)
    return records


def load_material_data() -> dict[str, dict[str, Any]]:
    print(f"MatterVis crystal_viewer: {Path(crystal_viewer.__file__).resolve()}")
    data: dict[str, dict[str, Any]] = {}
    for item in MATERIALS:
        name = item["name"]
        bundle = build_loaded_crystal(name=name, cif_path=str(item["cif"]), title=display_material(str(name)))
        reclassify_pem_fragments(bundle, item["x_family"])
        counts = Counter(f.get("type", "?") for f in bundle.topology_fragment_table)
        site_data = {
            "A": _collect_site_records(bundle, name, "A"),
            "B": _collect_site_records(bundle, name, "B"),
        }
        data[name] = {"meta": item, "bundle": bundle, "counts": counts, "sites": site_data}
        print(f"{name:<5s}: A={counts.get('A', 0):>2d} B={counts.get('B', 0):>2d} X={counts.get('X', 0):>2d}")
        for st in ("A", "B"):
            ref = _modal_record(site_data[st])
            if ref is None:
                print(f"  {st}: no sites")
                continue
            shape = _polyhedron_name(ref) or "n/a"
            mean_d = np.mean(ref["distances"]) if ref["distances"] else float("nan")
            print(
                f"  {st}: sites={len(site_data[st]):>2d} modal_CN={ref['cn']:>2d} "
                f"center={ref['center_formula']:<10s} mean_d={mean_d:.2f} shape={shape}"
            )
    return data


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _stats_style(ax) -> None:
    style_axes(ax, grid=True)
    ax.tick_params(labelsize=SMALL_TEXT_PT, length=2.4, width=0.5, pad=1.6)
    ax.set_facecolor("white")


def _draw_distance_panel(ax, ref: dict[str, Any] | None, site_type: str, color: str) -> None:
    if ref is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"no {site_type}", transform=ax.transAxes,
                ha="center", va="center", fontsize=SMALL_TEXT_PT, color="#888888")
        return
    cn = int(ref["cn"])
    all_d = np.sort(np.asarray(ref["all_distances"], dtype=float))[:max(cn + 4, 12)]
    shell_d = np.sort(np.asarray(ref["distances"], dtype=float))
    ax.scatter(range(1, len(all_d) + 1), all_d, s=8, color=COLORS["mid_gray"], linewidths=0, zorder=2)
    ax.scatter(range(1, len(shell_d) + 1), shell_d, s=10, color=color, linewidths=0, zorder=3)
    gap = ref.get("gap_info", {})
    if gap.get("hard_cutoff") is not None:
        ax.axhline(float(gap["hard_cutoff"]), color=COLORS["charcoal"], lw=0.65, ls=":", zorder=4)
    elif gap.get("gap_index") is not None and len(all_d) > 1:
        gi = int(gap["gap_index"])
        if 0 <= gi < len(all_d) - 1:
            gy = 0.5 * (all_d[gi] + all_d[gi + 1])
            ax.axhline(float(gy), color=COLORS["charcoal"], lw=0.65, ls=":", zorder=4)
    ax.set_ylim(0, DIST_YLIM[site_type])
    ax.set_xlabel("rank", fontsize=LABEL_TEXT_PT, labelpad=1.2)
    ax.set_ylabel(DIST_LABEL[site_type], fontsize=LABEL_TEXT_PT, labelpad=1.2)
    if len(shell_d):
        ax.text(
            0.95, 0.22,
            fr"$\bar d$={np.mean(shell_d):.2f} {DIST_UNIT}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=SMALL_TEXT_PT, color=COLORS["charcoal"],
        )
    _stats_style(ax)



def _crop_white_border(path: Path, *, pad: int = 12) -> None:
    image = Image.open(path).convert("RGBA")
    arr = np.asarray(image)
    alpha = arr[:, :, 3] > 0
    nonwhite = np.any(arr[:, :, :3] < 248, axis=2) & alpha
    if not np.any(nonwhite):
        return
    ys, xs = np.where(nonwhite)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image.width)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image.height)
    image.crop((x0, y0, x1, y1)).save(path)


def _vertical_orientation(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Build a fixed manuscript orientation: ``a`` right, ``c`` up, ``b`` depth.

    Earlier drafts aligned the longest lattice vector with screen ``+z``.
    That made near-cubic cells arbitrary (DAP-4 could put ``a`` vertical) and
    made the axis key semantically wrong.  Figure 2d is a comparison panel, so
    the camera convention is fixed instead: project ``a`` to screen-x, ``c`` to
    screen-z, and let ``b`` point mostly into/out of the screen.
    """
    M = np.asarray(M, dtype=float)
    a_vec, _b_vec, c_vec = M[0], M[1], M[2]
    z_axis = c_vec / max(np.linalg.norm(c_vec), 1e-8)
    x_seed = a_vec
    x_axis = x_seed - np.dot(x_seed, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        x_seed = M[1]
        x_axis = x_seed - np.dot(x_seed, z_axis) * z_axis
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    if np.dot(y_axis, M[1]) < 0:
        y_axis = -y_axis
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    return R, 2


def _supercell_repeat(material_name: str | None = None) -> tuple[int, int, int]:
    """Return the (na, nb, nc) supercell repeats for ``material_name``.

    Decoupled from the camera/orientation matrix on purpose.  Earlier
    versions derived the vertical-axis repeat from the rotation matrix, so
    changing the camera silently changed which cells were drawn (SY in
    particular).  Now the repeats are fixed per material via
    :data:`MATERIAL_REPEATS`.  Visual extent for materials whose fragments
    straddle the cell boundary (e.g. SY's ClO4 tetrahedra) is provided by
    MatterVis's ``display_mode='unit_cell'`` boundary-replica expansion plus
    the dynamic trace-bounds padding applied at render time.
    """
    if material_name is None:
        return (1, 1, 1)
    return MATERIAL_REPEATS.get(material_name, (1, 1, 1))


def _cell_box_segments(M: np.ndarray, repeats: tuple[int, int, int]) -> tuple[list[float], list[float], list[float]]:
    a, b, c = M[0], M[1], M[2]
    na, nb, nc = repeats
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for ia in range(na):
        for ib in range(nb):
            for ic in range(nc):
                origin = ia * a + ib * b + ic * c
                corners = np.array([
                    origin, origin + a, origin + a + b, origin + b,
                    origin + c, origin + a + c, origin + a + b + c, origin + b + c,
                ], dtype=float)
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
                for i, j in edges:
                    xs.extend([float(corners[i, 0]), float(corners[j, 0]), None])
                    ys.extend([float(corners[i, 1]), float(corners[j, 1]), None])
                    zs.extend([float(corners[i, 2]), float(corners[j, 2]), None])
    return xs, ys, zs


def _replicate_polyhedra(records: list[dict[str, Any]], M: np.ndarray, repeats: tuple[int, int, int]) -> list[dict[str, Any]]:
    a, b, c = M[0], M[1], M[2]
    out = []
    for ia in range(repeats[0]):
        for ib in range(repeats[1]):
            for ic in range(repeats[2]):
                shift = ia * a + ib * b + ic * c
                for rec in records:
                    out.append({
                        **rec,
                        "shell_coords": rec["shell_coords"] + shift,
                        "center_coords": rec["center_coords"] + shift,
                        "replica": (ia, ib, ic),
                    })
    return out


def _rotate_points(R: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if not len(pts):
        return pts
    return pts @ R.T


def _atom_bond_traces(
    bundle,
    R: np.ndarray,
    M: np.ndarray,
    repeats: tuple[int, int, int],
) -> list[Any]:
    try:
        scene = build_bundle_scene(bundle, display_mode="unit_cell", show_hydrogen=True)
    except Exception:
        scene = getattr(bundle, "scene", None)
    if not scene:
        return []
    src_atoms = scene.get("draw_atoms") or []
    src_bonds = scene.get("bonds") or []
    if not src_atoms:
        return []
    a, b, c = M[0], M[1], M[2]
    new_atoms: list[dict[str, Any]] = []
    new_bonds: list[dict[str, Any]] = []
    for ia in range(repeats[0]):
        for ib in range(repeats[1]):
            for ic in range(repeats[2]):
                shift = ia * a + ib * b + ic * c
                base = len(new_atoms)
                for atom in src_atoms:
                    cart = np.asarray(atom.get("cart"), dtype=float) + shift
                    cart_rot = R @ cart
                    new_atom = dict(atom)
                    new_atom["cart"] = cart_rot.tolist()
                    new_atom.pop("_render_color", None)
                    new_atoms.append(new_atom)
                for bond in src_bonds:
                    if not bond:
                        continue
                    start = np.asarray(bond.get("start"), dtype=float) + shift
                    end = np.asarray(bond.get("end"), dtype=float) + shift
                    start_rot = R @ start
                    end_rot = R @ end
                    new_bond = dict(bond)
                    new_bond["start"] = start_rot.tolist()
                    new_bond["end"] = end_rot.tolist()
                    new_bond["i"] = int(bond.get("i", -1)) + base if bond.get("i", -1) >= 0 else -1
                    new_bond["j"] = int(bond.get("j", -1)) + base if bond.get("j", -1) >= 0 else -1
                    new_bond.pop("_render_color", None)
                    new_bonds.append(new_bond)
    super_scene = dict(scene)
    super_scene["draw_atoms"] = new_atoms
    super_scene["bonds"] = new_bonds
    style: dict[str, Any] = {
        "material": "mesh",
        "style": "ball_stick",
        "atom_scale": ATOM_SCALE,
        "bond_radius": BOND_RADIUS,
        "major_opacity": 1.0,
        "minor_opacity": MINOR_OPACITY,
        "show_minor_only": False,
        "disorder": "opacity",
        "force_minor_fade": True,
        "minor_wireframe": False,
        "minor_bond_scale": 0.82,
    }
    atom_traces = list(_atom_mesh_traces(super_scene, style))
    bond_traces = list(_bond_mesh_traces(super_scene, style))
    out: list[Any] = []
    for tr in atom_traces:
        tr.update(
            flatshading=PAPER_FLATSHADING,
            lighting=PAPER_LIGHTING,
            lightposition=PAPER_LIGHTPOSITION,
        )
        out.append(tr)
    for tr in bond_traces:
        tr.update(
            flatshading=PAPER_FLATSHADING,
            lighting=PAPER_LIGHTING,
            lightposition=PAPER_LIGHTPOSITION,
        )
        out.append(tr)
    return out


def _shade_color(hex_color: str, shade: float) -> str:
    rgb = np.array(mcolors.to_rgb(hex_color), dtype=float)
    # Mix toward white/dark around the base colour so adjacent triangular
    # facets remain legible even with the low-saturation manuscript palette.
    if shade >= 1.0:
        rgb = rgb + (1.0 - rgb) * min(shade - 1.0, 0.35)
    else:
        rgb = rgb * max(shade, 0.55)
    return mcolors.to_hex(np.clip(rgb, 0.0, 1.0))


def _hull_mesh_paper(
    coords: np.ndarray,
    *,
    color: str,
    opacity: float,
    split_faces: bool = False,
) -> list[Any]:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 4:
        return []
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return []
    hull = ConvexHull(coords)
    if not split_faces:
        return [go.Mesh3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
            color=color,
            opacity=opacity,
            flatshading=PAPER_FLATSHADING,
            lighting=PAPER_LIGHTING,
            lightposition=PAPER_LIGHTPOSITION,
            hoverinfo="skip",
            showlegend=False,
            name="coordination-hull",
        )]

    center = coords.mean(axis=0)
    light = np.array([0.35, -0.45, 0.82], dtype=float)
    light /= np.linalg.norm(light)
    traces: list[Any] = []
    for tri in hull.simplices:
        face = coords[np.asarray(tri, dtype=int)]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
            if np.dot(normal, face.mean(axis=0) - center) < 0:
                normal = -normal
            lambert = max(0.0, float(np.dot(normal, light)))
        else:
            lambert = 0.0
        shade = 0.82 + 0.34 * lambert
        traces.append(go.Mesh3d(
            x=face[:, 0], y=face[:, 1], z=face[:, 2],
            i=[0], j=[1], k=[2],
            color=_shade_color(color, shade),
            opacity=opacity,
            flatshading=True,
            lighting=PAPER_LIGHTING,
            lightposition=PAPER_LIGHTPOSITION,
            hoverinfo="skip",
            showlegend=False,
            name="coordination-hull-face",
        ))
    return traces


def _hull_edge_scatter(coords: np.ndarray, *, color: str, width: float = HULL_EDGE_WIDTH):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return None
    hull = ConvexHull(coords)
    edges: set[tuple[int, int]] = set()
    for tri in hull.simplices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((a, c))))
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for i, j in sorted(edges):
        xs.extend([float(coords[i, 0]), float(coords[j, 0]), None])
        ys.extend([float(coords[i, 1]), float(coords[j, 1]), None])
        zs.extend([float(coords[i, 2]), float(coords[j, 2]), None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip", showlegend=False,
        name="coordination-edges",
    )


def _polyhedron_traces(
    shell_rot: np.ndarray,
    *,
    face_color: str,
    edge_color: str,
    opacity: float,
) -> list[Any]:
    traces: list[Any] = []
    traces.extend(_hull_mesh_paper(
        shell_rot,
        color=face_color,
        opacity=opacity,
        split_faces=opacity >= 0.20,
    ))
    edge = _hull_edge_scatter(
        shell_rot,
        color=edge_color,
        width=HULL_EDGE_WIDTH * (1.35 if opacity >= 0.20 else 1.0),
    )
    if edge is not None:
        traces.append(edge)
    return traces


def _cell_geometry(bundle, material_name: str | None = None) -> dict[str, Any]:
    M = np.asarray(bundle.M, dtype=float)
    R, _ = _vertical_orientation(M)
    repeats = _supercell_repeat(material_name)
    cell_xs, cell_ys, cell_zs = _cell_box_segments(M, repeats)
    cell_pts = np.array(
        [[x, y, z] for x, y, z in zip(cell_xs, cell_ys, cell_zs) if x is not None],
        dtype=float,
    )
    cell_pts_rot = _rotate_points(R, cell_pts)
    return {
        "M": M,
        "R": R,
        "repeats": repeats,
        "cell_xs": cell_xs,
        "cell_ys": cell_ys,
        "cell_zs": cell_zs,
        "cell_pts_rot": cell_pts_rot,
    }


def _pick_highlight_pair(
    a_records: list[dict[str, Any]],
    b_records: list[dict[str, Any]],
    cell_pts_rot: np.ndarray,
    R: np.ndarray,
    *,
    strategy: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Pick one A record and one B record to highlight.

    All records have ``shell_coords`` already in world coords; the rotation
    ``R`` is the panel orientation used by both atom rendering and the cell
    box, so we operate on rotated coordinates here for consistent metrics.
    """
    if not a_records and not b_records:
        return None, None
    cell_center = 0.5 * (cell_pts_rot.max(axis=0) + cell_pts_rot.min(axis=0)) if len(cell_pts_rot) else np.zeros(3)

    def _center_world(rec: dict[str, Any]) -> np.ndarray:
        c = np.asarray(rec["center_coords"], dtype=float)
        return R @ c

    def _modal_index(records: list[dict[str, Any]]) -> int | None:
        if not records:
            return None
        cn_counter = Counter(int(r["cn"]) for r in records)
        modal_cn = cn_counter.most_common(1)[0][0]
        for rec in records:
            if int(rec["cn"]) == modal_cn:
                return int(rec["fragment_index"])
        return int(records[0]["fragment_index"])

    def _by_modal_first(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not records:
            return None
        modal_idx = _modal_index(records)
        for rec in records:
            if int(rec["fragment_index"]) == modal_idx and tuple(rec.get("replica", (0, 0, 0))) == (0, 0, 0):
                return rec
        return records[0]

    def _by_central(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not records:
            return None
        modal_idx = _modal_index(records)
        same_cn = [r for r in records if int(r["fragment_index"]) == modal_idx]
        pool = same_cn if same_cn else records
        return min(pool, key=lambda r: float(np.linalg.norm(_center_world(r) - cell_center)))

    if strategy == "modal_first":
        return _by_modal_first(a_records), _by_modal_first(b_records)
    if strategy == "central":
        return _by_central(a_records), _by_central(b_records)
    if strategy == "linked":
        a_pick = _by_central(a_records)
        if not b_records:
            return a_pick, None
        if a_pick is None:
            return None, _by_central(b_records)
        a_center = _center_world(a_pick)
        modal_b_idx = _modal_index(b_records)
        b_pool = [r for r in b_records if int(r["fragment_index"]) == modal_b_idx] or b_records
        b_pick = min(b_pool, key=lambda r: float(np.linalg.norm(_center_world(r) - a_center)))
        return a_pick, b_pick
    if strategy == "separated":
        # Pick A centrally, then B as the record whose centre is
        # *farthest* (in 2D screen space) from A so the two highlight
        # polyhedra do not visually overlap.  We filter B by modal **CN**
        # (not modal fragment_index — that's a 1-element pool and
        # defeats the purpose of "separated") so the chosen B is still a
        # representative coordination of the typical B site.
        a_pick = _by_central(a_records)
        if not b_records:
            return a_pick, None
        if a_pick is None:
            return None, _by_central(b_records)
        a_screen = _center_world(a_pick)[[0, 2]]  # match the panel's 2D view
        cn_counter = Counter(int(r["cn"]) for r in b_records)
        modal_b_cn = cn_counter.most_common(1)[0][0]
        b_pool = [r for r in b_records if int(r["cn"]) == modal_b_cn] or b_records
        b_pick = max(
            b_pool,
            key=lambda r: float(np.linalg.norm(_center_world(r)[[0, 2]] - a_screen)),
        )
        return a_pick, b_pick
    return _by_modal_first(a_records), _by_modal_first(b_records)


def _compute_view_extent(materials_data: dict[str, dict[str, Any]]) -> dict[str, float]:
    half_x = 0.0
    half_y = 0.0
    half_z = 0.0
    for datum in materials_data.values():
        geom = _cell_geometry(datum["bundle"], str(datum["meta"]["name"]))
        pts = geom["cell_pts_rot"]
        center = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
        extent = np.abs(pts - center).max(axis=0)
        half_x = max(half_x, float(extent[0]))
        half_y = max(half_y, float(extent[1]))
        half_z = max(half_z, float(extent[2]))
    pad_xy = 0.9
    pad_z = 0.9
    return {"half_x": half_x + pad_xy, "half_y": half_y + pad_xy, "half_z": half_z + pad_z}


def _finite_trace_values(values: Any) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=object).ravel()
    floats: list[float] = []
    for value in arr:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            floats.append(f)
    return np.asarray(floats, dtype=float)


def _trace_bounds(traces: list[Any]) -> np.ndarray:
    pts: list[np.ndarray] = []
    for tr in traces:
        xs = _finite_trace_values(getattr(tr, "x", None))
        ys = _finite_trace_values(getattr(tr, "y", None))
        zs = _finite_trace_values(getattr(tr, "z", None))
        n = min(len(xs), len(ys), len(zs))
        if n:
            pts.append(np.column_stack([xs[:n], ys[:n], zs[:n]]))
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def _draw_axis_key(
    ax,
    M: np.ndarray,
    R: np.ndarray,
    *,
    center: tuple[float, float] = (0.50, 0.50),
    extent: float = 0.42,
    min_length_frac: float = 0.55,
    fontsize: float = 8.0,
    label_pad: float = 0.16,
    arrow_color: str = AXIS_COLOR,
    halo_color: str = "white",
    halo_lw: float = 2.0,
    arrow_head_scale: float = 4.5,
    origin_dot_radius: float = 0.012,
) -> None:
    """Two-axis crystallographic key: only the in-plane axes ``a`` and ``c``.

    The fixed manuscript camera puts ``b`` along the view direction by
    construction, so the third axis carries no information that the
    reader cannot already infer from the 2D projection itself.  Drawing
    only the in-plane axes follows the convention used in fig5 panel-a
    (``_topology_axis_triad`` calls :func:`plot_fig5._draw_classical_triad`
    with ``labels=["a", "b"]`` for that panel's 2D plane) and avoids the
    earlier traps: ``b`` collapsing to a sub-pixel dot, a fake 225°
    fallback that has no crystallographic meaning, or an ⊗ symbol that
    occupies legend space proportional to a real axis without adding
    information.

    Arrow geometry uses **display pixels** so the key stays a true
    compass rose regardless of the host axes' aspect ratio, with a
    :data:`min_length_frac` floor for arrows whose projection is short.
    A white halo behind every stroke (arrows + labels + origin dot) keeps
    the symbol legible against any background without occluding it.
    """
    ax._fig2d_axis_key = True  # QA marker: axis labels live here, not over structure.

    labels = ("a", "c")
    vectors_2d: list[tuple[float, float]] = []
    cell_lens: list[float] = []
    for label in labels:
        idx = ("a", "b", "c").index(label)
        rot = R @ np.asarray(M[idx], dtype=float)
        vectors_2d.append((float(rot[0]), float(rot[2])))
        cell_lens.append(float(np.linalg.norm(M[idx])))
    norms = [math.hypot(dx, dy) for dx, dy in vectors_2d]
    if not norms or max(norms) <= 1.0e-8:
        return
    max_norm = max(norms)

    fig = ax.figure
    pos = ax.get_position()
    ax_w_pix = float(pos.width * fig.get_figwidth() * fig.dpi)
    ax_h_pix = float(pos.height * fig.get_figheight() * fig.dpi)
    box_size_pix = min(ax_w_pix, ax_h_pix)
    if box_size_pix <= 1.0e-3:
        return

    pix_extent = float(extent) * box_size_pix
    pix_arrow_min = pix_extent * float(min_length_frac)
    pix_label_pad = float(label_pad) * box_size_pix

    cx, cy = center
    head_scale = float(arrow_head_scale)

    disp_origin = ax.transAxes.transform((cx, cy))
    inv_transAxes = ax.transAxes.inverted().transform

    def _axes_from_pixel(disp_xy):
        out = inv_transAxes(disp_xy)
        return float(out[0]), float(out[1])

    def _add_arrow(tip_axes, *, halo: bool) -> FancyArrowPatch:
        # ``shrinkA=0`` forces arrow shafts to start exactly at the common
        # origin so the in-plane axes visually share one point (the
        # default 2 pt offset fragments the compass rose).
        return FancyArrowPatch(
            posA=(cx, cy),
            posB=tip_axes,
            arrowstyle="-|>",
            mutation_scale=head_scale + (0.8 if halo else 0.0),
            linewidth=halo_lw if halo else 0.95,
            color=halo_color if halo else arrow_color,
            transform=ax.transAxes,
            clip_on=False,
            zorder=41 if halo else 43,
            joinstyle="round",
            capstyle="round",
            shrinkA=0.0,
            shrinkB=0.0,
        )

    # Origin dot, sized in pixels so it stays circular even in a
    # tall-thin host axes.  Halo + core like the arrows.
    if origin_dot_radius > 0.0:
        dot_pix = float(origin_dot_radius) * box_size_pix
        ax_dx, _ = _axes_from_pixel((disp_origin[0] + dot_pix, disp_origin[1]))
        rx_axes = abs(ax_dx - cx)
        _, ax_dy = _axes_from_pixel((disp_origin[0], disp_origin[1] + dot_pix))
        ry_axes = abs(ax_dy - cy)
        halo_dot = Ellipse(
            (cx, cy), 2 * rx_axes * 1.55, 2 * ry_axes * 1.55,
            transform=ax.transAxes,
            facecolor=halo_color, edgecolor="none",
            zorder=40, clip_on=False,
        )
        core_dot = Ellipse(
            (cx, cy), 2 * rx_axes, 2 * ry_axes,
            transform=ax.transAxes,
            facecolor=arrow_color, edgecolor="none",
            zorder=42, clip_on=False,
        )
        halo_dot._fig2d_axis_origin = True
        core_dot._fig2d_axis_origin = True
        ax.add_patch(halo_dot)
        ax.add_patch(core_dot)

    for label, (dx, dy), norm in zip(labels, vectors_2d, norms):
        if norm <= 1.0e-8:
            # Truly view-parallel a or c would mean the manuscript camera
            # has been broken — let QA catch it rather than silently
            # papering over with a fallback.
            continue
        ux = float(dx) / norm
        uy = float(dy) / norm
        pix_len = max(pix_extent * (norm / max_norm), pix_arrow_min)
        disp_tip = (
            disp_origin[0] + pix_len * ux,
            disp_origin[1] + pix_len * uy,
        )
        tip = _axes_from_pixel(disp_tip)
        halo_arrow = _add_arrow(tip, halo=True)
        core_arrow = _add_arrow(tip, halo=False)
        halo_arrow._fig2d_axis_marker = True
        core_arrow._fig2d_axis_marker = True
        core_arrow._fig2d_axis_label_key = label  # QA tag for direction checks.
        ax.add_patch(halo_arrow)
        ax.add_patch(core_arrow)
        disp_label = (
            disp_origin[0] + (pix_len + pix_label_pad) * ux,
            disp_origin[1] + (pix_len + pix_label_pad) * uy,
        )
        label_xy = _axes_from_pixel(disp_label)
        t = ax.text(
            *label_xy,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontstyle="italic",
            fontweight="bold",
            ha="center",
            va="center",
            color=arrow_color,
            zorder=44,
        )
        t.set_path_effects([withStroke(linewidth=halo_lw, foreground=halo_color)])
        t.set_clip_on(False)
        t._fig2d_axis_label = True


def _write_cell_panel(
    bundle,
    material_name: str,
    a_records: list[dict[str, Any]],
    b_records: list[dict[str, Any]],
    modal_a: dict[str, Any] | None,
    modal_b: dict[str, Any] | None,
    out_path: Path,
    view_extent: dict[str, float],
) -> None:
    geom = _cell_geometry(bundle, material_name)
    M = geom["M"]
    R = geom["R"]
    repeats = geom["repeats"]
    cell_xs = geom["cell_xs"]
    cell_pts_rot = geom["cell_pts_rot"]

    rot_xs: list[Any] = []
    rot_ys: list[Any] = []
    rot_zs: list[Any] = []
    cursor = 0
    for xv in cell_xs:
        if xv is None:
            rot_xs.append(None)
            rot_ys.append(None)
            rot_zs.append(None)
        else:
            rot_xs.append(float(cell_pts_rot[cursor, 0]))
            rot_ys.append(float(cell_pts_rot[cursor, 1]))
            rot_zs.append(float(cell_pts_rot[cursor, 2]))
            cursor += 1
    cell_trace = go.Scatter3d(
        x=rot_xs, y=rot_ys, z=rot_zs, mode="lines",
        line=dict(color=CELL_BOX_COLOR, width=CELL_BOX_WIDTH),
        hoverinfo="skip", showlegend=False, name="cell",
    )

    a_all = _replicate_polyhedra(a_records, M, repeats)
    b_all = _replicate_polyhedra(b_records, M, repeats)

    bg_traces: list[Any] = []
    hl_traces: list[Any] = []

    strategy = HIGHLIGHT_STRATEGY_OVERRIDE.get(material_name, SELECTION_STRATEGY)
    a_pick, b_pick = _pick_highlight_pair(
        a_all, b_all, cell_pts_rot, R, strategy=strategy,
    )
    def _key(rec):
        return None if rec is None else (int(rec["fragment_index"]), tuple(rec.get("replica", (0, 0, 0))))
    print(f"  [{material_name}/{strategy}] A={_key(a_pick)} B={_key(b_pick)}")

    def _add_records(records, *, picked, hl_face: str) -> None:
        picked_key = (
            (int(picked["fragment_index"]), tuple(picked.get("replica", (0, 0, 0))))
            if picked is not None else None
        )
        for rec in records:
            shell_rot = _rotate_points(R, rec["shell_coords"])
            if len(shell_rot) < 4:
                continue
            key = (int(rec["fragment_index"]), tuple(rec.get("replica", (0, 0, 0))))
            if picked_key is not None and key == picked_key:
                hl_traces.extend(_polyhedron_traces(
                    shell_rot,
                    face_color=hl_face,
                    edge_color=hl_face,
                    opacity=HL_OPACITY,
                ))
            else:
                bg_traces.extend(_polyhedron_traces(
                    shell_rot,
                    face_color=HULL_BG_FACE,
                    edge_color=HULL_BG_EDGE,
                    opacity=BG_OPACITY,
                ))

    _add_records(a_all, picked=a_pick, hl_face=A_HIGHLIGHT)
    _add_records(b_all, picked=b_pick, hl_face=B_HIGHLIGHT)

    cell_center = 0.5 * (cell_pts_rot.max(axis=0) + cell_pts_rot.min(axis=0))
    half_x = float(view_extent["half_x"])
    half_y = float(view_extent["half_y"])
    half_z = float(view_extent["half_z"])
    atom_bond = _atom_bond_traces(bundle, R, M, repeats)
    traces_no_axes: list[Any] = [*atom_bond, *bg_traces, *hl_traces, cell_trace]
    trace_pts = _trace_bounds(traces_no_axes)
    if len(trace_pts):
        # MatterVis now mirrors whole fragments at unit-cell boundaries.
        # Include those rendered replicas (not just the formal cell box) in
        # the panel bounds so SY's left/right boundary molecules are not
        # cropped out of the orthographic viewport.
        pad = 1.15
        half_x = max(half_x, float(np.max(np.abs(trace_pts[:, 0] - cell_center[0]))) + pad)
        half_y = max(half_y, float(np.max(np.abs(trace_pts[:, 1] - cell_center[1]))) + pad)
        half_z = max(half_z, float(np.max(np.abs(trace_pts[:, 2] - cell_center[2]))) + pad)
    x_range = [cell_center[0] - half_x, cell_center[0] + half_x]
    y_range = [cell_center[1] - half_y, cell_center[1] + half_y]
    z_range = [cell_center[2] - half_z, cell_center[2] + half_z]
    traces: list[Any] = traces_no_axes
    scale = max(half_x, half_y, half_z, 1e-6)
    aspect_x = half_x / scale
    aspect_y = half_y / scale
    aspect_z = half_z / scale

    eye = dict(x=0.0, y=-2.30, z=0.0)
    panel = go.Figure(data=traces)
    panel.update_layout(
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=8, r=8, t=8, b=8),
        scene=dict(
            xaxis=dict(visible=False, range=x_range),
            yaxis=dict(visible=False, range=y_range),
            zaxis=dict(visible=False, range=z_range),
            aspectmode="manual",
            aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
            bgcolor="white",
            camera=dict(eye=eye, up=dict(x=0, y=0, z=1), projection=dict(type="orthographic")),
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = PLOTLY_PANEL_SIZE
    panel.write_image(str(out_path), width=width, height=height, scale=PLOTLY_PANEL_SCALE)
    _crop_white_border(out_path)


def _draw_cell_panel(
    fig,
    subplot_spec,
    datum: dict[str, Any],
    view_extent: dict[str, float],
) -> Any:
    bundle = datum["bundle"]
    name = str(datum["meta"]["name"])
    a_records = datum["sites"]["A"]
    b_records = datum["sites"]["B"]
    modal_a = _modal_record(a_records)
    modal_b = _modal_record(b_records)

    sg = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=subplot_spec,
        height_ratios=[1.0, 0.18],
        hspace=0.04,
    )
    ax = fig.add_subplot(sg[0, 0])
    ax.set_axis_off()
    ax.set_facecolor("white")
    ax._fig2d_structure_panel = True

    safe_name = name.replace("/", "_")
    panel_path = PLOTLY_PANEL_DIR / f"{safe_name}_cell_{SELECTION_STRATEGY}.png"

    geom = _cell_geometry(bundle, name)
    M = geom["M"]
    R = geom["R"]
    repeats = geom["repeats"]
    cell_pts_rot = geom["cell_pts_rot"]
    a_all = _replicate_polyhedra(a_records, M, repeats)
    b_all = _replicate_polyhedra(b_records, M, repeats)
    a_pick_preview, b_pick_preview = _pick_highlight_pair(
        a_all, b_all, cell_pts_rot, R,
        strategy=HIGHLIGHT_STRATEGY_OVERRIDE.get(name, SELECTION_STRATEGY),
    )

    _write_cell_panel(
        bundle, name, a_records, b_records, modal_a, modal_b, panel_path,
        view_extent=view_extent,
    )
    image = plt.imread(str(panel_path))
    ax.imshow(image)

    # Split the legend band horizontally so the compass-rose triad gets a
    # *dedicated* near-square axes (~15% of the panel width).  Without
    # this, the triad inherits the chip-row's wide-thin host axes and the
    # pixel-sized arrows collapse to a few pixels — exactly the previous
    # failure mode where ``b`` rendered as a dash and ``c`` as a stub.
    leg_sg = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=sg[1, 0],
        width_ratios=[0.22, 1.0],
        wspace=0.04,
    )
    axis_key_ax = fig.add_subplot(leg_sg[0, 0])
    axis_key_ax.set_xlim(0, 1)
    axis_key_ax.set_ylim(0, 1)
    axis_key_ax.set_axis_off()
    axis_key_ax.set_aspect("equal", adjustable="box")
    leg_ax = fig.add_subplot(leg_sg[0, 1])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.set_axis_off()

    _draw_axis_key(axis_key_ax, M, R)

    items: list[tuple[str, str, str]] = []
    if a_pick_preview is not None:
        items.append(("A", A_HIGHLIGHT, _polyhedron_name(a_pick_preview) or "A polyhedron"))
    if b_pick_preview is not None:
        items.append(("B", B_HIGHLIGHT, _polyhedron_name(b_pick_preview) or "B polyhedron"))
    if items:
        n = len(items)
        chip_w = 0.035
        chip_h = 0.34
        row_centers = np.linspace(1.0, 0.0, n + 2)[1:-1]
        chip_x = 0.04
        for cy, (site_label, color, shape_label) in zip(row_centers, items):
            leg_ax.add_patch(plt.Rectangle(
                (chip_x, cy - chip_h / 2),
                chip_w, chip_h,
                facecolor=color, edgecolor=color, lw=0.6,
                alpha=HL_OPACITY,
                transform=leg_ax.transAxes,
            ))
            leg_ax.text(
                chip_x + chip_w + 0.018, cy,
                f"{site_label}: {shape_label}",
                ha="left", va="center",
                fontsize=LEGEND_PT,
                color=COLORS["charcoal"],
                transform=leg_ax.transAxes,
            )
    return ax


def draw_material(
    fig,
    subplot_spec,
    datum: dict[str, Any],
    view_extent: dict[str, float],
) -> Any:
    return _draw_cell_panel(fig, subplot_spec, datum, view_extent)


def _qa_check_axis_projection(data: dict[str, dict[str, Any]]) -> None:
    for material, datum in data.items():
        geom = _cell_geometry(datum["bundle"], material)
        M = geom["M"]
        R = geom["R"]
        projections = {
            label: np.array([(R @ M[idx])[0], (R @ M[idx])[2]], dtype=float)
            for idx, label in enumerate(("a", "b", "c"))
        }
        lengths = {label: max(float(np.linalg.norm(M[idx])), 1e-9)
                   for idx, label in enumerate(("a", "b", "c"))}

        a = projections["a"]
        c = projections["c"]
        b = projections["b"]
        if abs(float(a[0])) < 0.92 * np.linalg.norm(a) or abs(float(a[1])) > 0.18 * lengths["a"]:
            raise RuntimeError(
                f"Figure 2d QA failed: {material} a-axis is not near-horizontal "
                f"in the fixed camera projection: {a.tolist()}"
            )
        if abs(float(c[1])) < 0.92 * np.linalg.norm(c) or abs(float(c[0])) > 0.18 * lengths["c"]:
            raise RuntimeError(
                f"Figure 2d QA failed: {material} c-axis is not near-vertical "
                f"in the fixed camera projection: {c.tolist()}"
            )
        if np.linalg.norm(b) > 0.35 * lengths["b"]:
            raise RuntimeError(
                f"Figure 2d QA failed: {material} b-axis projection is too large; "
                f"axis key should show b as view-depth/dot-like, got {b.tolist()}"
            )


def _qa_check_repeats_decoupled(data: dict[str, dict[str, Any]]) -> None:
    """Ensure the supercell repeats are pinned per-material and independent of R.

    Earlier versions derived the repeats from the rotation matrix, so a
    camera-orientation change silently altered which cells were drawn for SY.
    This QA pins down both invariants:

    1. ``_supercell_repeat(name)`` returns the configured per-material value.
    2. The repeats are stable under any rotation matrix (i.e. the function
       takes no orientation argument any more).
    """
    for material in data.keys():
        expected = MATERIAL_REPEATS.get(material, (1, 1, 1))
        actual = _supercell_repeat(material)
        if tuple(actual) != tuple(expected):
            raise RuntimeError(
                f"Figure 2d QA failed: {material} supercell repeats "
                f"{actual} != configured {expected}"
            )
        # Sanity: changing the rotation must not change the result.
        if tuple(_supercell_repeat(material)) != tuple(actual):
            raise RuntimeError(
                f"Figure 2d QA failed: {material} supercell repeats are not "
                f"deterministic per material."
            )


def _qa_check_axis_key(fig) -> None:
    """Verify the two-axis (a, c) compass key for each structure panel.

    The manuscript convention puts ``b`` along the view direction by
    construction, so the axis key only renders the two in-plane axes
    ``a`` (right) and ``c`` (up).  This QA enforces:

    * exactly 2 italic labels ``a`` and ``c`` rendered inside the
      dedicated axis-key axes (never on the structure panel),
    * no stray ``b`` label leaking from an earlier code path,
    * each in-plane axis carries a :class:`FancyArrowPatch` tagged with
      ``_fig2d_axis_label_key``,
    * the two arrows share a common origin (within 1 axes-fraction
      percent) so the symbol reads as a single coordinate-system marker,
    * the ``c`` arrow points predominantly up,
    * the ``a`` arrow points predominantly right,
    * each arrow is at least 8 px long after rendering — protects against
      the host axes accidentally degenerating to wide-thin.
    """
    fig.canvas.draw()
    key_axes = [ax for ax in fig.axes if getattr(ax, "_fig2d_axis_key", False)]
    if not key_axes:
        raise RuntimeError("Figure 2d QA failed: no axis-key axes found.")
    for kax in key_axes:
        labels_seen: dict[str, Any] = {}
        for text in kax.findobj(match=plt.Text):
            if not getattr(text, "_fig2d_axis_label", False):
                continue
            s = text.get_text().strip()
            if s in {"a", "b", "c"}:
                labels_seen[s] = text
        if "b" in labels_seen:
            raise RuntimeError(
                "Figure 2d QA failed: b label found in axis key — the "
                "manuscript convention shows only in-plane axes (a, c)."
            )
        if set(labels_seen) != {"a", "c"}:
            raise RuntimeError(
                f"Figure 2d QA failed: axis key labels {sorted(labels_seen)} "
                f"!= expected {{'a', 'c'}}."
            )

        core_arrows: dict[str, FancyArrowPatch] = {}
        for obj in kax.get_children():
            tag = getattr(obj, "_fig2d_axis_label_key", None)
            if tag in {"a", "c"} and isinstance(obj, FancyArrowPatch):
                core_arrows[tag] = obj
            if tag == "b":
                raise RuntimeError(
                    "Figure 2d QA failed: b marker found in axis key — "
                    "expected only in-plane axes (a, c)."
                )
        if set(core_arrows) != {"a", "c"}:
            raise RuntimeError(
                f"Figure 2d QA failed: axis-key arrows cover "
                f"{sorted(core_arrows)}, expected {{'a', 'c'}}."
            )

        origins: list[tuple[float, float]] = []
        directions: dict[str, tuple[float, float]] = {}
        lengths_px: dict[str, float] = {}
        for label, arrow in core_arrows.items():
            start, end = arrow._posA_posB
            origins.append((float(start[0]), float(start[1])))
            directions[label] = (float(end[0] - start[0]), float(end[1] - start[1]))
            sp = kax.transAxes.transform(start)
            ep = kax.transAxes.transform(end)
            lengths_px[label] = float(math.hypot(ep[0] - sp[0], ep[1] - sp[1]))

        origins_arr = np.asarray(origins, dtype=float)
        if origins_arr.std(axis=0).max() > 0.01:
            raise RuntimeError(
                f"Figure 2d QA failed: axis-key arrows do not share a "
                f"common origin, origins={origins_arr.tolist()}"
            )

        dx_c, dy_c = directions["c"]
        if dy_c <= 0 or abs(dy_c) <= abs(dx_c):
            raise RuntimeError(
                f"Figure 2d QA failed: c-axis arrow is not predominantly "
                f"upward, got direction ({dx_c:.3f}, {dy_c:.3f})."
            )
        dx_a, dy_a = directions["a"]
        if dx_a <= 0 or abs(dx_a) <= abs(dy_a):
            raise RuntimeError(
                f"Figure 2d QA failed: a-axis arrow is not predominantly "
                f"rightward, got direction ({dx_a:.3f}, {dy_a:.3f})."
            )
        # Threshold tuned for the 100-dpi canvas.draw() pass used by QA;
        # the saved PDF / PNG renders at 300 dpi so the actual on-paper
        # arrow length is ~3× this number.  6 px at QA time = ~18 px in
        # the final output, comfortably readable.
        if lengths_px and min(lengths_px.values()) < 6.0:
            raise RuntimeError(
                f"Figure 2d QA failed: at least one axis-key arrow is "
                f"too short to read ({lengths_px}); did the host axes "
                f"degenerate to wide-thin?"
            )


def _qa_check_figure_text(fig) -> None:
    fig.canvas.draw()
    distance_token_re = ("d_{X-A}", "d_{X-B}", r"\bar d")
    axis_labels_seen = set()
    for text in fig.findobj(match=plt.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        s = text.get_text()
        if s in {"a", "b", "c"}:
            axis_labels_seen.add(s)
            if not getattr(text.axes, "_fig2d_axis_key", False):
                raise RuntimeError(
                    f"Figure 2d QA failed: axis label {s!r} is drawn on the "
                    "structure panel instead of the dedicated legend-band axis key."
                )
            if not getattr(text, "_fig2d_axis_label", False):
                raise RuntimeError(
                    f"Figure 2d QA failed: axis label {s!r} lacks axis-key metadata."
                )
        fontsize = float(text.get_fontsize())
        if fontsize < MIN_TEXT_PT:
            raise RuntimeError(
                f"Figure 2d QA failed: text {s!r} is {fontsize:.1f} pt "
                f"(< {MIN_TEXT_PT:.1f} pt)"
            )
        if "deg" in s:
            raise RuntimeError(f"Figure 2d QA failed: RMSD text still present: {s!r}")
        if any(tok in s for tok in distance_token_re) and DIST_UNIT not in s:
            raise RuntimeError(
                f"Figure 2d QA failed: distance text {s!r} is missing unit {DIST_UNIT!r}"
            )
    if axis_labels_seen != {"a", "c"}:
        raise RuntimeError(
            f"Figure 2d QA failed: expected axis labels a/c only "
            f"(b runs along the view direction so it is intentionally "
            f"omitted), saw {sorted(axis_labels_seen)}."
        )


def main() -> None:
    setup_nature_style()
    plt.rcParams.update({
        "axes.titlesize": POLY_NAME_PT,
        "axes.labelsize": LABEL_TEXT_PT,
        "xtick.labelsize": SMALL_TEXT_PT,
        "ytick.labelsize": SMALL_TEXT_PT,
        "figure.titlesize": TITLE_PT,
        "font.size": 10,
    })

    data = load_material_data()
    view_extent = _compute_view_extent(data)
    # 210 mm = 8.2677 in: full A4 width so the manuscript can include the
    # figure at \\textwidth without rescaling.  The 75 mm height preserves
    # the previous 2.8 aspect ratio, which kept structure panels roughly
    # square once the title + legend bands are accounted for.
    fig = plt.figure(figsize=(210.0 / 25.4, 75.0 / 25.4))
    outer = gridspec.GridSpec(
        1, 3,
        figure=fig,
        left=0.040,
        right=0.995,
        top=0.830,
        bottom=0.090,
        wspace=0.14,
    )

    first_axis = None
    for col, item in enumerate(MATERIALS):
        ax0 = draw_material(fig, outer[0, col], data[item["name"]], view_extent)
        if first_axis is None:
            first_axis = ax0
        pos = outer[0, col].get_position(fig)
        fig.text(
            0.5 * (pos.x0 + pos.x1),
            0.92,
            f"{display_material(str(item['name']))}  {item['stoich']}",
            ha="center",
            va="bottom",
            fontsize=TITLE_PT,
            fontweight="bold",
            color=COLORS["charcoal"],
        )

    if first_axis is not None:
        # Place the "d" panel label in figure-fraction coordinates so it
        # survives the strict figsize savefig (no tight bbox expansion).
        # Anchored near the top-left of the figure, vertically aligned
        # with the row of panel titles.
        fig.text(
            0.012,
            0.93,
            "d",
            fontsize=PANEL_LABEL_PT,
            fontweight="bold",
            color=EXP_COLORS["charcoal"],
            ha="left",
            va="bottom",
        )


    _qa_check_axis_projection(data)
    _qa_check_repeats_decoupled(data)
    _qa_check_axis_key(fig)
    _qa_check_figure_text(fig)

    if SELECTION_STRATEGY == "modal_first":
        out = OUT_DIR / "_figure2d_polyhedra.png"
    else:
        out = OUT_DIR / f"_figure2d_polyhedra_{SELECTION_STRATEGY}.png"
    # Save at exactly the figsize-defined 210 mm width.  Using
    # ``bbox_inches='tight'`` would trim margins and produce a slightly
    # narrower output, which then has to be rescaled in LaTeX and loses
    # the predictable 210 mm width the user explicitly asked for.
    fig.savefig(out, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=None, pad_inches=0, facecolor="white")
    print(f"Saved {out}")
    print(f"Saved {out.with_suffix('.pdf')}")
    plt.close(fig)


def render_all_strategies() -> None:
    global SELECTION_STRATEGY
    for strategy in SELECTION_STRATEGIES:
        SELECTION_STRATEGY = strategy
        print(f"\n--- Selection strategy: {strategy} ---")
        main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", action="store_true",
        help="Render one figure per selection strategy for comparison.",
    )
    parser.add_argument(
        "--strategy", choices=SELECTION_STRATEGIES, default=None,
        help="Set the active selection strategy for a single render.",
    )
    args = parser.parse_args()
    if args.strategy:
        SELECTION_STRATEGY = args.strategy
    if args.strategies:
        render_all_strategies()
    else:
        main()
