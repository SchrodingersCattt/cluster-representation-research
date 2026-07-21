#!/usr/bin/env python3
"""Figure 5 -- synthesis and validation for the DEP-derived ABX4 branch.

This manuscript-side driver centralizes the assembled Figure 5 workflow under
`manuscript/figures/` while reading curated release assets from `data/abx4/`
and reusable visualization modules from `src/`.

The original working `ABX4_expdata/` tree is intentionally not required by
the code-availability package.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sys
import warnings
from collections import Counter
from functools import reduce
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.text as mtext
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.patheffects import withStroke

try:
    from ase.io import read as ase_read
except Exception:
    ase_read = None
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


warnings.filterwarnings(
    "ignore",
    message=r"crystal system '.*' is not interpreted for space group .*",
    category=UserWarning,
)

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
SRC_DIR = ROOT / "src"
EXP_DIR = ROOT / "experiments"
PEMS_DATA_ROOT = ROOT / "data" / "pems"
MOLCRYSKIT_ROOT = Path(os.environ["MOLCRYSKIT_ROOT"]) if os.environ.get("MOLCRYSKIT_ROOT") else None

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXP_DIR))
if MOLCRYSKIT_ROOT is not None and MOLCRYSKIT_ROOT.exists():
    sys.path.insert(0, str(MOLCRYSKIT_ROOT))
from paper_plot_style import save_png_pdf, setup_nature_style  # noqa: E402

from stoich_cluster_learning.data.fig5 import ABX4_CIF_DIR, load_pxrd_specs  # noqa: E402
from stoich_cluster_learning.viz import polyhedra as fig2d_polyhedra  # noqa: E402
from stoich_cluster_learning.viz import topology_projection as topology_helpers  # noqa: E402

NEW_CIF_DIR = ABX4_CIF_DIR

# Programmatic figure-QA helpers (see FIGURE_QA.md for the binding rules).
sys.path.insert(0, str(THIS_DIR))
import _qa_check as _qa  # noqa: E402
from figure_style import display_material  # noqa: E402

# Panel b's underlying RGBA buffer is rendered at this DPI -- the QA helper
# uses it to convert label font sizes between points and pixels for bbox math.
PANEL_B_QA_DPI = 600.0

try:  # crystal_viewer renders panel b (formula_unit) with shared scale/occlusion
    from crystal_viewer.presets import DEFAULT_STYLE, deep_merge  # noqa: E402
    from crystal_viewer.renderer import build_figure, uniform_viewport  # noqa: E402
    from crystal_viewer.scene import build_scene_from_cif  # noqa: E402
except Exception:  # pragma: no cover - panel b will fall back to a warning tile
    DEFAULT_STYLE = None
    deep_merge = None
    build_figure = None
    uniform_viewport = None
    build_scene_from_cif = None


COLORS = {
    "SY": "#3B4F7A",
    "PEP": "#A0522D",
    "MPEP": "#6B5B7B",
    "HPEP": "#2E7D6A",
    "DAP-4": "#B88A3A",
    "EAP-4": "#4E8C9A",
    "SY_sim": "#8A9BBF",
    "PEP_sim": "#C9956E",
    "MPEP_sim": "#A99AB8",
    "HPEP_sim": "#7BB5A3",
    "A_site": "#8A5A67",
    "B_site": "#6B7C4E",
    "X_site": "#5A6D7B",
    "charcoal": "#2F2F2F",
    "mid_gray": "#D6D6D6",
    "mid_gray_dark": "#8A8A8A",
    "faint_grid": "#ECECEC",
    "highlight_gold": "#9E8B5E",
    "pred_blue": "#4A6274",
    "note_fill": "#FBFBFB",
}

# ABX4 site identities for each material (used in panel a composition annotation).
# All four share the same B-site (H2en2+) and X-site (ClO4-); only A-site differs.
ABX4_SITES = {
    "PEP": {
        "A": r"H$_2$pz$^{2+}$",
        "B": r"H$_2$en$^{2+}$",
        "X": r"ClO$_4^-$",
    },
    "MPEP": {
        "A": r"MeHpz$^{2+}$",
        "B": r"H$_2$en$^{2+}$",
        "X": r"ClO$_4^-$",
    },
    "HPEP": {
        "A": r"H$_2$hpz$^{2+}$",
        "B": r"H$_2$en$^{2+}$",
        "X": r"ClO$_4^-$",
    },
}

# Reference / experimental constants only. Model predictions (D_pred, D_pred_std)
# are loaded at runtime from experiments/pems_ood_5fold_exp7a.json
# via _load_v3_ensemble_predictions() and injected into this dict in main().
NEW_MATERIALS = {
    "PEP": {
        "formula": "C6H22Cl4N4O16",
        "rho": 1.88,
        "Td": 310.6,
        "D_KJ": 9090,
        "OB_ref": -14.6,
        "sg": "Pbca",
        "sg_title": r"$Pbca$",
    },
    "MPEP": {
        "formula": "C7H24Cl4N4O16",
        "rho": 1.82,
        "Td": 292.6,
        "D_KJ": 8729,
        "OB_ref": -22.8,
        "sg": "P2$_1$/c",
        "sg_title": r"$P2_1/c$",
    },
    "HPEP": {
        "formula": "C7H24Cl4N4O16",
        "rho": 1.83,
        "Td": 324.1,
        "D_KJ": 8764,
        "OB_ref": -22.8,
        "sg": "P2$_1$/c",
        "sg_title": r"$P2_1/c$",
    },
}

# JSON source-of-truth for model predictions (multi-task baseline v3 5-fold ensemble,
# n = 15 samples per material across 5 folds x 3 cluster realizations n1/n2/n3).
PREDICTIONS_JSON = EXP_DIR / "pems_ood_5fold_exp7a.json"
CLUSTER_UMAP_CACHE = THIS_DIR / "_cluster_umap_cache.npz"
MATERIAL_POOLED_UMAP_CACHE = THIS_DIR / "_material_pooled_umap_cache.npz"

BENCHMARKS = {
    "TNT": {"formula": "C7H5N3O6", "Td": 290.0, "D_KJ": 6897, "OB_ref": -74.0},
    "RDX": {"formula": "C3H6N6O6", "Td": 210.0, "D_KJ": 8634, "OB_ref": -21.6},
    "HMX": {"formula": "C4H8N8O8", "Td": 279.0, "D_KJ": 8892, "OB_ref": -21.6},
    "CL-20": {"formula": "C6H6N12O12", "Td": 215.0, "D_KJ": 9507, "OB_ref": -11.0},
}

MIX_STARS = {
    # DEP-1 is the remeasured previously known ABX4 reference. DAP-4 and EAP-4
    # are representative high-performing MIX anchors. DAP-4 has both an onset
    # and a peak decomposition temperature; the SI benchmark uses onset for
    # consistency and marks the peak separately.
    "SY": {"Td": 359.6, "D_KJ": 8867},
    "DAP-4": {"Td": 358.0, "Td_peak": 385.0, "D_KJ": 8806},
    "EAP-4": {"Td": 208.0, "D_KJ": 8977},
}

CLASSIC_VDET_BENCHMARKS = {
    "TNT": 6783,
    "RDX": 8634,
    "HMX": 8892,
    "CL-20": 9507,
}

MATERIAL_ORDER = ["PEP", "MPEP", "HPEP"]
SOURCE_ROW_ORDER = ["PEP", "MPEP", "HPEP"]
DISPLAY_TITLES = {name: display_material(name) for name in MATERIAL_ORDER}

TRAINING_MATERIALS = [
    "DAI-1",
    "DAI-2",
    "DAI-4",
    "DAI-X1",
    "DAN-2",
    "DAP-1",
    "DAP-2",
    "DAP-3",
    "DAP-4",
    "DAP-5",
    "DAP-6",
    "DAP-7",
    "DAP-M4",
    "DAP-O2",
    "DAP-O4",
    "PAN-2",
    "PAN-H2",
    "PAN-M2",
    "PAP-1",
    "PAP-4",
    "PAP-5",
    "PAP-H4",
    "PAP-H5",
    "PAP-M4",
    "PAP-M5",
]

PXRD_SPECS = {
    name: {
        "meas": entries["measured"],
        "sim": entries["simulated"],
    }
    for name, entries in load_pxrd_specs().items()
}

DSC_SHEETS = {
    "PEP": "PEP-1",
    "MPEP": "1026-PT-PAP-H6(hpz_NH2OH_HClO4)",
    "HPEP": "HPEP",
}

ATOMIC_WEIGHTS = {
    "H": 1.00794,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "Cl": 35.453,
    "Na": 22.98977,
    "K": 39.0983,
    "Rb": 85.4678,
    "Ag": 107.8682,
    "I": 126.90447,
    "Ba": 137.327,
}

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")
STRUCTURE_ELEV = 18.0
STRUCTURE_AZIM = -58.0
STRUCTURE_ATOM_COLORS = {
    "C": "#5E5E5E",
    "N": "#2C61AF",
    "O": "#B85060",
    "Cl": "#218E6A",
    "H": "#DCDCDC",
}
STRUCTURE_ATOM_SIZES = {"C": 18, "N": 22, "O": 20, "Cl": 34, "H": 10}
STRUCTURE_COV_RADII = {
    "H": 0.31,
    "C": 0.77,
    "N": 0.75,
    "O": 0.73,
    "Cl": 0.99,
    "Na": 1.66,
    "K": 2.03,
    "Rb": 2.16,
    "Ag": 1.45,
    "I": 1.33,
    "Ba": 2.15,
}
METAL_ELEMENTS = {"Na", "K", "Rb", "Ag", "Ba"}


def style_axes_local(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_edgecolor(COLORS["charcoal"])
    ax.spines["bottom"].set_edgecolor(COLORS["charcoal"])
    ax.tick_params(direction="out", length=2.5, width=0.5, pad=2, color=COLORS["charcoal"])
    if grid_axis == "y":
        ax.yaxis.grid(True, color=COLORS["faint_grid"], linewidth=0.4, linestyle=":")
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=COLORS["faint_grid"], linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)


def add_panel_label_at(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label.lower(),
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=COLORS["charcoal"],
        ha="left",
        va="bottom",
    )


def add_panel_label_fig(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(
        x,
        y,
        label.lower(),
        fontsize=9,
        fontweight="bold",
        color=COLORS["charcoal"],
        ha="left",
        va="bottom",
    )


def draw_missing_panel(ax: plt.Axes, message: str, title: str) -> None:
    ax.set_title(title, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color=COLORS["mid_gray_dark"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": COLORS["note_fill"],
            "edgecolor": COLORS["mid_gray"],
            "linewidth": 0.5,
        },
    )


def _load_row_tiles(
    image_path: Path,
    source_order: list[str],
    y_frac: tuple[float, float],
    x_frac: tuple[float, float] = (0.0, 1.0),
    x_pad_frac: float = 0.04,
    y_pad_frac: float = 0.01,
    trim_pad: int = 16,
    trim_tiles: bool = True,
) -> dict[str, np.ndarray]:
    image = plt.imread(str(image_path))
    height, width = image.shape[:2]
    xg0 = int(width * x_frac[0])
    xg1 = int(width * x_frac[1])
    y0 = int(height * (y_frac[0] + y_pad_frac))
    y1 = int(height * (y_frac[1] - y_pad_frac))
    cropped = image[y0:y1, xg0:xg1]
    tile_width = cropped.shape[1] / len(source_order)
    tiles: dict[str, np.ndarray] = {}
    for index, name in enumerate(source_order):
        x0 = int(index * tile_width + x_pad_frac * tile_width)
        x1 = int((index + 1) * tile_width - x_pad_frac * tile_width)
        tile = cropped[:, x0:x1]
        tiles[name] = _trim_tile_white_margin(tile, pad=trim_pad) if trim_tiles else tile
    return tiles


def _trim_tile_white_margin(tile: np.ndarray, threshold: float = 0.985, pad: int = 16) -> np.ndarray:
    if tile.ndim == 2:
        keep = tile < threshold
    else:
        keep = np.any(tile[..., :3] < threshold, axis=-1)
    coords = np.argwhere(keep)
    if coords.size == 0:
        return tile
    y0, x0 = np.maximum(coords.min(axis=0) - pad, 0)
    y1, x1 = np.minimum(coords.max(axis=0) + pad + 1, tile.shape[:2])
    return tile[y0:y1, x0:x1]


def _add_axes_triad(
    ax: plt.Axes,
    origin: tuple[float, float] = (0.10, 0.08),
    scale: float = 0.065,
    labels: tuple[str, str, str] = ("a", "b", "c"),
) -> None:
    vectors = (
        (scale, 0.0),
        (0.0, scale),
        (-0.55 * scale, 0.55 * scale),
    )
    for (dx, dy), label in zip(vectors, labels):
        tip = (origin[0] + dx, origin[1] + dy)
        ax.annotate(
            "",
            xy=tip,
            xytext=origin,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#2F2F2F", "shrinkA": 0, "shrinkB": 0},
        )
        ax.text(
            tip[0] + 0.012 * np.sign(dx if abs(dx) > 1.0e-8 else 1.0),
            tip[1] + 0.012 * np.sign(dy if abs(dy) > 1.0e-8 else 1.0),
            label,
            transform=ax.transAxes,
            fontsize=8.0,
            fontstyle="italic",
            ha="center",
            va="center",
            color="#2F2F2F",
        )


def _plot_tiled_row(
    fig: plt.Figure,
    subspec,
    image_path: Path,
    source_order: list[str],
    target_order: list[str],
    y_frac: tuple[float, float],
    title_map: dict[str, str],
    x_frac: tuple[float, float] = (0.0, 1.0),
    x_pad_frac: float = 0.04,
    y_pad_frac: float = 0.01,
    trim_pad: int = 16,
    trim_tiles: bool = True,
) -> list[plt.Axes]:
    row = subspec.subgridspec(1, len(target_order), wspace=0.03)
    axes = [fig.add_subplot(row[0, idx]) for idx in range(len(target_order))]
    if not image_path.exists():
        draw_missing_panel(axes[0], f"Missing image asset:\n{image_path}", title_map[target_order[0]])
        for ax, name in zip(axes[1:], target_order[1:]):
            draw_missing_panel(ax, "Missing image asset", title_map[name])
        return axes

    tiles = _load_row_tiles(
        image_path,
        source_order=source_order,
        y_frac=y_frac,
        x_frac=x_frac,
        x_pad_frac=x_pad_frac,
        y_pad_frac=y_pad_frac,
        trim_pad=trim_pad,
        trim_tiles=trim_tiles,
    )
    for ax, name in zip(axes, target_order):
        ax.imshow(tiles[name], interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title_map[name], pad=1.8, fontsize=10.0, fontweight="bold")
    return axes


def parse_formula(formula: str) -> dict[str, float]:
    formula_dict: dict[str, float] = {}
    for elem, count_text in FORMULA_RE.findall(formula):
        count = float(count_text) if count_text else 1.0
        formula_dict[elem] = formula_dict.get(elem, 0.0) + count
    return formula_dict


def format_formula(formula_dict: dict[str, float]) -> str:
    ordered_elems = []
    for elem in ("C", "H"):
        if elem in formula_dict:
            ordered_elems.append(elem)
    ordered_elems.extend(sorted(elem for elem in formula_dict if elem not in {"C", "H"}))
    parts = []
    for elem in ordered_elems:
        count = formula_dict[elem]
        if abs(count - round(count)) < 1.0e-8:
            count = int(round(count))
        parts.append(elem if count == 1 else f"{elem}{count}")
    return "".join(parts)


def _extract_cif_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"{re.escape(tag)}\s+(?:'([^']*)'|\"([^\"]*)\"|([^\r\n#]+))")
    match = pattern.search(text)
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return group.strip()
    return None


def _extract_z_value(text: str) -> int | None:
    z_text = _extract_cif_tag(text, "_cell_formula_units_Z")
    if not z_text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", z_text)
    if not match:
        return None
    z_val = float(match.group(0))
    if abs(z_val - round(z_val)) < 1.0e-8 and z_val > 0:
        return int(round(z_val))
    return None


def _count_atom_site_symbols(text: str) -> Counter:
    lines = text.splitlines()
    counter: Counter = Counter()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line != "loop_":
            index += 1
            continue
        index += 1
        tags = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            tags.append(lines[index].strip().split()[0])
            index += 1
        if "_atom_site_type_symbol" not in tags:
            continue
        symbol_idx = tags.index("_atom_site_type_symbol")
        while index < len(lines):
            row = lines[index].strip()
            if not row or row.startswith("#"):
                index += 1
                continue
            if row == "loop_" or row.startswith("_") or row.startswith("data_"):
                break
            tokens = row.split()
            if len(tokens) <= symbol_idx:
                index += 1
                continue
            symbol = re.sub(r"[^A-Za-z]", "", tokens[symbol_idx]).capitalize()
            if symbol:
                counter[symbol] += 1
            index += 1
        if counter:
            return counter
    return counter


def formula_dict_from_cif(cif_path: Path) -> dict[str, float]:
    cif_text = cif_path.read_text(encoding="utf-8", errors="ignore")
    for tag in ("_chemical_formula_sum", "_chemical_formula_moiety"):
        formula_text = _extract_cif_tag(cif_text, tag)
        if formula_text:
            formula_dict = parse_formula(formula_text)
            if formula_dict:
                return formula_dict

    atom_counts = _count_atom_site_symbols(cif_text)
    if not atom_counts:
        raise ValueError(f"Could not resolve formula from CIF: {cif_path}")

    values = [int(value) for value in atom_counts.values() if value > 0]
    z_value = _extract_z_value(cif_text)
    if z_value and all(value % z_value == 0 for value in values):
        divisor = z_value
    else:
        divisor = reduce(gcd, values)
    return {elem: value / divisor for elem, value in sorted(atom_counts.items())}


def molecular_weight(formula_dict: dict[str, float]) -> float:
    return sum(ATOMIC_WEIGHTS[elem] * count for elem, count in formula_dict.items())


def oxygen_balance_co2(formula_dict: dict[str, float], mw: float) -> float:
    c_count = formula_dict.get("C", 0.0)
    h_count = formula_dict.get("H", 0.0)
    o_count = formula_dict.get("O", 0.0)
    cl_count = formula_dict.get("Cl", 0.0)
    return 1600.0 * (o_count - 2.0 * c_count - (h_count - cl_count) / 2.0) / mw


def calc_ob(formula: str) -> float:
    formula_dict = parse_formula(formula)
    return oxygen_balance_co2(formula_dict, molecular_weight(formula_dict))


def calc_ob_from_formula_dict(formula_dict: dict[str, float]) -> float:
    return oxygen_balance_co2(formula_dict, molecular_weight(formula_dict))


def verify_reference_ob() -> None:
    for name, record in {**NEW_MATERIALS, **BENCHMARKS}.items():
        ob_calc = calc_ob(str(record["formula"]))
        if abs(ob_calc - float(record["OB_ref"])) > 0.35:
            warnings.warn(
                f"{name}: calculated OB ({ob_calc:.2f}) differs from reference value ({record['OB_ref']:.2f}).",
                stacklevel=2,
            )


def read_xy_csv(path: Path, x_col: int, y_col: int) -> tuple[np.ndarray, np.ndarray]:
    x_vals = []
    y_vals = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) <= max(x_col, y_col):
                continue
            try:
                x_val = float(row[x_col])
                y_val = float(row[y_col])
            except (TypeError, ValueError):
                continue
            x_vals.append(x_val)
            y_vals.append(y_val)
    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def normalize(y_vals: np.ndarray) -> np.ndarray:
    y_vals = np.asarray(y_vals, dtype=float)
    if y_vals.size == 0:
        return y_vals
    y_vals = y_vals - np.nanmin(y_vals)
    y_max = np.nanmax(y_vals)
    return y_vals / y_max if y_max > 0 else y_vals


def normalize_signed(y_vals: np.ndarray) -> np.ndarray:
    y_vals = np.asarray(y_vals, dtype=float)
    if y_vals.size == 0:
        return y_vals
    y_vals = y_vals - np.nanmedian(y_vals)
    scale = np.nanmax(np.abs(y_vals))
    return y_vals / scale if scale > 0 else y_vals


def load_pxrd_data() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    pxrd_data = {}
    for name, spec in PXRD_SPECS.items():
        meas_path, meas_xcol, meas_ycol = spec["meas"]
        sim_path, sim_xcol, sim_ycol = spec["sim"]
        meas_x, meas_y = read_xy_csv(meas_path, meas_xcol, meas_ycol)
        sim_x, sim_y = read_xy_csv(sim_path, sim_xcol, sim_ycol)
        pxrd_data[name] = {
            "meas": (meas_x, normalize(meas_y)),
            "sim": (sim_x, normalize(sim_y)),
        }
    return pxrd_data


def _load_train_pems_from_source(table_path: Path, cif_dir: Path) -> list[dict[str, float | str]]:
    if not table_path.exists():
        return []

    table_by_name: dict[str, dict[str, str]] = {}
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material = (row.get("material") or "").strip()
            if material:
                table_by_name[material] = row

    loaded_rows = []
    for material in TRAINING_MATERIALS:
        row = table_by_name.get(material)
        if row is None:
            continue
        d_text = (row.get("D_km_s") or "").strip()
        cif_path = cif_dir / f"{material}.cif"
        if not d_text or not cif_path.exists():
            continue
        try:
            formula_dict = formula_dict_from_cif(cif_path)
            d_kj = float(d_text) * 1000.0
        except (ValueError, KeyError):
            continue
        loaded_rows.append(
            {
                "name": material,
                "formula": format_formula(formula_dict),
                "formula_dict": formula_dict,
                "D_KJ": d_kj,
            }
        )
    return loaded_rows


def load_train_pems_from_sources() -> list[dict[str, float | str]]:
    candidates = [
        (PEMS_DATA_ROOT / "pems.csv", PEMS_DATA_ROOT / "confs"),
    ]
    best_rows: list[dict[str, float | str]] = []
    for table_path, cif_dir in candidates:
        rows = _load_train_pems_from_source(table_path, cif_dir)
        if len(rows) == len(TRAINING_MATERIALS):
            return rows
        if len(rows) > len(best_rows):
            best_rows = rows
    return best_rows


def load_train_pems() -> list[dict[str, float | str]]:
    primary_rows = load_train_pems_from_sources()
    if len(primary_rows) == len(TRAINING_MATERIALS):
        return primary_rows

    return primary_rows


def inspect_missing_inputs(train_pems: list[dict[str, float | str]]) -> list[str]:
    missing = []
    for name, spec in PXRD_SPECS.items():
        for role in ("meas", "sim"):
            path = spec[role][0]
            if not path.exists():
                missing.append(f"Missing PXRD {role} file for {name}: {path}")
    for name in MATERIAL_ORDER:
        cif_path = NEW_CIF_DIR / f"{name}.cif"
        if not cif_path.exists():
            missing.append(f"Missing structure CIF for {name}: {cif_path}")
    if not CLUSTER_UMAP_CACHE.exists():
        missing.append(f"Missing cluster UMAP cache: {CLUSTER_UMAP_CACHE}")
    return missing


def _load_dsc_sheet(xl: pd.ExcelFile, sheet_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    df = xl.parse(sheet_name, header=None)
    start_rows = df[df[0] == "StartOfData"].index.tolist()
    if not start_rows:
        return None, None
    start_row = start_rows[0]
    data = df.iloc[start_row + 1 :].copy()
    data = data[pd.to_numeric(data[0], errors="coerce") >= 0]
    temp = pd.to_numeric(data[1], errors="coerce")
    heat_flow = pd.to_numeric(data[2], errors="coerce")
    mask = temp.notna() & heat_flow.notna()
    return temp[mask].to_numpy(dtype=float), heat_flow[mask].to_numpy(dtype=float)


_TOPOLOGY_BUNDLE_CACHE: dict[str, tuple[object, np.ndarray, list[str]]] = {}


def _load_topology_bundle(name: str) -> tuple[object, np.ndarray, list[str]]:
    if name in _TOPOLOGY_BUNDLE_CACHE:
        return _TOPOLOGY_BUNDLE_CACHE[name]
    cif_path = NEW_CIF_DIR / f"{name}.cif"
    crys = topology_helpers.read_mol_crystal(str(cif_path))
    sa = topology_helpers.StoichiometryAnalyzer(crys)
    _, mol_types = topology_helpers.classify(crys, sa)
    centroids, point_types = topology_helpers.get_centroids(crys, mol_types)
    bundle = (crys, centroids, point_types)
    _TOPOLOGY_BUNDLE_CACHE[name] = bundle
    return bundle


def _expand_axis_limits(ax: plt.Axes, factor: float = 1.08) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0) * factor
    hy = 0.5 * (y1 - y0) * factor
    ax.set_xlim(cx - hx, cx + hx)
    ax.set_ylim(cy - hy, cy + hy)


_TOPOLOGY_BUNDLE_CACHE: dict[str, tuple[object, np.ndarray, list[str]]] = {}


def _load_topology_bundle(name: str) -> tuple[object, np.ndarray, list[str]]:
    if name in _TOPOLOGY_BUNDLE_CACHE:
        return _TOPOLOGY_BUNDLE_CACHE[name]
    cif_path = NEW_CIF_DIR / f"{name}.cif"
    crys = topology_helpers.read_mol_crystal(str(cif_path))
    sa = topology_helpers.StoichiometryAnalyzer(crys)
    _, mol_types = topology_helpers.classify(crys, sa)
    centroids, point_types = topology_helpers.get_centroids(crys, mol_types)
    bundle = (crys, centroids, point_types)
    _TOPOLOGY_BUNDLE_CACHE[name] = bundle
    return bundle


def _expand_axis_limits(ax: plt.Axes, factor: float = 1.08) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0) * factor
    hy = 0.5 * (y1 - y0) * factor
    ax.set_xlim(cx - hx, cx + hx)
    ax.set_ylim(cy - hy, cy + hy)


def _enlarge_triad(ax: plt.Axes) -> None:
    """Bump the small 'a/b' axis-triad produced by draw_crystal_proj so it is legible."""
    for text in list(ax.texts):
        if text.get_text() in ("a", "b", "c"):
            text.set_fontsize(9.0)
            text.set_fontweight("normal")
            text.set_fontstyle("italic")
            text.set_color(COLORS["charcoal"])
    for child in list(ax.get_children()):
        if isinstance(child, mtext.Annotation):
            arrow = getattr(child, "arrow_patch", None)
            if arrow is None:
                continue
            arrow.set_linewidth(1.4)
            arrow.set_color(COLORS["charcoal"])
            if hasattr(arrow, "set_mutation_scale"):
                arrow.set_mutation_scale(14)


def _strip_topology_overlay_text(ax: plt.Axes) -> None:
    """Remove the in-axes 'a/b' triad arrows+labels drawn by plot_topology.

    plot_topology places both the arrow (via ``ax.annotate('', ...)``) and the
    label (via ``ax.text(...)``) in DATA coordinates so they survive xlim/ylim
    overrides. We strip them here so we can redraw a uniform, left-aligned
    triad below. The panel title is not in ``ax.texts`` (lives on ``ax.title``),
    so it is safe to remove every short 'a'/'b'/'c' text child.
    """
    for child in list(ax.get_children()):
        if isinstance(child, mtext.Annotation):
            arrow = getattr(child, "arrow_patch", None)
            if arrow is not None:
                child.remove()
    for text in list(ax.texts):
        if text.get_text() in ("a", "b", "c"):
            text.remove()


def _topology_uniform_limits(axes: list[plt.Axes]) -> None:
    """Stamp identical xlim/ylim extents on every topology tile.

    Each tile's own draw_crystal_proj already centres the focus cell, so we
    pad each tile out to the max half-span across the row. This forces a
    consistent physical scale without editing plot_topology.py.
    """
    half_widths = []
    half_heights = []
    centers = []
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        half_widths.append(0.5 * (x1 - x0))
        half_heights.append(0.5 * (y1 - y0))
        centers.append((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
    if not axes:
        return
    hw = max(half_widths)
    hh = max(half_heights)
    for ax, (cx, cy) in zip(axes, centers):
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hh, cy + hh)


def _draw_left_aligned_axis_key(
    ax: plt.Axes,
    labels: list[str],
    vectors_2d: list[tuple[float, float]],
    anchor: tuple[float, float] = (0.05, 0.08),
    row_gap: float = 0.115,
    arrow_scale: float = 0.13,
    label_pad: float = 0.055,
    fontsize: float = 8.8,
) -> None:
    """Draw a legend-style crystallographic axis key in the lower-left corner.

    The italic labels sit in a vertical column all sharing the same left
    x-coordinate (``anchor[0]``), rendered with ``ha='left'``. Each label is
    followed immediately to its right by a short arrow pointing in the
    projected direction of that crystallographic axis. Arrow lengths are
    normalised so the longest vector equals ``arrow_scale`` (in axes-fraction
    units), preserving the relative lengths of the projections.

    ``labels[0]`` sits on top, ``labels[-1]`` on the bottom — so callers that
    want the conventional (c top, a bottom) stacking should pass
    ``labels=['c', 'b', 'a']`` with matching projection vectors.

    Both the label text and the arrow patch have ``clip_on=False`` so the
    triad is rendered in full even when the anchor point is close to the
    axes frame — the previous round hit intermittent clipping when the
    arrow tip sat right on the border.
    """
    if not labels or not vectors_2d or len(labels) != len(vectors_2d):
        return
    norms = [math.hypot(float(dx), float(dy)) for dx, dy in vectors_2d]
    max_norm = max(norms) if norms else 0.0
    if max_norm <= 1.0e-8:
        return
    ax_x, ax_y = anchor
    n = len(labels)
    for idx, (label, (dx, dy), norm) in enumerate(zip(labels, vectors_2d, norms)):
        row_y = ax_y + (n - 1 - idx) * row_gap
        text_obj = ax.text(
            ax_x,
            row_y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontstyle="italic",
            ha="left",
            va="center",
            color=COLORS["charcoal"],
            zorder=31,
        )
        text_obj.set_clip_on(False)
        if norm <= 1.0e-8:
            continue
        arrow_start = (ax_x + label_pad, row_y)
        length = arrow_scale * (norm / max_norm)
        arrow_end = (
            arrow_start[0] + length * (float(dx) / norm),
            arrow_start[1] + length * (float(dy) / norm),
        )
        arrow = FancyArrowPatch(
            posA=arrow_start,
            posB=arrow_end,
            arrowstyle="-|>",
            mutation_scale=6.0,
            linewidth=1.0,
            color=COLORS["charcoal"],
            transform=ax.transAxes,
            clip_on=False,
            zorder=30,
        )
        ax.add_patch(arrow)


def _draw_classical_triad(
    ax: plt.Axes,
    labels: list[str],
    vectors_2d: list[tuple[float, float]],
    center: tuple[float, float] = (0.86, 0.86),
    extent: float = 0.13,
    min_length_frac: float = 0.55,
    fontsize: float = 8.0,
    label_pad: float = 0.020,
    arrow_color: str = "#1A1A1A",
    halo_color: str = "white",
    halo_lw: float = 2.0,
    arrow_head_scale: float = 4.5,
    origin_dot_radius: float = 0.0055,
) -> None:
    """Classical axis triad with arrows emanating from a single origin.

    *Transparent* compass-rose style: no opaque background box. The user
    objected to the earlier white-box version on the grounds that it
    obstructed atoms/topology behind it (i.e. it occluded the data instead
    of merely overlaying a key). Instead each arrow and label is drawn
    twice — a wide white "halo" pass behind a narrower charcoal core pass
    via :func:`matplotlib.patheffects.withStroke`. The reader sees the
    triad pop against any background (white margin, atoms, bonds, dots)
    without any visual region being hidden.

    Arrow geometry: all arrows originate at ``center`` (axes-fraction).
    Lengths are computed in **display pixels**, not axes-fraction, so a
    wide-short tile (such as panel a's topology projection) does not stretch
    the horizontal axis or squash the vertical one — basis-vector keys are
    expected to read as a true compass-rose with equal physical arm lengths
    regardless of the host axes' aspect ratio.

    ``extent`` is interpreted as a fraction of the **smaller** axes-box
    dimension (in pixels), then turned into a fixed pixel arrow length.
    Each arrow tip is computed in display coords as
    ``origin_pixel + length_pixel * unit_direction`` and converted back to
    axes-fraction so :class:`FancyArrowPatch` (which still uses
    ``transform=ax.transAxes``) renders an arrow of exactly that pixel length
    along the requested direction. ``norm / max_norm`` is retained for callers
    that want to encode relative lengths; for unit-vector input — the typical
    basis-vector case — every arrow ends up at the same pixel length.
    """
    if not labels or not vectors_2d or len(labels) != len(vectors_2d):
        return
    norms = [math.hypot(float(dx), float(dy)) for dx, dy in vectors_2d]
    max_norm = max(norms) if norms else 0.0
    if max_norm <= 1.0e-8:
        return

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

    def _add_arrow(tip_axes, halo: bool):
        # ``shrinkA=0`` forces the arrow to start *exactly* at the origin
        # rather than leaving FancyArrowPatch's default 2-pt gap that
        # makes the three axes look detached from a common centre.
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

    # A tiny solid dot at the origin so the three arrows visually share
    # one common starting point even after antialiasing. Drawn UNDER the
    # arrows (lower zorder) so the arrow shafts hide its inner pixels.
    # Rendered as an Ellipse sized in pixels (then converted back to axes
    # fraction) so it stays circular regardless of axes aspect ratio.
    if origin_dot_radius > 0.0:
        from matplotlib.patches import Ellipse as _Ellipse
        dot_pix = float(origin_dot_radius) * box_size_pix
        # Convert the desired pixel radius to axes-fraction along each axis
        # independently so the displayed shape is a true circle.
        ax_dx, ax_dy = _axes_from_pixel((disp_origin[0] + dot_pix, disp_origin[1]))
        rx_axes = abs(ax_dx - cx)
        ax_dx, ax_dy = _axes_from_pixel((disp_origin[0], disp_origin[1] + dot_pix))
        ry_axes = abs(ax_dy - cy)
        halo_dot = _Ellipse(
            (cx, cy), 2 * rx_axes * 1.55, 2 * ry_axes * 1.55,
            transform=ax.transAxes,
            facecolor=halo_color, edgecolor="none",
            zorder=40, clip_on=False,
        )
        core_dot = _Ellipse(
            (cx, cy), 2 * rx_axes, 2 * ry_axes,
            transform=ax.transAxes,
            facecolor=arrow_color, edgecolor="none",
            zorder=42, clip_on=False,
        )
        ax.add_patch(halo_dot)
        ax.add_patch(core_dot)

    for label, (dx, dy), norm in zip(labels, vectors_2d, norms):
        if norm <= 1.0e-8:
            # Out-of-plane axis: label-only at the origin, italic + halo'd.
            t = ax.text(
                cx, cy - label_pad * 0.6,
                label,
                transform=ax.transAxes,
                fontsize=max(8.0, fontsize * 0.9),
                fontstyle="italic",
                fontweight="bold",
                ha="center",
                va="top",
                color=arrow_color,
                zorder=44,
            )
            t.set_path_effects([withStroke(linewidth=halo_lw, foreground=halo_color)])
            t.set_clip_on(False)
            continue
        ux = float(dx) / norm
        uy = float(dy) / norm
        pix_len = max(pix_extent * (norm / max_norm), pix_arrow_min)
        disp_tip = (
            disp_origin[0] + pix_len * ux,
            disp_origin[1] + pix_len * uy,
        )
        tip = _axes_from_pixel(disp_tip)
        ax.add_patch(_add_arrow(tip, halo=True))
        ax.add_patch(_add_arrow(tip, halo=False))
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


def _normalise_axis_vectors(vectors_2d: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for dx, dy in vectors_2d:
        norm = math.hypot(float(dx), float(dy))
        if norm <= 1.0e-8:
            out.append((0.0, 0.0))
        else:
            out.append((float(dx) / norm, float(dy) / norm))
    return out


def _topology_axis_triad(ax: plt.Axes,
                          vh_norm: np.ndarray | None = None,
                          vv_norm: np.ndarray | None = None) -> None:
    """Panel-a axis key: classical triad with arrows from a common origin.
    
    Arrows are drawn with **equal length** (basis vectors) regardless of the
    actual lattice parameter magnitudes. The direction is preserved from the
    projected lattice vectors, but both are normalized to unit length so
    _draw_classical_triad renders them identically."""
    if vh_norm is not None and vv_norm is not None:
        # Ensure both vectors have exactly norm=1.0 for equal arrow lengths
        vh_unit = vh_norm / np.linalg.norm(vh_norm)
        vv_unit = vv_norm / np.linalg.norm(vv_norm)
        vectors = [
            (float(vh_unit[0]), float(vh_unit[1])),  # a
            (float(vv_unit[0]), float(vv_unit[1])),  # b
        ]
    else:
        vectors = [(1.0, 0.0), (0.0, 1.0)]
    # Anchor the triad in the dedicated whitespace lane *below* the
    # tile, sharing its horizontal strip with the A/B/X composition
    # text but staying in the leftmost ~15% reserved for it (the
    # composition entries start at tile_w * 0.30). ``clip_on=False`` is
    # already set inside ``_draw_classical_triad`` so arrows + labels
    # at a negative axes-fraction y stay visible. See FIGURE_QA
    # section 3.2.
    _draw_classical_triad(
        ax,
        labels=["a", "b"],
        vectors_2d=vectors,
        center=(0.05, -0.10),
        extent=0.060,
        fontsize=8.0,
    )


def _topology_projected_axes(name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return projected in-plane unit vectors for lattice axes shown in the
    topology tile (matching plot_topology's ``pick_layer_proj``)."""
    try:
        crys, centroids, point_types = _load_topology_bundle(name)
    except Exception:
        return None, None
    try:
        lat = np.asarray(crys.lattice, dtype=float)
        stk, ip = topology_helpers.pick_layer_proj(crys, centroids, point_types)
        eh = lat[ip[0]].copy()
        eh /= np.linalg.norm(eh)
        ev = lat[stk].copy()
        ev -= np.dot(ev, eh) * eh
        norm_ev = np.linalg.norm(ev)
        if norm_ev < 1.0e-8:
            return None, None
        ev /= norm_ev
    except Exception:
        return None, None

    def to2d(vec):
        return np.array([float(np.dot(vec, eh)), float(np.dot(vec, ev))])

    vh = to2d(lat[ip[0]])
    vv = to2d(lat[stk])
    if np.linalg.norm(vh) < 1.0e-8 or np.linalg.norm(vv) < 1.0e-8:
        return None, None
    rot_deg = float(topology_helpers.best_proj_rotation(vh, vv))
    vh = topology_helpers.rotate2d(vh[None, :], rot_deg)[0]
    vv = topology_helpers.rotate2d(vv[None, :], rot_deg)[0]
    vh /= np.linalg.norm(vh)
    vv /= np.linalg.norm(vv)
    return vh, vv


_FIG2D_POLY_MODULE = None


def _load_fig2d_polyhedra_module():
    """Load Figure 2d's unit-cell polyhedron renderer for Figure 5a."""
    global _FIG2D_POLY_MODULE
    if _FIG2D_POLY_MODULE is not None:
        return _FIG2D_POLY_MODULE
    _FIG2D_POLY_MODULE = fig2d_polyhedra
    return _FIG2D_POLY_MODULE


def _configure_fig5a_polyhedra_renderer(module) -> None:
    """Retarget Figure 2d's renderer to the three newly synthesized *PEP cells."""
    materials = [
        {
            "name": name,
            "stoich": r"ABX$_4$",
            "cif": NEW_CIF_DIR / f"{name}.cif",
            "x_family": "ClO4",
            "repeats": (1, 1, 1),
        }
        for name in MATERIAL_ORDER
    ]
    module.MATERIALS = materials
    module.MATERIAL_REPEATS = {item["name"]: tuple(item["repeats"]) for item in materials}
    module.PLOTLY_PANEL_DIR = THIS_DIR / "_figure5a_plotly_panels"

    def _crop_fig5a_panel(path: Path, *, pad: int = 14) -> None:
        if Image is None:
            return
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

    module._crop_white_border = _crop_fig5a_panel
    # Figure 5a legend: drop modifiers (irregular, distorted, etc.) — just
    # show the bare polyhedron name (e.g. "icosahedron", "bicapped cube").
    _original_polyhedron_name = module._polyhedron_name

    def _fig5a_polyhedron_name(record):
        name = _original_polyhedron_name(record)
        # Strip known modifiers; keep only the base shape name.
        for modifier in ("irregular", "distorted", "clean", "ambiguous"):
            name = name.replace(f"{modifier} ", "").replace(f" {modifier}", "")
        return name.strip() or "polyhedron"

    module._polyhedron_name = _fig5a_polyhedron_name
    # Keep the representative A/B polyhedra spatially separated where possible,
    # matching Figure 2d's reader-facing cell-box logic rather than the old
    # abstract topology projection.
    module.HIGHLIGHT_STRATEGY_OVERRIDE = {name: "separated" for name in MATERIAL_ORDER}
    original_shift = module.extract_coordination_shell.__globals__.get("_shift_hull_payload")

    def _robust_shift_hull_payload(hull, delta):
        vertices = (hull or {}).get("vertices") or []
        vertices_arr = np.asarray(vertices, dtype=float)
        if vertices_arr.size == 0 or vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
            return {"vertices": [], "simplices": [], "edges": []}
        if original_shift is not None:
            return original_shift(hull, delta)
        out = dict(hull)
        out["vertices"] = (vertices_arr + delta).tolist()
        return out

    # Some *PEP cleaned CIF fragments carry an empty MolCrysKit hull payload
    # encoded as ``[[]]``. Figure 2d's renderer can ignore such static hulls
    # because it rebuilds visible A/B polyhedra from shell coordinates.
    module.extract_coordination_shell.__globals__["_shift_hull_payload"] = _robust_shift_hull_payload

    def _robust_replicate_polyhedra(records, M, repeats):
        a, b, c = np.asarray(M, dtype=float)[0], np.asarray(M, dtype=float)[1], np.asarray(M, dtype=float)[2]
        out = []
        for ia in range(repeats[0]):
            for ib in range(repeats[1]):
                for ic in range(repeats[2]):
                    shift = ia * a + ib * b + ic * c
                    for rec in records:
                        shell_coords = np.asarray(rec.get("shell_coords", []), dtype=float)
                        if shell_coords.size == 0:
                            shell_coords = np.zeros((0, 3), dtype=float)
                        elif shell_coords.ndim == 1:
                            shell_coords = shell_coords.reshape((-1, 3))
                        center_coords = np.asarray(rec.get("center_coords", [0.0, 0.0, 0.0]), dtype=float)
                        out.append({
                            **rec,
                            "shell_coords": shell_coords + shift,
                            "center_coords": center_coords + shift,
                            "replica": (ia, ib, ic),
                        })
        return out

    module._replicate_polyhedra = _robust_replicate_polyhedra
    original_collect = module._collect_site_records

    def _classify_fallback_shell(shell_coords, center_coords):
        try:
            shape = module._molcrys_classify_shell(shell_coords, center=center_coords)
        except Exception:
            shape = None
        if not shape or not shape.get("primary_label"):
            return None
        return {
            "name": str(shape["primary_label"]),
            "label_modifier": str(shape.get("label_modifier") or "clean"),
            "cshm": shape.get("cshm_value"),
        }

    def _nearest_gap_bx_record(bundle, material_name: str, frag: dict) -> dict | None:
        M = np.asarray(bundle.M, dtype=float)
        center = np.asarray(frag.get("center"), dtype=float)
        x_frags = [item for item in bundle.topology_fragment_table if item.get("type") == "X"]
        candidates = []
        for x_frag in x_frags:
            x_center = np.asarray(x_frag.get("center"), dtype=float)
            for ia in (-1, 0, 1):
                for ib in (-1, 0, 1):
                    for ic in (-1, 0, 1):
                        shift = ia * M[0] + ib * M[1] + ic * M[2]
                        coord = x_center + shift
                        dist = float(np.linalg.norm(coord - center))
                        candidates.append((dist, int(x_frag["index"]), (ia, ib, ic), coord, x_frag))
        if len(candidates) < 4:
            return None
        candidates.sort(key=lambda row: row[0])
        dists = np.asarray([row[0] for row in candidates], dtype=float)
        search_mask = dists <= 12.0
        search_n = int(np.count_nonzero(search_mask))
        if search_n < 4:
            return None
        # Match the DEP/Figure-2d intent: use the natural distance gap, not an
        # arbitrary hard radius. For these *PEP B-X shells the dominant gap is
        # after ten ClO4 neighbours, the same CN returned by MCK for MPEP.
        max_rank = min(search_n, 14)
        gaps = np.diff(dists[: max_rank + 1])
        if gaps.size == 0:
            return None
        cn = int(np.argmax(gaps) + 1)
        if cn < 4:
            return None
        picked = candidates[:cn]
        shell_coords = np.asarray([row[3] for row in picked], dtype=float)
        shell_distances = [float(row[0]) for row in picked]
        all_distances = [float(row[0]) for row in candidates[:max(search_n, cn)]]
        shell = [
            {
                "index": int(row[1]),
                "center": row[3].tolist(),
                "distance": float(row[0]),
                "image_shift": row[2],
            }
            for row in picked
        ]
        best_match = _classify_fallback_shell(shell_coords, center)
        return {
            "material_name": material_name,
            "site_type": "B",
            "bundle": bundle,
            "center_fragment": frag,
            "center_formula": module._formula(frag, bundle),
            "label": frag.get("label", f"B{frag['index']}"),
            "fragment_index": int(frag["index"]),
            "cn": cn,
            "distances": shell_distances,
            "all_distances": all_distances,
            "gap_info": {
                "coordination_number": cn,
                "mode": "nearest-gap fallback",
                "primary_gap_cn": cn,
                "gap_index": cn - 1,
                "gap_value": float(gaps[cn - 1]) if cn - 1 < len(gaps) else None,
                "enclosed": None,
                "enclosure_expanded": False,
                "cutoff": None,
                "search_cutoff": 12.0,
                "hard_cutoff": None,
            },
            "shell_coords": shell_coords,
            "center_coords": center,
            "shell_fragments": shell,
            "raw_shell": {
                "candidate_fragments": shell,
                "pool_coords": [row[3].tolist() for row in candidates[:search_n]],
                "all_distances": all_distances,
            },
            "best_match": best_match,
        }

    def _collect_fig5a_site_records(bundle, material_name: str, site_type: str):
        records = original_collect(bundle, material_name, site_type)
        if site_type != "B":
            return records
        out = []
        for record in records:
            if int(record.get("cn", 0)) >= 4:
                out.append(record)
                continue
            fallback = _nearest_gap_bx_record(bundle, material_name, record["center_fragment"])
            out.append(fallback if fallback is not None else record)
        return out

    module._collect_site_records = _collect_fig5a_site_records
    original_pick = module._pick_highlight_pair

    def _pick_visible_highlight_pair(a_records, b_records, cell_pts_rot, R, *, strategy):
        a_pick, b_pick = original_pick(a_records, b_records, cell_pts_rot, R, strategy=strategy)

        def _visible(rec):
            if rec is None or int(rec.get("cn", 0)) < 4:
                return None
            coords = np.asarray(rec.get("shell_coords", []), dtype=float)
            if coords.size == 0 or len(coords.reshape((-1, 3))) < 4:
                return None
            return rec

        return _visible(a_pick), _visible(b_pick)

    module._pick_highlight_pair = _pick_visible_highlight_pair


_PANEL_A_SUPERCELL_OVERRIDE = {
    # (nh, nv) = cells along the 2D projection's horizontal/vertical axes.
    # The one-row topology strip uses wide-short tiles. We aim for ~4-6
    # polyhedra per tile so all three compounds read at the same density
    # (PEP: 2x2 = 4, PEP-M: 3x2 = 6, PEP-H: 2x3 = 6); over-replicating one
    # tile makes the polyhedra look smaller than the others' and breaks
    # cross-tile visual comparability (FIGURE_QA section 1.2).
    "PEP":  (2, 2),
    "MPEP": (3, 2),
    "HPEP": (2, 3),
}


def _patch_topology_supercell_override() -> dict | None:
    """Replace plot_topology.draw_crystal_proj's local supercell override via
    module globals. plot_topology defines a dict literal inside the function,
    so we instead patch the module-level fallback by monkey-patching the
    function itself. This is the least invasive way to keep the four panels
    visually balanced."""
    if topology_helpers is None:
        return None
    original_override = getattr(topology_helpers, "_PANEL_A_SUPERCELL_OVERRIDE", None)
    topology_helpers._PANEL_A_SUPERCELL_OVERRIDE = dict(_PANEL_A_SUPERCELL_OVERRIDE)
    return original_override


def _draw_abx_composition_strip(
    fig: plt.Figure,
    axes_row: list[plt.Axes],
    names_row: list[str],
) -> None:
    """Draw a horizontal composition strip below a row of topology tiles.

    Shows A, B, X site identities for each material in a compact horizontal
    layout using color-coded text. Positioned in figure coordinates below the
    row of tiles. Each material gets three colored text entries spread
    horizontally within its tile's x-extent.
    """
    if not axes_row or not names_row:
        return

    site_colors = {
        "A": "#8A5A67",
        "B": "#6B7C4E",
        "X": "#5A6D7B",
    }

    # Get the y-position just below the row of axes
    fig.canvas.draw()
    y_bottom = min(ax.get_position().y0 for ax in axes_row)
    # Place the strip *below* the axes box (transparent figure margin)
    # so it does not overlay the periodic mu4-X lattice and so it leaves
    # a clean horizontal lane that the panel-a triad can share with it
    # without bumping into the data layer (FIGURE_QA section 1.1).
    strip_y = y_bottom - 0.005

    for ax, name in zip(axes_row, names_row):
        if name not in ABX4_SITES:
            continue

        sites = ABX4_SITES[name]
        pos = ax.get_position()
        tile_w = pos.x1 - pos.x0

        # Draw three color-coded entries spread horizontally; the
        # leftmost ~18% of the tile width is reserved for the axis triad.
        site_entries = [
            ("A", sites["A"]),
            ("B", sites["B"]),
            ("X", sites["X"]),
        ]
        for idx, (site_label, site_formula) in enumerate(site_entries):
            x_pos = pos.x0 + tile_w * (0.30 + 0.28 * idx)
            text = f"{site_label}: {site_formula}"
            fig.text(
                x_pos,
                strip_y,
                text,
                ha="center",
                va="top",
                fontsize=8.0,
                fontweight="bold",
                color=site_colors[site_label],
            )


def _retune_topology_artists(ax: plt.Axes) -> None:
    """Match panel-a line weights and make projected mu4-X markers visible."""
    purple = np.asarray(mcolors.to_rgb("#7C5CBF"))
    cell = np.asarray(mcolors.to_rgb("#2F2F2F"))
    grid = np.asarray(mcolors.to_rgb("#C0C0C0"))
    x_site = np.asarray(mcolors.to_rgb("#5A6D7B"))
    poly_gray = "#A8A8A8"
    mu4 = "#9B7B55"

    for line in ax.lines:
        color = np.asarray(mcolors.to_rgb(line.get_color()))
        if np.linalg.norm(color - purple) < 0.03:
            line.set_color(poly_gray)
            line.set_alpha(0.72)
            line.set_linewidth(0.5)
        elif np.linalg.norm(color - cell) < 0.03:
            line.set_linewidth(0.5)
        elif np.linalg.norm(color - grid) < 0.03:
            line.set_linewidth(0.25)

    for patch in ax.patches:
        edge = patch.get_edgecolor()
        face = patch.get_facecolor()
        edge_is_poly = (
            edge is not None
            and len(edge) >= 3
            and np.linalg.norm(np.asarray(edge[:3]) - purple) < 0.03
        )
        face_is_poly = (
            face is not None
            and len(face) >= 3
            and np.linalg.norm(np.asarray(face[:3]) - purple) < 0.03
        )
        if edge_is_poly or face_is_poly:
            patch.set_edgecolor(poly_gray)
            patch.set_facecolor("#D9D9D9")
            patch.set_alpha(0.08)
            patch.set_linewidth(0.5)

    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        face = collection.get_facecolor()
        if face.size == 0:
            continue
        rgb = np.asarray(face[0][:3], dtype=float)
        if np.linalg.norm(rgb - x_site) < 0.04:
            # Do not sparsify or choose representative mu4-X sites: the
            # projection should show the full periodic X-point lattice.
            collection.set_facecolor(mu4)
            collection.set_alpha(0.58)
            collection.set_sizes(np.asarray(collection.get_sizes()) * 1.12)


def panel_a_topology_row(fig: plt.Figure, subspec) -> list[plt.Axes]:
    """Panel a: Figure-2d-style unit cells with A/B coordination polyhedra."""
    poly = _load_fig2d_polyhedra_module()
    _configure_fig5a_polyhedra_renderer(poly)
    data = poly.load_material_data()
    view_extents = {
        name: poly._compute_view_extent({name: datum})
        for name, datum in data.items()
    }
    n_mat = len(MATERIAL_ORDER)
    n_cols = n_mat
    n_rows = (n_mat + n_cols - 1) // n_cols
    row = subspec.subgridspec(n_rows, n_cols, wspace=0.10, hspace=0.0)
    axes: list[plt.Axes] = []
    for idx, name in enumerate(MATERIAL_ORDER):
        slot = row[idx // n_cols, idx % n_cols]
        try:
            ax = poly.draw_material(fig, slot, data[name], view_extents[name])
        except Exception as exc:
            ax = fig.add_subplot(slot)
            draw_missing_panel(ax, f"Topology redraw failed:\n{exc}", name)
        axes.append(ax)

        pos = slot.get_position(fig)
        fig.text(
            0.5 * (pos.x0 + pos.x1),
            pos.y1 + 0.003,
            display_material(name),
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color=COLORS["charcoal"],
        )

    return axes


def panel_a_legend(fig: plt.Figure, axes_a: list[plt.Axes]) -> None:
    """Figure 2d renderer supplies per-tile A/B polyhedron keys."""
    return


def panel_b_legend(fig: plt.Figure, axes_b: list[plt.Axes]) -> None:
    """Element-colour key for panel b's ball-and-stick clusters.

    Without this key the reader has to guess that red = O, green = Cl, etc.
    Every Nature structural figure carries such a key. The swatch colours
    are pulled from ``STRUCTURE_ATOM_COLORS`` — the same dict that's fed
    into crystal_viewer's ``element_colors`` style override — so the legend
    cannot drift out of sync with the rendered atoms (a previous version
    used textbook CPK colours that did **not** match the muted palette
    actually drawn).
    """
    if not axes_b:
        return
    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=color,
               markeredgecolor="#3F3F3F" if elem == "H" else "white",
               markeredgewidth=0.45,
               markersize=6.0,
               label=elem)
        for elem, color in (
            ("H", STRUCTURE_ATOM_COLORS["H"]),
            ("C", STRUCTURE_ATOM_COLORS["C"]),
            ("N", STRUCTURE_ATOM_COLORS["N"]),
            ("O", STRUCTURE_ATOM_COLORS["O"]),
            ("Cl", STRUCTURE_ATOM_COLORS["Cl"]),
        )
    ]
    fig.canvas.draw()
    # Place element legend INSIDE panel b's row near the top, below the
    # axis triads (which sit ~bottom-right of each tile).  Centre it
    # horizontally and push it down just enough to clear the top edge while
    # still sitting above the actual molecular renderings.
    y_top = max(ax.get_position().y1 for ax in axes_b)
    fig_legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y_top - 0.004),
        ncol=len(handles),
        frameon=False,
        handletextpad=0.45,
        columnspacing=2.1,
        fontsize=8.0,
        handlelength=1.0,
    )
    fig_legend.set_zorder(40)


def _structure_camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elev = np.deg2rad(STRUCTURE_ELEV)
    azim = np.deg2rad(STRUCTURE_AZIM)
    cam = np.asarray(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )
    cam /= np.linalg.norm(cam)
    forward = -cam
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1.0e-8:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_img = np.cross(right, forward)
    up_img /= np.linalg.norm(up_img)
    return cam, right, up_img


def _project_structure(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam, right, up_img = _structure_camera_basis()
    x_vals = points @ right
    y_vals = points @ up_img
    depths = points @ cam
    return x_vals, y_vals, depths


def _select_structure_indices(symbols: list[str], positions: np.ndarray, keep_non_h: int = 38) -> list[int]:
    non_h = [idx for idx, sym in enumerate(symbols) if sym != "H"]
    if not non_h:
        return list(range(len(symbols)))
    center = positions[non_h].mean(axis=0)
    dist = np.linalg.norm(positions - center, axis=1)
    picked = sorted(non_h, key=lambda idx: dist[idx])[:keep_non_h]
    return sorted(picked)


def _structure_bonds(symbols: list[str], positions: np.ndarray) -> list[tuple[int, int]]:
    bonds = []
    for idx_a in range(len(symbols)):
        for idx_b in range(idx_a + 1, len(symbols)):
            sym_a = symbols[idx_a]
            sym_b = symbols[idx_b]
            if "H" in (sym_a, sym_b):
                continue
            if sym_a in METAL_ELEMENTS or sym_b in METAL_ELEMENTS:
                continue
            cutoff = STRUCTURE_COV_RADII.get(sym_a, 0.80) + STRUCTURE_COV_RADII.get(sym_b, 0.80) + 0.38
            if np.linalg.norm(positions[idx_a] - positions[idx_b]) < cutoff:
                bonds.append((idx_a, idx_b))
    return bonds


def _structure_label_texts(symbols: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    labels = []
    for sym in symbols:
        counts[sym] = counts.get(sym, 0) + 1
        labels.append(f"{sym}{counts[sym]}")
    return labels


def _pick_structure_label_indices(symbols: list[str], x_vals: np.ndarray, y_vals: np.ndarray, depths: np.ndarray) -> list[int]:
    projected = np.column_stack([x_vals, y_vals])
    chosen: list[int] = []
    for symbol, limit in (("N", 4), ("Cl", 2)):
        candidates = [idx for idx, sym in enumerate(symbols) if sym == symbol]
        candidates.sort(key=lambda idx: depths[idx], reverse=True)
        for idx in candidates:
            if len([j for j in chosen if symbols[j] == symbol]) >= limit:
                break
            if all(np.linalg.norm(projected[idx] - projected[j]) > 0.65 for j in chosen):
                chosen.append(idx)
    return chosen


def _draw_structure_labels(
    ax: plt.Axes,
    symbols: list[str],
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    depths: np.ndarray,
) -> None:
    chosen = _pick_structure_label_indices(symbols, x_vals, y_vals, depths)
    if not chosen:
        return
    label_texts = _structure_label_texts(symbols)
    center = np.array([np.mean(x_vals), np.mean(y_vals)], dtype=float)
    span = max(np.ptp(x_vals), np.ptp(y_vals), 1.0)
    for rank, idx in enumerate(chosen):
        point = np.array([x_vals[idx], y_vals[idx]], dtype=float)
        direction = point - center
        if np.linalg.norm(direction) < 1.0e-8:
            direction = np.array([1.0, 0.35 + 0.12 * rank], dtype=float)
        direction /= np.linalg.norm(direction)
        tangent = np.array([-direction[1], direction[0]], dtype=float)
        jitter = ((rank % 3) - 1) * 0.030 * span
        label_xy = point + 0.12 * span * direction + jitter * tangent
        ax.plot([point[0], label_xy[0]], [point[1], label_xy[1]], color="#9A9A9A", lw=0.45, zorder=20)
        ax.text(
            label_xy[0],
            label_xy[1],
            label_texts[idx],
            fontsize=8.0,
            ha="center",
            va="center",
            color=COLORS["charcoal"],
            zorder=21,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.96},
        )


# Per-compound element scope for panel b. We label every distinct atom of the
# listed elements (cation N centres and ClO4- chlorines for these
# perchlorate energetic salts); pixel positions of each labelled atom are
# detected from the rendered image so arrows always meet the atoms regardless
# of the auto-selected camera. C and O are left to the colour legend so the
# tile does not become unreadable.
PANEL_B_LABEL_ELEMENTS = ("N", "Cl")


def _detect_atom_centroids_from_image(
    image: np.ndarray,
    target_hex: str,
    *,
    color_tol: float = 70.0,
    min_area_frac: float = 4.0e-4,
) -> list[tuple[float, float]]:
    """Find pixel centroids of every atom whose dominant colour is ``target_hex``.

    Strategy:
    1. Mask pixels within Euclidean RGB distance ``color_tol`` of the target.
    2. Erode once to break thin bond bridges that share the half-bond colour.
    3. Connected-component label and keep every blob whose area exceeds
       ``min_area_frac`` of the canvas area (so the count adapts to the
       compound -- e.g. four cation N atoms vs. two perchlorate Cl atoms).
    4. Return centroids as ``(x_pixel, y_pixel)`` (matplotlib data coords for
       an ``imshow`` of the image), ordered left-to-right then top-to-bottom
       so numbering is stable across runs.
    """
    if image is None or image.size == 0:
        return []
    rgb = image[..., :3].astype(np.int32)
    if image.shape[-1] == 4:
        alpha = image[..., 3]
    else:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    target = np.array(
        [int(target_hex[i:i + 2], 16) for i in (1, 3, 5)],
        dtype=np.int32,
    )
    diff = np.linalg.norm(rgb - target, axis=2)
    mask = (diff < color_tol) & (alpha > 200)
    if not mask.any():
        return []
    try:
        from scipy import ndimage as ndi
    except Exception:
        return []
    eroded = ndi.binary_erosion(mask, iterations=1)
    if not eroded.any():
        eroded = mask
    labelled, n = ndi.label(eroded)
    if n == 0:
        return []
    sizes = ndi.sum(eroded, labelled, index=np.arange(1, n + 1))
    min_area = max(8.0, min_area_frac * image.shape[0] * image.shape[1])
    keep_idx = [int(idx + 1) for idx, sz in enumerate(sizes) if sz >= min_area]
    if not keep_idx:
        keep_idx = [int(np.argmax(sizes)) + 1]
    centroids: list[tuple[float, float]] = []
    for idx in keep_idx:
        ys, xs = np.where(labelled == idx)
        centroids.append((float(xs.mean()), float(ys.mean())))
    centroids.sort(key=lambda p: (p[0], p[1]))
    return centroids


def _draw_panel_b_atom_labels(ax: plt.Axes, image: np.ndarray, name: str) -> None:
    """Auto-place every N and Cl atom label for the panel-b rasterised renders.

    Placement uses :func:`_qa_check.whitespace_label_placement` (a greedy,
    distance-transform-driven search that prefers the deepest whitespace
    around each atom). Once the labels have been placed we re-validate the
    full panel against the rules in ``FIGURE_QA.md`` -- any violation
    raises so the bug shows up immediately rather than slipping into a
    silent overlap.

    The detector itself is image-based (colour-mask + connected
    components) so the arrow tails always land on the requested atom no
    matter which camera ``auto_view_dir`` picked for the compound.
    """
    if image is None:
        return

    height, width = image.shape[:2]
    fontsize_pt = 8.0

    # ---------------- gather every labelled atom ----------------
    atoms: list[tuple[str, tuple[float, float]]] = []
    for elem in PANEL_B_LABEL_ELEMENTS:
        target_hex = STRUCTURE_ATOM_COLORS.get(elem)
        if not target_hex:
            continue
        centroids = _detect_atom_centroids_from_image(image, target_hex)
        for idx, atom_xy in enumerate(centroids, start=1):
            atoms.append((f"{elem}{idx}", (float(atom_xy[0]), float(atom_xy[1]))))
    if not atoms:
        return

    cluster_center = (
        float(np.mean([a[1][0] for a in atoms])),
        float(np.mean([a[1][1] for a in atoms])),
    )

    forbidden_rects = [
        # axis triad lower-left
        (0.0, 0.66 * height, 0.30 * width, float(height)),
        # compound-name label bottom-centre
        (0.30 * width, 0.86 * height, 0.70 * width, float(height)),
        # element-colour legend top strip
        (0.0, 0.0, float(width), 0.14 * height),
    ]

    placement = _qa.whitespace_label_placement(
        image=image,
        atoms=atoms,
        cluster_center_px=cluster_center,
        fontsize_pt=fontsize_pt,
        forbidden_rects_px=forbidden_rects,
        atom_color_hexes=None,  # validate against every coloured pixel
        # 4 px at 600 dpi is ~0.17 mm at print -- enough to keep the
        # white label patch from touching an atom while leaving room
        # for the dense ClO4 / amine groups.
        min_atom_clearance_px=4.0,
        # 0.3 x text-height edge gap -- visually distinct without forcing
        # ClO4 quartet labels into the next tile.
        min_label_edge_distance_factor=0.3,
        dpi=PANEL_B_QA_DPI,
        # Tight polar grid just outside the atom's halo. The smallest
        # radius has to clear the H-atom envelope around peripheral N
        # atoms (~100 px at 600 dpi) before any "label adjacent" slot
        # is geometrically possible. Larger radii are tried only if no
        # close slot survives the QA filters.
        angle_step_deg=8.0,
        radius_search_factors=(
            0.30, 0.36, 0.42, 0.50, 0.58, 0.68, 0.78, 0.90,
        ),
        # Cap the leader at 6 x text-height (~400 px at 600 dpi for
        # 8 pt) -- enough room for ClO4 corner labels without letting
        # any label drift across half the tile.
        max_arrow_factor=6.0,
        # Skip the immediate covalent shell of the target atom (H/C
        # neighbours bonded to it) when checking "leader through atom"
        # -- those atoms visually belong to the target and a leader
        # that grazes them is not "pointing at the wrong sphere".
        # A typical N-H bond renders at ~70-80 px at 600 dpi.
        target_atom_radius_px=85.0,
    )

    for rec in placement:
        ax.annotate(
            rec.label,
            xy=rec.atom_xy_px,
            xytext=rec.text_xy_px,
            xycoords="data",
            textcoords="data",
            ha="center",
            va="center",
            fontsize=fontsize_pt,
            color=COLORS["charcoal"],
            zorder=55,
            arrowprops={
                "arrowstyle": "-",
                "color": "#7B7B7B",
                "lw": 0.45,
                "shrinkA": 1.5,
                "shrinkB": 4.0,
            },
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
        )

    errors = _qa.validate_atom_label_panel(
        panel=f"figure5/panel-b/{name}",
        image=image,
        records=placement,
        cluster_center_px=cluster_center,
        atom_color_hexes=None,
        min_atom_clearance_px=4.0,
        min_label_edge_distance_factor=0.3,
        dpi=PANEL_B_QA_DPI,
        max_arrow_factor=6.0,
        target_atom_radius_px=85.0,
    )
    if errors:
        for err in errors:
            print(f"FIGURE-QA  {err}", file=sys.stderr)
        raise RuntimeError(
            f"panel b QA failed for {name} -- see stderr for details"
        )


def _draw_axes_triad_2d(ax: plt.Axes, origin: tuple[float, float] = (0.08, 0.08), scale: float = 0.095) -> None:
    for (dx, dy), label in zip(((scale, 0.0), (0.0, scale), (-0.58 * scale, 0.58 * scale)), ("a", "b", "c")):
        arrow = FancyArrowPatch(
            posA=origin,
            posB=(origin[0] + dx, origin[1] + dy),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=COLORS["charcoal"],
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(arrow)
        ax.text(
            origin[0] + dx + 0.010,
            origin[1] + dy + 0.010,
            label,
            transform=ax.transAxes,
            fontsize=8.0,
            fontstyle="italic",
            ha="center",
            va="center",
            color=COLORS["charcoal"],
        )


def draw_structure_panel(ax: plt.Axes, name: str) -> None:
    cif_path = NEW_CIF_DIR / f"{name}.cif"
    if not cif_path.exists():
        draw_missing_panel(ax, f"Missing {name}.cif", name)
        return
    if ase_read is None:
        draw_missing_panel(ax, "ASE unavailable", name)
        return
    try:
        atoms = ase_read(str(cif_path))
    except Exception as exc:
        draw_missing_panel(ax, f"Could not read CIF:\n{exc}", name)
        return

    symbols_all = atoms.get_chemical_symbols()
    positions_all = np.asarray(atoms.get_positions(), dtype=float)
    keep_idx = _select_structure_indices(symbols_all, positions_all, keep_non_h=38)
    symbols = [symbols_all[idx] for idx in keep_idx]
    positions = positions_all[keep_idx] - positions_all[keep_idx].mean(axis=0)

    x_vals, y_vals, depths = _project_structure(positions)
    bonds = _structure_bonds(symbols, positions)
    bond_depth = [0.5 * (depths[idx_a] + depths[idx_b]) for idx_a, idx_b in bonds]
    for bond_idx in np.argsort(bond_depth):
        idx_a, idx_b = bonds[int(bond_idx)]
        ax.plot(
            [x_vals[idx_a], x_vals[idx_b]],
            [y_vals[idx_a], y_vals[idx_b]],
            color="#B9B9B9",
            lw=0.8,
            alpha=0.9,
            solid_capstyle="round",
            zorder=1,
        )
    for atom_idx in np.argsort(depths):
        sym = symbols[int(atom_idx)]
        ax.scatter(
            x_vals[int(atom_idx)],
            y_vals[int(atom_idx)],
            s=1.18 * STRUCTURE_ATOM_SIZES.get(sym, 18),
            c=STRUCTURE_ATOM_COLORS.get(sym, "#909090"),
            edgecolors="white",
            linewidths=0.35,
            zorder=2 + atom_idx / max(len(symbols), 1),
        )

    _draw_structure_labels(ax, symbols, x_vals, y_vals, depths)
    x_pad = 0.10 * max(np.ptp(x_vals), 1.0)
    y_pad = 0.16 * max(np.ptp(y_vals), 1.0)
    ax.set_xlim(np.min(x_vals) - x_pad, np.max(x_vals) + x_pad)
    ax.set_ylim(np.min(y_vals) - y_pad, np.max(y_vals) + y_pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _draw_axes_triad_2d(ax)
    ax.set_title(DISPLAY_TITLES[name], pad=1.8, fontsize=10.0, fontweight="bold")


# ── Panel b: formula-unit structure row via crystal_viewer ──────────────────
STRUCTURE_ROW_ELEV = 18.0      # camera elevation above the xy plane (deg)
STRUCTURE_ROW_AZIM = -58.0     # camera azimuth from +x (deg, CCW)
STRUCTURE_ROW_DPI = 600
# Plotly renders with aspectmode='cube' so the 3D scene fits the SHORTER
# canvas dimension. Square canvas + square matplotlib tiles means the cube
# fills the tile in BOTH dimensions, with no pillarbox margin where stray
# pixels go to waste. The 1x4-strip tile is also square (panel_b_height ≈
# tile_w) so the cluster occupies the whole tile, not just its centre.
STRUCTURE_ROW_PANEL_W_IN = 1.82
STRUCTURE_ROW_PANEL_H_IN = 1.82
STRUCTURE_ROW_TRIAD_SCALE = 0.13
STRUCTURE_ROW_TRIAD_ORIGIN = (0.09, 0.09)


def _structure_row_view_vectors() -> tuple[np.ndarray, np.ndarray]:
    elev = np.deg2rad(STRUCTURE_ROW_ELEV)
    azim = np.deg2rad(STRUCTURE_ROW_AZIM)
    view = np.asarray(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(view, up))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    return view, up


def _structure_row_basis(view_vec: np.ndarray, up_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (view_x, view_y) in 3D — the screen right/up directions used by crystal_viewer."""
    z = np.asarray(view_vec, dtype=float)
    z /= np.linalg.norm(z)
    up = np.asarray(up_vec, dtype=float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1.0e-6:
        x = np.cross(np.asarray([1.0, 0.0, 0.0], dtype=float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return x, y


def _structure_row_style() -> dict:
    if DEFAULT_STYLE is None or deep_merge is None:
        return {}
    return deep_merge(DEFAULT_STYLE, {
        "display_mode": "formula_unit",
        "show_title": False,
        "show_labels": False,
        "show_axes": False,
        "show_unit_cell": False,
        "show_hydrogen": True,
        # ORTEP rendering — anisotropic 50%-probability thermal ellipsoids
        # built from the CIF Uij. ``atom_scale = 1.0`` so the ellipsoid is
        # a *true* 50% probability surface (any other scale would lie
        # about the underlying ADP). Atoms with only Uiso (e.g. H) get an
        # equivalent isotropic ellipsoid; atoms with neither fall back to
        # the covalent display sphere.
        "atom_render": "ortep",
        "ortep_probability": 0.50,
        "atom_scale": 1.0,
        # Dense ellipsoid mesh — kaleido has no GPU MSAA so triangulation
        # edges become visible "jaggies" unless the mesh is fine enough
        # that adjacent normals nearly coincide. 30×48 = ~1440 triangles
        # per ellipsoid is the threshold where edges disappear at print
        # DPI even with the supersampling already done in
        # ``_structure_row_scene_to_image``.
        "ortep_lat_steps": 30,
        "ortep_lon_steps": 48,
        # H atoms are drawn as a small fixed sphere (no ADP — SHELX
        # constrains H Uiso to 1.2× parent which would give ridiculous
        # ellipsoids). 0.20 Å sits just slightly larger than the
        # ``bond_radius`` (0.16 Å) so an H atom reads as a small bead at
        # the bond's end, which is what crystallographers expect on an
        # ORTEP-style figure.
        "ortep_hydrogen_radius": 0.20,
        # Softer plotly Mesh3d lighting — the defaults produce a harsh
        # flat-shaded look with blown-out highlights. These values give
        # matte, evenly-lit ellipsoids with enough specular bite to read
        # as 3D without glare.
        "mesh_ambient": 0.52,
        "mesh_diffuse": 0.90,
        "mesh_specular": 0.18,
        "mesh_roughness": 0.82,
        "mesh_fresnel": 0.06,
        "bond_radius": 0.16,
        # Disorder visualisation: minor sites (SHELX PART 2 / alt-loc B)
        # are pushed to ~0.22 opacity so they read as a faint "ghost"
        # behind the major occupant — the standard ORTEP convention. The
        # renderer's opacity floor is now exposed as ``minor_opacity_floor``
        # (0.18 here, vs. the former hard-coded 0.48 that washed out the
        # disorder contrast).
        "major_opacity": 1.0,
        "minor_opacity": 0.22,
        "minor_opacity_floor": 0.18,
        "minor_bond_scale": 0.65,
        "element_colors": STRUCTURE_ATOM_COLORS,
        "element_colors_light": STRUCTURE_ATOM_COLORS,
        "background": "#FFFFFF",
        "topology_enabled": False,
        "fast_rendering": False,
        # The axis triad is now drawn post-hoc as a matplotlib overlay in
        # ``panel_b_structure_row`` using ``scene["projected_axes"]``. This
        # gives a single-origin ("compass-rose") triad on an opaque white
        # badge that shares the same style + font-size as panel a, and is
        # immune to imshow downscaling (which previously blurred the
        # in-plotly triad).
        "show_axis_key": False,
    })


def _structure_row_preset(name: str, style: dict) -> dict:
    """Preset that triggers crystal_viewer's auto_view_dir per compound.

    Omitting ``view_direction`` / ``up`` / ``camera`` makes
    ``_resolve_view`` fall back to ``ops.auto_view_dir``, which scores many
    candidate view vectors and picks the one that exposes the most atoms with
    minimal occlusion — the "bad view" issue we hit with a fixed camera.
    """
    return {
        "version": 1,
        "style": style,
        "structures": {
            name: {
                "show_hydrogen": True,
            },
        },
    }


def _structure_row_scene_to_image(scene: dict, style: dict, width_px: int, height_px: int) -> np.ndarray:
    """Render a plotly scene to a PNG with 4× supersampling antialiasing.

    Headless kaleido has **no GPU MSAA** — its WebGL output rasterises
    triangle edges as hard pixel steps, which then survive matplotlib's
    bilinear imshow scaling and look like ladder/jaggy artefacts on the
    final figure. Standard remedy is supersampled antialiasing: render
    plotly at 4× the target resolution, then downsample to the target
    with PIL's LANCZOS filter (a separable Lanczos-3 windowed sinc that
    averages ~6×6 source pixels per output pixel, which is enough to
    fully suppress any single-pixel ladder). The resulting PNG that
    enters matplotlib is already smooth, so imshow's nearest-neighbour
    fallback is enough to preserve sharpness without re-introducing
    aliasing during the matplotlib resize.
    """
    fig_plotly = build_figure(scene, style)
    fig_plotly.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    super_factor = 4
    png_bytes = fig_plotly.to_image(
        format="png", width=int(width_px), height=int(height_px), scale=super_factor
    )
    with Image.open(io.BytesIO(png_bytes)) as im:
        target = (int(width_px), int(height_px))
        if im.size != target:
            im = im.resize(target, Image.LANCZOS)
        return np.array(im.convert("RGBA"))


def _structure_row_triad(ax: plt.Axes, cell_vectors: np.ndarray | None,
                         view_vec: np.ndarray, up_vec: np.ndarray) -> None:
    """Panel-b axis key: ``c`` (top), ``b``, ``a`` (bottom) stacked, left-aligned.

    Arrows beside each label point in the direction that crystallographic
    axis projects onto the panel's 2D screen. Lengths are scaled so the
    longest projection fills the reserved key width; relative proportions are
    preserved so readers can still compare a/b/c lengths at a glance.
    """
    if cell_vectors is None:
        # Fallback: stylised identity triad when the cell is unavailable.
        vectors = [(-0.6, 0.6), (1.0, 0.0), (1.0, 0.0)]  # c, b, a
    else:
        x_hat, y_hat = _structure_row_basis(view_vec, up_vec)
        projected = np.column_stack([cell_vectors @ x_hat, cell_vectors @ y_hat])
        # cell rows are (a, b, c); we stack top→bottom as (c, b, a).
        vectors = [
            (float(projected[2, 0]), float(projected[2, 1])),
            (float(projected[1, 0]), float(projected[1, 1])),
            (float(projected[0, 0]), float(projected[0, 1])),
        ]
    _draw_left_aligned_axis_key(
        ax,
        labels=["c", "b", "a"],
        vectors_2d=vectors,
    )


def _panel_b_cell_vectors(name: str) -> np.ndarray | None:
    cif_path = NEW_CIF_DIR / f"{name}.cif"
    if not cif_path.exists() or ase_read is None:
        return None
    try:
        atoms = ase_read(str(cif_path))
    except Exception:
        return None
    try:
        cell = np.asarray(atoms.get_cell(), dtype=float)
    except Exception:
        return None
    if cell.shape != (3, 3):
        return None
    return cell


def panel_b_structure_row(fig: plt.Figure, subspec) -> list[plt.Axes]:
    """Panel b: 1x3 strip of formula-unit ball-and-stick renders.

    A single row of square-ish tiles is the geometry that lets the
    (roughly cubic) plotly cluster fill its tile. The previous 2x2 layout
    produced very wide tiles with the cluster centred in pillarbox white-
    space, wasting ~50% of the panel's pixels.

    Each compound gets its own best camera via ``auto_view_dir`` (occlusion-
    aware scoring). A shared world-cube viewport pins the four tiles to a
    single physical scale. Axis triads are drawn post-hoc as matplotlib
    overlays using ``scene["projected_axes"]`` so they match panel a's
    classical-triad style and stay readable at publication scale.
    """
    n_mat = len(MATERIAL_ORDER)
    # Treat the whole panel-b strip as a canvas and place enlarged square
    # image axes explicitly. This avoids GridSpec's tight cell clipping while
    # letting each rendered structure occupy more of the available row.
    bbox = subspec.get_position(fig)
    fig_w, fig_h = fig.get_size_inches()
    tile_h = bbox.height * 1.13
    tile_w = tile_h * (fig_h / fig_w)  # square in physical inches
    centers = np.linspace(
        bbox.x0 + bbox.width / (2.0 * n_mat),
        bbox.x1 - bbox.width / (2.0 * n_mat),
        n_mat,
    )
    y0 = bbox.y0 + 0.5 * (bbox.height - tile_h) + 0.005
    axes = [
        fig.add_axes([cx - 0.5 * tile_w, y0, tile_w, tile_h])
        for cx in centers
    ]

    if (build_scene_from_cif is None or build_figure is None
            or uniform_viewport is None or Image is None):
        for ax, name in zip(axes, MATERIAL_ORDER):
            draw_missing_panel(ax, "crystal_viewer unavailable", DISPLAY_TITLES[name])
        return axes

    style = _structure_row_style()

    scenes: list[dict] = []
    for name in MATERIAL_ORDER:
        cif_path = NEW_CIF_DIR / f"{name}.cif"
        if not cif_path.exists():
            scenes.append({"_error": FileNotFoundError(cif_path), "name": name})
            continue
        try:
            scene = build_scene_from_cif(
                name=name,
                cif_path=str(cif_path),
                title=display_material(name),
                preset=_structure_row_preset(name, style),
                show_hydrogen=True,
                display_mode="formula_unit",
            )
        except Exception as exc:  # noqa: BLE001
            scenes.append({"_error": exc, "name": name})
            continue
        scenes.append(scene)

    real_scenes = [s for s in scenes if "_error" not in s]
    if real_scenes:
        # Uniform viewport keeps all four compounds at the **same physical
        # length scale** (cube side = largest compound's radius-aware span),
        # at the cost of smaller compounds occupying less of their tile. We
        # shrink the cube by ~8% so the largest compound visually fills its
        # panel — aspectmode='cube' + a positive camera-eye distance in
        # Plotly leaves a bit of headroom regardless — while smaller
        # compounds still render big enough to read.
        # padding=0.25 Å so the cube has a small breathing margin around
        # the outermost ORTEP ellipsoid (each ellipsoid extends up to
        # ~0.3 Å beyond the atom centre at 50% probability). The atom-bbox
        # already uses the full ellipsoid extent internally; this extra
        # margin kills the last 1–2 % of edge clipping at the corners.
        vps = uniform_viewport(real_scenes, style=style, padding=0.25)
        for scene, vp in zip(real_scenes, vps):
            # Tighter shrink -> the largest compound fills ~80% of its
            # cube, smaller compounds stay comparable-but-smaller; this
            # keeps relative size information visible without collapsing
            # the smallest compound to a speck. ``uniform_viewport`` sets
            # half_span = 0.5 * max(radius_spans), i.e. the tightest cube
            # that just fits the largest cluster — so any shrink < 1.0
            # **literally clips that largest cluster** (atoms whose centres
            # sit beyond the shrunk cube are cut, and bonds dangle into
            # the void). Use shrink = 1.0 to guarantee zero truncation,
            # and rely on the now-square tile geometry to make the cluster
            # appear large rather than zooming the camera.
            shrink = 1.0
            cx, cy, cz = vp["center"]
            half = vp["half_span"] * shrink
            scene["viewport"] = {
                "x": [cx - half, cx + half],
                "y": [cy - half, cy + half],
                "z": [cz - half, cz + half],
                "center": [cx, cy, cz],
                "half_span": half,
            }

    width_px = int(STRUCTURE_ROW_PANEL_W_IN * STRUCTURE_ROW_DPI)
    height_px = int(STRUCTURE_ROW_PANEL_H_IN * STRUCTURE_ROW_DPI)

    for ax, name, scene in zip(axes, MATERIAL_ORDER, scenes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if "_error" in scene:
            draw_missing_panel(ax, f"Render failed:\n{scene['_error']}", DISPLAY_TITLES[name])
            continue
        try:
            image = _structure_row_scene_to_image(scene, style, width_px, height_px)
        except Exception as exc:  # noqa: BLE001
            draw_missing_panel(ax, f"Render failed:\n{exc}", DISPLAY_TITLES[name])
            continue
        # ``interpolation="lanczos"`` is the antialiased downsampler matching
        # the supersampling we already do inside
        # ``_structure_row_scene_to_image``: when matplotlib later writes the
        # figure at PDF/PNG output DPI it inevitably has to resample our
        # 1×-of-tile image to whatever pixel grid the canvas uses, and
        # ``"none"`` would alias hard on the resampled output. Lanczos kills
        # the residual ladder/jaggies on slanted ellipsoid edges, arrow
        # tips, and bond cylinders without softening the structure.
        ax.imshow(image, interpolation="lanczos")
        _draw_panel_b_atom_labels(ax, image, name)
        # Classical triad drawn post-hoc in matplotlib so it shares the
        # exact same style + font size as panel a. ``projected_axes`` is
        # exposed by build_scene_from_atoms in (a, b, c) order, as
        # (screen_right, screen_up) components — so we can use them
        # directly with the axes-fraction triad helper.
        proj = scene.get("projected_axes")
        labels = list(scene.get("axis_labels") or ["a", "b", "c"])[:3]
        if proj is not None and len(proj) >= len(labels):
            vectors = [(float(proj[i][0]), float(proj[i][1]))
                       for i in range(len(labels))]
            _draw_classical_triad(
                ax,
                labels=labels,
                vectors_2d=_normalise_axis_vectors(vectors),
                center=(0.10, 0.13),
                extent=0.085,
                fontsize=8.0,
            )
    return axes


def _benchmark_bar_colors(names: list[str]) -> list[str]:
    colors: list[str] = []
    for name in names:
        if name in NEW_MATERIALS:
            colors.append(COLORS.get(name, COLORS["mid_gray_dark"]))
        elif name in MIX_STARS:
            colors.append("#5A5A5A")
        else:
            colors.append(COLORS["mid_gray_dark"])
    return colors


def _style_benchmark_axis(
    ax: plt.Axes,
    names: list[str],
    values: list[float | None],
    ylabel: str,
    accent: str,
    ylim: tuple[float, float],
    title: str | None = None,
    show_xticklabels: bool = True,
) -> None:
    x_pos = np.arange(len(names), dtype=float)
    colors = _benchmark_bar_colors(names)
    missing_height = 0.055 * (ylim[1] - ylim[0])
    for x, name, value, color in zip(x_pos, names, values, colors):
        if value is None:
            ax.bar(
                x,
                missing_height,
                bottom=ylim[0],
                width=0.62,
                color="white",
                edgecolor=COLORS["mid_gray_dark"],
                linewidth=0.5,
                hatch="///",
                zorder=3,
            )
            ax.text(
                x,
                ylim[0] + missing_height + 0.015 * (ylim[1] - ylim[0]),
                "n/a",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=COLORS["mid_gray_dark"],
                rotation=90,
                zorder=4,
            )
        else:
            # MIX stars use white hatching on dark grey to read as
            # "benchmark group" without needing a legend entry; the
            # hatch colour is controlled by the bar's edgecolor.
            ax.bar(
                x,
                float(value),
                width=0.62,
                color=color,
                edgecolor="white" if name in MIX_STARS else "none",
                linewidth=0.0,
                hatch="///" if name in MIX_STARS else None,
                zorder=3,
            )
    ax.set_xlim(-0.65, len(names) - 0.35)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, color=COLORS["charcoal"], fontweight="normal")
    ax.set_xticks(x_pos)
    if show_xticklabels:
        # Compact rotated labels: 50° + ha="right" keeps each tick glyph
        # mostly under its own bar rather than trailing diagonally into
        # the adjacent subplot below. Font is trimmed to 5.5pt to fit in
        # the slimmer bottom-row budget created by the A4 figure height
        # (was 6.5pt at 32° — bled into panel e).
        ax.set_xticklabels([display_material(name) for name in names], rotation=60, ha="right", fontsize=8.0)
        for tick in ax.get_xticklabels():
            tick.set_color(COLORS["charcoal"])
        ax.tick_params(axis="x", pad=1.0)
    else:
        ax.set_xticklabels([])
    if title:
        ax.set_title(title, pad=3)
    style_axes_local(ax, grid_axis="y")
    ax.tick_params(axis="y", colors=COLORS["charcoal"])
    ax.tick_params(axis="x", colors=COLORS["charcoal"])


def panel_d_benchmark_bars(fig: plt.Figure, subspec) -> plt.Axes:
    """Panel d: detonation-velocity benchmark bar chart."""
    outer = subspec.subgridspec(2, 1, height_ratios=[1.0, 0.46], hspace=0.0)
    ax = fig.add_subplot(outer[0, 0])
    benchmark_names = ["TNT", "RDX", "HMX", "CL-20"]
    star_names = ["SY", "DAP-4", "EAP-4"]
    d_names = [*benchmark_names, *star_names, *MATERIAL_ORDER]
    d_values: list[float | None] = (
        [float(BENCHMARKS[name]["D_KJ"]) for name in benchmark_names]
        + [float(MIX_STARS[name]["D_KJ"]) for name in star_names]
        + [float(NEW_MATERIALS[name]["D_KJ"]) for name in MATERIAL_ORDER]
    )
    _style_benchmark_axis(
        ax,
        d_names,
        d_values,
        ylabel=r"$V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)",
        accent=COLORS["charcoal"],
        ylim=(6300.0, 9700.0),
        show_xticklabels=True,
    )
    # The bar groupings (reported / MIX stars / this work) are read from
    # bar styling alone -- solid grey (reported), white-hatched dark grey
    # (MIX stars) and the per-material accent colours (this work). A
    # floating legend would just repeat that information and crowd the
    # tile; removed per FIGURE_QA section 1.5 (information density).
    return ax


def render_si_td_benchmark() -> None:
    """Supplementary thermal-decomposition benchmark bar chart."""
    benchmark_names = ["TNT", "RDX", "HMX", "CL-20"]
    star_names = ["SY", "DAP-4", "EAP-4"]
    names = [*benchmark_names, *star_names, *MATERIAL_ORDER]
    values: list[float | None] = (
        [float(BENCHMARKS[name]["Td"]) for name in benchmark_names]
        + [float(MIX_STARS[name]["Td"]) for name in star_names]
        + [float(NEW_MATERIALS[name]["Td"]) for name in MATERIAL_ORDER]
    )
    fig, ax = plt.subplots(figsize=(6.7, 2.45))
    _style_benchmark_axis(
        ax,
        names,
        values,
        ylabel=r"$T_{\mathrm{d}}$ ($^\circ$C)",
        accent=COLORS["charcoal"],
        ylim=(180.0, 405.0),
        title="Thermal-decomposition benchmark",
        show_xticklabels=True,
    )
    dap_x = names.index("DAP-4")
    dap_peak = float(MIX_STARS["DAP-4"]["Td_peak"])
    ax.scatter(
        [dap_x],
        [dap_peak],
        marker="v",
        s=28,
        facecolor="#5A5A5A",
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    ax.text(
        dap_x + 0.15,
        dap_peak + 2.5,
        "peak",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=COLORS["charcoal"],
    )
    ax.legend(
        [
            Patch(facecolor=COLORS["mid_gray"], edgecolor="none"),
            Patch(facecolor="#5A5A5A", edgecolor="white", hatch="///"),
            tuple(Patch(facecolor=COLORS[name], edgecolor="none") for name in MATERIAL_ORDER),
        ],
        ["Reported", "MIX stars", "This work"],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.04),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.4,
        columnspacing=0.8,
        handlelength=2.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    )
    fig.tight_layout(pad=0.4)
    save_png_pdf(fig, THIS_DIR / "_si_td_benchmark")
    plt.close(fig)


def panel_c_pxrd(ax: plt.Axes) -> None:
    try:
        pxrd_data = load_pxrd_data()
    except FileNotFoundError as exc:
        draw_missing_panel(ax, f"Missing PXRD input:\n{exc}", "PXRD validation")
        return

    ax.set_title("PXRD validation", pad=3, loc="left")
    gap = 1.55
    sim_offset = 0.64
    line_scale = 0.78
    for index, name in enumerate(MATERIAL_ORDER):
        base_y = (len(MATERIAL_ORDER) - 1 - index) * gap
        meas_x, meas_y = pxrd_data[name]["meas"]
        sim_x, sim_y = pxrd_data[name]["sim"]
        ax.plot(meas_x, base_y + line_scale * meas_y, color=COLORS[name], lw=0.95, zorder=2)
        ax.plot(sim_x, base_y + sim_offset + 0.62 * sim_y, color=COLORS[f"{name}_sim"], lw=0.90, zorder=1)
        ax.text(49.5, base_y + 0.22, display_material(name), ha="right", va="center", fontsize=8.0, color=COLORS[name], fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=COLORS["charcoal"], lw=0.9, label="Measured"),
        Line2D([0], [0], color=COLORS["mid_gray_dark"], lw=0.9, label="Simulated"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax.set_xlim(5, 50)
    ax.set_ylim(-0.2, gap * len(MATERIAL_ORDER) - 0.15)
    ax.set_xlabel(r"2$\theta$ (deg)", labelpad=4)
    ax.set_ylabel("Normalized intensity", labelpad=4)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    style_axes_local(ax, grid_axis=None)
    ax.spines["left"].set_visible(False)


def panel_d_dumbbell(ax: plt.Axes) -> None:
    materials = MATERIAL_ORDER
    d_ref = np.asarray([float(NEW_MATERIALS[name]["D_KJ"]) for name in materials], dtype=float)
    d_pred = np.asarray([float(NEW_MATERIALS[name]["D_pred"]) for name in materials], dtype=float)
    d_std = np.asarray([float(NEW_MATERIALS[name]["D_pred_std"]) for name in materials], dtype=float)
    y_pos = np.arange(len(materials), dtype=float)

    for idx, name in enumerate(materials):
        ax.plot([d_ref[idx], d_pred[idx]], [y_pos[idx], y_pos[idx]], color=COLORS["charcoal"], lw=0.8, zorder=1)
        ax.scatter(
            d_ref[idx],
            y_pos[idx],
            s=40,
            facecolors="white",
            edgecolors=COLORS[name],
            linewidths=1.2,
            zorder=3,
        )
        ax.errorbar(
            d_pred[idx],
            y_pos[idx],
            xerr=d_std[idx],
            fmt="o",
            color=COLORS[name],
            markersize=5.5,
            capsize=2.5,
            zorder=4,
        )

    x_min = min(np.min(d_ref), np.min(d_pred - d_std)) - 65
    x_max = max(np.max(d_ref), np.max(d_pred + d_std)) + 65
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_material(name) for name in materials])
    ax.invert_yaxis()
    ax.set_ylim(len(materials) - 0.45, -0.45)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    style_axes_local(ax, grid_axis="x")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["charcoal"], markeredgewidth=1.0, markersize=4.8, label="K-J ref."),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["charcoal"], markeredgecolor=COLORS["charcoal"], markersize=4.8, label="Pred."),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.16),
        ncol=2,
        frameon=False,
        handletextpad=0.28,
        columnspacing=0.85,
        borderaxespad=0.0,
    )


def panel_e_td_bar(ax: plt.Axes) -> None:
    materials = MATERIAL_ORDER
    td_vals = [float(NEW_MATERIALS[name]["Td"]) for name in materials]
    x_pos = np.arange(len(materials), dtype=float)
    ax.bar(
        x_pos,
        td_vals,
        color=[COLORS[name] for name in materials],
        edgecolor="white",
        linewidth=0.6,
        width=0.62,
        zorder=3,
    )
    for benchmark, linestyle in (("RDX", ":"), ("HMX", "--"), ("CL-20", "-.")):
        td_ref = float(BENCHMARKS[benchmark]["Td"])
        ax.axhline(td_ref, color=COLORS["mid_gray_dark"], lw=0.8, ls=linestyle, zorder=1)
        ax.text(3.48, td_ref + 2.5, benchmark, fontsize=8.0, color=COLORS["mid_gray_dark"], ha="left", va="bottom")
    ax.set_title("ABX$_4$ thermal zoom", pad=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([display_material(name) for name in materials])
    for tick, name in zip(ax.get_xticklabels(), materials):
        tick.set_color(COLORS[name])
        tick.set_fontweight("bold")
    ax.set_ylabel(r"$T_{\mathrm{d}}$ ($^\circ$C)")
    ax.set_ylim(180, 380)
    style_axes_local(ax, grid_axis="y")
    add_panel_label_at(ax, "f")


def load_cluster_umap_cache(cache_path: Path = CLUSTER_UMAP_CACHE) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing cluster UMAP cache: {cache_path}\n"
            "Run _compute_cluster_umap.py before rendering Fig. 5."
        )
    with np.load(str(cache_path), allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def _cluster_umap_records(cache: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    materials = [str(item) for item in cache["materials"]]
    sets = [str(item) for item in cache["material_set"]]
    centroids = np.asarray(cache["material_centroid"], dtype=float)
    return {
        material: {"set": material_set, "xy": centroids[index]}
        for index, (material, material_set) in enumerate(zip(materials, sets))
    }


def load_material_pooled_umap_cache(
    cache_path: Path = MATERIAL_POOLED_UMAP_CACHE,
) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing material-pooled UMAP cache: {cache_path}\n"
            "Build per-material 256D descriptors before rendering Fig. 5f."
        )
    with np.load(str(cache_path), allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def _standardize_descriptor_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (matrix - mean) / std


def _material_pooled_umap_xy(material_emb: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    from umap import UMAP

    matrix = _standardize_descriptor_matrix(material_emb)
    n_pca = min(20, matrix.shape[0] - 1, matrix.shape[1])
    pca_coords = PCA(n_components=n_pca, random_state=42).fit_transform(matrix)
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(8, pca_coords.shape[0] - 1),
        min_dist=0.30,
        metric="euclidean",
        random_state=42,
    )
    xy = reducer.fit_transform(pca_coords)
    return (xy - xy.mean(axis=0, keepdims=True)) / xy.std(axis=0, keepdims=True)


def _material_pooled_umap_records(cache: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    materials = [str(item) for item in cache["materials"]]
    sets = [str(item) for item in cache["material_set"]]
    xy = _material_pooled_umap_xy(np.asarray(cache["material_emb"], dtype=float))
    return {
        material: {"set": material_set, "xy": xy[index]}
        for index, (material, material_set) in enumerate(zip(materials, sets))
    }


def _set_material_pooled_umap_limits(ax: plt.Axes, xy: np.ndarray) -> None:
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xrange = float(xmax - xmin)
    yrange = float(ymax - ymin)
    xpad_left = max(0.60, 0.18 * xrange)
    xpad_right = max(0.96, 0.28 * xrange)
    ypad_bottom = max(0.54, 0.18 * yrange)
    ypad_top = max(1.05, 0.34 * yrange)
    ax.set_xlim(float(xmin - xpad_left), float(xmax + xpad_right))
    ax.set_ylim(float(ymin - ypad_bottom), float(ymax + ypad_top))


def _set_umap_limits(ax: plt.Axes, xy: np.ndarray) -> None:
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xpad = max(1.2, 0.18 * float(xmax - xmin))
    ypad = max(1.6, 0.24 * float(ymax - ymin))
    ax.set_xlim(float(xmin - xpad), float(xmax + xpad))
    ax.set_ylim(float(ymin - ypad), float(ymax + 3.0 * ypad))


def _draw_point_mask_image(
    *,
    width: int,
    height: int,
    points_px: list[tuple[float, float]],
    colors: list[str],
    radius_px: float = 4.5,
) -> np.ndarray:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    for (x_px, y_px), color in zip(points_px, colors):
        rgb = np.asarray(mcolors.to_rgb(color), dtype=float) * 255.0
        mask = (xx - x_px) ** 2 + (yy - y_px) ** 2 <= radius_px ** 2
        image[mask] = rgb.astype(np.uint8)
    return image


def _data_to_panel_px(ax: plt.Axes, xy: np.ndarray) -> list[tuple[float, float]]:
    bbox = ax.bbox
    points_px: list[tuple[float, float]] = []
    for x_val, y_val in xy:
        display_x, display_y = ax.transData.transform((float(x_val), float(y_val)))
        points_px.append((float(display_x - bbox.x0), float(bbox.y1 - display_y)))
    return points_px


def _panel_px_to_data(ax: plt.Axes, xy_px: tuple[float, float]) -> tuple[float, float]:
    bbox = ax.bbox
    display_xy = (bbox.x0 + xy_px[0], bbox.y1 - xy_px[1])
    data_xy = ax.transData.inverted().transform(display_xy)
    return float(data_xy[0]), float(data_xy[1])


def _annotate_umap_materials(
    ax: plt.Axes,
    *,
    all_xy: np.ndarray,
    all_colors: list[str],
    label_xy: dict[str, np.ndarray],
    label_colors: dict[str, str],
    fontsize: float = 8.0,
    preferred_offsets_pt: dict[str, tuple[float, float]] | None = None,
) -> None:
    if preferred_offsets_pt is not None:
        leader_kw = dict(
            arrowstyle="-",
            lw=0.55,
            color=COLORS["mid_gray_dark"],
            shrinkA=1.0,
            shrinkB=2.0,
        )
        for label, xy in label_xy.items():
            dx, dy = preferred_offsets_pt[label]
            ax.annotate(
                display_material(label),
                xy=(float(xy[0]), float(xy[1])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                color=label_colors[label],
                arrowprops=leader_kw,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
                zorder=6,
            )
        return

    # Convert the plotted points into a lightweight image mask so the same
    # placement helper used by the structure panels can keep labels off data.
    ax.figure.canvas.draw()
    bbox = ax.bbox
    width = max(1, int(round(bbox.width)))
    height = max(1, int(round(bbox.height)))
    all_points_px = _data_to_panel_px(ax, all_xy)
    image = _draw_point_mask_image(
        width=width,
        height=height,
        points_px=all_points_px,
        colors=all_colors,
        radius_px=2.2,
    )
    label_names = list(label_xy)
    label_points_px = _data_to_panel_px(
        ax,
        np.asarray([label_xy[name] for name in label_names], dtype=float),
    )
    records = _qa.whitespace_label_placement(
        image=image,
        atoms=list(zip(label_names, label_points_px)),
        cluster_center_px=(width / 2.0, height / 2.0),
        atom_color_hexes=list(label_colors.values()),
        fontsize_pt=fontsize,
        min_atom_clearance_px=2.0,
        min_label_edge_distance_factor=0.05,
        dpi=float(ax.figure.dpi),
        radius_search_factors=(0.14, 0.20, 0.26, 0.34, 0.44, 0.56, 0.70, 0.90),
        target_atom_radius_px=2.5,
        max_arrow_factor=10.0,
    )
    leader_kw = dict(
        arrowstyle="-",
        lw=0.55,
        color=COLORS["mid_gray_dark"],
        shrinkA=1.0,
        shrinkB=2.0,
    )
    for rec in records:
        target_xy = label_xy[rec.label]
        text_xy = _panel_px_to_data(ax, rec.text_xy_px)
        ax.annotate(
            display_material(rec.label),
            xy=(float(target_xy[0]), float(target_xy[1])),
            xytext=text_xy,
            textcoords="data",
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=label_colors[rec.label],
            arrowprops=leader_kw,
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=6,
        )


def plot_cluster_umap_panel(
    ax: plt.Axes,
    *,
    focus_materials: tuple[str, ...],
    focus_set: str,
    focus_label: str,
    focus_colors: dict[str, str],
    show_ylabel: bool = True,
    label_offsets_pt: dict[str, tuple[float, float]] | None = None,
) -> None:
    cache = load_cluster_umap_cache()
    records = _cluster_umap_records(cache)
    ind_materials = [
        material for material, record in records.items()
        if record["set"] == "ind25"
    ]
    missing_focus = [material for material in focus_materials if material not in records]
    if missing_focus:
        raise KeyError(f"Missing focus materials from cluster UMAP cache: {missing_focus}")
    wrong_set = [
        material for material in focus_materials
        if records[material]["set"] != focus_set
    ]
    if wrong_set:
        raise RuntimeError(f"Unexpected UMAP material-set labels for {wrong_set}")

    ind_xy = np.asarray([records[material]["xy"] for material in ind_materials], dtype=float)
    focus_xy = np.asarray([records[material]["xy"] for material in focus_materials], dtype=float)
    all_xy = np.vstack([ind_xy, focus_xy])
    _set_umap_limits(ax, all_xy)

    ax.scatter(
        ind_xy[:, 0],
        ind_xy[:, 1],
        c=COLORS["mid_gray"],
        s=18,
        alpha=0.82,
        edgecolors="none",
        label="IND",
        zorder=2,
    )
    for material in focus_materials:
        xy = np.asarray(records[material]["xy"], dtype=float)
        ax.scatter(
            xy[0],
            xy[1],
            c=focus_colors[material],
            s=56,
            edgecolors="white",
            linewidths=0.7,
            label=focus_label if material == focus_materials[0] else None,
            zorder=4,
        )

    label_xy = {
        material: np.asarray(records[material]["xy"], dtype=float)
        for material in focus_materials
    }
    label_colors = {material: focus_colors[material] for material in focus_materials}
    all_colors = [COLORS["mid_gray"]] * len(ind_xy) + [
        focus_colors[material] for material in focus_materials
    ]
    _annotate_umap_materials(
        ax,
        all_xy=all_xy,
        all_colors=all_colors,
        label_xy=label_xy,
        label_colors=label_colors,
        preferred_offsets_pt=label_offsets_pt,
    )

    ax.set_xlabel("UMAP1", labelpad=1)
    ax.set_ylabel("UMAP2" if show_ylabel else "", labelpad=1)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    style_axes_local(ax, grid_axis=None)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def panel_f_cluster_umap(ax: plt.Axes) -> None:
    cache = load_material_pooled_umap_cache()
    records = _material_pooled_umap_records(cache)
    ood_new = tuple(MATERIAL_ORDER)
    ood_holdout = ("SY", "DAC-4", "TAP-2", "DPPE-1", "EAP-4")
    alias = {"DPPE-1": "DAI-1_0.5 4_0.5"}
    no_leader_labels = {"SY", "TAP-2", "DPPE-1", "DAC-4"}
    no_leader_align = {
        "SY": ("right", "center"),
        "TAP-2": ("center", "bottom"),
        "DPPE-1": ("left", "center"),
        "DAC-4": ("left", "top"),
    }
    label_offsets_pt = {
        "PEP": (6.0, 14.0),
        "MPEP": (-18.0, -13.0),
        "HPEP": (34.0, 2.0),
        "SY": (-9.0, 6.0),
        "EAP-4": (48.0, -18.0),
        "DAC-4": (9.0, -3.0),
        "TAP-2": (0.0, 13.0),
        "DPPE-1": (9.0, 0.0),
    }

    missing = [material for material in (*ood_new, *ood_holdout) if material not in records]
    if missing:
        raise KeyError(f"Missing materials from material-pooled UMAP cache: {missing}")

    ind_materials = [
        material for material, record in records.items()
        if material not in ood_new and str(record["set"]).lower() != "heldout"
    ]
    all_xy = np.asarray([record["xy"] for record in records.values()], dtype=float)
    _set_material_pooled_umap_limits(ax, all_xy)

    ind_xy = np.asarray([records[material]["xy"] for material in ind_materials], dtype=float)
    ax.scatter(
        ind_xy[:, 0],
        ind_xy[:, 1],
        c="#CCD1D5",
        s=22,
        alpha=0.86,
        edgecolors="white",
        linewidths=0.45,
        label="IND material",
        zorder=2,
    )

    for material in ood_holdout:
        xy = np.asarray(records[material]["xy"], dtype=float)
        ax.scatter(
            xy[0],
            xy[1],
            s=46,
            marker="s",
            facecolors="white",
            edgecolors=COLORS["charcoal"],
            linewidths=1.15,
            label="OOD-holdout" if material == ood_holdout[0] else None,
            zorder=5,
        )

    for material in ood_new:
        xy = np.asarray(records[material]["xy"], dtype=float)
        ax.scatter(
            xy[0],
            xy[1],
            s=58,
            marker="^",
            c=COLORS[material],
            edgecolors="none",
            linewidths=0.0,
            label="OOD-new" if material == ood_new[0] else None,
            zorder=7,
        )

    for material in (*ood_holdout, *ood_new):
        xy = np.asarray(records[material]["xy"], dtype=float)
        dx, dy = label_offsets_pt[material]
        color = COLORS[material] if material in ood_new else COLORS["charcoal"]
        ha, va = no_leader_align.get(material, ("center", "center"))
        ax.annotate(
            display_material(alias.get(material, material)),
            xy=(float(xy[0]), float(xy[1])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.9 if material in ood_holdout else 7.4,
            fontweight="bold" if material in ood_new else "normal",
            color=color,
            arrowprops=None if material in no_leader_labels else {
                "arrowstyle": "-",
                "lw": 0.54,
                "color": COLORS["mid_gray_dark"],
                "shrinkA": 1,
                "shrinkB": 3,
            },
            bbox={
                "boxstyle": "round,pad=0.09",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.90,
            },
            zorder=20,
        )

    ax.set_xlabel("UMAP1", labelpad=1)
    ax.set_ylabel("UMAP2", labelpad=1)
    style_axes_local(ax, grid_axis=None)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#CCD1D5",
             markeredgecolor="white", markersize=4.5, label="IND material"),
         Line2D([0], [0], marker="^", color="none", markerfacecolor="#B23A48",
             markeredgecolor="none", markeredgewidth=0.0, markersize=6,
               label="OOD-new"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["charcoal"], markeredgewidth=1.2,
             markersize=5.5, label="OOD-holdout"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )


def panel_f_landscape(ax: plt.Axes, train_pems: list[dict[str, float | str]]) -> None:
    plotted_obs: list[float] = []
    plotted_ds: list[float] = []
    train_rows = []
    for row in train_pems:
        try:
            formula_dict = row.get("formula_dict")
            if isinstance(formula_dict, dict):
                ob_val = calc_ob_from_formula_dict(formula_dict)
            else:
                ob_val = calc_ob(str(row["formula"]))
            d_val = float(row["D_KJ"])
        except (KeyError, TypeError, ValueError):
            continue
        train_rows.append((str(row["name"]), ob_val, d_val))

    if train_rows:
        train_ob = np.asarray([row[1] for row in train_rows], dtype=float)
        train_d = np.asarray([row[2] for row in train_rows], dtype=float)
        ax.scatter(
            train_ob,
            train_d,
            c=COLORS["mid_gray"],
            s=20,
            alpha=0.75,
            edgecolors="none",
            label="PEMs (train)",
            zorder=1,
        )
        plotted_obs.extend(train_ob.tolist())
        plotted_ds.extend(train_d.tolist())
    else:
        ax.text(
            0.03,
            0.06,
            "25-PEM training set\nnot found in workspace",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color=COLORS["mid_gray_dark"],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": COLORS["note_fill"],
                "edgecolor": COLORS["mid_gray"],
                "linewidth": 0.5,
            },
        )

    leader_kw = dict(
        arrowstyle="-",
        color=COLORS["mid_gray_dark"],
        lw=0.4,
        shrinkA=1.5,
        shrinkB=2.5,
    )

    # Offsets are tuned to avoid label-label collisions in the upper-right
    # cluster (PEP-M/PEP-H/PEP sit within ~10 m s^-1 and ~5 % OB of each
    # other). Each compound points outward from the cluster.
    new_offsets = {
        "HPEP": (-44, -2),
        "MPEP": (24, -16),
        "PEP":  (20, 14),
    }
    for index, name in enumerate(MATERIAL_ORDER):
        record = NEW_MATERIALS[name]
        ob_val = calc_ob(str(record["formula"]))
        ax.scatter(
            ob_val,
            float(record["D_KJ"]),
            c=COLORS[name],
            s=58,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            display_material(name),
            (ob_val, float(record["D_KJ"])),
            xytext=new_offsets[name],
            textcoords="offset points",
            fontsize=8.0,
            color=COLORS[name],
            fontweight="bold",
            arrowprops=dict(**leader_kw),
        )
        plotted_obs.append(ob_val)
        plotted_ds.append(float(record["D_KJ"]))

    bench_offsets = {"TNT": (10, -12), "RDX": (-26, -10), "HMX": (12, -12), "CL-20": (-32, 10)}
    for index, (name, record) in enumerate(BENCHMARKS.items()):
        ob_val = calc_ob(str(record["formula"]))
        ax.scatter(
            ob_val,
            float(record["D_KJ"]),
            c=COLORS["mid_gray_dark"],
            s=40,
            marker="D",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        # Benchmark labels were previously rendered in mid_gray_dark, which
        # is the same colour as some of the training-set background dots —
        # CL-20 in particular sat invisibly against the lavender MPEP fill.
        # Switch to charcoal so all four benchmarks read clearly.
        ax.annotate(
            name,
            (ob_val, float(record["D_KJ"])),
            xytext=bench_offsets[name],
            textcoords="offset points",
            fontsize=8.0,
            color=COLORS["charcoal"],
            arrowprops=dict(**leader_kw),
        )
        plotted_obs.append(ob_val)
        plotted_ds.append(float(record["D_KJ"]))

    ax.set_xlim(min(plotted_obs) - 4.5, max(plotted_obs) + 5.5)
    y_lo = min(plotted_ds) - 200.0
    y_hi = max(plotted_ds) + 900.0
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"Oxygen balance (CO$_2$ basis, %)")
    ax.set_ylabel(r"$V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    style_axes_local(ax, grid_axis="y")
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLORS["mid_gray"], markeredgecolor="none", markersize=5.5, label="PEMs (train)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["charcoal"], markeredgewidth=1.0, markersize=6.0, label="New ABX4"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=COLORS["mid_gray_dark"], markeredgecolor="white", markeredgewidth=0.5, markersize=5.8, label="Benchmark"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def _load_v3_ensemble_predictions(
    json_path: Path = PREDICTIONS_JSON,
    materials: tuple[str, ...] = tuple(MATERIAL_ORDER),
) -> dict[str, dict[str, float]]:
    """Read multi-task baseline v3 5-fold ensemble predictions for the three new ABX$_4$ compounds.

    Convention (matches manuscript Table 2 and infer_pems.py):
      1. Within each fold, average the 3 cluster realizations (n1, n2, n3) into
         a single fold-level prediction. This pools cluster (structure-realization)
         noise into the mean.
      2. Across the 5 fold-level predictions, report the mean (``D_pred``) and the
         sample standard deviation with ddof=1, n=5 (``D_pred_std``). The error
         bar represents pure model deviation; cluster noise is not double-counted.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"Required v3 ensemble JSON missing: {json_path}\n"
            "Re-run experiments/infer_pems.py to regenerate."
        )
    with json_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    by_material = {row["material"]: row for row in records if isinstance(row, dict)}
    out: dict[str, dict[str, float]] = {}
    for name in materials:
        record = by_material.get(name)
        if record is None:
            raise KeyError(
                f"{name} missing from {json_path.name}; "
                f"available materials: {sorted(by_material)}"
            )
        # Use the JSON's pre-computed grand_mean_m_s / model_std_m_s fields,
        # which infer_pems.py writes following the convention above.
        out[name] = {
            "D_pred": float(record["grand_mean_m_s"]),
            "D_pred_std": float(record["model_std_m_s"]),
            "n_folds": int(record.get("n_folds", len(record["predictions"]))),
        }
    return out


def _inject_v3_predictions_into_materials() -> None:
    preds = _load_v3_ensemble_predictions()
    for name, stats in preds.items():
        NEW_MATERIALS[name]["D_pred"] = stats["D_pred"]
        NEW_MATERIALS[name]["D_pred_std"] = stats["D_pred_std"]
    print(
        f"Loaded v3 ensemble predictions from {PREDICTIONS_JSON.name} "
        f"(error bar = pooled-cluster fold-mean std, ddof=1, n={preds[MATERIAL_ORDER[0]]['n_folds']} folds):"
    )
    for name in MATERIAL_ORDER:
        stats = preds[name]
        print(f"  {name}: D_pred = {stats['D_pred']:7.2f} m·s⁻¹, model_std = {stats['D_pred_std']:6.2f} m·s⁻¹")


def main() -> None:
    setup_nature_style()
    plt.rcParams.update({"axes.titlesize": 10.0, "legend.fontsize": 8.0})
    verify_reference_ob()

    _inject_v3_predictions_into_materials()
    train_pems = load_train_pems()
    missing = inspect_missing_inputs(train_pems)
    print("Figure 5 input check:")
    if missing:
        for item in missing:
            print(f"  - Missing: {item}")
    else:
        print("  - All required inputs found.")

    # Layout:
    #     a  (one row x three topology panels)
    #     b  (one row x three molecular structure panels)
    #     c d
    #     c e f
    # Rows 0 / 1 are full-width strips for panels a and b. Row 2-3 share a
    # two-row left block (panel c = PXRD) and a right block that holds panel
    # d on its top row and panels e + f side-by-side on its bottom row.
    fig = plt.figure(figsize=(8.27, 8.15))
    grid = fig.add_gridspec(
        4,
        4,
        width_ratios=[1.00, 1.00, 1.00, 1.00],
        # Panel a collapsed from 2x2 to 1x3, so the whole figure can be
        # shorter while panel b keeps near-square structure tiles.
        height_ratios=[2.35, 2.25, 1.45, 1.10],
        left=0.060,
        right=0.985,
        top=0.980,
        bottom=0.055,
        wspace=0.42,
        hspace=0.36,
    )

    axes_a = panel_a_topology_row(fig, grid[0, :])
    panel_a_legend(fig, axes_a)
    axes_b = panel_b_structure_row(fig, grid[1, :])
    panel_b_legend(fig, axes_b)
    ax_c = fig.add_subplot(grid[2:4, 0:2])
    panel_c_pxrd(ax_c)
    ax_d = panel_d_benchmark_bars(fig, grid[2, 2:4])
    ax_e = fig.add_subplot(grid[3, 2])
    ax_f = fig.add_subplot(grid[3, 3])

    panel_d_dumbbell(ax_e)
    panel_f_cluster_umap(ax_f)

    offset_left = 0.020
    a_pos = axes_a[0].get_position()
    b_pos = axes_b[0].get_position()
    c_pos = ax_c.get_position()
    d_pos = ax_d.get_position()
    e_pos = ax_e.get_position()
    f_pos = ax_f.get_position()
    # Use common x for "a" and "b" labels so they're left-aligned
    label_x = 0.040  # grid left (0.060) - offset_left (0.020)
    add_panel_label_fig(fig, label_x, a_pos.y1 + 0.003, "a")
    add_panel_label_fig(fig, label_x, b_pos.y1 + 0.004, "b")
    add_panel_label_fig(fig, c_pos.x0 - offset_left, c_pos.y1 + 0.003, "c")
    add_panel_label_fig(fig, d_pos.x0 - offset_left, d_pos.y1 + 0.008, "d")
    add_panel_label_fig(fig, e_pos.x0 - offset_left, e_pos.y1 + 0.014, "e")
    add_panel_label_fig(fig, f_pos.x0 - offset_left, f_pos.y1 + 0.014, "f")

    save_png_pdf(fig, THIS_DIR / "figure5")
    plt.close(fig)
    render_si_td_benchmark()


if __name__ == "__main__":
    main()
