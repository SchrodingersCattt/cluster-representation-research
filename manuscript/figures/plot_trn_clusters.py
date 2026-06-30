#!/usr/bin/env python3
"""
Plot training and OOD cluster structures as grid figures.

Training layout (one figure per cluster variant; restructured 2026-05-03):
    5 rows x 5 cols of materials, one figure per n1/n2/n3 variant
    -> manuscript/figures/_trn_clusters_n1.png/.pdf
       manuscript/figures/_trn_clusters_n2.png/.pdf
       manuscript/figures/_trn_clusters_n3.png/.pdf

OOD layout (split by group; restructured 2026-05-03):
    OOD-holdout: 5 rows (DAC-4, TAP-2, DAI-1_0.5 4_0.5, EAP-4, DEP) x 3 cols (n1/n2/n3)
        -> manuscript/figures/_ood_clusters_heldout.png/.pdf
    OOD-new:     3 rows (PEP, PEP-M, PEP-H)              x 3 cols (n1/n2/n3)
        -> manuscript/figures/_ood_clusters_new.png/.pdf

Style: Arial font, PDF fonttype 42, white background, 600 dpi.
Rendering pipeline:
    crystal_viewer (Mesh3d spheres + cylinder bonds, radius-aware viewport,
    shared world-cube via uniform_viewport) renders each panel to a PNG,
    matplotlib composes the grid and overlays panel titles, the xyz axis
    triad, and the shared element legend.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

from figure_style import display_material

# ---- Workaround: choreographer's _libs_ok mis-detects deps when
# /usr/bin/google-chrome is a wrapper shell script (ldd returns "not a
# dynamic executable" -> exit 1, which choreographer treats as "ldd failed"
# -> sets missing_libs=True, causing spurious BrowserDepsError on any
# transient browser death such as a watchdog timeout). Real chrome libs
# are present (ldd /opt/google/chrome/chrome resolves cleanly), so we
# force the deps check to pass.
try:
    from choreographer.browsers import chromium as _choreo_chromium
    _choreo_chromium.Chromium._libs_ok = lambda self: True
except Exception:
    pass

# ---- crystal_viewer (MatterVis) imports ----------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CRYSTAL_VIEWER_ROOT = os.path.join(
    REPO_ROOT, "ABX4_expdata", "crystal_viewer"
)
if CRYSTAL_VIEWER_ROOT not in sys.path:
    sys.path.insert(0, CRYSTAL_VIEWER_ROOT)

from crystal_viewer.presets import DEFAULT_STYLE, deep_merge  # noqa: E402
from crystal_viewer.renderer import build_figure, uniform_viewport  # noqa: E402
from crystal_viewer.scene import build_scene_from_cif  # noqa: E402


# ---- Global style (Arial, fonttype 42, white bg) -------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---- Paths ---------------------------------------------------------------
TRN_CIF_BASE = os.path.join(
    REPO_ROOT, "experiments_davis2024", "00_data_prep", "pems_cluster_cifs",
)
OOD_CIF_BASE = os.path.join(
    REPO_ROOT, "experiments_davis2024", "00_data_prep", "pems_cluster_cifs_ood",
)

TRN_MATERIALS = [
    "DAI-1", "DAI-2", "DAI-4", "DAI-X1", "DAN-2",
    "DAP-1", "DAP-2", "DAP-3", "DAP-4", "DAP-5",
    "DAP-6", "DAP-7", "DAP-M4", "DAP-O2", "DAP-O4",
    "PAN-2", "PAN-H2", "PAN-M2", "PAP-1", "PAP-4",
    "PAP-5", "PAP-H4", "PAP-H5", "PAP-M4", "PAP-M5",
]
assert len(TRN_MATERIALS) == 25

# Split OOD into the two reporting categories used in the SI ablation tables.
OOD_HELDOUT_MATERIALS = ["DAC-4", "TAP-2", "DPPE-1", "EAP-4", "SY"]
OOD_NEW_MATERIALS = ["PEP", "MPEP", "HPEP"]
OOD_MATERIALS = OOD_HELDOUT_MATERIALS + OOD_NEW_MATERIALS  # legacy alias

VARIANTS = ["cluster_n1", "cluster_n2", "cluster_n3"]
VARIANT_LABELS = ["n1", "n2", "n3"]
VARIANT_DISPLAY_LABELS = [r"$n_1$", r"$n_2$", r"$n_3$"]


# ---- Element palette overrides (elements not in the vendored table) -----
ELEM_COLOR_OVERRIDES = {
    "H":  "#DCDCDC",
    "C":  "#5E5E5E",
    "N":  "#2C61AF",
    "O":  "#B85060",
    "Cl": "#218E6A",
    "S":  "#D4A017",
    "Na": "#E6D11E",
    "K":  "#AB82FF",
    "Rb": "#D100D1",
    "I":  "#940094",
    "Ag": "#C0C0C0",
}

# Legend (shown in every figure); keep the ordered short list the reviewers
# expect rather than auto-generating from the union of elements.
LEGEND_ELEMENTS = ["H", "C", "N", "O", "Cl"]


# ---- Fixed 3D view -------------------------------------------------------
FIXED_ELEV = 18.0    # degrees above the xy plane
FIXED_AZIM = -58.0   # degrees CCW from +x in the xy plane


def _view_vectors():
    elev = math.radians(FIXED_ELEV)
    azim = math.radians(FIXED_AZIM)
    view = np.array(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=float,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(view, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=float)
    return view, up


def _view_rotation(view_vec, up_vec):
    """Return 3x3 rotation whose rows are (view_x, view_y, view_z)."""
    z = np.array(view_vec, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array(up_vec, dtype=float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.array([x, y, z])


# ---- Crystal-viewer style assembly --------------------------------------
def _build_style():
    """Render tuning for publication-quality cluster panels."""
    return deep_merge(DEFAULT_STYLE, {
        "display_mode": "cluster",
        "show_title": False,    # matplotlib overlays the panel title instead
        "show_labels": False,   # atom labels hide the geometry at grid scale
        "show_axes": False,     # matplotlib draws a clean xyz triad overlay
        "show_hydrogen": True,
        "show_unit_cell": False,
        "atom_scale": 0.85,
        "bond_radius": 0.13,
        "major_opacity": 1.0,
        "minor_opacity": 1.0,
        "element_colors": ELEM_COLOR_OVERRIDES,
        "element_colors_light": ELEM_COLOR_OVERRIDES,
        "background": "#FFFFFF",
        "topology_enabled": False,
        "fast_rendering": False,
    })


def _preset_with_view(name, view_dir, up):
    """Pin the crystal-viewer preset so every scene shares the same camera."""
    style = _build_style()
    return {
        "version": 1,
        "style": style,
        "structures": {
            name: {
                "view_direction": [float(v) for v in view_dir],
                "up": [float(v) for v in up],
                "show_hydrogen": True,
            },
        },
    }


def _elem_color(sym):
    return ELEM_COLOR_OVERRIDES.get(sym, "#909090")


# ---- Scene -> PNG via crystal_viewer Plotly renderer --------------------
def _scene_to_png(scene, style, width_px, height_px, max_attempts=4):
    """Render ``scene`` to an RGBA numpy image using the crystal_viewer
    Mesh3d path. This is the rendering path that makes panels
    - atom-bond occlusion correct (Mesh3d depth ordering)
    - uniform across the grid (scene["viewport"] pinned beforehand)
    - free of edge clipping (radius-aware bounds)

    Retries up to ``max_attempts`` times on transient Kaleido/Chromium
    failures (watchdog timeout, BrowserDepsError, browser closed)."""
    fig = build_figure(scene, style)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    last_exc = None
    for attempt in range(max_attempts):
        try:
            png_bytes = fig.to_image(
                format="png",
                width=int(width_px),
                height=int(height_px),
                scale=2,
            )
            with Image.open(io.BytesIO(png_bytes)) as im:
                return np.array(im.convert("RGBA"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                print(
                    f"    retry {attempt + 1}/{max_attempts - 1} "
                    f"after kaleido error: {type(exc).__name__}",
                    flush=True,
                )
                time.sleep(2.0 * (attempt + 1))
                continue
            raise last_exc


# ---- Matplotlib overlays (title, axis triad, legend) --------------------
AXIS_TRIAD_SCALE = 0.26
AXIS_COLORS = {"x": "#D1495B", "y": "#2C9A5F", "z": "#2C61AF"}
ROW_FRAME_COLOR = "#B8B8B8"
PANEL_TITLE_Y = 0.925


def _add_row_frames(fig: plt.Figure, row_axes: list[list[plt.Axes]]) -> None:
    """Draw one frame around each data row to bind titles to their clusters."""
    fig.canvas.draw()
    for axes in row_axes:
        boxes = [ax.get_position() for ax in axes]
        x0 = min(box.x0 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        x1 = max(box.x1 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        pad_x, pad_y = 0.004, 0.006
        fig.add_artist(mpatches.Rectangle(
            (x0 - pad_x, y0 - pad_y),
            (x1 - x0) + 2 * pad_x,
            (y1 - y0) + 2 * pad_y,
            transform=fig.transFigure,
            fill=False,
            edgecolor=ROW_FRAME_COLOR,
            linewidth=0.8,
            zorder=100,
            clip_on=False,
        ))


def _draw_axes_triad(ax):
    view_vec, up_vec = _view_vectors()
    R = _view_rotation(view_vec, up_vec)
    view_x, view_y = R[0], R[1]
    origin = np.array([0.105, 0.105], dtype=float)
    axis_len = AXIS_TRIAD_SCALE * 0.50
    for axis_name, direction in (
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ):
        d2 = np.array([direction @ view_x, direction @ view_y], dtype=float)
        norm = np.linalg.norm(d2)
        if norm < 1e-8:
            continue
        d2 /= norm
        end = origin + axis_len * d2
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            transform=ax.transAxes,
            color=AXIS_COLORS[axis_name],
            lw=1.1,
            solid_capstyle="round",
            zorder=20,
        )
        label_pos = origin + axis_len * 1.35 * d2
        ax.text(
            label_pos[0], label_pos[1], f"${axis_name}$",
            transform=ax.transAxes,
            fontsize=8.0,
            color=AXIS_COLORS[axis_name],
            ha="center", va="center",
            zorder=21,
        )


def _paint_panel(ax, image, title):
    ax.imshow(image)
    _draw_axes_triad(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5, PANEL_TITLE_Y, title,
        transform=ax.transAxes,
        fontsize=8.0, fontweight="bold",
        ha="center", va="top", color="#111111",
    )


def _paint_error(ax, title, exc):
    ax.set_axis_off()
    short = type(exc).__name__
    ax.text(
        0.5, 0.5, f"ERR\n{short}",
        transform=ax.transAxes, fontsize=8.0,
        ha="center", va="center", color="#C00000",
    )
    ax.text(
        0.5, PANEL_TITLE_Y, title, transform=ax.transAxes,
        fontsize=8.0, fontweight="bold",
        ha="center", va="top", color="#111111",
    )


# ---- Dataset helpers ----------------------------------------------------
def training_cif_path(material, variant):
    return os.path.join(TRN_CIF_BASE, variant, f"{material}.cif")


def ood_cif_path(material, variant):
    return os.path.join(OOD_CIF_BASE, material, f"{variant}.cif")


def _load_degenerate_pairs():
    """Return {material: set_of_pair_strings} from pems_manifest.json."""
    manifest_path = os.path.join(
        REPO_ROOT, "experiments_davis2024", "00_data_prep", "pems_manifest.json"
    )
    try:
        import json
        with open(manifest_path) as f:
            mf = json.load(f)
        return {
            mat: set(info.get("degenerate_pairs", []))
            for mat, info in mf.get("cluster_diversity", {}).items()
        }
    except Exception:
        return {}


def _build_scenes_for_variant(materials, cif_path_fn, variant):
    """Parse every CIF for one variant, build scenes, skip missing gracefully."""
    view_dir, up = _view_vectors()
    scenes = []
    for material in materials:
        cif_path = cif_path_fn(material, variant)
        try:
            scene = build_scene_from_cif(
                name=f"{material}_{variant}",
                cif_path=cif_path,
                title=display_material(material),
                preset=_preset_with_view(f"{material}_{variant}", view_dir, up),
                show_hydrogen=True,
                display_mode="cluster",
            )
        except Exception as exc:  # noqa: BLE001
            scenes.append({"_error": exc, "name": material, "_variant": variant})
            continue
        scenes.append(scene)
    return scenes


# Uniform padding inside every rendered panel. Keep this global, not per-image,
# so all clusters within a grid share the same zoom and aligned titles/frames.
VIEWPORT_PADDING = 0.08

# DPI for matplotlib saving. Embedded Plotly PNGs are rendered at
# RENDER_DPI px per inch and then doubled by Kaleido scale=2, so the
# effective print resolution is ~2x RENDER_DPI.
RENDER_DPI = 600


# ---- OOD grid: rows = materials, cols = n1/n2/n3 -----------------------
def render_ood_grid(materials, cif_path_fn, out_stem, group_label=None):
    """Render an OOD grid: rows = materials, cols = n1/n2/n3.

    Used for both the OOD-holdout group (5 rows) and the new group (3 rows);
    the layout intentionally matches across the two figures so the SI
    pages are visually consistent.
    """
    n_mats = len(materials)
    n_vars = len(VARIANTS)

    # Square panels minimize whitespace from the radius-aware viewport.
    PANEL_W = 1.75
    PANEL_H = 1.75
    ROW_LABEL_W = 1.18
    COL_HDR_H = 0.34
    LEG_H = 0.44
    PANEL_W_PX = int(PANEL_W * RENDER_DPI)
    PANEL_H_PX = int(PANEL_H * RENDER_DPI)

    FIG_W = ROW_LABEL_W + PANEL_W * n_vars
    FIG_H = COL_HDR_H + PANEL_H * n_mats + LEG_H
    print(
        f"OOD grid '{out_stem}': {FIG_W:.1f} x {FIG_H:.1f} inches "
        f"({n_mats} rows x {n_vars} cols)",
        flush=True,
    )

    all_scenes_by_variant = []
    for variant in VARIANTS:
        vscenes = _build_scenes_for_variant(materials, cif_path_fn, variant)
        all_scenes_by_variant.append(vscenes)

    style = _build_style()
    all_real = [
        s for vscenes in all_scenes_by_variant for s in vscenes if "_error" not in s
    ]
    if all_real:
        uniform_viewport(all_real, style=style, padding=VIEWPORT_PADDING)

    degenerate_pairs = _load_degenerate_pairs()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        n_mats + 2, n_vars + 1,
        figure=fig,
        width_ratios=[ROW_LABEL_W / PANEL_W] + [1.0] * n_vars,
        height_ratios=[COL_HDR_H / PANEL_H] + [1.0] * n_mats + [LEG_H / PANEL_H],
        hspace=0.0, wspace=0.0,
        left=0.005, right=0.998, top=0.998, bottom=0.005,
    )

    for vi, label in enumerate(VARIANT_DISPLAY_LABELS):
        ax_hdr = fig.add_subplot(gs[0, vi + 1])
        ax_hdr.set_axis_off()
        ax_hdr.text(
            0.5, 0.5, label, transform=ax_hdr.transAxes,
            fontsize=8.0, fontweight="bold",
            ha="center", va="center", color="#333333",
        )
    fig.add_subplot(gs[0, 0]).set_axis_off()

    n_panels = n_mats * n_vars
    panel_i = 0
    row_axes = []
    for mat_idx, material in enumerate(materials):
        ax_lbl = fig.add_subplot(gs[mat_idx + 1, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(
            0.95, 0.5, display_material(material), transform=ax_lbl.transAxes,
            fontsize=8.0, fontweight="bold",
            ha="right", va="center", color="#111111",
        )

        mat_deg = degenerate_pairs.get(material, set())
        current_row_axes = [ax_lbl]
        for vi, (variant, vlabel) in enumerate(zip(VARIANTS, VARIANT_LABELS)):
            scene = all_scenes_by_variant[vi][mat_idx]
            ax = fig.add_subplot(gs[mat_idx + 1, vi + 1])
            current_row_axes.append(ax)
            panel_i += 1
            print(f"  [{panel_i:2d}/{n_panels}] {material} {vlabel} ...", flush=True)

            if "_error" in scene:
                _paint_error(ax, f"{display_material(material)} ({vlabel})", scene["_error"])
                continue
            try:
                image = _scene_to_png(scene, style, PANEL_W_PX, PANEL_H_PX)
            except Exception as exc:  # noqa: BLE001
                _paint_error(ax, f"{display_material(material)} ({vlabel})", exc)
                continue
            _paint_panel(ax, image, "")

            tag_a = VARIANT_LABELS[vi]
            is_degenerate = any(
                (tag_a in pair and any(tag_b in pair for tag_b in VARIANT_LABELS[:vi]))
                for pair in mat_deg
            )
            if is_degenerate:
                ax.text(
                    0.94, 0.06, "●", transform=ax.transAxes,
                    fontsize=8.0, ha="center", va="center",
                    color="#999999", zorder=25,
                )
        row_axes.append(current_row_axes)

    ax_leg = fig.add_subplot(gs[n_mats + 1, 1:])
    ax_leg.axis("off")
    handles = [mpatches.Patch(color=_elem_color(e), label=e)
               for e in LEGEND_ELEMENTS]
    ax_leg.legend(
        handles=handles, loc="center", fontsize=8.0,
        frameon=False, ncol=len(handles),
        borderpad=0.2, handlelength=1.0,
    )
    _add_row_frames(fig, row_axes)

    out_base = os.path.join(SCRIPT_DIR, out_stem)
    print("Saving PNG ...", flush=True)
    fig.savefig(out_base + ".png", dpi=RENDER_DPI,
                bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print("Saving PDF ...", flush=True)
    fig.savefig(out_base + ".pdf", dpi=RENDER_DPI,
                bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"Done -> {out_base}.png / .pdf", flush=True)


# ---- Training grid: 5 x 5 materials, single variant per figure ---------
def render_trn_variant_grid(materials, variant, cif_path_fn, out_stem,
                            n_cols=5):
    """Render the 25 training materials in an n_cols x n_rows grid for
    a single cluster variant (n1, n2 or n3). One figure per variant.
    """
    n_mats = len(materials)
    n_rows = math.ceil(n_mats / n_cols)
    vlabel = VARIANT_LABELS[VARIANTS.index(variant)]
    display_vlabel = VARIANT_DISPLAY_LABELS[VARIANTS.index(variant)]

    PANEL_W = 1.65
    PANEL_H = 1.75       # extra space reserved for the in-panel material label
    HDR_H = 0.40
    LEG_H = 0.42
    PANEL_W_PX = int(PANEL_W * RENDER_DPI)
    PANEL_H_PX = int(PANEL_H * RENDER_DPI)

    FIG_W = n_cols * PANEL_W
    FIG_H = HDR_H + n_rows * PANEL_H + LEG_H
    print(
        f"Training grid '{out_stem}' ({vlabel}): "
        f"{FIG_W:.1f} x {FIG_H:.1f} inches ({n_rows} rows x {n_cols} cols)",
        flush=True,
    )

    scenes = _build_scenes_for_variant(materials, cif_path_fn, variant)
    style = _build_style()
    real_scenes = [s for s in scenes if "_error" not in s]
    if real_scenes:
        uniform_viewport(real_scenes, style=style, padding=VIEWPORT_PADDING)

    degenerate_pairs = _load_degenerate_pairs()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = GridSpec(
        n_rows + 2, n_cols,
        figure=fig,
        height_ratios=[HDR_H / PANEL_H] + [1.0] * n_rows + [LEG_H / PANEL_H],
        hspace=0.0, wspace=0.0,
        left=0.005, right=0.998, top=0.998, bottom=0.005,
    )

    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.5, 0.4,
        f"Training cluster set - variant {display_vlabel}",
        transform=ax_hdr.transAxes,
        fontsize=10, fontweight="bold",
        ha="center", va="center", color="#222222",
    )

    row_axes = [[] for _ in range(n_rows)]
    for idx, material in enumerate(materials):
        row = idx // n_cols + 1
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        row_axes[row - 1].append(ax)
        scene = scenes[idx]
        print(f"  [{idx + 1:2d}/{n_mats}] {material} {vlabel} ...", flush=True)
        if "_error" in scene:
            _paint_error(ax, f"{display_material(material)} ({vlabel})", scene["_error"])
        else:
            try:
                image = _scene_to_png(scene, style, PANEL_W_PX, PANEL_H_PX)
            except Exception as exc:  # noqa: BLE001
                _paint_error(ax, f"{display_material(material)} ({vlabel})", exc)
                continue
            _paint_panel(ax, image, display_material(material))

        # Degenerate-pair indicator: this variant matches a sibling variant
        mat_deg = degenerate_pairs.get(material, set())
        is_degenerate = any(
            vlabel in pair and any(other in pair for other in VARIANT_LABELS if other != vlabel)
            for pair in mat_deg
        )
        if is_degenerate:
            ax.text(
                0.94, 0.06, "●", transform=ax.transAxes,
                fontsize=8.0, ha="center", va="center",
                color="#999999", zorder=25,
            )

    # Pad any unused trailing slots with blank axes for clean spacing
    for idx in range(n_mats, n_rows * n_cols):
        row = idx // n_cols + 1
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_axis_off()
        row_axes[row - 1].append(ax)

    ax_leg = fig.add_subplot(gs[n_rows + 1, :])
    ax_leg.axis("off")
    handles = [mpatches.Patch(color=_elem_color(e), label=e)
               for e in LEGEND_ELEMENTS]
    ax_leg.legend(
        handles=handles, loc="center", fontsize=8.0,
        frameon=False, ncol=len(handles),
        borderpad=0.2, handlelength=1.0,
    )
    _add_row_frames(fig, row_axes)

    out_base = os.path.join(SCRIPT_DIR, out_stem)
    print("Saving PNG ...", flush=True)
    fig.savefig(out_base + ".png", dpi=RENDER_DPI,
                bbox_inches="tight", facecolor="white", pad_inches=0.02)
    print("Saving PDF ...", flush=True)
    fig.savefig(out_base + ".pdf", dpi=RENDER_DPI,
                bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"Done -> {out_base}.png / .pdf", flush=True)


# ---- Legacy combined grid (kept for back-compat with old wrappers) -----
def render_grid(materials, cif_path_fn, out_stem):
    """Legacy combined "rows x 3 cols" rendering (single PDF).

    Retained so plot_ood_clusters_only.py and other older entry points
    continue to function. New SI figures should call render_ood_grid /
    render_trn_variant_grid instead.
    """
    return render_ood_grid(materials, cif_path_fn, out_stem)


def main():
    # Per-variant 5x5 training grids (one PDF each).
    for variant in VARIANTS:
        vlabel = VARIANT_LABELS[VARIANTS.index(variant)]
        print(f"\nRendering training clusters — variant {vlabel}", flush=True)
        render_trn_variant_grid(
            TRN_MATERIALS, variant, training_cif_path,
            f"_trn_clusters_{vlabel}",
        )

    # OOD grids split by reporting category.
    print("\nRendering OOD-holdout clusters (5 rows x 3 cols)", flush=True)
    render_ood_grid(
        OOD_HELDOUT_MATERIALS, ood_cif_path, "_ood_clusters_heldout",
    )

    print("\nRendering OOD-new clusters (3 rows x 3 cols)", flush=True)
    render_ood_grid(
        OOD_NEW_MATERIALS, ood_cif_path, "_ood_clusters_new",
    )


if __name__ == "__main__":
    main()
