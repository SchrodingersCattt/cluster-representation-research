"""plot_dataset_overview.py — Publication figures for the PEMs dataset.

Figure 1a  (_dataset_grid.pdf)
    Combined figure:
      - Top: A×X data grid with text labels (A-site rows, X-site columns, B-site markers)
      - Below: A-site molecular structures (2×3 grid)
      - Below: X-site molecular structures (1×4 row)
      - Below: B-site legend (1 row)
    Target width: half A4 (105 mm).

Figure 1b  (_asite_structures.pdf)
    Stand-alone 2×3 grid of A-site cation structures.

Usage
-----
    python manuscript/figures/plot_dataset_overview.py

Output
------
    manuscript/figures/_dataset_grid.{png,pdf}
    manuscript/figures/_asite_structures.{png,pdf}
"""

from __future__ import annotations

import sys
import warnings
import re as _re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV  = REPO_ROOT / "data" / "pems" / "pems.csv"
OUT_DIR   = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from paper_plot_style import setup_nature_style, style_axes, save_png_pdf

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CANONICAL_MATERIALS = {
    "DAI-1", "DAI-2", "DAI-4", "DAI-X1", "DAN-2",
    "DAP-1", "DAP-2", "DAP-3", "DAP-4", "DAP-5", "DAP-6", "DAP-7",
    "DAP-M4", "DAP-O2", "DAP-O4",
    "PAN-2", "PAN-H2", "PAN-M2",
    "PAP-1", "PAP-4", "PAP-5", "PAP-H4", "PAP-H5", "PAP-M4", "PAP-M5",
}


def load_pems() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(DATA_CSV, index_col=False)
    df = df[df["material"].isin(CANONICAL_MATERIALS)].copy()
    df["D_km_s"] = pd.to_numeric(df["D_km_s"], errors="coerce")
    df["vdet"]   = df["D_km_s"] * 1000.0
    for col in ["A_site", "B_site", "X_site"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df

# ---------------------------------------------------------------------------
# Canonical orderings
# ---------------------------------------------------------------------------

X_ORDER = ["ClO4-", "NO3-", "IO4-", "ClO3-"]

def _normalise_xsite_for_grid(x: str) -> str:
    if x == "H4IO6-":
        return "IO4-"
    return x

A_ORDER = [
    "H2dabco2+",
    "MeHdabco2+",
    "H2odabco2+",
    "H2pz2+",
    "H2hpz2+",
    "MeHpz2+",
    "HQ+",
    "Huru+",
]

B_MARKERS = {
    "Na+":              "o",
    "K+":               "s",
    "NH4+":             "D",
    "Ag+":              "^",
    "Rb+":              "v",
    "NH3OH+":           "P",
    "NH2NH3+":          "*",
    "Ba2+":             "h",
    "Na+/NH4+ (ordered)": "X",
    "CH3NH3+":          "<",
    "H3O+":             ">",
}
B_DEFAULT_MARKER = "o"

B_LABELS = {
    "Na+":              r"$\mathrm{Na^+}$",
    "K+":               r"$\mathrm{K^+}$",
    "NH4+":             r"$\mathrm{NH_4^+}$",
    "Ag+":              r"$\mathrm{Ag^+}$",
    "Rb+":              r"$\mathrm{Rb^+}$",
    "NH3OH+":           r"$\mathrm{NH_3OH^+}$",
    "NH2NH3+":          r"$\mathrm{NH_2NH_3^+}$",
    "Ba2+":             r"$\mathrm{Ba^{2+}}$",
    "Na+/NH4+ (ordered)": r"$\mathrm{Na^+/NH_4^+}$",
    "CH3NH3+":          r"$\mathrm{CH_3NH_3^+}$",
    "H3O+":             r"$\mathrm{H_3O^+}$",
}

X_LABELS = {
    "ClO4-":  "Perchlorate",
    "NO3-":   "Nitrate",
    "IO4-":   "Periodate",
    "ClO3-":  "Chlorate",
}

A_LABELS = {
    "H2dabco2+":  r"$\mathregular{H_2dabco^{2+}}$",
    "MeHdabco2+": r"$\mathregular{MeHdabco^{2+}}$",
    "H2odabco2+": r"$\mathregular{H_2odabco^{2+}}$",
    "H2pz2+":     r"$\mathregular{H_2pz^{2+}}$",
    "H2hpz2+":    r"$\mathregular{H_2hpz^{2+}}$",
    "MeHpz2+":    r"$\mathregular{MeHpz^{2+}}$",
    "HQ+":        r"$\mathregular{HQ^+}$",
    "Huru+":      r"$\mathregular{Huru^+}$",
}

X_SITE_COLORS = {
    "ClO4-":  "#5A6D7B",
    "NO3-":   "#8B7355",
    "IO4-":   "#7A6B8A",
    "ClO3-":  "#6B7D6A",
}
X_SITE_DEFAULT = "#5A6D7B"
GRAY_NO_DATA = "#BBBBBB"

# ---------------------------------------------------------------------------
# Molecule drawing helpers
# ---------------------------------------------------------------------------

def _parse_svg_color(style_str: str, attr: str):
    m = _re.search(rf'{attr}:#([0-9a-fA-F]{{6}})', style_str)
    if m:
        h = m.group(1)
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    m = _re.search(rf'{attr}:([a-zA-Z]+)', style_str)
    if m and m.group(1) != "none":
        return m.group(1)
    return None


def _parse_svg_path_d(d: str):
    import matplotlib.path as mpath
    tokens = _re.findall(r'[MLQZmlqz]|[-+]?[0-9]*\.?[0-9]+', d)
    verts: list[list[float]] = []
    codes: list[int] = []
    i = 0
    cmd = "M"
    while i < len(tokens):
        t = tokens[i]
        if t in "MLQZmlqz":
            cmd = t
            i += 1
            continue
        if cmd == "M":
            verts.append([float(tokens[i]), float(tokens[i + 1])])
            codes.append(mpath.Path.MOVETO)
            i += 2
        elif cmd == "L":
            verts.append([float(tokens[i]), float(tokens[i + 1])])
            codes.append(mpath.Path.LINETO)
            i += 2
        elif cmd == "Q":
            verts.append([float(tokens[i]), float(tokens[i + 1])])
            codes.append(mpath.Path.CURVE3)
            verts.append([float(tokens[i + 2]), float(tokens[i + 3])])
            codes.append(mpath.Path.CURVE3)
            i += 4
        elif cmd == "Z":
            verts.append([0.0, 0.0])
            codes.append(mpath.Path.CLOSEPOLY)
        else:
            i += 1
    if not verts:
        return None
    return mpath.Path(np.array(verts), codes)


# RDKit default hex → project palette
_SVG_COLOUR_REMAP = {
    "#000000": "#5E5E5E",  # C / bonds
    "#0000FF": "#125CCA",  # N
    "#FF0000": "#D63C53",  # O
    "#00CC00": "#0EB87F",  # Cl
    "#940094": "#6B5B7B",  # I
}


def _remap_svg_colours(svg_text: str) -> str:
    result = svg_text
    for old, new in _SVG_COLOUR_REMAP.items():
        result = _re.sub(_re.escape(old), new, result, flags=_re.IGNORECASE)
    return result


def _draw_mol_svg_on_ax(
    ax,
    svg_text: str,
    x_offset: float,
    cell_w: float,
    cell_h: float,
    y_offset: float = 0.0,
) -> None:
    """Parse a RDKit SVG string and draw all paths/text onto a matplotlib Axes."""
    import xml.etree.ElementTree as ET
    import matplotlib.path as mpath

    svg_text = svg_text.strip()
    if svg_text.startswith("<?xml"):
        svg_text = svg_text[svg_text.index("?>") + 2:].strip()
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(svg_text)

    def flip_y(y: float) -> float:
        return cell_h - y + y_offset

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "path":
            d = elem.get("d", "")
            style = elem.get("style", "")
            fill_attr = elem.get("fill", "none")
            path = _parse_svg_path_d(d)
            if path is None:
                continue
            verts = path.vertices.copy()
            verts[:, 0] += x_offset
            verts[:, 1] = flip_y(verts[:, 1])
            flipped = mpath.Path(verts, path.codes)
            stroke_color = _parse_svg_color(style, "stroke")
            fill_color = _parse_svg_color(style, "fill")
            if fill_attr and fill_attr not in ("none", ""):
                fill_color = fill_attr
            lw_m = _re.search(r'stroke-width:([\d.]+)px', style)
            lw = float(lw_m.group(1)) if lw_m else 1.0
            patch = mpatches.PathPatch(
                flipped,
                facecolor=fill_color if fill_color and fill_color != "none" else "none",
                edgecolor=stroke_color if stroke_color else "none",
                linewidth=lw,
                transform=ax.transData,
                clip_on=False,
            )
            ax.add_patch(patch)

        elif tag == "text":
            try:
                x = float(elem.get("x", 0)) + x_offset
                y = flip_y(float(elem.get("y", 0)))
                fill = elem.get("fill", "#000000")
                fs_str = elem.get("font-size", "12")
                fs = float(_re.sub(r'[^0-9.]', '', fs_str))
                # RDKit SVG atom labels: <text x=.. y=..>N<tspan dy=..>+</tspan></text>
                # Strategy: render elem.text (atom symbol) only; skip tspan children
                # that are purely formal-charge annotations (+, -, 2+, etc.).
                # This suppresses N⁺, Cl⁺, I⁺ charges while keeping atom symbols.
                atom_symbol = elem.text or ""
                # Also collect any non-charge tspan content (e.g. subscript digits)
                extra = ""
                for t in elem:
                    t_text = (t.text or "") + (t.tail or "")
                    is_charge = (
                        bool(_re.fullmatch(r'\s*[+\-]?\d*[+\-]?\s*', t_text))
                        and any(c in t_text for c in '+-')
                    )
                    if not is_charge:
                        extra += t_text
                text = atom_symbol + extra
                if not text:
                    continue
                ax.text(x, y, text, color=fill, fontsize=fs * 0.75,
                        ha="center", va="center", transform=ax.transData,
                        fontfamily="Arial", clip_on=False)
            except Exception:
                pass


def _make_xsite_mol_from_scratch(name: str):
    """Build X-site anion mol from scratch with explicit bond types.

    RDKit collapses hypervalent Cl/I to all-single-bond on SMILES parse, so we
    build the molecule atom-by-atom with explicit BondType assignments and call
    AllChem.Compute2DCoords to get a clean 2-D layout.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import BondType

    rw = Chem.RWMol()

    if name == "ClO4-":
        cl = rw.AddAtom(Chem.Atom(17))
        o1 = rw.AddAtom(Chem.Atom(8)); rw.GetAtomWithIdx(o1).SetFormalCharge(-1)
        o2 = rw.AddAtom(Chem.Atom(8))
        o3 = rw.AddAtom(Chem.Atom(8))
        o4 = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(cl, o1, BondType.SINGLE)
        rw.AddBond(cl, o2, BondType.DOUBLE)
        rw.AddBond(cl, o3, BondType.DOUBLE)
        rw.AddBond(cl, o4, BondType.DOUBLE)
        for i in [cl, o1, o2, o3, o4]:
            rw.GetAtomWithIdx(i).SetNoImplicit(True)

    elif name == "NO3-":
        n = rw.AddAtom(Chem.Atom(7)); rw.GetAtomWithIdx(n).SetFormalCharge(1)
        o1 = rw.AddAtom(Chem.Atom(8))
        o2 = rw.AddAtom(Chem.Atom(8)); rw.GetAtomWithIdx(o2).SetFormalCharge(-1)
        o3 = rw.AddAtom(Chem.Atom(8)); rw.GetAtomWithIdx(o3).SetFormalCharge(-1)
        rw.AddBond(n, o1, BondType.DOUBLE)
        rw.AddBond(n, o2, BondType.SINGLE)
        rw.AddBond(n, o3, BondType.SINGLE)
        for i in [n, o1, o2, o3]:
            rw.GetAtomWithIdx(i).SetNoImplicit(True)

    elif name == "IO4-":
        io = rw.AddAtom(Chem.Atom(53))
        o1 = rw.AddAtom(Chem.Atom(8)); rw.GetAtomWithIdx(o1).SetFormalCharge(-1)
        o2 = rw.AddAtom(Chem.Atom(8))
        o3 = rw.AddAtom(Chem.Atom(8))
        o4 = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(io, o1, BondType.SINGLE)
        rw.AddBond(io, o2, BondType.DOUBLE)
        rw.AddBond(io, o3, BondType.DOUBLE)
        rw.AddBond(io, o4, BondType.DOUBLE)
        for i in [io, o1, o2, o3, o4]:
            rw.GetAtomWithIdx(i).SetNoImplicit(True)

    elif name == "H4IO6-":
        # orthoperiodate anion [H4IO6]⁻ (orthoperiodic acid H4IO6):
        # Reference geometry: I at center, axial =O (top) and O⁻ (bottom),
        # equatorial 4×OH at NW/NE/SW/SE (45°, 135°, 225°, 315°).
        # Atom indices: io=0, od=1(=O top), om=2(O⁻ bottom),
        #               oh1=3(NE), oh2=4(NW), oh3=5(SW), oh4=6(SE)
        io  = rw.AddAtom(Chem.Atom(53))   # I center
        od  = rw.AddAtom(Chem.Atom(8))    # =O  axial top
        om  = rw.AddAtom(Chem.Atom(8))    # O⁻  axial bottom
        rw.GetAtomWithIdx(om).SetFormalCharge(-1)
        oh1 = rw.AddAtom(Chem.Atom(8))    # OH  NE equatorial
        oh2 = rw.AddAtom(Chem.Atom(8))    # OH  NW equatorial
        oh3 = rw.AddAtom(Chem.Atom(8))    # OH  SW equatorial
        oh4 = rw.AddAtom(Chem.Atom(8))    # OH  SE equatorial
        rw.AddBond(io, od,  BondType.DOUBLE)
        rw.AddBond(io, om,  BondType.SINGLE)
        rw.AddBond(io, oh1, BondType.SINGLE)
        rw.AddBond(io, oh2, BondType.SINGLE)
        rw.AddBond(io, oh3, BondType.SINGLE)
        rw.AddBond(io, oh4, BondType.SINGLE)
        for i in [io, od, om]:
            rw.GetAtomWithIdx(i).SetNoImplicit(True)
        for i in [oh1, oh2, oh3, oh4]:
            rw.GetAtomWithIdx(i).SetNumExplicitHs(1)
            rw.GetAtomWithIdx(i).SetNoImplicit(True)

    else:
        return None

    mol = rw.GetMol()

    if name == "H4IO6-":
        # Manual 2-D coordinates matching the reference image:
        # I at center; =O axial top (90°); O⁻ axial bottom (270°);
        # 4×OH equatorial at 45°, 135°, 225°, 315° (NE, NW, SW, SE).
        from rdkit.Chem import Conformer
        import math
        conf = Conformer(mol.GetNumAtoms())
        r = 1.4   # bond length in Å
        r_eq = 1.4
        # io=0 at center
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        # od=1: axial top (90°)
        conf.SetAtomPosition(1, (0.0, r, 0.0))
        # om=2: axial bottom (270°)
        conf.SetAtomPosition(2, (0.0, -r, 0.0))
        # oh1=3: NE (45°)
        conf.SetAtomPosition(3, ( r_eq * math.cos(math.radians(45)),  r_eq * math.sin(math.radians(45)),  0.0))
        # oh2=4: NW (135°)
        conf.SetAtomPosition(4, ( r_eq * math.cos(math.radians(135)), r_eq * math.sin(math.radians(135)), 0.0))
        # oh3=5: SW (225°)
        conf.SetAtomPosition(5, ( r_eq * math.cos(math.radians(225)), r_eq * math.sin(math.radians(225)), 0.0))
        # oh4=6: SE (315°)
        conf.SetAtomPosition(6, ( r_eq * math.cos(math.radians(315)), r_eq * math.sin(math.radians(315)), 0.0))
        mol.AddConformer(conf, assignId=True)
    else:
        AllChem.Compute2DCoords(mol)

    return mol


def _prepare_display_mol(name: str, smi: str, strip_charges: bool):
    # For X-site anions, build from scratch to preserve correct bond orders.
    if name in {"ClO4-", "NO3-", "IO4-", "H4IO6-"}:
        return _make_xsite_mol_from_scratch(name)

    from rdkit import Chem

    mol = None
    if smi:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
        RDLogger.EnableLog("rdApp.*")
    if mol is None:
        return None

    if strip_charges:
        from rdkit import Chem as _Chem
        rw = _Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() != 8:
                atom.SetFormalCharge(0)
                # Suppress implicit H display: after zeroing charge, secondary-amine N
                # would gain 1 implicit H in the neutral form. Mark no-implicit so RDKit
                # doesn't render "NH" labels — the schematic shows connectivity only.
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(0)
        mol = rw.GetMol()
    else:
        # Keep formal charges but suppress explicit H on N atoms so labels read
        # "N⁺" instead of "NH₂⁺" — avoids wide labels overlapping ring bonds.
        from rdkit import Chem as _Chem
        rw = _Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 7:   # nitrogen only
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(0)
        mol = rw.GetMol()

    return mol


def _render_structures_in_ax(
    ax,
    names: list[str],
    smiles_dict: dict[str, str],
    labels_dict: dict[str, str],
    n_cols: int,
    cell_w_pts: float,
    cell_h_pts: float,
    label_h_pts: float,
    gap_x_pts: float,
    gap_y_pts: float,
    font_size_pt: float,
    bond_lw: float,
    padding: float,
    strip_charges: bool = False,
    fixed_bond_length: float = -1.0,
) -> None:
    """Draw multiple molecules in a grid layout within a single axes.

    The axes xlim/ylim must be set externally to match the total canvas size.
    Molecules are laid out left-to-right, top-to-bottom in n_cols columns.
    If strip_charges=True, all formal charges are zeroed before drawing so that
    charge annotations (e.g. Cl³⁺, N⁺) are not rendered — useful for X-site
    anions where the bond topology is what matters, not the formal charge bookkeeping.
    If fixed_bond_length > 0, all molecules are drawn with that bond length in pixels,
    ensuring consistent ring sizes across molecules of different complexity.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdCoordGen import AddCoords

    n = len(names)
    n_rows = int(np.ceil(n / n_cols))
    fs_svg = max(12, int(font_size_pt / 0.75))

    for i, name in enumerate(names):
        row = i // n_cols
        col = i % n_cols
        x_off = col * (cell_w_pts + gap_x_pts)
        # y increases upward in matplotlib; row 0 is at top
        row_bottom = (n_rows - 1 - row) * (cell_h_pts + label_h_pts + gap_y_pts)

        smi = smiles_dict.get(name, "")
        mol = _prepare_display_mol(name, smi, strip_charges=strip_charges)
        if mol is None:
            continue
        # Only call AddCoords if the mol doesn't already have 2-D coordinates
        # (mols built by _make_xsite_mol_from_scratch already have Compute2DCoords applied).
        conf = mol.GetNumConformers()
        if conf == 0:
            AddCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(int(cell_w_pts), int(cell_h_pts))
        drawer.drawOptions().addStereoAnnotation = False
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().bondLineWidth = bond_lw
        drawer.drawOptions().padding = padding
        drawer.drawOptions().minFontSize = fs_svg
        drawer.drawOptions().maxFontSize = fs_svg
        if fixed_bond_length > 0:
            try:
                drawer.drawOptions().fixedBondLength = fixed_bond_length
            except AttributeError:
                pass  # older RDKit versions may not support this
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = _remap_svg_colours(drawer.GetDrawingText())
        _draw_mol_svg_on_ax(
            ax, svg_text, x_off, cell_w_pts, cell_h_pts,
            y_offset=row_bottom + label_h_pts,
        )
        label = labels_dict.get(name, name)
        ax.text(
            x_off + cell_w_pts / 2,
            row_bottom + label_h_pts / 2,
            label,
            ha="center", va="center",
            fontsize=font_size_pt, fontfamily="Arial",
            transform=ax.transData, clip_on=False,
        )


def _draw_bond(ax, p0, p1, color, lw=1.2, order=1, sep=2.2):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    nrm = np.linalg.norm(v)
    if nrm < 1e-6:
        return
    perp = np.array([-v[1], v[0]]) / nrm
    if order == 1:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, solid_capstyle="round")
    else:
        off = perp * (sep / 2)
        for s in (-1, 1):
            q0 = p0 + s * off
            q1 = p1 + s * off
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=color, lw=lw, solid_capstyle="round")


def _draw_atom_label(ax, xy, text, color, fs=14, dx=0.0, dy=0.0):
    ax.text(
        xy[0] + dx, xy[1] + dy, text,
        color=color, fontsize=fs, ha="center", va="center",
        fontfamily="Arial", clip_on=False,
    )


def _render_manual_xsite_in_ax(
    ax,
    names: list[str],
    labels_dict: dict[str, str],
    cell_w_pts: float,
    cell_h_pts: float,
    label_h_pts: float,
    gap_y_pts: float,
    font_size_pt: float,
) -> None:
    n_rows = len(names)
    for i, name in enumerate(names):
        row_bottom = (n_rows - 1 - i) * (cell_h_pts + label_h_pts + gap_y_pts)
        cx = cell_w_pts / 2
        cy = row_bottom + label_h_pts + cell_h_pts / 2 + 2.0
        r = min(cell_w_pts, cell_h_pts) * 0.30

        if name == "ClO4-":
            center = np.array([cx, cy])
            pts = {
                "left": np.array([cx - r, cy]),
                "top": np.array([cx, cy + r]),
                "right": np.array([cx + r, cy]),
                "bottom": np.array([cx, cy - r]),
            }
            _draw_bond(ax, center, pts["left"], "#0EB87F", order=1)
            for key in ["top", "right", "bottom"]:
                _draw_bond(ax, center, pts[key], "#0EB87F", order=2)
            _draw_atom_label(ax, center, "Cl", "#0EB87F", fs=14)
            _draw_atom_label(ax, pts["left"], r"O$^{-}$", "#D63C53", fs=14, dx=-6)
            _draw_atom_label(ax, pts["top"], "O", "#D63C53", fs=14, dy=7)
            _draw_atom_label(ax, pts["right"], "O", "#D63C53", fs=14, dx=6)
            _draw_atom_label(ax, pts["bottom"], "O", "#D63C53", fs=14, dy=-7)
        elif name == "NO3-":
            center = np.array([cx, cy])
            pts = {
                "left": np.array([cx - r * 0.9, cy + r * 0.35]),
                "right": np.array([cx + r * 0.9, cy + r * 0.35]),
                "bottom": np.array([cx, cy - r]),
            }
            _draw_bond(ax, center, pts["top"] if False else pts["right"], "#125CCA", order=2)
            _draw_bond(ax, center, pts["left"], "#125CCA", order=1)
            _draw_bond(ax, center, pts["bottom"], "#125CCA", order=1)
            _draw_atom_label(ax, center, r"N$^{+}$", "#125CCA", fs=14)
            _draw_atom_label(ax, pts["right"], "O", "#D63C53", fs=14, dx=7, dy=3)
            _draw_atom_label(ax, pts["left"], r"O$^{-}$", "#D63C53", fs=14, dx=-8, dy=3)
            _draw_atom_label(ax, pts["bottom"], r"O$^{-}$", "#D63C53", fs=14, dy=-8)
        elif name == "IO4-":
            center = np.array([cx, cy])
            pts = {
                "left": np.array([cx - r, cy]),
                "top": np.array([cx, cy + r]),
                "right": np.array([cx + r, cy]),
                "bottom": np.array([cx, cy - r]),
            }
            _draw_bond(ax, center, pts["left"], "#6B5B7B", order=1)
            for key in ["top", "right", "bottom"]:
                _draw_bond(ax, center, pts[key], "#6B5B7B", order=2)
            _draw_atom_label(ax, center, "I", "#6B5B7B", fs=14)
            _draw_atom_label(ax, pts["left"], r"O$^{-}$", "#D63C53", fs=14, dx=-6)
            _draw_atom_label(ax, pts["top"], "O", "#D63C53", fs=14, dy=7)
            _draw_atom_label(ax, pts["right"], "O", "#D63C53", fs=14, dx=6)
            _draw_atom_label(ax, pts["bottom"], "O", "#D63C53", fs=14, dy=-7)
        elif name == "H4IO6-":
            center = np.array([cx, cy])
            pts = {
                "tl": np.array([cx - r * 0.8, cy + r * 0.45]),
                "tr": np.array([cx + r * 0.8, cy + r * 0.45]),
                "ml": np.array([cx - r * 0.95, cy - r * 0.05]),
                "mr": np.array([cx + r * 0.95, cy - r * 0.05]),
                "b": np.array([cx, cy - r * 0.95]),
                "t": np.array([cx, cy + r * 0.95]),
            }
            _draw_bond(ax, center, pts["t"], "#6B5B7B", order=2)
            _draw_bond(ax, center, pts["tl"], "#6B5B7B", order=2)
            _draw_bond(ax, center, pts["tr"], "#6B5B7B", order=1)
            _draw_bond(ax, center, pts["ml"], "#6B5B7B", order=1)
            _draw_bond(ax, center, pts["mr"], "#6B5B7B", order=1)
            _draw_bond(ax, center, pts["b"], "#6B5B7B", order=1)
            _draw_atom_label(ax, center, "I", "#6B5B7B", fs=14)
            _draw_atom_label(ax, pts["t"], "O", "#D63C53", fs=14, dy=6)
            _draw_atom_label(ax, pts["tl"], "O", "#D63C53", fs=14, dx=-6, dy=3)
            _draw_atom_label(ax, pts["tr"], "OH", "#D63C53", fs=14, dx=8, dy=3)
            _draw_atom_label(ax, pts["ml"], r"O$^{-}$", "#D63C53", fs=14, dx=-8)
            _draw_atom_label(ax, pts["mr"], "OH", "#D63C53", fs=14, dx=9)
            _draw_atom_label(ax, pts["b"], "OH", "#D63C53", fs=14, dy=-7)

        label = labels_dict.get(name, name)
        ax.text(
            cx,
            row_bottom + label_h_pts / 2,
            label,
            ha="center",
            va="center",
            fontsize=font_size_pt,
            fontfamily="Arial",
            transform=ax.transData,
            clip_on=False,
        )


# ---------------------------------------------------------------------------
# Ion data
# ---------------------------------------------------------------------------

A_SITE_SMILES: dict[str, str] = {
    "H2dabco2+":  "C1C[NH+]2CC[NH+]1CC2",
    "MeHdabco2+": "C[N+]12CC[NH+](CC1)CC2",
    "H2odabco2+": "O[N+]12CC[NH+](CC1)CC2",
    "H2pz2+":     "[NH2+]1CC[NH2+]CC1",
    "H2hpz2+":    "[NH2+]1CCC[NH2+]CC1",
    "MeHpz2+":    "C[NH+]1CC[NH2+]CC1",
    "HQ+":        "C1C[NH+]2CCC1CC2",
    "Huru+":      "[NH+]12CN3CN(C1)CN(C2)C3",
}

A_STRUCT_LABELS: dict[str, str] = {
    "H2dabco2+":  r"$\mathregular{H_2dabco^{2+}}$",
    "MeHdabco2+": r"$\mathregular{MeHdabco^{2+}}$",
    "H2odabco2+": r"$\mathregular{H_2odabco^{2+}}$",
    "H2pz2+":     r"$\mathregular{H_2pz^{2+}}$",
    "H2hpz2+":    r"$\mathregular{H_2hpz^{2+}}$",
    "MeHpz2+":    r"$\mathregular{MeHpz^{2+}}$",
    "HQ+":        r"$\mathregular{HQ^+}$",
    "Huru+":      r"$\mathregular{Huru^+}$",
}

X_SITE_SMILES: dict[str, str] = {
    # Correct Lewis-structure SMILES with proper double bonds.
    # Central atoms (Cl, N, I) have 0 formal charge in these representations.
    "ClO4-":  "[O-][Cl](=O)(=O)=O",              # perchlorate: 1×Cl–O⁻ + 3×Cl=O
    "NO3-":   "[O-][N+](=O)=O",                   # nitrate: N⁺ with 1×N–O⁻ + 2×N=O
    "IO4-":   "[O-][I](=O)(=O)=O",                # metaperiodate: 1×I–O⁻ + 3×I=O
    "H4IO6-": "OI(O)(O)(=O)(=O)[O-]",             # orthoperiodate: 3×I–OH + 2×I=O + 1×I–O⁻
}

X_STRUCT_LABELS: dict[str, str] = {
    "ClO4-":  r"$\mathregular{ClO_4^-}$",
    "NO3-":   r"$\mathregular{NO_3^-}$",
    "IO4-":   r"$\mathregular{IO_4^-}$",
    "H4IO6-": r"$\mathregular{[H_4IO_6]^-}$",
}

# A-site cations shown in the combined figure (6 present in training set)
_ASITE_NAMES_SHOWN = [
    "H2dabco2+", "MeHdabco2+", "H2odabco2+",
    "H2pz2+",    "H2hpz2+",    "MeHpz2+",
]

# X-site anions shown in the combined figure
_XSITE_NAMES_SHOWN = ["ClO4-", "NO3-", "IO4-", "H4IO6-"]


# ---------------------------------------------------------------------------
# Cell offset helper for data grid
# ---------------------------------------------------------------------------

def _cell_offsets(n: int) -> list[tuple[float, float]]:
    if n == 1:
        return [(0.0, 0.0)]
    if n <= 4:
        xs = np.linspace(-0.30, 0.30, n)
        return [(float(x), 0.0) for x in xs]
    ncols = int(np.ceil(n / 2))
    xs = np.linspace(-0.32, 0.32, ncols)
    ys = [0.16, -0.16]
    offsets = []
    for r in range(2):
        for c in range(ncols):
            if len(offsets) < n:
                offsets.append((float(xs[c]), float(ys[r])))
    return offsets


# ---------------------------------------------------------------------------
# Figure 1a — Combined dataset grid
# ---------------------------------------------------------------------------

def plot_dataset_grid(df: pd.DataFrame) -> None:
    """Combined figure with reference layout on one A4-page width."""
    setup_nature_style()

    a_present = [a for a in A_ORDER if a in df["A_site"].values]
    x_present = [x for x in X_ORDER if x in df["X_site"].values]
    n_rows = len(a_present)
    n_cols = len(x_present)

    dpi = 72.0

    outer_left = 0.34
    outer_right = 0.18
    outer_top = 0.18
    outer_bottom = 0.18
    gap_main = 0.05      # gap between left grid block and right block
    gap_right_v = 0.16   # vertical gap between legend and lower panels
    gap_right_h = 0.20   # horizontal gap between X-site and A-site panels

    left_margin = 0.85
    right_margin = 0.08
    top_margin = 0.50
    bottom_margin_grid = 0.06
    grid_cell_h = 0.50
    grid_h = n_rows * grid_cell_h

    legend_h = 0.90

    # X-site: 2 columns × 2 rows.
    # Cell size matches A-site so rows align when both panels are top-aligned.
    x_cell_w = 90.0
    x_cell_h = 90.0
    x_label_h = 26.0
    x_gap_x = 10.0
    x_gap_y = 6.0
    x_total_w = 2 * x_cell_w + x_gap_x
    x_total_h = 2 * (x_cell_h + x_label_h) + 1 * x_gap_y
    x_panel_h = x_total_h / dpi

    # A-site: 3 columns × 2 rows — same cell height as X-site for row alignment
    a_cell_w = 90.0
    a_cell_h = 90.0
    a_label_h = 26.0
    a_gap_x = 10.0
    a_gap_y = 6.0
    a_total_w = 3 * a_cell_w + 2 * a_gap_x
    a_total_h = 2 * (a_cell_h + a_label_h) + 1 * a_gap_y
    a_panel_h = a_total_h / dpi

    lower_h = max(x_panel_h, a_panel_h)
    right_block_h = legend_h + gap_right_v + lower_h
    left_block_h = right_block_h

    # Compute panel widths in inches from content sizes (no squishing)
    # Use tight margins: 8 pts each side for X-site, 10 pts each side for A-site
    x_w = (x_total_w + 16.0) / dpi
    a_w = (a_total_w + 20.0) / dpi
    right_block_w = x_w + gap_right_h + a_w

    # Left block: wide enough for grid labels (3 columns × ~55 pts each + margins)
    grid_label_w_pts = n_cols * 55.0 + left_margin * dpi + right_margin * dpi + 10.0
    left_block_w = grid_label_w_pts / dpi   # no artificial floor — use exact content width

    # Right block starts just after the grid axes right edge (eliminates dead right_margin space).
    # grid_axes_right = outer_left + 0.10 + (left_block_w - left_margin - right_margin)
    grid_axes_right = outer_left + 0.10 + (left_block_w - left_margin - right_margin)
    right_x_abs_pre = grid_axes_right + gap_main

    # Figure width based on actual content extent (not left_block_w which includes dead space)
    fig_w = max(210 / 25.4, right_x_abs_pre + right_block_w + outer_right)

    fig_h = outer_top + right_block_h + outer_bottom

    fig = plt.figure(figsize=(fig_w, fig_h))

    left = outer_left / fig_w
    bottom = outer_bottom / fig_h
    left_w = left_block_w / fig_w
    left_h = left_block_h / fig_h

    right_x_abs = right_x_abs_pre   # already computed above (grid_axes_right + gap_main)
    right_y_abs = outer_bottom
    lower_h_abs = lower_h
    legend_bottom_abs = right_y_abs + lower_h_abs + gap_right_v

    right_x = right_x_abs / fig_w
    right_w = right_block_w / fig_w
    legend_bottom = legend_bottom_abs / fig_h
    legend_h_frac = legend_h / fig_h
    lower_bottom = right_y_abs / fig_h
    lower_h_frac = lower_h_abs / fig_h
    x_w_frac = x_w / fig_w
    a_w_frac = a_w / fig_w
    a_x = (right_x_abs + x_w + gap_right_h) / fig_w

    ax_grid = fig.add_axes([left, bottom, left_w, left_h])
    ax_grid.set_xlim(-0.5, n_cols - 0.5)
    ax_grid.set_ylim(-0.5, n_rows - 0.5)
    ax_grid.invert_yaxis()

    for xi in range(n_cols + 1):
        ax_grid.axvline(xi - 0.5, color="#E0E0E0", linewidth=0.4, zorder=0)
    for yi in range(n_rows + 1):
        ax_grid.axhline(yi - 0.5, color="#E0E0E0", linewidth=0.4, zorder=0)

    b_seen: set[str] = set()
    for row_i, a in enumerate(a_present):
        for col_i, x in enumerate(x_present):
            x_norm = _normalise_xsite_for_grid(x)
            subset = df[
                (df["A_site"] == a)
                & (df["X_site"].apply(_normalise_xsite_for_grid) == x_norm)
            ]
            if subset.empty:
                continue
            offsets = _cell_offsets(len(subset))
            color = X_SITE_COLORS.get(x, X_SITE_DEFAULT)
            for k, (_, row) in enumerate(subset.iterrows()):
                b = row["B_site"]
                b_seen.add(b)
                marker = B_MARKERS.get(b, B_DEFAULT_MARKER)
                fc = GRAY_NO_DATA if pd.isna(row["vdet"]) else color
                dx, dy = offsets[k]
                ax_grid.scatter(
                    col_i + dx, row_i + dy,
                    marker=marker, s=28, color=fc,
                    edgecolors="#444444", linewidths=0.3, zorder=3,
                )

    ax_grid.set_xticks(range(n_cols))
    ax_grid.set_xticklabels([X_LABELS.get(x, x) for x in x_present], fontsize=14)
    ax_grid.xaxis.set_ticks_position("top")
    ax_grid.xaxis.set_label_position("top")
    ax_grid.set_xlabel("$X$-site anion", fontsize=14, labelpad=4)
    ax_grid.set_yticks(range(n_rows))
    ax_grid.set_yticklabels([A_LABELS.get(a, a) for a in a_present], fontsize=14)
    ax_grid.set_ylabel("$A$-site cation", fontsize=14, labelpad=4)
    ax_grid.spines["right"].set_visible(False)
    ax_grid.spines["bottom"].set_visible(False)
    ax_grid.spines["top"].set_linewidth(0.5)
    ax_grid.spines["left"].set_linewidth(0.5)
    ax_grid.tick_params(width=0.5, length=2)
    ax_grid.set_position([
        left + 0.10 / fig_w,
        bottom + 0.04 / fig_h,
        (left_block_w - left_margin - right_margin) / fig_w,
        (right_block_h - top_margin - bottom_margin_grid - 0.02) / fig_h,
    ])

    ax_leg = fig.add_axes([right_x, legend_bottom, right_w, legend_h_frac])
    ax_leg.axis("off")
    b_legend_order = [b for b in B_MARKERS if b in b_seen]
    legend_handles = [
        mlines.Line2D(
            [0], [0],
            marker=B_MARKERS[b], color="w",
            markerfacecolor="#888888", markeredgecolor="#444444",
            markeredgewidth=0.3, markersize=6,
            label=B_LABELS.get(b, b),
        )
        for b in b_legend_order
    ]
    ax_leg.legend(
        handles=legend_handles,
        title="$B$ site",
        title_fontsize=14,
        fontsize=14,
        loc="center",
        ncol=len(legend_handles),
        frameon=False,
        handlelength=0.8,
        handletextpad=0.3,
        borderpad=0.2,
        labelspacing=0.2,
        columnspacing=0.6,
    )

    margin_x = max(8.0, (x_w * dpi - x_total_w) / 2)
    # Use exact x_panel_h for X-site axes (not lower_h) to avoid vertical stretching.
    # Top-align with the A-site panel.
    x_bottom_abs = right_y_abs + lower_h - x_panel_h
    x_bottom_frac = x_bottom_abs / fig_h
    ax_xsite = fig.add_axes([right_x, x_bottom_frac, x_w_frac, x_panel_h / fig_h])
    ax_xsite.set_xlim(-margin_x, x_total_w + margin_x)
    ax_xsite.set_ylim(-x_label_h, x_total_h + 20)
    ax_xsite.axis("off")
    ax_xsite.text(
        x_total_w / 2, x_total_h + 10,
        "$X$ site",
        ha="center", va="bottom", fontsize=14, fontfamily="Arial",
        transform=ax_xsite.transData, clip_on=False,
    )
    _render_structures_in_ax(
        ax_xsite,
        names=_XSITE_NAMES_SHOWN,
        smiles_dict=X_SITE_SMILES,
        labels_dict=X_STRUCT_LABELS,
        n_cols=2,
        cell_w_pts=x_cell_w,
        cell_h_pts=x_cell_h,
        label_h_pts=x_label_h,
        gap_x_pts=x_gap_x,
        gap_y_pts=x_gap_y,
        font_size_pt=14.0,
        bond_lw=1.2,
        padding=0.22,
        strip_charges=False,
    )

    margin_a = max(8.0, (a_w * dpi - a_total_w) / 2)
    # Top-align A-site with X-site: both have 2 rows, use exact panel height
    a_bottom_abs = right_y_abs + lower_h - a_panel_h
    a_bottom_frac = a_bottom_abs / fig_h
    ax_asite = fig.add_axes([a_x, a_bottom_frac, a_w_frac, a_panel_h / fig_h])
    ax_asite.set_xlim(-margin_a, a_total_w + margin_a)
    ax_asite.set_ylim(-a_label_h, a_total_h + 20)
    ax_asite.axis("off")
    ax_asite.text(
        a_total_w / 2, a_total_h + 10,
        "$A$ site",
        ha="center", va="bottom", fontsize=14, fontfamily="Arial",
        transform=ax_asite.transData, clip_on=False,
    )
    _render_structures_in_ax(
        ax_asite,
        names=_ASITE_NAMES_SHOWN,
        smiles_dict=A_SITE_SMILES,
        labels_dict=A_STRUCT_LABELS,
        n_cols=3,
        cell_w_pts=a_cell_w,
        cell_h_pts=a_cell_h,
        label_h_pts=a_label_h,
        gap_x_pts=a_gap_x,
        gap_y_pts=a_gap_y,
        font_size_pt=14.0,
        bond_lw=1.2,
        padding=0.18,
        strip_charges=False,
        fixed_bond_length=20.0,
    )

    out = OUT_DIR / "_dataset_grid.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    fig.savefig(str(out.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} and .pdf")


# ---------------------------------------------------------------------------
# Figure 1b — Stand-alone A-site structures (2×3 grid)
# ---------------------------------------------------------------------------

def plot_asite_structures() -> None:
    """Draw the 6 A-site cation structures in a 2-row × 3-column grid."""
    setup_nature_style()

    names = _ASITE_NAMES_SHOWN
    n_cols = 3
    n_rows = 2

    cell_w = 160.0
    cell_h = 140.0
    label_h = 32.0
    gap_x = 8.0
    gap_y = 14.0

    total_w_pts = n_cols * cell_w + (n_cols - 1) * gap_x
    total_h_pts = n_rows * (cell_h + label_h) + (n_rows - 1) * gap_y

    margin_x = 8.0
    margin_y = 8.0

    dpi = 72.0
    fig_w = (total_w_pts + 2 * margin_x) / dpi
    fig_h = (total_h_pts + 2 * margin_y) / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-margin_x, total_w_pts + margin_x)
    ax.set_ylim(-margin_y, total_h_pts + margin_y)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0)

    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.rdCoordGen import AddCoords

    for i, name in enumerate(names):
        row = i // n_cols
        col = i % n_cols
        x_off = col * (cell_w + gap_x)
        row_bottom = (n_rows - 1 - row) * (cell_h + label_h + gap_y)

        smi = A_SITE_SMILES[name]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  WARNING: could not parse SMILES for {name}: {smi}")
            continue
        AddCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(int(cell_w), int(cell_h))
        drawer.drawOptions().addStereoAnnotation = False
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().bondLineWidth = 1.6
        drawer.drawOptions().padding = 0.20
        drawer.drawOptions().minFontSize = 19
        drawer.drawOptions().maxFontSize = 19
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = _remap_svg_colours(drawer.GetDrawingText())
        _draw_mol_svg_on_ax(ax, svg_text, x_off, cell_w, cell_h,
                            y_offset=row_bottom + label_h)

        label = A_STRUCT_LABELS.get(name, name)
        ax.text(x_off + cell_w / 2, row_bottom + label_h / 2, label,
                ha="center", va="center",
                fontsize=14.0, transform=ax.transData, fontfamily="Arial",
                clip_on=False)

    out = OUT_DIR / "_asite_structures.png"
    save_png_pdf(fig, out)
    plt.close(fig)
    print(f"Saved {out} and .pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_pems()
    print(f"Loaded {len(df)} materials")
    plot_dataset_grid(df)
    plot_asite_structures()


if __name__ == "__main__":
    main()
