"""plot_anova_strip.py — ANOVA strip-plot: Vdet grouped by X-site, B-site, A-site.

Reads data/pems/pems.csv, computes one-way eta-squared for each site factor,
and produces a three-panel strip-plot saved as
  manuscript/figures/_anova_strip.{png,pdf}

Usage
-----
  python manuscript/figures/plot_anova_strip.py

Requirements: numpy, pandas, matplotlib (all in the dpa3 conda env).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]   # manuscript/figures/ → repo root
DATA_CSV  = REPO_ROOT / "data" / "pems" / "pems.csv"
OUT_DIR   = Path(__file__).resolve().parent        # save alongside other figures
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style (paper_plot_style.py lives in experiments_davis2024/)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "experiments_davis2024"))
from paper_plot_style import setup_nature_style, style_axes, add_panel_label, save_png_pdf

# ---------------------------------------------------------------------------
# Per-panel color palettes (distinct hue families per site type)
# ---------------------------------------------------------------------------
X_SITE_COLORS = {
    "ClO4-":  "#5A6D7B",   # slate blue
    "NO3-":   "#8B7355",   # warm brown
    "IO4-":   "#7A6B8A",   # muted purple
    "H4IO6-": "#A07DA0",   # lighter purple (orthoperiodate)
    "ClO3-":  "#6B7D6A",   # muted sage (fallback for any extra X-site)
}
X_SITE_DEFAULT = "#5A6D7B"

B_SITE_COLORS = {
    "Na+":      "#A0AD80",
    "K+":       "#8B9C68",
    "NH4+":     "#6B7C4E",
    "Ag+":      "#4A5C30",
    "NH3OH+":   "#3A4C20",
    "NH2NH3+":  "#5C6E40",
    "Rb+":      "#7A8E5A",
    "Ba2+":     "#2E3E18",
}
B_SITE_DEFAULT = "#6B7C4E"

A_SITE_COLOR = "#8A5A67"   # single muted rose for all A-sites (many families)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eta_squared(values: np.ndarray, groups: list[np.ndarray]) -> float:
    """One-way eta-squared from pre-split group arrays."""
    grand_mean = values.mean()
    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def _jitter(n: int, width: float = 0.18, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.uniform(-width, width, n)


# Explicit LaTeX tick labels for every ion that appears in the dataset.
_ION_LATEX: dict[str, str] = {
    # X-site
    "ClO4-":   r"$\mathrm{ClO_4^-}$",
    "NO3-":    r"$\mathrm{NO_3^-}$",
    "IO4-":    r"$\mathrm{IO_4^-}$",
    "H4IO6-":  r"$\mathrm{[H_4IO_6]^-}$",
    "ClO3-":   r"$\mathrm{ClO_3^-}$",
    # B-site
    "Na+":     r"$\mathrm{Na^+}$",
    "K+":      r"$\mathrm{K^+}$",
    "NH4+":    r"$\mathrm{NH_4^+}$",
    "Ag+":     r"$\mathrm{Ag^+}$",
    "Rb+":     r"$\mathrm{Rb^+}$",
    "Ba2+":    r"$\mathrm{Ba^{2+}}$",
    "NH3OH+":  r"$\mathrm{NH_3OH^+}$",
    "NH2NH3+": r"$\mathrm{NH_2NH_3^+}$",
    # A-site
    "H2dabco2+":  r"$\mathrm{H_2dabco^{2+}}$",
    "H2odabco2+": r"$\mathrm{H_2odabco^{2+}}$",
    "H2pz2+":     r"$\mathrm{H_2pz^{2+}}$",
    "H2hpz2+":    r"$\mathrm{H_2hpz^{2+}}$",
    "MeHpz2+":    r"$\mathrm{MeHpz^{2+}}$",
    "MeHdabco2+": r"$\mathrm{MeHdabco^{2+}}$",
    "Huru+":      r"$\mathrm{Huru^+}$",
    "Na+/NH4+ (ordered)": r"$\mathrm{Na^+/NH_4^+}$*",
}


def _ion_label(s: str) -> str:
    """Return a LaTeX math-text label for an ion string."""
    return _ION_LATEX.get(s, s)


def _normalize_site(raw) -> str:
    """Collapse minor spelling variants to a canonical label."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    replacements = {
        "ClO4-":  ["ClO4-", "ClO4−", "perchlorate"],
        "NO3-":   ["NO3-",  "NO3−",  "nitrate"],
        "IO4-":   ["IO4-",  "IO4−",  "periodate"],
        "H4IO6-": ["H4IO6-", "orthoperiodate"],
        "ClO3-":  ["ClO3-", "ClO3−", "chlorate"],
        "NH4+":  ["NH4+",  "NH4＋"],
        "Na+":   ["Na+",   "Na＋"],
        "K+":    ["K+",    "K＋"],
        "Ag+":   ["Ag+",   "Ag＋"],
        "Rb+":   ["Rb+",   "Rb＋"],
        "Ba2+":  ["Ba2+",  "Ba2＋", "Ba²+"],
        "NH3OH+":  ["NH3OH+"],
        "NH2NH3+": ["NH2NH3+"],
    }
    for canonical, variants in replacements.items():
        if s in variants:
            return canonical
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CANONICAL_MATERIALS = {
    "DAI-1", "DAI-2", "DAI-4", "DAI-X1", "DAN-2",
    "DAP-1", "DAP-2", "DAP-3", "DAP-4", "DAP-5", "DAP-6", "DAP-7",
    "DAP-M4", "DAP-O2", "DAP-O4",
    "PAN-2", "PAN-H2", "PAN-M2",
    "PAP-1", "PAP-4", "PAP-5", "PAP-H4", "PAP-H5", "PAP-M4", "PAP-M5",
}

_EXCLUDE_XSITES = {"H4IO6-"}

# Explicit group orderings to match composition matrix (plot_dataset_overview.py)
X_SITE_ORDER = ["ClO4-", "NO3-", "IO4-"]   # Perchlorate, Nitrate, Periodate
A_SITE_ORDER = [
    "H2dabco2+", "MeHdabco2+", "H2odabco2+",   # dabco family
    "H2pz2+",    "H2hpz2+",    "MeHpz2+",       # pz family
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, index_col=False)
    df = df[df["material"].isin(CANONICAL_MATERIALS)].copy()
    df["D_km_s"] = pd.to_numeric(df["D_km_s"], errors="coerce")
    df = df[df["D_km_s"].notna()].copy()
    df["vdet"] = df["D_km_s"] * 1000.0   # km·s⁻¹ → m·s⁻¹
    df["x_site"] = df["X_site"].fillna("").apply(_normalize_site)
    df["b_site"] = df["B_site"].fillna("").apply(_normalize_site)
    df["a_site"] = df["A_site"].fillna("").apply(_normalize_site)
    df = df[~df["x_site"].isin(_EXCLUDE_XSITES)].copy()
    return df


def _draw_panel_vertical(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    color_map: dict[str, str],
    default_color: str,
    panel_label: str,
    eta2: float,
    show_ylabel: bool,
    rng: np.random.Generator,
    group_order: list[str] | None = None,
) -> None:
    """Draw one vertical strip-plot panel (groups on x-axis, Vdet on y-axis)."""
    present = set(df[group_col].unique())
    if group_order is not None:
        # Keep only present groups, in the specified order; append any extras at end
        groups = [g for g in group_order if g in present]
        groups += sorted(g for g in present if g not in group_order)
    else:
        groups = sorted(present)

    # Thin horizontal grid lines
    ax.yaxis.grid(True, color="#ECECEC", linewidth=0.3, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    for xi, grp in enumerate(groups):
        mask = df[group_col] == grp
        vals = df.loc[mask, "vdet"].values
        n = len(vals)
        color = color_map.get(grp, default_color)

        # Vertical range line
        ax.vlines(xi, vals.min(), vals.max(), color="#C0C0C0", linewidth=0.5, zorder=1)

        # Jittered dots (jitter on x-axis)
        jx = xi + _jitter(n, width=0.18, rng=rng)
        ax.scatter(jx, vals, color=color, s=12, alpha=0.70, linewidths=0,
                   zorder=3, clip_on=True)

        # Mean tick (horizontal bar)
        mean_val = vals.mean()
        ax.hlines(mean_val, xi - 0.28, xi + 0.28,
                  color=color, linewidth=1.2, zorder=4)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(
        [_ion_label(g) for g in groups],
        fontsize=14,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(groups) - 0.4)

    # Panel title above each panel: site letter + eta²
    site_letter = panel_label[0]   # "X", "B", or "A"
    ax.set_title(
        f"${site_letter}$-site  ($\\eta^2 = {eta2:.2f}$)",
        fontsize=14,
        pad=4,
        loc="center",
        color="#2F2F2F",
    )

    style_axes(ax, grid=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5)

    # All panels share y-axis (Vdet); only the leftmost panel shows the y-axis label.
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="y", labelsize=14)
    if show_ylabel:
        ax.set_ylabel(r"$V_\mathrm{det}$ (m$\cdot$s$^{-1}$)", fontsize=14)
    else:
        ax.tick_params(axis="y", labelleft=False)


def main() -> None:
    setup_nature_style()
    rng = np.random.default_rng(42)

    df = load_data()

    def _eta2_for(col: str) -> float:
        groups = [df.loc[df[col] == g, "vdet"].values for g in df[col].unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return 0.0
        all_vals = np.concatenate(groups)
        return _eta_squared(all_vals, groups)

    eta2_x = _eta2_for("x_site")
    eta2_b = _eta2_for("b_site")
    eta2_a = _eta2_for("a_site")

    print(f"η²(X-site) = {eta2_x:.3f}")
    print(f"η²(B-site) = {eta2_b:.3f}")
    print(f"η²(A-site) = {eta2_a:.3f}")

    # Count groups per panel to size columns proportionally
    n_x = df["x_site"].nunique()
    n_b = df["b_site"].nunique()
    n_a = df["a_site"].nunique()

    # Figure layout: 3 panels side by side (horizontal).
    # Groups on x-axis, Vdet on y-axis (vertical strip plot).
    # Total width = 2/3 A4 = 140 mm.
    # Width proportional to number of groups; height fixed.
    fig_w = 140 / 25.4 * 1.5   # 5.51 inches × 1.5  (2/3 A4 × 1.5)
    col_w = 0.52          # inches per group column
    fig_h = 3.2           # fixed height (inches)

    fig, axes = plt.subplots(
        1, 3,
        sharey=True,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [n_x, n_b, n_a]},
    )

    panel_specs = [
        # (ax, group_col, color_map, default_color, panel_label, eta2, show_ylabel, group_order)
        (axes[0], "x_site", X_SITE_COLORS, X_SITE_DEFAULT, "X-site", eta2_x, True,  X_SITE_ORDER),
        (axes[1], "b_site", B_SITE_COLORS, B_SITE_DEFAULT, "B-site", eta2_b, False, None),
        (axes[2], "a_site", {},            A_SITE_COLOR,   "A-site", eta2_a, False, A_SITE_ORDER),
    ]

    for ax, gcol, cmap, dcol, plabel, eta2, show_yl, gorder in panel_specs:
        _draw_panel_vertical(ax, df, gcol, cmap, dcol, plabel, eta2, show_yl, rng,
                             group_order=gorder)

    # left:   room for y-axis label + tick labels on leftmost panel
    # bottom: room for rotated x-tick labels (ion names)
    # top:    room for panel titles
    # wspace: gap between panels
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.32, wspace=0.08)

    out_path = OUT_DIR / "_anova_strip.png"
    save_png_pdf(fig, out_path)
    plt.close(fig)
    print(f"Saved {out_path} and .pdf")


if __name__ == "__main__":
    main()
