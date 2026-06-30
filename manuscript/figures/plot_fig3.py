#!/usr/bin/env python3
"""Figure 3 -- Why few-shot adaptation works and how well it predicts.

This script rebuilds the manuscript figure directly from refreshed experiment
outputs under ``experiments``.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
EXP_DIR = REPO_ROOT / "experiments"
OUT_PATH = THIS_DIR / "figure3.png"
SI_DESC_DIST_PATH = THIS_DIR / "figure_si_descriptor_distance.png"
OOD_HELDOUT_PATH = EXP_DIR / "pems_ood_heldout_exp7_all.json"

sys.path.insert(0, str(EXP_DIR))
from paper_plot_style import save_png_pdf, style_axes  # noqa: E402
from figure_style import display_material  # noqa: E402

TARGET_MATERIALS = {
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
}

COLORS = {
    "mt": "#4A6274",
    "st": "#7A4B58",
    "scratch": "#8C8C8C",
    "zero_shot": "#C8B8A8",
    "composition": "#C8B8A8",
    "clo4": "#5A6D7B",
    "no3": "#8B7355",
    "io4": "#7A6B8A",
    "pems": "#7A4B58",
    "charcoal": "#2F2F2F",
    "grid": "#E8E8E8",
    "light_gray": "#F1F1F1",
}

X_ORDER = ["ClO4-", "NO3-", "IO4-"]
X_LABELS = {
    "ClO4-": r"ClO$_4^-$",
    "NO3-": r"NO$_3^-$",
    "IO4-": r"IO$_4^-$",
    "All": "All",
}
X_COLORS = {
    "ClO4-": COLORS["clo4"],
    "NO3-": COLORS["no3"],
    "IO4-": COLORS["io4"],
}

PERT_ORDER = ["rotation", "translation", "dap4_template"]
PERT_LABELS = {
    "rotation": "Rotation",
    "translation": "Translation",
    "dap4_template": "Template\nswap",
}
PERT_SHORT = {
    "rotation": "R",
    "translation": "T",
    "dap4_template": "S",
}

MODEL_ORDER = ["exp7a", "exp7c", "exp7d"]
MODEL_COLORS = {
    "exp7a": COLORS["mt"],
    "exp7c": COLORS["st"],
    "exp7d": COLORS["scratch"],
}
MODEL_SHORT = {
    "exp7a": "MT-FT",
    "exp7c": "ST-FT",
    "exp7d": "ST-TFS",
}

CROSS_ORDER = ["exp7a", "exp7b", "exp7c", "exp7d", "exp8a"]
CROSS_LABELS = {
    "exp7a": "MT-FT\ncluster",
    "exp7b": "MT-FT-aux\ncluster",
    "exp7c": "ST-FT\ncluster",
    "exp7d": "ST-TFS\ncluster",
    "exp8a": "MT-FT\ncrystal",
}

# Refreshed exp4d zero-shot MAE on cleaned PEM clusters. The refresh did not keep a
# dedicated per-material JSON under experiments, so the scalar summary is
# stored directly for the adaptation ladder.
ZERO_SHOT_MAE_M_S = 1422.6

FS_TITLE = 11
FS_LABEL = 10
FS_TICK = 9
FS_LEGEND = 9
FS_ANNOT = 9
FS_PANEL = 12
FS_ROW = 10.5


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.03,
        label.lower(),
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["charcoal"],
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": COLORS["charcoal"],
            "axes.labelcolor": COLORS["charcoal"],
            "axes.edgecolor": COLORS["charcoal"],
            "xtick.color": COLORS["charcoal"],
            "ytick.color": COLORS["charcoal"],
        }
    )


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_x(raw: str) -> str:
    raw = (raw or "").strip()
    return "IO4-" if raw == "H4IO6-" else raw


def load_pems_metadata() -> dict[str, dict[str, object]]:
    path = REPO_ROOT / "data" / "pems" / "pems.csv"
    meta: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material = (row.get("material") or "").strip()
            value = (row.get("D_km_s") or "").strip()
            if material not in TARGET_MATERIALS or not value:
                continue
            meta[material] = {
                "material": material,
                "A_site": (row.get("A_site") or "").strip(),
                "B_site": (row.get("B_site") or "").strip(),
                "X_site": canonical_x(row.get("X_site") or ""),
                "vdet": float(value) * 1000.0,
            }
    return meta


def load_splits() -> list[list[str]]:
    data = _load_json(EXP_DIR / "00_data_prep" / "pems_5fold_splits_v2.json")
    return [list(v) for _, v in sorted(data["folds"].items(), key=lambda kv: int(kv[0]))]


def compute_mean_predictor(meta: dict[str, dict[str, object]]) -> tuple[float, float]:
    vals = np.array([float(rec["vdet"]) for rec in meta.values()], dtype=float)
    mean_val = float(np.mean(vals))
    mean_mae = float(np.mean(np.abs(vals - mean_val)))
    return mean_val, mean_mae


def compute_xsite_baseline(
    meta: dict[str, dict[str, object]], splits: list[list[str]]
) -> dict[str, object]:
    materials = sorted(meta)
    preds: dict[str, float] = {}
    fold_group_mae: dict[str, list[float]] = defaultdict(list)
    for val_mats in splits:
        train_mats = [m for m in materials if m not in val_mats]
        train_by_x: dict[str, list[float]] = defaultdict(list)
        for mat in train_mats:
            train_by_x[str(meta[mat]["X_site"])].append(float(meta[mat]["vdet"]))
        global_mean = float(np.mean([float(meta[m]["vdet"]) for m in train_mats]))
        for mat in val_mats:
            x_site = str(meta[mat]["X_site"])
            x_vals = train_by_x.get(x_site, [])
            preds[mat] = float(np.mean(x_vals)) if x_vals else global_mean
        for grp in X_ORDER + ["All"]:
            group_mats = [m for m in val_mats if grp == "All" or str(meta[m]["X_site"]) == grp]
            if not group_mats:
                continue
            fold_group_mae[grp].append(
                float(np.mean([abs(preds[m] - float(meta[m]["vdet"])) for m in group_mats]))
            )

    group_stats: dict[str, dict[str, float]] = {}
    for grp in X_ORDER + ["All"]:
        group_mats = [m for m in materials if grp == "All" or str(meta[m]["X_site"]) == grp]
        abs_err = [abs(preds[m] - float(meta[m]["vdet"])) for m in group_mats]
        group_stats[grp] = {
            "mae": float(np.mean(abs_err)),
            "fold_std": float(np.std(fold_group_mae[grp])) if fold_group_mae[grp] else 0.0,
            "n": float(len(group_mats)),
        }

    return {
        "preds": preds,
        "overall_mae": group_stats["All"]["mae"],
        "group_stats": group_stats,
    }


def load_family_predictions(family: str) -> dict[str, dict[str, float | int]]:
    rows = _load_json(EXP_DIR / f"pems_ood_model_deviation_{family}.json")
    out: dict[str, dict[str, float | int]] = {}
    for row in rows:
        material = row.get("material")
        if material not in TARGET_MATERIALS:
            continue
        if row.get("is_ood") is not False or row.get("honest_pred_m_s") is None:
            continue
        out[str(material)] = {
            "pred": float(row["honest_pred_m_s"]),
            "gt": float(row["exp_m_s"]),
            "abs_error": float(row["honest_abs_error_m_s"]),
            "signed_error": float(row["honest_error_m_s"]),
            "model_std": float(row["model_std_m_s"]),
            "heldout_fold": int(row["heldout_fold"]),
        }
    return out


def compute_family_metrics(rows: dict[str, dict[str, float | int]]) -> dict[str, float]:
    y_true = np.array([float(rec["gt"]) for rec in rows.values()], dtype=float)
    y_pred = np.array([float(rec["pred"]) for rec in rows.values()], dtype=float)
    abs_err = np.array([float(rec["abs_error"]) for rec in rows.values()], dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {
        "mae": float(np.mean(abs_err)),
        "r2": float(r2),
        "mape": float(np.mean(100.0 * abs_err / y_true)),
    }


def compute_family_group_stats(
    rows: dict[str, dict[str, float | int]],
    meta: dict[str, dict[str, object]],
    splits: list[list[str]],
) -> dict[str, dict[str, float]]:
    fold_group_mae: dict[str, list[float]] = defaultdict(list)
    for val_mats in splits:
        for grp in X_ORDER + ["All"]:
            group_mats = [m for m in val_mats if grp == "All" or str(meta[m]["X_site"]) == grp]
            if not group_mats:
                continue
            fold_group_mae[grp].append(
                float(np.mean([float(rows[m]["abs_error"]) for m in group_mats]))
            )

    out: dict[str, dict[str, float]] = {}
    for grp in X_ORDER + ["All"]:
        group_mats = [m for m in rows if grp == "All" or str(meta[m]["X_site"]) == grp]
        # Sample std across folds (ddof=1, n=5) — matches the
        # pool-clusters-then-fold-std convention used elsewhere.
        fg = fold_group_mae[grp]
        if len(fg) >= 2:
            fold_std = float(np.std(fg, ddof=1))
        else:
            fold_std = 0.0
        out[grp] = {
            "mae": float(np.mean([float(rows[m]["abs_error"]) for m in group_mats])),
            "fold_std": fold_std,
            "n": float(len(group_mats)),
        }
    return out


def load_cross_representation() -> tuple[np.ndarray, list[str], list[str]]:
    raw = _load_json(EXP_DIR / "cross_infer_rep.json")
    matrix = np.array(
        [[float(raw[row][col]["mean_mae"]) for col in ["cluster_n1", "crystal"]] for row in CROSS_ORDER],
        dtype=float,
    )
    return matrix, [CROSS_LABELS[row] for row in CROSS_ORDER], ["Cluster", "Crystal"]


def load_sensitivity() -> dict[str, dict[str, dict[str, float]]]:
    raw = _load_json(EXP_DIR / "pems_sensitivity_summary.json")["sensitivity_comparison"]
    exp_map = {
        "exp7a_5fold_cv": "exp7a",
        "exp7c_5fold_cv": "exp7c",
        "exp7d_5fold_cv": "exp7d",
    }
    out: dict[str, dict[str, dict[str, float]]] = {fam: {} for fam in MODEL_ORDER}
    for rec in raw:
        family = exp_map.get(rec.get("experiment", ""))
        perturb = rec.get("perturbation")
        if family is None or perturb not in PERT_ORDER:
            continue
        out[family][str(perturb)] = {
            "baseline_mae": float(rec["baseline_mae_mean"]),
            "delta_mae": float(rec["delta_mae"]),
            "delta_pct": float(rec["delta_mae_pct"]),
        }
    return out


def load_descriptor_distances() -> tuple[dict[str, dict[str, float | bool]], float]:
    raw = _load_json(
        EXP_DIR / "exp_ood_pretrained_domain" / "_descriptor_distances" / "descriptor_distances.json"
    )
    return raw["pems_cluster_distances"], float(raw["coverage_thresholds"]["dap_max_knn"])


def load_pretrained_domain_ood() -> dict[str, dict[str, float | int]]:
    """Load the pretrained-domain OOD split summary refreshed against the
    Apr 25 v3 retraining checkpoints.

    The JSON groups results by experiment key ``pretrained_domain_{mt,st,sd}``
    and stores the in-distribution (non-DAP, ``non_dap_mae_m_s``) and the
    pretrained-domain held-out (``dap_mae_m_s``) MAEs that drive Panel g.
    """
    raw = _load_json(EXP_DIR / "exp_ood_pretrained_domain" / "pretrained_domain_results.json")
    out: dict[str, dict[str, float | int]] = {}
    for key, exp_key in (("mt", "pretrained_domain_mt"), ("st", "pretrained_domain_st"), ("sd", "pretrained_domain_sd")):
        rec = raw[exp_key]
        out[key] = {
            "ind_mae": float(rec["non_dap_mae_m_s"]),
            "ood_mae": float(rec["dap_mae_m_s"]),
            "n_ind": int(rec["n_train"]),
            "n_ood": int(rec["n_test"]),
        }
    return out


def load_ood_heldout_comparison(
    materials: tuple[str, ...] = ("DAC-4", "TAP-2", "EAP-4", "SY"),
    series: tuple[str, ...] = ("exp7a", "exp7c", "exp7d"),
) -> dict[str, dict[str, object]]:
    """Load OOD-holdout absolute errors for the MT/ST/TFS comparison panel.

    The double-perovskite fail case is deliberately not included here; it is retained in the data file
    but treated as a fail-case material outside the main Figure 3 summary.
    """
    raw = _load_json(OOD_HELDOUT_PATH)
    families = {str(fam["series"]): fam for fam in raw["families"]}
    out: dict[str, dict[str, object]] = {}
    for fam_key in series:
        rows = {
            str(row["material"]): {
                "exp_m_s": float(row["exp_m_s"]),
                "pred_m_s": float(row["pred_m_s"]),
                "ae_m_s": float(row["ae_m_s"]),
            }
            for row in families[fam_key]["materials"]
            if str(row["material"]) in materials
        }
        aes = np.array([float(rows[m]["ae_m_s"]) for m in materials], dtype=float)
        out[fam_key] = {
            "materials": rows,
            "mean_ae_m_s": float(np.mean(aes)),
        }
    return out


def plot_panel_a(
    ax: plt.Axes,
    meta: dict[str, dict[str, object]],
    mean_vdet: float,
    mean_mae: float,
    model_mae: float,
) -> None:
    values = np.array([float(rec["vdet"]) for rec in meta.values()], dtype=float)
    bins = np.linspace(6000.0, 9500.0, 11)
    ax.hist(
        values,
        bins=bins,
        color=COLORS["pems"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.axvspan(mean_vdet - mean_mae, mean_vdet + mean_mae, color=COLORS["scratch"], alpha=0.10)
    ax.axvspan(mean_vdet - model_mae, mean_vdet + model_mae, color=COLORS["mt"], alpha=0.16)
    ax.axvline(mean_vdet, color="black", ls="--", lw=1.0)
    ax.set_xlim(6000.0, 9500.0)
    ax.set_xlabel(r"$V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.set_ylabel("Count")
    y_top = max(ax.get_ylim())
    ax.text(mean_vdet + 18.0, y_top * 0.96, "Mean", fontsize=FS_ANNOT - 0.4, color=COLORS["charcoal"])
    text = (
        "MAE (m$\\cdot$s$^{-1}$)\n"
        f"Mean {mean_mae:.0f}\n"
        f"MT-FT {model_mae:.0f}"
    )
    ax.text(
        0.03,
        0.94,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_ANNOT - 1.0,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.94},
    )
    style_axes(ax, grid=False)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.6, linestyle=":")
    ax.set_axisbelow(True)


def plot_panel_b(
    ax: plt.Axes,
    mean_mae: float,
    exp7a_metrics: dict[str, float],
    exp7c_metrics: dict[str, float],
    exp7d_metrics: dict[str, float],
) -> None:
    ladder = [
        ("Zero-shot", ZERO_SHOT_MAE_M_S, COLORS["zero_shot"]),
        ("ST-FT", exp7c_metrics["mae"], COLORS["st"]),
        ("ST-TFS", exp7d_metrics["mae"], COLORS["scratch"]),
        ("MT-FT", exp7a_metrics["mae"], COLORS["mt"]),
    ]
    labels = [item[0] for item in ladder]
    values = [item[1] for item in ladder]
    bar_colors = [item[2] for item in ladder]
    y = np.arange(len(ladder), dtype=float)
    ax.barh(y, values, color=bar_colors, height=0.62, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(mean_mae, color="black", ls=":", lw=1.0, zorder=2)
    ax.text(
        mean_mae + 8.0,
        0.02,
        "Mean predictor",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        fontsize=FS_ANNOT - 0.9,
        color=COLORS["charcoal"],
        ha="left",
        va="bottom",
    )
    for yi, value in zip(y, values):
        ax.text(value + 18.0, yi, f"{value:.0f}", va="center", ha="left", fontsize=FS_ANNOT)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max(values) * 1.18)
    ax.set_xlabel("MAE (m$\\cdot$s$^{-1}$)")
    style_axes(ax, grid=True)


def plot_panel_c_representation(
    ax: plt.Axes,
    cax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> None:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cross_repr",
        ["#2F5B73", "#7B96A4", "#D1C8BA", "#F4EEE6"],
    )
    norm = mcolors.Normalize(vmin=float(np.min(matrix)), vmax=float(np.max(matrix)))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            text_color = "white" if norm(value) < 0.38 else COLORS["charcoal"]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=FS_ANNOT, color=text_color)
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Cross-representation MAE", pad=4)
    ax.tick_params(axis="x", labelrotation=0, length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("MAE (m$\\cdot$s$^{-1}$)", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)


def plot_panel_c_sensitivity(
    ax: plt.Axes,
    cax: plt.Axes,
    sensitivity: dict[str, dict[str, dict[str, float]]],
) -> None:
    row_keys = ["exp7a", "exp7c", "exp7d"]
    row_labels = ["MT-FT", "ST-FT", "ST-TFS"]
    col_labels = [PERT_LABELS[p] for p in PERT_ORDER]
    data = np.array(
        [[float(sensitivity[row][pert]["delta_mae"]) for pert in PERT_ORDER] for row in row_keys],
        dtype=float,
    )
    vmax = float(max(abs(np.min(data)), abs(np.max(data))))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sens_div",
        [COLORS["mt"], "#F7F7F7", COLORS["st"]],
    )
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = float(data[i, j])
            text_color = "white" if abs(value) > 0.50 * vmax else COLORS["charcoal"]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=FS_ANNOT, color=text_color)
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Sensitivity ΔMAE", pad=4)
    ax.tick_params(axis="x", labelrotation=0, length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\Delta$MAE (m$\cdot$s$^{-1}$)", fontsize=FS_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)


def plot_panel_d(
    ax: plt.Axes,
    exp7a_rows: dict[str, dict[str, float | int]],
    metrics: dict[str, float],
    mean_mae: float,
) -> None:
    mats = sorted(exp7a_rows, key=lambda m: float(exp7a_rows[m]["gt"]))
    y_true = np.array([float(exp7a_rows[m]["gt"]) for m in mats], dtype=float)
    y_pred = np.array([float(exp7a_rows[m]["pred"]) for m in mats], dtype=float)
    y_std = np.array([float(exp7a_rows[m]["model_std"]) for m in mats], dtype=float)
    lo = min(float(np.min(y_true)), float(np.min(y_pred - y_std))) - 120.0
    hi = max(float(np.max(y_true)), float(np.max(y_pred + y_std))) + 120.0
    xline = np.linspace(lo, hi, 256)
    ax.fill_between(xline, 0.99 * xline, 1.01 * xline, color=COLORS["light_gray"], alpha=0.90, linewidth=0, edgecolor="none", zorder=1)
    ax.fill_between(xline, xline - metrics["mae"], xline + metrics["mae"], color="#D9D9D9", alpha=0.48, linewidth=0, edgecolor="none", zorder=1)
    ax.plot(xline, xline, color="black", lw=0.9, zorder=2)
    ax.errorbar(
        y_true,
        y_pred,
        yerr=y_std,
        fmt="none",
        ecolor=COLORS["mt"],
        elinewidth=0.7,
        capsize=2.0,
        capthick=0.7,
        zorder=3,
    )
    ax.scatter(
        y_true,
        y_pred,
        s=56,
        marker="o",
        facecolors="none",
        edgecolors=COLORS["mt"],
        linewidths=0.9,
        zorder=4,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Reference $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.set_ylabel(r"Predicted $V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    stats_text = (
        f"MAD = {mean_mae:.0f} m$\\cdot$s$^{{-1}}$\n"
        f"MAE = {metrics['mae']:.0f} m$\\cdot$s$^{{-1}}$\n"
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"MAPE = {metrics['mape']:.1f}%"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_ANNOT,
        color=COLORS["charcoal"],
    )
    style_axes(ax, grid=False)


def plot_panel_e(
    ax: plt.Axes,
    exp7a_rows: dict[str, dict[str, float | int]],
    meta: dict[str, dict[str, object]],
    mae: float,
) -> None:
    mats = sorted(exp7a_rows, key=lambda m: float(exp7a_rows[m]["gt"]))
    errs = np.array([float(exp7a_rows[m]["signed_error"]) for m in mats], dtype=float)
    colors = [X_COLORS[str(meta[m]["X_site"])] for m in mats]
    ypos = np.arange(len(mats), dtype=float)
    ax.axvspan(-mae, mae, facecolor=COLORS["light_gray"], alpha=1.0, linewidth=0, edgecolor="none", zorder=1)
    ax.axvline(0.0, color="black", lw=0.9, zorder=2)
    ax.hlines(ypos, 0.0, errs, colors=colors, linewidth=1.1, zorder=3)
    ax.scatter(errs, ypos, s=34, facecolors="none", edgecolors=colors, linewidths=0.9, zorder=4)
    ax.set_yticks(ypos, mats)
    ax.invert_yaxis()
    ax.set_xlabel("Signed error (m$\\cdot$s$^{-1}$)")
    ax.set_title("Signed error by material", pad=4)
    xmax = float(max(np.max(np.abs(errs)), mae) * 1.20)
    ax.set_xlim(-xmax, xmax)
    ax.text(0.02, 0.95, r"$\pm$MAE band", transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT - 0.3)
    ax.tick_params(axis="y", labelsize=FS_TICK - 1.3)
    style_axes(ax, grid=True)


def plot_panel_f(
    ax: plt.Axes,
    model_stats: dict[str, dict[str, float]],
) -> None:
    order = ["ClO4-", "NO3-", "IO4-", "All"]
    x = np.arange(len(order), dtype=float)
    width = 0.55
    model_mae = [model_stats[g]["mae"] for g in order]
    model_std = [model_stats[g]["fold_std"] for g in order]
    ns = [int(model_stats[g]["n"]) for g in order]
    all_color = COLORS["charcoal"]
    bar_colors = [X_COLORS.get(g, all_color) for g in order]
    bar_tops = [model_mae[i] + model_std[i] for i in range(len(order))]
    upper = max(bar_tops) * 1.18
    ax.bar(
        x,
        model_mae,
        width,
        yerr=model_std,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=2.6,
        error_kw={"elinewidth": 0.5, "capthick": 0.5},
        zorder=3,
    )
    for idx, n in enumerate(ns):
        ax.text(idx, bar_tops[idx] + upper * 0.018, rf"$n$={n}", ha="center", va="bottom", fontsize=FS_ANNOT - 0.6, color=COLORS["charcoal"])
    ax.set_xticks(x, [X_LABELS[g] for g in order])
    ax.set_ylabel("MAE (m$\\cdot$s$^{-1}$)")
    ax.set_title("MT-FT MAE by X-site", pad=4, fontsize=FS_LABEL)
    ax.set_ylim(0.0, upper)
    style_axes(ax, grid=True)


def plot_panel_g(ax: plt.Axes, sensitivity: dict[str, dict[str, dict[str, float]]]) -> None:
    jit = {"rotation": -5.5, "translation": 0.0, "dap4_template": 5.5}
    all_x: list[float] = []
    all_y: list[float] = []
    for family in MODEL_ORDER:
        fam_color = MODEL_COLORS[family]
        for perturb in PERT_ORDER:
            rec = sensitivity[family][perturb]
            x = rec["baseline_mae"] + jit[perturb]
            y = rec["delta_mae"]
            all_x.append(x)
            all_y.append(y)
            ax.scatter(x, y, s=62, color=fam_color, edgecolors="white", linewidths=0.6, zorder=3)
            ax.text(x, y + 4.5, PERT_SHORT[perturb], ha="center", va="bottom", fontsize=FS_ANNOT - 0.7, color=COLORS["charcoal"])
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Clean MAE (m$\\cdot$s$^{-1}$)")
    ax.set_ylabel(r"$\Delta$MAE (m$\cdot$s$^{-1}$)")
    ax.set_title("Sensitivity", pad=4)
    ax.set_xlim(min(all_x) - 20.0, max(all_x) + 20.0)
    ymin = min(all_y) - 18.0
    ymax = max(all_y) + 18.0
    ax.set_ylim(ymin, ymax)
    ax.text(0.04, 0.96, "R = rotation\nT = translation\nS = template", transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT - 0.8)
    ax.legend(
        handles=[
            mlines.Line2D([], [], color=MODEL_COLORS["exp7a"], marker="o", ls="", markersize=6, label="MT-FT"),
            mlines.Line2D([], [], color=MODEL_COLORS["exp7c"], marker="o", ls="", markersize=6, label="ST-FT"),
            mlines.Line2D([], [], color=MODEL_COLORS["exp7d"], marker="o", ls="", markersize=6, label="ST-TFS"),
        ],
        loc="lower left",
        frameon=False,
        fontsize=FS_LEGEND - 0.7,
        ncol=1,
        handletextpad=0.3,
    )
    style_axes(ax, grid=True)


def plot_panel_h(ax: plt.Axes, sensitivity: dict[str, dict[str, dict[str, float]]]) -> None:
    x = np.arange(len(PERT_ORDER), dtype=float)
    width = 0.24
    offsets = {"exp7a": -width, "exp7c": 0.0, "exp7d": width}
    labels = {"exp7a": "MT-FT", "exp7c": "ST-FT", "exp7d": "ST-TFS"}
    for family in MODEL_ORDER:
        vals = [sensitivity[family][pert]["delta_mae"] for pert in PERT_ORDER]
        ax.bar(
            x + offsets[family],
            vals,
            width,
            color=MODEL_COLORS[family],
            edgecolor="white",
            linewidth=0.6,
            label=labels[family],
            zorder=3,
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x, [PERT_LABELS[p] for p in PERT_ORDER])
    ax.set_ylabel(r"$\Delta$MAE (m$\cdot$s$^{-1}$)")
    ax.set_title("By perturbation", pad=4)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=FS_LEGEND - 0.5,
        handletextpad=0.35,
        ncol=3,
        columnspacing=0.8,
    )
    style_axes(ax, grid=True)


def _draw_split_inset(ax: plt.Axes) -> None:
    """2x2 coverage grid showing which training stage saw which material
    family.

    Top row = backbone DFT pre-training (deepEMS-LAM): covers the DAP
    family but not the non-DAP A-sites (pz/hpz/mepz/odabco/iodabco).
    Bottom row = property head fine-tuning: trains on the 17 non-DAP
    materials and is held out on the 8 DAP-core materials. The two
    stages cover complementary subsets of the panel materials, so the
    held-out cell (bottom-right) is the OOD test for the property head
    yet sits inside the DFT pre-trained domain. This asymmetric coverage
    is what the bars below probe across the MT / ST / Scratch variants.
    """
    x_lab_left = 0.00
    x_lab_right = 0.30
    x_c1 = 0.32
    x_c2 = 0.64
    cell_w = 0.30
    cell_h = 0.12
    y_row0 = 0.72
    y_row1 = 0.58
    y_hdr = y_row0 + cell_h + 0.012

    for cx, name, n in [(x_c1, "non-DAP", "17"), (x_c2, "DAP-core", "8")]:
        ax.text(
            cx + cell_w / 2,
            y_hdr,
            f"{name}\n$n$={n}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FS_ANNOT - 1.0,
            color=COLORS["charcoal"],
            linespacing=1.2,
        )

    x_lab_center = (x_lab_left + x_lab_right) / 2
    for ry, line1, line2 in [
        (y_row0, "DFT", "pre-train"),
        (y_row1, "Property", "head"),
    ]:
        ax.text(
            x_lab_center,
            ry + cell_h / 2,
            f"{line1}\n{line2}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_ANNOT - 1.0,
            color=COLORS["charcoal"],
            linespacing=1.2,
        )

    cells = [
        (x_c1, y_row0, "not seen", False),
        (x_c2, y_row0, "trained", True),
        (x_c1, y_row1, "trained", True),
        (x_c2, y_row1, "held-out", False),
    ]
    fill = mcolors.to_rgba(COLORS["mt"], alpha=0.20)
    for cx, cy, label, is_seen in cells:
        if is_seen:
            ax.add_patch(plt.Rectangle(
                (cx, cy),
                cell_w,
                cell_h,
                transform=ax.transAxes,
                facecolor=fill,
                edgecolor=COLORS["mt"],
                linewidth=0.7,
                zorder=4,
                clip_on=False,
            ))
        else:
            ax.add_patch(plt.Rectangle(
                (cx, cy),
                cell_w,
                cell_h,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=COLORS["charcoal"],
                linewidth=0.6,
                linestyle=(0, (3, 2)),
                zorder=4,
                clip_on=False,
            ))
        ax.text(
            cx + cell_w / 2,
            cy + cell_h / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FS_ANNOT - 1.0,
            color=COLORS["charcoal"],
            zorder=6,
        )


def plot_panel_g_ood(
    ax: plt.Axes,
    ood: dict[str, dict[str, float | int]],
) -> None:
    """Paired bar chart for the pretrained-domain DAP OOD split.

    Two splits ("Train, non-DAP" and "Held-out, DAP-core") are placed on
    the x-axis, with three bars per split for MT-FT,
    ST-FT, and ST-TFS variants. Bar colors mirror
    Panel b's MAE ladder so readers can carry the same legend across
    panels without a new key. A set-diagram inset above the bars shows
    that both groups sit inside the DFT pre-trained domain.
    """
    splits = ["ind_mae", "ood_mae"]
    split_labels = [
        f"Train (non-DAP)\n$n$={ood['mt']['n_ind']}",
        f"Held-out (DAP-core)\n$n$={ood['mt']['n_ood']}",
    ]
    series = [
        ("mt", "MT-FT", COLORS["mt"]),
        ("st", "ST-FT", COLORS["st"]),
        ("sd", "ST-TFS", COLORS["scratch"]),
    ]
    width = 0.20
    x = np.arange(len(splits), dtype=float)
    offsets = {"mt": -width, "st": 0.0, "sd": width}
    upper = 0.0
    for key, label, color in series:
        vals = [float(ood[key][s]) for s in splits]
        ax.bar(
            x + offsets[key],
            vals,
            width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
        upper = max(upper, max(vals))
    # Lift ymax so the coverage-grid inset sits above the tallest bar
    # and its value label without crowding.
    upper *= 2.10
    for key, _label, color in series:
        for split_idx, split_key in enumerate(splits):
            value = float(ood[key][split_key])
            ax.text(
                split_idx + offsets[key],
                value + upper * 0.012,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=FS_ANNOT - 1.0,
                color="black",
            )
    ax.set_xticks(x, split_labels)
    ax.set_ylim(0.0, upper)
    ax.set_ylabel("MAE (m$\\cdot$s$^{-1}$)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=False,
        fontsize=FS_LEGEND - 1.0,
        ncol=3,
        handletextpad=0.3,
        columnspacing=0.65,
    )
    _draw_split_inset(ax)
    style_axes(ax, grid=True)


def plot_panel_ood_compare(
    ax: plt.Axes,
    ood: dict[str, dict[str, object]],
    materials: tuple[str, ...] = ("DAC-4", "TAP-2", "EAP-4", "SY"),
) -> None:
    series = [
        ("exp7a", "MT-FT", COLORS["mt"], "o", -0.16),
        ("exp7c", "ST-FT", COLORS["st"], "^", 0.00),
        ("exp7d", "ST-TFS", COLORS["scratch"], "s", 0.16),
    ]
    x = np.arange(len(materials), dtype=float)
    ref_vals = np.array(
        [float(ood["exp7a"]["materials"][mat]["exp_m_s"]) for mat in materials],
        dtype=float,
    )
    all_vals = [*ref_vals.tolist()]
    for key, _label, _color, _marker, _offset in series:
        all_vals.extend(float(ood[key]["materials"][mat]["pred_m_s"]) for mat in materials)

    for xi, ref in zip(x, ref_vals):
        ax.hlines(
            ref,
            xi - 0.34,
            xi + 0.34,
            color=COLORS["charcoal"],
            linewidth=1.0,
            zorder=2,
        )
    for key, label, color, marker, offset in series:
        vals = np.array([float(ood[key]["materials"][mat]["pred_m_s"]) for mat in materials], dtype=float)
        ax.scatter(
            x + offset,
            vals,
            s=48,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidths=0.6,
            label=label,
            zorder=4,
        )
    ax.set_xticks(x, [display_material(m) for m in materials])
    ax.set_ylabel(r"$V_{\mathrm{det}}$ (m$\cdot$s$^{-1}$)")
    ax.set_ylim(6500.0, 10000.0)
    ax.set_xlim(-0.55, len(materials) - 0.45)
    ax.set_xlabel("OOD-holdout materials")
    handles = [
        mlines.Line2D([], [], color=COLORS["charcoal"], lw=1.0, label="Reference"),
        *[
            mlines.Line2D(
                [],
                [],
                color=color,
                marker=marker,
                ls="",
                markersize=5.8,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=label,
            )
            for _key, label, color, marker, _offset in series
        ],
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
        fontsize=FS_LEGEND - 1.2,
        ncol=4,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.55,
    )
    style_axes(ax, grid=True)


def plot_panel_i(
    ax: plt.Axes,
    exp7a_rows: dict[str, dict[str, float | int]],
    meta: dict[str, dict[str, object]],
    descriptor_distances: dict[str, dict[str, float | bool]],
    coverage_boundary: float,
) -> None:
    mats = [m for m in sorted(exp7a_rows) if m in descriptor_distances]
    xs = np.array([float(descriptor_distances[m]["knn_mean_l2"]) for m in mats], dtype=float)
    ys = np.array([float(exp7a_rows[m]["abs_error"]) for m in mats], dtype=float)
    groups = [str(meta[m]["X_site"]) for m in mats]
    rho_all, _p_all = spearmanr(xs, ys)
    non_dai_mask = np.array([not m.startswith("DAI") for m in mats], dtype=bool)
    if non_dai_mask.sum() >= 2:
        rho_no_dai, _p_no_dai = spearmanr(xs[non_dai_mask], ys[non_dai_mask])
    else:
        rho_no_dai = float("nan")
    colors = [X_COLORS[g] for g in groups]
    ax.scatter(xs, ys, s=44, c=colors, edgecolors="white", linewidths=0.6, zorder=3)
    ax.axvline(coverage_boundary, color=COLORS["composition"], ls=":", lw=1.0, zorder=2)
    dai_points = [(m, float(descriptor_distances[m]["knn_mean_l2"]), float(exp7a_rows[m]["abs_error"])) for m in mats if m.startswith("DAI")]
    if dai_points:
        dai_x = float(np.mean([p[1] for p in dai_points]))
        dai_y = float(np.mean([p[2] for p in dai_points]))
        ax.annotate(
            "DAI series",
            xy=(dai_x, dai_y),
            xytext=(0.73, 0.84),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": COLORS["charcoal"]},
            fontsize=FS_ANNOT - 0.4,
            color=COLORS["charcoal"],
        )
    ax.set_xlabel("kNN descriptor distance")
    ax.set_ylabel(r"|CV error| (m$\cdot$s$^{-1}$)")
    ax.text(
        0.04,
        0.96,
        f"$\\rho_{{\\mathrm{{all}}}}$ = {rho_all:.2f}\n"
        f"$\\rho_{{\\mathrm{{no\\,DAI}}}}$ = {rho_no_dai:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_ANNOT,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.94},
    )
    ax.text(
        coverage_boundary + 0.004,
        0.92,
        "DAP boundary",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        fontsize=FS_ANNOT - 1.0,
        color=COLORS["composition"],
        va="top",
    )
    legend_handles = [
        mlines.Line2D([], [], color=X_COLORS[grp], marker="o", ls="", markersize=6.0, label=X_LABELS[grp])
        for grp in X_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        frameon=False,
        fontsize=FS_LEGEND - 0.4,
        handletextpad=0.35,
        ncol=3,
        columnspacing=0.9,
    )
    style_axes(ax, grid=True)


def plot_panel_j(
    ax: plt.Axes,
    exp7a_rows: dict[str, dict[str, float | int]],
    uq_summary: dict[str, float],
) -> None:
    stds = np.array([float(rec["model_std"]) for rec in exp7a_rows.values()], dtype=float)
    errs = np.array([float(rec["abs_error"]) for rec in exp7a_rows.values()], dtype=float)
    max_val = float(max(np.max(stds), np.max(errs)) * 1.07)
    ax.scatter(stds, errs, s=42, color=COLORS["mt"], edgecolors="white", linewidths=0.6, alpha=0.88, zorder=3)
    ax.plot([0.0, max_val], [0.0, max_val], color="black", ls="--", lw=0.8, zorder=2)
    ax.set_xlim(0.0, max_val)
    ax.set_ylim(0.0, max_val)
    ax.set_xlabel(r"Model $\sigma$ across folds (m$\cdot$s$^{-1}$)")
    ax.set_ylabel(r"|CV error| (m$\cdot$s$^{-1}$)")
    ax.text(
        0.96,
        0.04,
        f"$\\rho$ = {float(uq_summary['spearman_rho']):.2f}\n"
        f"$p$ = {float(uq_summary['spearman_p']):.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_ANNOT,
        color=COLORS["charcoal"],
    )
    style_axes(ax, grid=True)


def main() -> None:
    setup_style()
    meta = load_pems_metadata()
    _mean_vdet, mean_mae = compute_mean_predictor(meta)

    exp7a_rows = load_family_predictions("exp7a")
    exp7a_metrics = compute_family_metrics(exp7a_rows)

    ood_summary = load_pretrained_domain_ood()
    ood_heldout = load_ood_heldout_comparison()
    uq_summary = _load_json(EXP_DIR / "pems_uq_calibration.json")["exp7a"]

    # Five-panel composite layout: signed-error panel at left, with parity,
    # OOD-holdout, pretrained-domain, and calibration panels in a right 2x2.
    fig = plt.figure(figsize=(8.27, 6.35))

    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.065, right=0.985, top=0.955, bottom=0.075,
        wspace=0.14,
        width_ratios=[0.82, 2.08],
    )

    ax_a = fig.add_subplot(outer[0, 0])

    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0, 1],
        hspace=0.40,
        wspace=0.34,
        height_ratios=[1.0, 1.0],
    )
    ax_b = fig.add_subplot(gs_right[0, 0])
    ax_c = fig.add_subplot(gs_right[0, 1])
    ax_d = fig.add_subplot(gs_right[1, 0])
    ax_e = fig.add_subplot(gs_right[1, 1])

    plot_panel_e(ax_a, exp7a_rows, meta, exp7a_metrics["mae"])
    plot_panel_d(ax_b, exp7a_rows, exp7a_metrics, mean_mae)
    plot_panel_ood_compare(ax_c, ood_heldout)
    plot_panel_g_ood(ax_d, ood_summary)
    plot_panel_j(ax_e, exp7a_rows, uq_summary)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], list("abcde")):
        add_panel_label(ax, label)

    save_png_pdf(fig, OUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")
    print(f"Saved {OUT_PATH.with_suffix('.pdf')}")

    # The stand-alone descriptor-distance SI export overlaps with main Fig. 3
    # and is not included in the current SI.


def render_si_descriptor_distance(
    exp7a_rows: dict[str, dict[str, float | int]],
    meta: dict[str, dict[str, object]],
    descriptor_distances: dict[str, dict[str, float | bool]],
    coverage_boundary: float,
) -> None:
    """Render the descriptor-distance versus |CV error| panel as a stand-alone
    SI figure, replacing what used to be Panel g of Figure 3."""
    fig = plt.figure(figsize=(4.4, 3.2))
    ax = fig.add_axes([0.16, 0.18, 0.78, 0.74])
    plot_panel_i(ax, exp7a_rows, meta, descriptor_distances, coverage_boundary)
    save_png_pdf(fig, SI_DESC_DIST_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved {SI_DESC_DIST_PATH}")
    print(f"Saved {SI_DESC_DIST_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
