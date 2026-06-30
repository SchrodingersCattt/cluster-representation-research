#!/usr/bin/env python3
"""Extended Data LOSO hierarchy figure.

Outputs:
    manuscript/figures/_ed_loso_hierarchy.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
EXP = ROOT / "experiments_davis2024"
DATA_CSV = ROOT / "data" / "pems" / "pems.csv"
LOSO_JSON = EXP / "exp_ood_loso" / "loso_results_summary.json"
ANOVA_JSON = EXP / "mechanism_results" / "mechanism_m4b_results.json"
OUT = THIS / "_ed_loso_hierarchy"

from figure_style import (
    CHARCOAL,
    MID_GRAY,
    MODEL_COLORS,
    SITE_COLORS,
    heatmap_text_color,
    save_figure,
    setup_style,
    style_axes,
)
from _qa_check import bbox_center, bbox_edge_distance, bboxes_overlap, segments_intersect

LEVEL_COLORS = {"X": SITE_COLORS["X"], "A": SITE_COLORS["A"], "B": SITE_COLORS["B"]}

LEVEL_ORDER = ("X", "A", "B")
X_FAMILY_ORDER = ("ClO4", "NO3", "IO4-family")


def _x_family(raw: str) -> str:
    raw = str(raw).strip()
    if raw == "ClO4-":
        return "ClO4"
    if raw == "NO3-":
        return "NO3"
    if raw in {"IO4-", "H4IO6-"}:
        return "IO4-family"
    return raw


def _withheld_label(split_id: str) -> str:
    labels = {
        "loso_x_clo4": r"$\mathrm{ClO_4^-}$",
        "loso_x_no3": r"$\mathrm{NO_3^-}$",
        "loso_x_io4": r"$\mathrm{IO_4^-}$",
        "loso_a_dabco": r"$\mathrm{H_2dabco^{2+}}$",
        "loso_a_pz": r"$\mathrm{H_2pz^{2+}}$",
        "loso_a_hpz": r"$\mathrm{H_2hpz^{2+}}$",
        "loso_a_mepz": r"$\mathrm{MeHpz^{2+}}$",
        "loso_b_k": r"$\mathrm{K^+}$",
        "loso_b_na": r"$\mathrm{Na^+}$",
        "loso_b_nh4": r"$\mathrm{NH_4^+}$",
        "loso_b_ag": r"$\mathrm{Ag^+}$",
    }
    return labels.get(split_id, split_id.replace("loso_", "").replace("_", " "))


def _load_loso() -> list[dict[str, object]]:
    data = json.loads(LOSO_JSON.read_text(encoding="utf-8"))
    rows = []
    for rec in data.values():
        if rec.get("config_type") != "mt":
            continue
        rows.append(
            {
                "split_id": str(rec["split_id"]),
                "level": str(rec["level"]),
                "mae": float(rec["mae_m_s"]),
                "n_train": int(rec["n_train"]),
                "n_val": int(rec["n_val"]),
                "train_materials": list(rec["train_materials"]),
            }
        )
    rows.sort(key=lambda r: (LEVEL_ORDER.index(str(r["level"])), str(r["split_id"])))
    return rows


def _load_x_assignments() -> dict[str, str]:
    df = pd.read_csv(DATA_CSV, index_col=False)
    return {
        str(row["material"]): _x_family(str(row["X_site"]))
        for _, row in df.iterrows()
        if pd.notna(row.get("material")) and pd.notna(row.get("X_site"))
    }


def _panel_anova(ax: plt.Axes) -> None:
    anova = json.loads(ANOVA_JSON.read_text(encoding="utf-8"))["anova"]
    factors = ["X_type", "B_type", "A_type"]
    labels = ["X site", "B site", "A site"]
    one_way = [anova[f]["eta2"] for f in factors]
    partial = [anova[f]["partial_eta2"] for f in factors]
    x = np.arange(len(factors))
    width = 0.32
    ax.bar(x - width / 2, one_way, width, color=CHARCOAL, label=r"One-way $\eta^2$")
    ax.bar(x + width / 2, partial, width, color=MODEL_COLORS["zero_shot"], label=r"Type II partial $\eta^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Effect size")
    ax.set_title("a  Site-level variance decomposition", loc="left", fontweight="bold", pad=5)
    ax.legend(frameon=False, loc="upper right")
    style_axes(ax, grid=True)


def _panel_loso(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    pos = {level: i for i, level in enumerate(LEVEL_ORDER)}
    level_values: dict[str, list[float]] = {level: [] for level in LEVEL_ORDER}
    jitter_offsets = {"X": [-0.20, 0.0, 0.20], "A": [-0.27, -0.09, 0.09, 0.27], "B": [-0.27, -0.09, 0.09, 0.27]}
    label_offsets = {
        "loso_x_clo4": (0, 14),
        "loso_x_no3": (8, 14),
        "loso_x_io4": (-2, 14),
        "loso_a_dabco": (0, 14),
        "loso_a_pz": (14, 6),
        "loso_a_hpz": (-8, -8),
        "loso_a_mepz": (14, 16),
        "loso_b_k": (14, 8),
        "loso_b_na": (14, 8),
        "loso_b_nh4": (14, -14),
        "loso_b_ag": (-14, -6),
    }
    counters = {level: 0 for level in LEVEL_ORDER}
    for row in rows:
        level = str(row["level"])
        mae = float(row["mae"])
        idx = counters[level]
        counters[level] += 1
        xpos = pos[level] + jitter_offsets[level][idx]
        level_values[level].append(mae)
        ax.scatter(xpos, mae, s=38, color=LEVEL_COLORS[level], edgecolor="white", linewidth=0.5, zorder=3)
        split_id = str(row["split_id"])
        dx, dy = label_offsets[split_id]
        ax.annotate(
            _withheld_label(str(row["split_id"])),
            xy=(xpos, mae),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.0,
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5, shrinkA=1, shrinkB=1),
            zorder=4,
        )
    for level, vals in level_values.items():
        mean = float(np.mean(vals))
        x0 = pos[level]
        ax.hlines(mean, x0 - 0.30, x0 + 0.30, color=LEVEL_COLORS[level], linewidth=1.2, zorder=2)
    mean_text = "\n".join(f"{level} mean {np.mean(vals):.0f}" for level, vals in level_values.items())
    ax.text(
        0.98,
        0.95,
        mean_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=CHARCOAL,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=MID_GRAY, alpha=0.92),
        zorder=5,
    )
    ax.set_xticks([pos[level] for level in LEVEL_ORDER])
    ax.set_xticklabels(["X-site holdout", "A-site holdout", "B-site holdout"])
    ax.set_ylabel(r"LOSO MAE (m$\cdot$s$^{-1}$)")
    ax.set_xlim(-0.55, 2.62)
    ax.set_ylim(0, 2050)
    ax.set_title("b  Leave-one-site-out generalization", loc="left", fontweight="bold", pad=5)
    style_axes(ax, grid=True)


def _panel_confound(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    x_assign = _load_x_assignments()
    counts = []
    row_labels = []
    for row in rows:
        train = row["train_materials"]
        fam_counts = {fam: 0 for fam in X_FAMILY_ORDER}
        for material in train:
            fam = x_assign.get(str(material))
            if fam in fam_counts:
                fam_counts[fam] += 1
        counts.append([fam_counts[fam] for fam in X_FAMILY_ORDER])
        row_labels.append(f"{row['level']}: {_withheld_label(str(row['split_id']))}")
    arr = np.asarray(counts, dtype=float)
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=17, aspect="auto")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.0)
    ax.set_xticks(np.arange(len(X_FAMILY_ORDER)))
    ax.set_xticklabels([r"$\mathrm{ClO_4^-}$", r"$\mathrm{NO_3^-}$", r"$\mathrm{IO_4^-}$ family"], fontsize=8.0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = int(arr[i, j])
            color = heatmap_text_color(im.cmap, im.norm, val)
            weight = "bold" if val == 0 else "normal"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8.0, color=color, fontweight=weight)
    for i in range(1, len(rows)):
        if rows[i]["level"] != rows[i - 1]["level"]:
            ax.axhline(i - 0.5, color="white", linewidth=1.2)
    ax.set_title("c  Coupled X-site coverage after holdout", loc="left", fontweight="bold", pad=5)
    ax.tick_params(length=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Training count", fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def _validate_loso_annotations(fig: plt.Figure, texts: list[plt.Text], anchors_data: list[tuple[float, float]]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [txt.get_window_extent(renderer).expanded(1.04, 1.12) for txt in texts]
    bbox_tuples = [(bb.x0, bb.y0, bb.x1, bb.y1) for bb in bboxes]
    centers = [bbox_center(bb) for bb in bbox_tuples]
    anchors = [tuple(texts[0].axes.transData.transform(anchor)) for anchor in anchors_data]

    errors: list[str] = []
    for i in range(len(texts)):
        arrow_len = np.hypot(centers[i][0] - anchors[i][0], centers[i][1] - anchors[i][1])
        if arrow_len > 4.0 * bboxes[i].height:
            errors.append(f"{texts[i].get_text()} leader exceeds 4x text height")
        for j in range(i + 1, len(texts)):
            label_i = texts[i].get_text()
            label_j = texts[j].get_text()
            if bboxes_overlap(bbox_tuples[i], bbox_tuples[j]):
                errors.append(f"{label_i} overlaps {label_j}")
                continue
            min_edge = 0.4 * min(bboxes[i].height, bboxes[j].height)
            edge = bbox_edge_distance(bbox_tuples[i], bbox_tuples[j])
            if edge < min_edge:
                errors.append(f"{label_i} too close to {label_j}: {edge:.1f}px < {min_edge:.1f}px")
            if segments_intersect(anchors[i], centers[i], anchors[j], centers[j]):
                errors.append(f"{label_i} leader crosses {label_j} leader")

    if errors:
        raise RuntimeError("LOSO annotation QA failed: " + "; ".join(errors))


def main() -> None:
    setup_style()
    rows = _load_loso()
    fig = plt.figure(figsize=(8.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], width_ratios=[0.85, 1.25])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    _panel_anova(ax_a)
    _panel_loso(ax_b, rows)
    _panel_confound(ax_c, rows)
    fig.tight_layout()
    save_figure(fig, OUT)
    print(f"Saved {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
