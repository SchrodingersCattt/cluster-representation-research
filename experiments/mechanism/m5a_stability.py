"""M5a -- Cross-fold descriptor stability (mean pairwise cosine across CV folds)"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from . import paths, runtime
from .constants import (
    COLORS, MODEL_DISPLAY_NAMES, PERTURBATION_STYLE, PERTURBATION_TYPES_M1,
    SCALE_FACTORS, N_SEEDS, RIDGE_ALPHAS, KJ_RHO_COEF,
    METAL_ELEMENTS, PEM_BOND_THRESHOLDS,
)
from .io_data import (
    read_cluster_system, get_materials, get_m1_heldout_mats,
    load_gt_vdet, get_family, load_densities,
    compute_composition_and_ob, build_probe_targets,
)
from .io_models import load_property_model, load_descriptor_model
from .inference import predict_single, extract_descriptor, extract_descriptor_per_atom
from .stats import bootstrap_r2_ci
from .plot_helpers import (
    setup_nature_style, style_axes, add_panel_label, save_figure,
    rounded_limits, plot_mean_with_individuals, disp,
)


def run_m5a(output_dir: Path) -> None:
    """Compute cross-fold descriptor stability for MT (exp7a), ST-pretrained (exp7c),
    and ST-scratch (exp7d).

    For each model and each IND training material, extract the material-level
    descriptor (mean over all atoms) from each of the 5 fold checkpoints, then
    compute the mean pairwise cosine similarity over the C(5,2)=10 pairs.

    Output: mechanism_m5a_results.json
    """
    print("\n" + "=" * 60 + "\nM5a: Cross-fold descriptor stability\n" + "=" * 60)

    materials = get_materials()
    models_to_run = ["exp7a", "exp7c", "exp7d"]

    # per_material[model_name][material] -> {cos_mean, cos_std, pairs}
    per_material: dict[str, dict[str, dict]] = {mn: {} for mn in models_to_run}

    for mn in models_to_run:
        print(f"\n  Model: {mn}")
        # Collect fold descriptors: fold_descs[fi] = {mat: np.ndarray(256,)}
        fold_descs: dict[int, dict[str, np.ndarray]] = {}
        for fi in runtime.ACTIVE_FOLD_IDS:
            print(f"    fold {fi}")
            dp = load_descriptor_model(mn, fi)
            fd: dict[str, np.ndarray] = {}
            for mat in materials:
                sys_dir = paths.CLUSTER_N1_DIR / mat
                if not sys_dir.exists():
                    continue
                coord, syms, _ = read_cluster_system(sys_dir)
                fd[mat] = extract_descriptor(dp, coord, syms)
            fold_descs[fi] = fd

        # Compute pairwise cosine per material
        fold_ids_list = list(runtime.ACTIVE_FOLD_IDS)
        for mat in materials:
            vecs = []
            for fi in fold_ids_list:
                v = fold_descs.get(fi, {}).get(mat)
                if v is not None:
                    vecs.append((fi, v))
            if len(vecs) < 2:
                continue
            pairs_out = []
            cos_vals = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    fi, vi = vecs[i]
                    fj, vj = vecs[j]
                    n_i = np.linalg.norm(vi)
                    n_j = np.linalg.norm(vj)
                    if n_i < 1e-12 or n_j < 1e-12:
                        cos = float("nan")
                    else:
                        cos = float(np.dot(vi, vj) / (n_i * n_j))
                    pairs_out.append([fi, fj, cos])
                    if not np.isnan(cos):
                        cos_vals.append(cos)
            cos_mean = float(np.mean(cos_vals)) if cos_vals else float("nan")
            cos_std = float(np.std(cos_vals)) if len(cos_vals) > 1 else 0.0
            per_material[mn][mat] = {
                "cos_mean": cos_mean,
                "cos_std": cos_std,
                "pairs": pairs_out,
                "n_folds": len(vecs),
            }

    # Aggregate per model
    aggregated: dict[str, dict] = {}
    for mn in models_to_run:
        vals = [v["cos_mean"] for v in per_material[mn].values() if not np.isnan(v["cos_mean"])]
        aggregated[mn] = {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            "median": float(np.median(vals)) if vals else float("nan"),
            "n_materials": len(vals),
        }
        print(f"  {mn}: mean_cos={aggregated[mn]['mean']:.4f}, median={aggregated[mn]['median']:.4f}, n={aggregated[mn]['n_materials']}")

    results = {
        "fold_ids": list(runtime.ACTIVE_FOLD_IDS),
        "models": models_to_run,
        "per_material": per_material,
        "aggregated": aggregated,
    }
    out_path = output_dir / "mechanism_m5a_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    plot_m5a(results, output_dir)


# ===========================================================================
# Plot (added 2026-04-19)
# ===========================================================================

def plot_m5a(results: dict, output_dir) -> None:
    """Two-panel summary of M5a cross-fold descriptor stability.

    Panel A (left)  -- per-model strip plot: each dot is one material's
                       mean pairwise cosine across the C(5,2)=10 fold pairs.
                       IQR box and median bar overlay; per-model summary
                       (median, n) annotated on top.
    Panel B (right) -- 5x5 fold-fold cosine similarity heatmap, averaged
                       over materials, for the multi-task model (exp7a).
                       Identity diagonal omitted for clarity.
    """
    model_keys = ["exp7a", "exp7c", "exp7d"]
    pm = results.get("per_material", {})
    aggr = results.get("aggregated", {})
    fold_ids = results.get("fold_ids", [0, 1, 2, 3, 4])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2),
                              gridspec_kw={"width_ratios": [1.5, 1.0]})

    # --- Panel A: strip plot -------------------------------------------------
    ax = axes[0]
    rng = np.random.default_rng(42)
    for i, mn in enumerate(model_keys):
        per_mat = pm.get(mn, {})
        vals = np.array(
            [v["cos_mean"] for v in per_mat.values() if not np.isnan(v["cos_mean"])]
        )
        if vals.size == 0:
            continue
        xj = i + rng.uniform(-0.18, 0.18, vals.size)
        ax.scatter(xj, vals, s=14, color=COLORS[mn], alpha=0.65,
                   edgecolors="white", linewidths=0.3, zorder=4)
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        ax.fill_betweenx([q25, q75], i - 0.16, i + 0.16,
                         color=COLORS[mn], alpha=0.18, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [q50, q50], color=COLORS[mn], lw=1.4, zorder=5)
        ax.text(i, q50 + 0.0015, f"{q50:.3f}", ha="center", va="bottom",
                fontsize=6, color=COLORS[mn])
        ag = aggr.get(mn, {})
        if ag:
            ax.text(i, 1.005, f"n={ag.get('n_materials', vals.size)}",
                    ha="center", va="bottom", fontsize=6, color="#444444")

    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels([disp(m) for m in model_keys], fontsize=7)
    ax.set_ylabel("Mean pairwise cosine across 5 folds")
    ax.set_title("Per-material descriptor stability", loc="left", pad=4)
    lo_min = 1.0
    for mn in model_keys:
        for v in pm.get(mn, {}).values():
            if not np.isnan(v["cos_mean"]):
                lo_min = min(lo_min, v["cos_mean"])
    ax.set_ylim(max(0.0, lo_min - 0.02), 1.02)
    style_axes(ax, grid=True)
    ax.grid(axis="x", visible=False)
    add_panel_label(ax, "A")

    # --- Panel B: fold-fold heatmap (exp7a) ---------------------------------
    ax2 = axes[1]
    nf = len(fold_ids)
    mat_avg = np.full((nf, nf), np.nan)
    pm_a = pm.get("exp7a", {})
    pair_acc: dict[tuple[int, int], list[float]] = {}
    for mat, info in pm_a.items():
        for pair in info.get("pairs", []):
            i, j, c = int(pair[0]), int(pair[1]), float(pair[2])
            if np.isnan(c):
                continue
            key = (min(i, j), max(i, j))
            pair_acc.setdefault(key, []).append(c)
    for (i, j), vals in pair_acc.items():
        m = float(np.mean(vals))
        mat_avg[i, j] = m
        mat_avg[j, i] = m
    np.fill_diagonal(mat_avg, np.nan)

    im = ax2.imshow(mat_avg, vmin=np.nanmin(mat_avg), vmax=1.0,
                    cmap="viridis", origin="lower")
    for i in range(nf):
        for j in range(nf):
            v = mat_avg[i, j]
            if np.isnan(v):
                continue
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if v < 0.96 else "#222222")
    ax2.set_xticks(range(nf))
    ax2.set_yticks(range(nf))
    ax2.set_xticklabels([str(f) for f in fold_ids], fontsize=7)
    ax2.set_yticklabels([str(f) for f in fold_ids], fontsize=7)
    ax2.set_xlabel("Fold j")
    ax2.set_ylabel("Fold i")
    ax2.set_title("Fold-fold cosine (exp7a, mat-avg)", loc="left", pad=4)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Cosine", fontsize=7)
    style_axes(ax2)
    add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M5a_stability_supp",
        supplementary=True,
        legacy_png_name="figure_m5a_stability.png",
    )
    plt.close(fig)
    print("Saved Figure M5a")
