"""M4a -- Embedding compactness, silhouette, LOO R^2 probes"""
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


def run_m4a(output_dir: Path) -> None:
    print("\n" + "=" * 60 + "\nM4a: Embedding compactness\n" + "=" * 60)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score, silhouette_score

    materials = get_materials()
    gt = load_gt_vdet()
    densities = load_densities()

    per_fold = {}
    emb_acc = {"exp7a": None, "exp7c": None}
    nf = len(runtime.ACTIVE_FOLD_IDS)

    def loo_r2(Xv, yv):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict
        rcv = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=5)
        rcv.fit(Xv, yv)
        yp = cross_val_predict(Ridge(alpha=rcv.alpha_), Xv, yv, cv=LeaveOneOut())
        return float(r2_score(yv, yp))

    for fi in runtime.ACTIVE_FOLD_IDS:
        print(f"\n--- checkpoint fold {fi} ---")
        per_fold[str(fi)] = {}
        for mn in ["exp7a", "exp7c"]:
            print(f"  {mn}")
            dp = load_descriptor_model(mn, fi)
            embs = {mat: extract_descriptor(dp, *read_cluster_system(paths.CLUSTER_N1_DIR / mat)[:2]) for mat in materials}
            X = np.array([embs[m] for m in materials])
            if emb_acc[mn] is None:
                emb_acc[mn] = X.copy()
            else:
                emb_acc[mn] += X
            compactness = float(np.mean(np.linalg.norm(X - X.mean(axis=0), axis=1)))
            families = [get_family(m) for m in materials]
            sil = float(silhouette_score(X, families)) if len(set(families)) >= 2 else float("nan")
            yv = np.array([gt.get(m, np.nan) for m in materials])
            vv = ~np.isnan(yv)
            r2v = loo_r2(X[vv], yv[vv]) if vv.sum() >= 5 else float("nan")
            yd = np.array([densities.get(m, np.nan) for m in materials])
            vd = ~np.isnan(yd)
            r2d = loo_r2(X[vd], yd[vd]) if vd.sum() >= 5 else float("nan")
            per_fold[str(fi)][mn] = {
                "compactness": compactness,
                "silhouette": sil,
                "r2_vdet": r2v,
                "r2_density": r2d,
                "embedding_dim": int(X.shape[1]),
            }
            print(f"    compact={compactness:.4f}, sil={sil:.3f}, R2(Vdet)={r2v:.3f}, R2(dens)={r2d:.3f}")

    all_emb = {mn: emb_acc[mn] / float(nf) for mn in ["exp7a", "exp7c"]}
    aggregated = {}
    for mn in ["exp7a", "exp7c"]:
        comps = [per_fold[fk][mn]["compactness"] for fk in per_fold]
        sils = [per_fold[fk][mn]["silhouette"] for fk in per_fold]
        r2vs = [per_fold[fk][mn]["r2_vdet"] for fk in per_fold]
        r2ds = [per_fold[fk][mn]["r2_density"] for fk in per_fold]
        Xmean = all_emb[mn]
        families = [get_family(m) for m in materials]
        sil_mean_emb = float(silhouette_score(Xmean, families)) if len(set(families)) >= 2 else float("nan")
        aggregated[mn] = {
            "compactness": float(np.mean(comps)),
            "compactness_std_across_folds": float(np.std(comps)) if nf > 1 else 0.0,
            "silhouette": sil_mean_emb,
            "silhouette_std_across_folds": float(np.std(sils)) if nf > 1 else 0.0,
            "r2_vdet": float(np.mean(r2vs)),
            "r2_vdet_std_across_folds": float(np.std(r2vs)) if nf > 1 else 0.0,
            "r2_density": float(np.mean(r2ds)),
            "r2_density_std_across_folds": float(np.std(r2ds)) if nf > 1 else 0.0,
            "embedding_dim": per_fold[str(runtime.ACTIVE_FOLD_IDS[0])][mn]["embedding_dim"],
        }

    results = {"fold_ids": list(runtime.ACTIVE_FOLD_IDS), "per_fold": per_fold, "aggregated": aggregated}
    (output_dir / "mechanism_m4a_results.json").write_text(json.dumps(results, indent=2))
    plot_m4a(aggregated, all_emb, materials, gt, output_dir)


def plot_m4a(results: dict, all_emb: dict, materials: list, gt: dict, output_dir: Path) -> None:
    vv = np.array([gt.get(m, np.nan) for m in materials])
    valid = ~np.isnan(vv)
    fig_u, axes_u = plt.subplots(1, 2, figsize=(7.0, 3.2))
    try:
        from umap import UMAP
        vmin, vmax = float(np.nanmin(vv[valid])), float(np.nanmax(vv[valid]))
        for col, (mn, ax) in enumerate(zip(["exp7a", "exp7c"], axes_u)):
            X = all_emb[mn][valid]
            v = vv[valid]
            X2 = UMAP(n_components=2, random_state=42, n_neighbors=max(2, min(10, len(X) - 1))).fit_transform(X)
            sc = ax.scatter(X2[:, 0], X2[:, 1], c=v, cmap="viridis", vmin=vmin, vmax=vmax, s=28, edgecolors="white", linewidths=0.3)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2" if col == 0 else "")
            ax.set_title(mn)
            style_axes(ax, grid=False)
            add_panel_label(ax, "AB"[col])
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Vdet (m/s)", fontsize=8)
    except ImportError:
        for col, ax in enumerate(axes_u):
            ax.text(0.5, 0.5, "UMAP unavailable", ha="center", va="center", transform=ax.transAxes)
            style_axes(ax)
            add_panel_label(ax, "AB"[col])
    fig_u.tight_layout()
    save_figure(
        fig_u,
        "Fig_M4a_umap_supp",
        supplementary=True,
        legacy_png_name="figure_m4a_embedding_umap.png",
    )
    plt.close(fig_u)

    fig_q, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.0, 1.8]})
    x0 = np.arange(2)
    compact_vals = [results["exp7a"]["compactness"], results["exp7c"]["compactness"]]
    axL.bar(x0, compact_vals, color=[COLORS["exp7a"], COLORS["exp7c"]], edgecolor="white", linewidth=0.5)
    axL.set_xticks(x0)
    axL.set_xticklabels(["exp7a", "exp7c"])
    axL.set_ylabel("Compactness")
    axL.set_title("Compactness")
    style_axes(axL, grid=True)
    add_panel_label(axL, "A")

    metrics = ["silhouette", "r2_vdet", "r2_density"]
    mlabels = ["Silhouette", "R²(Vdet)", "R²(density)"]
    x = np.arange(len(metrics))
    w = 0.26
    axR.bar(x - w / 2, [results["exp7a"][m] for m in metrics], w, color=COLORS["exp7a"], edgecolor="white", linewidth=0.5, label="exp7a")
    axR.bar(x + w / 2, [results["exp7c"][m] for m in metrics], w, color=COLORS["exp7c"], edgecolor="white", linewidth=0.5, label="exp7c")
    axR.set_xticks(x)
    axR.set_xticklabels(mlabels)
    axR.set_ylabel("Score")
    axR.axhline(0, color=COLORS["ref"], lw=1.0, ls="--")
    axR.set_ylim(-0.5, 1.05)
    axR.legend(frameon=False, loc="upper left")
    axR.set_title("Representation metrics")
    style_axes(axR, grid=True)
    add_panel_label(axR, "B")

    fig_q.tight_layout()
    save_figure(
        fig_q,
        "Fig_M4a_quant_main",
        supplementary=False,
        legacy_png_name="figure_m4a_quantitative.png",
    )
    plt.close(fig_q)
    print("Saved Figure M4a")
