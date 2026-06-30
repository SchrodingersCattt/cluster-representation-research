"""M3b -- Nonlinear probe (Ridge vs kernel Ridge RBF) on cluster descriptors"""
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
    METAL_ELEMENTS, PEM_BOND_THRESHOLDS, COMPOSITION_ELEMENTS,
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


def run_m3b(output_dir: Path) -> None:
    """M3b: Compare linear (Ridge) vs nonlinear (kernel Ridge RBF) probes on embeddings.

    Uses the same exp7 5-fold CV splits (pems_5fold_splits_v2.json) to ensure
    consistency with training fold assignments. For each target, we evaluate
    Ridge and KernelRidge(RBF) on embedding features and composition features.

    Design rationale for small-sample regime (n=20 per fold, d=256):
    - KernelRidge(RBF) is the standard nonlinear probe for n≪d: it has only
      n parameters (dual coefficients), so it cannot overfit more than Ridge.
    - Grid search over alpha ∈ RIDGE_ALPHAS and gamma ∈ {1e-4, 1e-3, 1e-2,
      0.1, 1.0} using 5-fold CV on the training fold.
    - StandardScaler and train-fold PCA are applied before kernel computation
      to control scale and high-dimensional distance concentration.

    Scientific note: KernelRidge(RBF) answers "does the embedding contain
    nonlinear structure beyond what linear Ridge can capture?" without adding
    a separate neural-network probe.
    """
    print("\n" + "=" * 60 + "\nM3b: Nonlinear probe (Ridge vs KernelRidge RBF)\n" + "=" * 60)
    from sklearn.linear_model import RidgeCV
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    # --- Load 5-fold splits (must match exp7) ---
    splits_data = json.loads(paths.M1_SPLITS_PATH.read_text(encoding="utf-8"))
    fold_dict = splits_data["folds"]  # {"0": [...], "1": [...], ...}
    num_folds = len(fold_dict)
    print(f"  Loaded {num_folds}-fold splits from {paths.M1_SPLITS_PATH.name}")

    materials = get_materials()
    gt = load_gt_vdet()

    # --- Build material-to-fold mapping ---
    mat_to_fold: dict[str, int] = {}
    for fold_idx_str, fold_mats in fold_dict.items():
        for m in fold_mats:
            mat_to_fold[m] = int(fold_idx_str)

    # --- Composition, OB, density (shared helpers) ---
    comp, ob_values = compute_composition_and_ob(materials)
    densities = load_densities()

    # KernelRidge RBF hyperparameter grid
    KRR_ALPHAS = RIDGE_ALPHAS  # same alpha grid as Ridge
    KRR_GAMMAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0]  # RBF bandwidth grid

    def _5fold_cv(X: np.ndarray, y: np.ndarray, mats: list[str],
                  model_factory, scale: bool = False) -> float:
        """Run 5-fold CV matching exp7 splits, return pooled R².

        StandardScaler is applied on the training fold only (no leakage).
        """
        fold_r2s: list[float] = []
        for fi in range(num_folds):
            val_mask = np.array([mat_to_fold.get(m, -1) == fi for m in mats])
            tr_mask = ~val_mask
            if val_mask.sum() < 1 or tr_mask.sum() < 2:
                continue
            Xtr, ytr = X[tr_mask], y[tr_mask]
            Xte, yte = X[val_mask], y[val_mask]
            if scale:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)
            mdl = model_factory()
            mdl.fit(Xtr, ytr)
            ypred = mdl.predict(Xte)
            # Per-fold R² can be negative for small folds; compute from pooled later
            fold_r2s.append((yte, ypred))
        # Pool all predictions across folds
        if not fold_r2s:
            return float("nan")
        y_all = np.concatenate([p[0] for p in fold_r2s])
        yp_all = np.concatenate([p[1] for p in fold_r2s])
        return float(r2_score(y_all, yp_all))

    def _5fold_cv_krr(X: np.ndarray, y: np.ndarray, mats: list[str]) -> tuple[float, float, float]:
        """KernelRidge(RBF) 5-fold CV with PCA pre-reduction + inner grid search.

        For each outer fold:
          1. Scale X (fit on train, apply to test — no leakage).
          2. PCA(n_components=min(20, n_train-1)) fit on train, applied to test.
             This reduces d=256 → ≤20 before RBF kernel, avoiding the
             distance-concentration curse in high dimensions.
          3. Inner GridSearchCV(KernelRidge, alpha×gamma grid, cv=min(3, n_tr-1))
             on the PCA-reduced training fold to select best (alpha, gamma).
          4. Refit on full training fold, predict test fold.

        Returns (pooled_r2, best_alpha_mean, best_gamma_mean).
        """
        from sklearn.decomposition import PCA as _PCA
        fold_r2s: list[tuple] = []
        best_alphas: list[float] = []
        best_gammas: list[float] = []
        param_grid = [
            {"alpha": KRR_ALPHAS, "gamma": KRR_GAMMAS}
        ]
        for fi in range(num_folds):
            val_mask = np.array([mat_to_fold.get(m, -1) == fi for m in mats])
            tr_mask = ~val_mask
            if val_mask.sum() < 1 or tr_mask.sum() < 2:
                continue
            Xtr, ytr = X[tr_mask], y[tr_mask]
            Xte, yte = X[val_mask], y[val_mask]
            # Step 1: StandardScaler (fit on train only)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
            # Step 2: PCA pre-reduction to avoid RBF distance concentration in d=256
            n_pca = min(20, int(tr_mask.sum()) - 1)
            pca = _PCA(n_components=n_pca)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
            # Step 3: Inner GridSearchCV for (alpha, gamma)
            inner_cv = min(3, int(tr_mask.sum()) - 1)
            gs = GridSearchCV(
                KernelRidge(kernel="rbf"),
                param_grid,
                cv=inner_cv,
                scoring="r2",
                n_jobs=1,
            )
            gs.fit(Xtr, ytr)
            best_alphas.append(float(gs.best_params_["alpha"]))
            best_gammas.append(float(gs.best_params_["gamma"]))
            ypred = gs.predict(Xte)
            fold_r2s.append((yte, ypred))
        if not fold_r2s:
            return float("nan"), float("nan"), float("nan")
        y_all = np.concatenate([p[0] for p in fold_r2s])
        yp_all = np.concatenate([p[1] for p in fold_r2s])
        r2 = float(r2_score(y_all, yp_all))
        return r2, float(np.mean(best_alphas)), float(np.mean(best_gammas))

    per_ckpt = {}
    for cfi in runtime.ACTIVE_FOLD_IDS:
        print(f"\n--- descriptor checkpoint fold {cfi} ---")
        per_ckpt[str(cfi)] = {}
        for mn in ["exp7a", "exp7c"]:
            print(f"  {mn}")
            dp = load_descriptor_model(mn, cfi)
            embs = {mat: extract_descriptor(dp, *read_cluster_system(paths.CLUSTER_N1_DIR / mat)[:2]).tolist()
                    for mat in materials}
            X = np.array([embs[m] for m in materials])
            Xc = np.array([[comp[m][e] for e in COMPOSITION_ELEMENTS] for m in materials])
            tgts = build_probe_targets(materials, gt, comp, ob_values, densities)

            pr = {}
            for tn, y in tgts.items():
                v = ~np.isnan(y)
                if v.sum() < 5:
                    continue
                Xv = X[v]
                Xcv = Xc[v]
                yv = y[v]
                mats_v = [m for m, ok in zip(materials, v) if ok]

                r2_linear = _5fold_cv(
                    Xv, yv, mats_v,
                    lambda: RidgeCV(alphas=np.array(RIDGE_ALPHAS)),
                    scale=False,
                )
                r2_comp = _5fold_cv(
                    Xcv, yv, mats_v,
                    lambda: RidgeCV(alphas=np.array(RIDGE_ALPHAS)),
                    scale=False,
                )

                r2_krr, best_alpha, best_gamma = _5fold_cv_krr(Xv, yv, mats_v)
                gap = r2_krr - r2_linear

                pr[tn] = {
                    "r2_linear": r2_linear,
                    "r2_krr": r2_krr,
                    "r2_krr_best_alpha": best_alpha,
                    "r2_krr_best_gamma": best_gamma,
                    "r2_composition": r2_comp,
                    "gap": gap,
                }
                print(f"    {tn}: linear={r2_linear:.3f}, KRR={r2_krr:.3f}, gap={gap:+.3f}, best_alpha={best_alpha:.4g}, best_gamma={best_gamma:.4g}")

            per_ckpt[str(cfi)][mn] = {"probe_results": pr}

    aggregated = {}
    for mn in ["exp7a", "exp7c"]:
        tn_all = set()
        for fk in per_ckpt:
            tn_all.update(per_ckpt[fk][mn]["probe_results"].keys())
        agg_pr = {}
        for tn in tn_all:
            r2_lin = [per_ckpt[fk][mn]["probe_results"][tn]["r2_linear"] for fk in per_ckpt]
            r2_cmp = [per_ckpt[fk][mn]["probe_results"][tn]["r2_composition"] for fk in per_ckpt]
            r2_krr_vals = [per_ckpt[fk][mn]["probe_results"][tn]["r2_krr"] for fk in per_ckpt]
            r2_lin_m = float(np.mean(r2_lin))
            r2_krr_m = float(np.mean(r2_krr_vals))
            agg_pr[tn] = {
                "r2_linear": r2_lin_m,
                "r2_linear_std_across_ckpt_folds": float(np.std(r2_lin)) if len(r2_lin) > 1 else 0.0,
                "r2_krr": r2_krr_m,
                "r2_krr_std_across_ckpt_folds": float(np.std(r2_krr_vals)) if len(r2_krr_vals) > 1 else 0.0,
                "r2_composition": float(np.mean(r2_cmp)),
                "r2_composition_std_across_ckpt_folds": float(np.std(r2_cmp)) if len(r2_cmp) > 1 else 0.0,
                "gap": r2_krr_m - r2_lin_m,
            }
        aggregated[mn] = {"probe_results": agg_pr}

    results = {"fold_ids": list(runtime.ACTIVE_FOLD_IDS), "per_checkpoint_fold": per_ckpt, "aggregated": aggregated}

    # --- Interpretation ---
    print("\n--- Interpretation (aggregated over checkpoint folds) ---")
    for mn in ["exp7a", "exp7c"]:
        pr = aggregated[mn]["probe_results"]
        for tn, d in pr.items():
            gap = d["gap"]
            if gap > 0.1:
                print(f"  {mn}/{tn}: gap={gap:+.3f} > 0.1 → information present but nonlinearly encoded")
            elif gap > 0.02:
                print(f"  {mn}/{tn}: gap={gap:+.3f} ∈ (0.02, 0.1] → moderate nonlinear gain")
            elif gap >= -0.05:
                print(f"  {mn}/{tn}: gap={gap:+.3f} ∈ [-0.05, 0.02] → information already linearized (or absent)")
            else:
                print(f"  {mn}/{tn}: gap={gap:+.3f} < -0.05 → KRR underperforms Ridge (unexpected; check data)")

    # --- Save ---
    out_path = output_dir / "mechanism_m3b_nonlinear_probe.json"
    out_path.write_text(json.dumps({
        "description": "M3b: Linear vs nonlinear (KernelRidge RBF) probe on embeddings (5-fold CV matching exp7 splits)",
        "krr_config": {
            "pipeline": "StandardScaler → KernelRidge(kernel='rbf')",
            "kernel": "rbf",
            "alpha_grid": KRR_ALPHAS,
            "gamma_grid": KRR_GAMMAS,
            "inner_cv": "min(3, n_train-1) folds",
            "_note": (
                "KernelRidge(RBF) has only n dual parameters, so it cannot overfit more than Ridge. "
                "Inner GridSearchCV selects (alpha, gamma) on training fold only (no leakage). "
                "StandardScaler and PCA are fit on train fold only."
            ),
        },
        "cv_splits": "pems_5fold_splits_v2.json",
        "results": results,
    }, indent=2))
    print(f"\nSaved results to {out_path}")
    plot_m3b(aggregated, output_dir)



def plot_m3b(results: dict, output_dir: Path) -> None:
    """Plot M3b: Linear (Ridge) vs nonlinear (KernelRidge RBF) probe comparison.

    Panel A: Grouped bars — Ridge vs KRR R² for each target × model.
    Panel B: Gap (KRR R² − Ridge R²) with nonlinear-gain shading.
    """
    setup_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5))

    order = ["Vdet", "density", "OB", "frac_N", "frac_O", "n_atoms"]
    label_map = {
        "Vdet": "Vdet", "density": "density", "OB": "OB%",
        "frac_N": "N frac", "frac_O": "O frac", "n_atoms": "atom count",
    }
    tns = [t for t in order if any(
        t in results.get(mn, {}).get("probe_results", {}) for mn in ["exp7a", "exp7c"]
    )]
    x = np.arange(len(tns))
    w = 0.18  # bar width

    # --- Panel A: R² bars (Ridge lighter, KRR solid) ---
    # Use lighter fill + border for Ridge, solid fill for KRR
    for i, (mn, color) in enumerate([("exp7a", COLORS["exp7a"]), ("exp7c", COLORS["exp7c"])]):
        pr = results.get(mn, {}).get("probe_results", {})
        ridge_vals = [pr.get(t, {}).get("r2_linear", np.nan) for t in tns]
        krr_vals = [pr.get(t, {}).get("r2_krr", np.nan) for t in tns]
        krr_stds = [pr.get(t, {}).get("r2_krr_std_across_ckpt_folds", 0.0) for t in tns]

        offset = (i - 0.5) * 2 * w  # -w for exp7a, +w for exp7c
        # Ridge bar (lighter fill + colored border)
        ax1.bar(
            x + offset - w / 2, ridge_vals, w * 0.9,
            color=color, alpha=0.25, edgecolor=color, linewidth=1.0,
            label=f"{mn} Ridge",
        )
        # KRR bar (solid) with error bars (std across checkpoint folds)
        ax1.bar(
            x + offset + w / 2, krr_vals, w * 0.9,
            color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
            yerr=krr_stds, error_kw=dict(lw=0.8, capsize=2, capthick=0.8),
            label=f"{mn} KRR",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map[t] for t in tns], rotation=25, ha="right")
    ax1.set_ylabel("R² (5-fold CV)")
    ax1.set_ylim(-0.9, 1.1)
    ax1.axhline(0, color=COLORS["ref"], lw=0.8, ls="--")
    ax1.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left",
               columnspacing=1.0, handlelength=1.5, handletextpad=0.4)
    style_axes(ax1, grid=True)
    add_panel_label(ax1, "A")

    # --- Panel B: Gap (KRR − Ridge) ---
    for mn, color, marker in [("exp7a", COLORS["exp7a"], "o"), ("exp7c", COLORS["exp7c"], "s")]:
        pr = results.get(mn, {}).get("probe_results", {})
        gaps = [pr.get(t, {}).get("gap", np.nan) for t in tns]
        ax2.plot(x, gaps, marker=marker, color=color, markersize=6,
                 linewidth=1.2, label=mn, markeredgecolor="white",
                 markeredgewidth=0.5, zorder=3)

    ax2.axhline(0, color=COLORS["ref"], lw=0.8, ls="--")
    # Compute ylim first, then shade
    all_gaps = []
    for mn in ["exp7a", "exp7c"]:
        pr = results.get(mn, {}).get("probe_results", {})
        all_gaps.extend([pr.get(t, {}).get("gap", 0.0) for t in tns])
    yabs = max(abs(min(all_gaps)), abs(max(all_gaps)), 0.3) * 1.15
    ax2.set_ylim(-yabs, yabs)
    ax2.set_xlim(-0.5, len(tns) - 0.3)  # leave room on right for labels
    # Shade nonlinear gain / overfit regions
    ax2.axhspan(0.1, yabs, alpha=0.06, color="#2ca02c", zorder=0)
    ax2.axhspan(-yabs, -0.1, alpha=0.06, color="#d62728", zorder=0)
    # Label the shaded regions (inside axes, not clipped)
    ax2.text(len(tns) - 1.0, yabs * 0.85, "nonlinear gain", fontsize=6,
             color="#2ca02c", alpha=0.7, ha="right", va="top")
    ax2.text(len(tns) - 1.0, -yabs * 0.85, "KRR underperforms", fontsize=6,
             color="#d62728", alpha=0.7, ha="right", va="bottom")

    ax2.set_xticks(x)
    ax2.set_xticklabels([label_map[t] for t in tns], rotation=25, ha="right")
    ax2.set_ylabel("Gap (KRR R² − Ridge R²)")
    ax2.legend(frameon=False, fontsize=7, loc="lower left")
    style_axes(ax2, grid=True)
    add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M3b_nonlinear_probe_main",
        supplementary=False,
        legacy_png_name="figure_m3b_nonlinear_probe.png",
    )
    plt.close(fig)
    print("Saved Figure M3b")


# ===========================================================================
# M4a
