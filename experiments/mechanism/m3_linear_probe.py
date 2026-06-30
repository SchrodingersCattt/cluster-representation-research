"""M3 -- Linear probe (Ridge) on cluster descriptors for compositional content"""
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


def run_m3(output_dir: Path) -> None:
    print("\n" + "=" * 60 + "\nM3: Linear probe\n" + "=" * 60)
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score, mean_absolute_error

    materials = get_materials()
    gt = load_gt_vdet()
    comp, ob_values = compute_composition_and_ob(materials)
    densities = load_densities()
    print(f"  Loaded densities for {len(densities)} materials (range {min(densities.values()):.2f}--{max(densities.values()):.2f} g/cm3)")
    if ob_values:
        print(f"  Computed OB for {len(ob_values)} materials (range {min(ob_values.values()):.1f}--{max(ob_values.values()):.1f} %)")

    tgts = build_probe_targets(materials, gt, comp, ob_values, densities)
    per_fold = {}
    emb_dim = None

    for fi in runtime.ACTIVE_FOLD_IDS:
        print(f"\n--- checkpoint fold {fi} ---")
        per_fold[str(fi)] = {}
        for mn in ["exp7a", "exp7c"]:
            print(f"  {mn}")
            dp = load_descriptor_model(mn, fi)
            embs = {mat: extract_descriptor(dp, *read_cluster_system(paths.CLUSTER_N1_DIR / mat)[:2]).tolist() for mat in materials}
            X = np.array([embs[m] for m in materials])
            emb_dim = int(X.shape[1])
            Xc = np.array([[comp[m][e] for e in COMPOSITION_ELEMENTS] for m in materials])

            pr = {}
            for tn, y in tgts.items():
                v = ~np.isnan(y)
                if v.sum() < 5:
                    continue
                Xv, Xcv, yv = X[v], Xc[v], y[v]
                # Select alpha once on full data (5-fold CV), then LOO with fixed alpha.
                # Avoids nested LOO-inside-LOO which is O(n^2 * alphas) and very slow.
                from sklearn.linear_model import Ridge
                from sklearn.model_selection import cross_val_predict
                rcv_e = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=5)
                rcv_e.fit(Xv, yv)
                ype = cross_val_predict(Ridge(alpha=rcv_e.alpha_), Xv, yv, cv=LeaveOneOut())
                rcv_c = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=5)
                rcv_c.fit(Xcv, yv)
                ypc = cross_val_predict(Ridge(alpha=rcv_c.alpha_), Xcv, yv, cv=LeaveOneOut())
                pr[tn] = {
                    "r2_embedding": float(r2_score(yv, ype)),
                    "mae_embedding": float(mean_absolute_error(yv, ype)),
                    "r2_composition": float(r2_score(yv, ypc)),
                    "mae_composition": float(mean_absolute_error(yv, ypc)),
                    "y_pred_emb": ype.tolist(),
                    "y_pred_comp": ypc.tolist(),
                    "y_true": yv.tolist(),
                }
                print(f"    {tn}: emb R2={pr[tn]['r2_embedding']:.3f}, comp R2={pr[tn]['r2_composition']:.3f}")
            per_fold[str(fi)][mn] = {"probe_results": pr, "embedding_dim": emb_dim}

    aggregated = {}
    for mn in ["exp7a", "exp7c"]:
        pr_agg = {}
        tn_list = set()
        for fk in per_fold:
            tn_list.update(per_fold[fk][mn]["probe_results"].keys())
        for tn in tn_list:
            r2e = [per_fold[fk][mn]["probe_results"][tn]["r2_embedding"] for fk in per_fold if tn in per_fold[fk][mn]["probe_results"]]
            r2c = [per_fold[fk][mn]["probe_results"][tn]["r2_composition"] for fk in per_fold if tn in per_fold[fk][mn]["probe_results"]]
            yts = None
            y_stack_e = []
            y_stack_c = []
            for fk in sorted(per_fold.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                if tn not in per_fold[fk][mn]["probe_results"]:
                    continue
                cell = per_fold[fk][mn]["probe_results"][tn]
                yts = np.array(cell["y_true"])
                y_stack_e.append(np.array(cell["y_pred_emb"]))
                y_stack_c.append(np.array(cell["y_pred_comp"]))
            y_mean_e = np.mean(np.stack(y_stack_e, axis=0), axis=0)
            y_mean_c = np.mean(np.stack(y_stack_c, axis=0), axis=0)
            lo_e, hi_e = bootstrap_r2_ci(yts, y_mean_e)
            lo_c, hi_c = bootstrap_r2_ci(yts, y_mean_c)
            entry = {
                "r2_embedding": float(np.mean(r2e)),
                "r2_embedding_std_across_folds": float(np.std(r2e)) if len(r2e) > 1 else 0.0,
                "r2_embedding_ci_low": lo_e,
                "r2_embedding_ci_high": hi_e,
                "mae_embedding": float(np.mean([per_fold[fk][mn]["probe_results"][tn]["mae_embedding"] for fk in per_fold if tn in per_fold[fk][mn]["probe_results"]])),
                "r2_composition": float(np.mean(r2c)),
                "r2_composition_std_across_folds": float(np.std(r2c)) if len(r2c) > 1 else 0.0,
                "r2_composition_ci_low": lo_c,
                "r2_composition_ci_high": hi_c,
                "mae_composition": float(np.mean([per_fold[fk][mn]["probe_results"][tn]["mae_composition"] for fk in per_fold if tn in per_fold[fk][mn]["probe_results"]])),
            }
            if tn == "density":
                entry["y_true"] = yts.tolist()
                entry["y_pred_emb"] = y_mean_e.tolist()
                entry["y_pred_comp"] = y_mean_c.tolist()
            pr_agg[tn] = entry
            print(f"  {mn} {tn} agg: emb R2={entry['r2_embedding']:.3f}+/-{entry['r2_embedding_std_across_folds']:.3f}")
        aggregated[mn] = {"probe_results": pr_agg, "embedding_dim": emb_dim or 256}

    results = {"fold_ids": list(runtime.ACTIVE_FOLD_IDS), "per_fold": per_fold, "aggregated": aggregated}
    (output_dir / "mechanism_m3_results.json").write_text(json.dumps(results, indent=2))
    plot_m3(aggregated, output_dir)



def plot_m3(results: dict, output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    order = ["Vdet", "density", "OB", "frac_N", "frac_O", "n_atoms"]
    label_map = {"Vdet": "Vdet", "density": "density", "OB": "OB%", "frac_N": "N frac", "frac_O": "O frac", "n_atoms": "atom count"}
    tns = [t for t in order if any(t in mr["probe_results"] for mr in results.values())]
    x = np.arange(len(tns))
    w = 0.22

    def _ci_err(mn, t, emb: bool):
        d = results[mn]["probe_results"].get(t, {})
        if emb:
            m, lo, hi = d.get("r2_embedding", np.nan), d.get("r2_embedding_ci_low", np.nan), d.get("r2_embedding_ci_high", np.nan)
        else:
            m, lo, hi = d.get("r2_composition", np.nan), d.get("r2_composition_ci_low", np.nan), d.get("r2_composition_ci_high", np.nan)
        if any(np.isnan(np.array([m, lo, hi], dtype=float))):
            return np.nan, np.nan
        return m - lo, hi - m

    err7a_e = np.array([_ci_err("exp7a", t, True) for t in tns]).T
    err7c_e = np.array([_ci_err("exp7c", t, True) for t in tns]).T
    err_comp = np.array([_ci_err("exp7a", t, False) for t in tns]).T
    # Clip to non-negative: bootstrap CIs can invert when n_folds is small
    ye_a = np.clip(err7a_e, 0, None) if np.all(np.isfinite(err7a_e)) else None
    ye_c = np.clip(err7c_e, 0, None) if np.all(np.isfinite(err7c_e)) else None
    yc_o = np.clip(err_comp, 0, None) if np.all(np.isfinite(err_comp)) else None

    ax1.bar(
        x - w,
        [results["exp7a"]["probe_results"].get(t, {}).get("r2_embedding", np.nan) for t in tns],
        w,
        yerr=ye_a,
        capsize=1.5,
        label="exp7a",
        color=COLORS["exp7a"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x,
        [results["exp7c"]["probe_results"].get(t, {}).get("r2_embedding", np.nan) for t in tns],
        w,
        yerr=ye_c,
        capsize=1.5,
        label="exp7c",
        color=COLORS["exp7c"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.bar(
        x + w,
        [results["exp7a"]["probe_results"].get(t, {}).get("r2_composition", np.nan) for t in tns],
        w,
        yerr=yc_o,
        capsize=1.5,
        label="composition",
        color=COLORS["ref"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map[t] for t in tns])
    ax1.set_ylabel("R² (LOO)")
    ax1.set_ylim(-0.5, 1.05)
    ax1.axhline(0, color=COLORS["ref"], lw=1.0, ls="--")
    ax1.legend(frameon=False, loc="upper left")
    style_axes(ax1, grid=True)
    add_panel_label(ax1, "A")

    if "density" in results.get("exp7a", {}).get("probe_results", {}):
        yt = results["exp7a"]["probe_results"]["density"]["y_true"]
        lo, hi = min(yt) - 0.05, max(yt) + 0.05
        for mn, color, marker in [("exp7a", COLORS["exp7a"], "o"), ("exp7c", COLORS["exp7c"], "s")]:
            dr = results[mn]["probe_results"]["density"]
            ax2.scatter(
                dr["y_true"],
                dr["y_pred_emb"],
                color=color,
                marker=marker,
                s=22,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
                label=f"{mn} (R²={dr['r2_embedding']:.2f})",
            )
        ax2.plot([lo, hi], [lo, hi], color=COLORS["ref"], lw=1.0, ls="--")
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.set_xlabel("True density (g/cm³)")
        ax2.set_ylabel("Probe-predicted density")
        ax2.legend(frameon=False, loc="upper left")
        style_axes(ax2, grid=False)
        add_panel_label(ax2, "B")
    else:
        ax2.text(0.5, 0.5, "Density data N/A", ha="center", va="center", transform=ax2.transAxes)
        style_axes(ax2, grid=False)
        add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M3_linear_probe_main",
        supplementary=False,
        legacy_png_name="figure_m3_linear_probe.png",
    )
    plt.close(fig)
    print("Saved Figure M3")


# ===========================================================================
# M3b — Nonlinear Probe Analysis
