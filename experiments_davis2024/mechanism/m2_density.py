"""M2 -- Distance scaling probe (exp7a vs exp7c)"""
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


def _aggregate_m2_across_folds(per_fold: dict[str, dict]) -> dict:
    """Average per-material predictions and sensitivity across checkpoint folds."""
    agg: dict = {}
    fold_keys = sorted(per_fold.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    mats = get_materials()
    for mn in ["exp7a", "exp7c"]:
        sens_folds = [
            per_fold[fk][mn]["mean_sensitivity"]
            for fk in fold_keys
            if fk in per_fold and mn in per_fold[fk]
        ]
        agg_mr: dict = {}
        for mat in mats:
            ss_union: set[str] = set()
            for fk in fold_keys:
                pm = per_fold.get(fk, {}).get(mn, {}).get("per_material", {})
                if mat in pm:
                    ss_union.update(pm[mat].keys())
            if not ss_union:
                continue
            agg_mr[mat] = {}
            for ss in sorted(ss_union, key=float):
                vals = [
                    float(per_fold[fk][mn]["per_material"][mat][ss])
                    for fk in fold_keys
                    if mat in per_fold.get(fk, {}).get(mn, {}).get("per_material", {})
                    and ss in per_fold[fk][mn]["per_material"][mat]
                ]
                if vals:
                    agg_mr[mat][ss] = float(np.mean(vals))
        agg[mn] = {
            "per_material": agg_mr,
            "mean_sensitivity": float(np.mean(sens_folds)) if sens_folds else 0.0,
            "std_sensitivity": float(np.std(sens_folds)) if sens_folds else 0.0,
            "sensitivity_std_across_folds": float(np.std(sens_folds)) if len(sens_folds) > 1 else 0.0,
        }
    return agg


def run_m2(output_dir: Path, skip_inference: bool = False) -> None:
    print("\n" + "=" * 60 + "\nM2: Distance scaling (exp7a vs exp7c)\n" + "=" * 60)
    materials = get_materials()
    cache_path = output_dir / "mechanism_m2_results.json"
    densities = load_densities()

    if skip_inference and cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        print("Loaded cached M2")
    else:
        per_fold: dict[str, dict] = {}
        for fi in runtime.ACTIVE_FOLD_IDS:
            print(f"\n--- checkpoint fold {fi} ---")
            per_fold[str(fi)] = {}
            for mn in ["exp7a", "exp7c"]:
                print(f"  {mn}")
                model = load_property_model(mn, fi)
                mr: dict = {}
                for mat in materials:
                    pbs: dict = {}
                    for sc in SCALE_FACTORS:
                        ss = f"{sc:.2f}"
                        sd = paths.MECHANISM_DIR / f"scaled_{ss}" / mat
                        if sd.exists():
                            c, s, _ = read_cluster_system(sd)
                            pbs[ss] = predict_single(model, c, s)
                    if pbs:
                        mr[mat] = pbs
                sens = []
                for m in materials:
                    if m in mr and "0.80" in mr[m] and "1.20" in mr[m] and "1.00" in mr[m] and mr[m]["1.00"]!=0:
                        sens.append((mr[m]["0.80"]-mr[m]["1.20"])/mr[m]["1.00"])
                per_fold[str(fi)][mn] = {"per_material":mr, "mean_sensitivity":float(np.mean(sens)) if sens else 0., "std_sensitivity":float(np.std(sens)) if sens else 0.}
                print(f"    mean sensitivity: {per_fold[str(fi)][mn]['mean_sensitivity']:.4f}")
        aggregated = _aggregate_m2_across_folds(per_fold)
        results = {"fold_ids": list(runtime.ACTIVE_FOLD_IDS), "per_fold": per_fold, "aggregated": aggregated}
        cache_path.write_text(json.dumps(results, indent=2)); print(f"Saved M2")

    plot_block = results.get("aggregated", results)
    plot_m2(plot_block, output_dir, densities)


def plot_m2(results: dict, output_dir: Path, densities: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for col, mn in enumerate(["exp7a", "exp7c"]):
        ax = axes[col]
        mr = results[mn]["per_material"]
        color = COLORS[mn]
        all_x = sorted(set(float(s) for series in mr.values() for s in series))
        mean_y = []
        for mat in sorted(mr):
            sc = sorted(mr[mat].keys(), key=float)
            ax.plot([float(s) for s in sc], [mr[mat][s] for s in sc], color=COLORS["ref"], alpha=0.25, lw=0.7)
        for sx in all_x:
            ss = f"{sx:.2f}"
            vals = [mr[m][ss] for m in mr if ss in mr[m]]
            mean_y.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(all_x, mean_y, color=color, lw=2.3, label=mn, zorder=10)
        if 1.0 in all_x:
            ccoef = KJ_RHO_COEF
            kj_lines = []
            for mat in sorted(mr):
                r0 = densities.get(mat)
                if r0 is None or "1.00" not in mr.get(mat, {}):
                    continue
                v1 = float(mr[mat]["1.00"])
                denom = 1.0 + ccoef * r0
                if denom <= 0:
                    continue
                kj_lines.append(
                    [v1 * (1.0 + ccoef * r0 / (s**3)) / denom for s in all_x]
                )
            if kj_lines:
                kj_mean = [float(np.mean([row[i] for row in kj_lines])) for i in range(len(all_x))]
                ax.plot(all_x, kj_mean, color=COLORS["kj"], lw=1.2, ls="--", label="K-J density scaling")
        ax.axvline(1.0, color=COLORS["ref"], lw=1.0, ls=":")
        ms = results[mn]["mean_sensitivity"]
        mstd = results[mn].get("sensitivity_std_across_folds", results[mn].get("std_sensitivity", 0.0))
        sens_txt = f"Cluster scaling sens. = {ms * 100:+.1f}%"
        if mstd and mstd > 0:
            sens_txt += f" (±{mstd * 100:.1f}% across folds)"
        ax.text(
            0.04,
            0.95,
            sens_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
        )
        ax.set_xlabel("Uniform scale factor s (ρ(s)≈ρ₀/s³)")
        ax.set_ylabel("Predicted Vdet (m/s)" if col == 0 else "")
        ax.set_title(mn)
        style_axes(ax, grid=False)
        add_panel_label(ax, "AB"[col])
        if col == 1:
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Uniform cluster scaling: ρ(s)=ρ₀/s³ matches isotropic compression; K-J line uses per-material ρ₀",
        fontsize=7,
        y=0.02,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(
        fig,
        "Fig_M2_scaling_main",
        supplementary=False,
        legacy_png_name="figure_m2_distance_scaling.png",
    )
    plt.close(fig)
    print("Saved Figure M2")
