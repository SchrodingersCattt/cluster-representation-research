"""M1 -- Composition probe + DAP-4 template invariance.

This module merges two related experiments into a single inference pass:

    destructive perturbations  -- geometry destroyed (scrambled / sphere /
        line / B-swap). Probes the model's reliance on geometric
        arrangement of the local polyhedra.
    polyhedron-preserving stretches -- A-X, B-X, and A-B COM distances are
        scaled while each molecular fragment remains internally rigid.
        Probes whether the model is insensitive to local polyhedron-size
        changes that preserve fragment chemistry and local shape.
    polyhedron-breaking swaps  -- A/B/X molecular fragments exchange COM
        positions, preserving internal fragment geometry while disrupting
        the local polyhedron tiling.
    template perturbations     -- chemistry preserved, geometry transplanted
        onto a universal local-polyhedra template (currently only DAP-4).
        Probes whether the model learns rotation/translation-invariant
        cluster-level features rather than memorising a specific cluster
        geometry. Merged from the standalone M5b experiment in 2026-04.

All evaluations use **held-out fold semantics**: each material is predicted
by the single fold model that did NOT see it during training. This is the
historical M1 convention; M5b previously used 5-fold ensemble averaging and
those predictions can drift by a few m/s after the merge -- see
``journal.md`` 2026-04-19 entry for the recorded delta.

Adding a new local-polyhedra template (e.g. PAP-6, DAN-4):
    1. Build the template dataset under ``00_data_prep/`` mirroring
       ``pems_dap4_template_systems``.
    2. Add a ``Perturbation(kind="template", ...)`` entry to
       ``mechanism/perturbations.py`` (see the docstring there).
    3. Re-run ``python run_mechanism_analysis.py --experiments m1``.

The new perturbation will appear automatically under
``aggregated.<model>.<perturbation_id>`` in
``mechanism_m1_results.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

from . import paths, perturbations, runtime
from .constants import COLORS, N_SEEDS, PERTURBATION_STYLE
from .io_data import get_m1_heldout_mats, get_materials, read_cluster_system
from .io_models import load_property_model
from .inference import predict_single
from .plot_helpers import (
    add_panel_label,
    disp,
    rounded_limits,
    save_figure,
    style_axes,
)


LADDER_ORDER = [
    "template_dap4",
    "rotation", "translation",
    "stretch_bx", "stretch_ax", "stretch_ab",
    "swap_a_b", "swap_b_x", "swap_a_x",
    "scrambled_swap", "scrambled_random", "random_sphere", "sorted_line",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_m1_across_folds(per_fold: dict[str, dict]) -> dict:
    """Pool each fold's held-out predictions and recompute Spearman per perturbation.

    Each material appears exactly once (as the held-out set of the fold whose
    model did NOT see it during training). This avoids averaging predictions
    from models that were trained on the same material.

    Iterates the perturbation registry so newly-added perturbations are picked
    up automatically.
    """
    fkeys = sorted(per_fold.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    agg: dict = {}

    # Union of models that appear anywhere in per_fold (handles the case where
    # different models cover different perturbations, e.g. only exp7d does
    # template_dap4 in addition to exp7a/exp7c).
    all_models: set[str] = set()
    for fk in fkeys:
        all_models.update(per_fold.get(fk, {}).keys())

    for mn in sorted(all_models):
        fold_mrs = [(fk, per_fold[fk][mn]) for fk in fkeys if mn in per_fold.get(fk, {})]
        if not fold_mrs:
            continue

        # Identity noise floor (only present when the model also ran destructive
        # perturbations; template-only models -- e.g. exp7d -- skip identity).
        id_stds = [x["identity"]["mean_std"] for _, x in fold_mrs if "identity" in x]
        mr_out: dict = {}
        if id_stds:
            mr_out["identity"] = {"mean_std": float(np.mean(id_stds))}

        # Pool original predictions from each fold's own held-out set
        orig_pooled: dict[str, float] = {}
        for fk, fmr in fold_mrs:
            fi = int(fk)
            heldout = set(get_m1_heldout_mats(fi))
            for mat, pred in fmr.get("original", {}).items():
                if mat in heldout:
                    orig_pooled[mat] = float(pred)
        mr_out["original"] = orig_pooled

        # Per-perturbation aggregation, registry-driven.
        for pert in perturbations.PERTURBATIONS:
            if mn not in pert.models:
                continue
            pid = pert.id
            pp_pooled: dict[str, list] = {}
            per_fold_sp: list[float] = []
            for fk, fmr in fold_mrs:
                fi = int(fk)
                heldout = set(get_m1_heldout_mats(fi))
                if pid not in fmr:
                    continue
                pt_data = fmr[pid]
                for mat, preds in pt_data.get("per_material", {}).items():
                    if mat in heldout:
                        pp_pooled[mat] = preds
                # Per-fold Spearman (small n approx 5, informative but noisy).
                ov_f = [float(orig_pooled[m]) for m in heldout if m in orig_pooled and m in pp_pooled]
                pv_f = [float(np.mean(pp_pooled[m])) for m in heldout if m in orig_pooled and m in pp_pooled]
                if len(ov_f) >= 3:
                    per_fold_sp.append(float(sp_stats.spearmanr(ov_f, pv_f).statistic))

            if not pp_pooled:
                continue

            all_mats = sorted(set(orig_pooled) & set(pp_pooled))
            ov = [orig_pooled[m] for m in all_mats]
            pv = [float(np.mean(pp_pooled[m])) for m in all_mats]
            sp = float(sp_stats.spearmanr(ov, pv).statistic) if len(ov) >= 3 else float("nan")
            deltas = [abs(pv[i] - ov[i]) for i in range(len(ov))]
            dm = float(np.mean(deltas)) if deltas else 0.0
            ds = float(np.std(deltas)) if deltas else 0.0
            # Fisher z-transform aggregation of per-fold Spearman rho values.
            # Direct averaging of rho is biased; z = atanh(rho) is approximately
            # normal, so we average z then back-transform.
            if per_fold_sp:
                z_vals = [np.arctanh(np.clip(r, -0.9999, 0.9999)) for r in per_fold_sp]
                sp_fold_mean = float(np.tanh(np.mean(z_vals)))
            else:
                sp_fold_mean = float("nan")
            mr_out[pid] = {
                "per_material": pp_pooled,
                "delta_mean": dm,
                "delta_std": ds,
                "spearman": sp,
                "spearman_per_fold": per_fold_sp,
                "spearman_per_fold_mean_fisher": sp_fold_mean,
                "n_materials": len(pp_pooled),
                "kind": pert.kind,
            }
        agg[mn] = mr_out
    return agg


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------

def _models_used_in_fold(fold_idx: int) -> tuple[str, ...]:
    """Union of model keys participating in any registered perturbation."""
    return tuple(sorted({m for p in perturbations.PERTURBATIONS for m in p.models}))


def _identity_baseline_models() -> tuple[str, ...]:
    """Models that need an identity-noise baseline for perturbation comparisons."""
    out: set[str] = set()
    baseline_kinds = {"destructive", "polyhedron_preserve", "polyhedron_break"}
    for p in perturbations.PERTURBATIONS:
        if p.kind in baseline_kinds:
            out.update(p.models)
    return tuple(sorted(out))


def run_m1(output_dir: Path, skip_inference: bool = False) -> None:
    print("\n" + "=" * 60 + "\nM1: Composition probe + DAP-4 template invariance\n" + "=" * 60)
    print("  Evaluation: held-out materials only (5-fold pooled)")
    cache_path = output_dir / "mechanism_m1_results.json"

    if skip_inference and cache_path.exists():
        results = json.loads(cache_path.read_text(encoding="utf-8"))
        print("Loaded cached M1 results")
    else:
        identity_models = set(_identity_baseline_models())
        per_fold: dict[str, dict] = {}
        for fi in runtime.ACTIVE_FOLD_IDS:
            print(f"\n--- checkpoint fold {fi} (held-out only) ---")
            heldout_mats = get_m1_heldout_mats(fi)
            per_fold[str(fi)] = {}
            models_this_fold = _models_used_in_fold(fi)
            for mn in models_this_fold:
                # Determine which perturbations this model participates in.
                model_perts = [p for p in perturbations.PERTURBATIONS if mn in p.models]
                if not model_perts:
                    continue
                print(f"  {mn}  (n_heldout={len(heldout_mats)}, n_perts={len(model_perts)})")
                model = load_property_model(mn, fi)
                mr: dict = {}

                if mn in identity_models:
                    print("  Identity baseline (5x) on held-out mats...")
                    id_preds: dict[str, list[float]] = {}
                    for mat in heldout_mats:
                        c, s, _ = read_cluster_system(paths.CLUSTER_N1_DIR / mat)
                        id_preds[mat] = [predict_single(model, c, s) for _ in range(5)]
                    id_std = float(np.mean([np.std(v) for v in id_preds.values()]))
                    mr["identity"] = {
                        "per_material": {
                            m: {"preds": p, "std": float(np.std(p))} for m, p in id_preds.items()
                        },
                        "mean_std": id_std,
                    }
                    print(f"  Identity noise floor: {id_std:.4f} m/s")

                print("  Original predictions on held-out mats...")
                orig: dict[str, float] = {}
                for mat in heldout_mats:
                    c, s, _ = read_cluster_system(paths.CLUSTER_N1_DIR / mat)
                    orig[mat] = predict_single(model, c, s)
                mr["original"] = orig

                for pert in model_perts:
                    ns = pert.n_seeds
                    print(f"  {pert.id} (kind={pert.kind}, x{ns})...")
                    pp: dict[str, list[float]] = {}
                    for mat in heldout_mats:
                        preds: list[float] = []
                        for si in range(ns):
                            sd = pert.system_path(mat, si)
                            if sd.exists():
                                c, s, _ = read_cluster_system(sd)
                                preds.append(predict_single(model, c, s))
                        if preds:
                            pp[mat] = preds

                    if pp:
                        deltas = [abs(np.mean(pp[m]) - orig[m]) for m in heldout_mats if m in pp and m in orig]
                        dm, ds = (float(np.mean(deltas)), float(np.std(deltas))) if deltas else (0.0, 0.0)
                        ov = [orig[m] for m in heldout_mats if m in pp]
                        pv = [float(np.mean(pp[m])) for m in heldout_mats if m in pp]
                        sp = float(sp_stats.spearmanr(ov, pv).statistic) if len(ov) >= 3 else float("nan")
                        mr[pert.id] = {
                            "per_material": dict(pp),
                            "delta_mean": dm,
                            "delta_std": ds,
                            "spearman": sp,
                            "n_materials": len(pp),
                            "kind": pert.kind,
                        }
                        print(f"    delta={dm:.1f}+/-{ds:.1f}, spearman={sp:.3f}, n={len(pp)}")

                per_fold[str(fi)][mn] = mr

        aggregated = _aggregate_m1_across_folds(per_fold)
        results = {
            "fold_ids": list(runtime.ACTIVE_FOLD_IDS),
            "evaluation": "held_out_5fold_pooled",
            "perturbation_groups": {
                "destructive": list(perturbations.DESTRUCTIVE_IDS),
                "polyhedron_preserve": list(perturbations.POLYHEDRON_PRESERVE_IDS),
                "polyhedron_break": list(perturbations.POLYHEDRON_BREAK_IDS),
                "template": list(perturbations.TEMPLATE_IDS),
            },
            "perturbations": {
                p.id: {
                    "kind": p.kind,
                    "n_seeds": p.n_seeds,
                    "models": list(p.models),
                    "seeded": p.seeded,
                }
                for p in perturbations.PERTURBATIONS
            },
            "per_fold": per_fold,
            "aggregated": aggregated,
        }
        cache_path.write_text(json.dumps(results, indent=2))
        print("Saved M1 results")

    plot_block = results.get("aggregated", results)
    plot_m1_destructive(plot_block, output_dir)
    if any("template_dap4" in plot_block.get(m, {}) for m in plot_block):
        plot_m1_template(plot_block, output_dir)
    plot_m1_direct_comparison(plot_block, output_dir)


# ---------------------------------------------------------------------------
# Destructive perturbations -- 2x2 panel (Fig_M1_composition_probe_main).
# ---------------------------------------------------------------------------

def plot_m1_destructive(results: dict, output_dir: Path) -> None:
    # Use materials present in results (held-out pooled = all 25) rather than get_materials().
    materials = sorted(set(results.get("exp7a", {}).get("original", {}).keys())
                       | set(results.get("exp7c", {}).get("original", {}).keys()))
    if not materials:
        materials = get_materials()
    main_pts = [
        "scrambled_swap",
        "scrambled_random",
        "random_sphere",
        "sorted_line",
        "swapped_bsite",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
    ymax = 0.0
    for mn in ["exp7a", "exp7c"]:
        for pt in main_pts:
            if pt in results.get(mn, {}):
                ymax = max(ymax, results[mn][pt]["delta_mean"] + results[mn][pt]["delta_std"])
    ymax *= 1.12

    for col, mn in enumerate(["exp7a", "exp7c"]):
        mr = results[mn]
        ax = axes[0, col]
        vals = [mr[pt]["delta_mean"] for pt in main_pts]
        errs = [mr[pt]["delta_std"] for pt in main_pts]
        colors = [PERTURBATION_STYLE[pt]["color"] for pt in main_pts]
        ax.bar(
            np.arange(len(main_pts)),
            vals,
            yerr=errs,
            capsize=2.0,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        if "identity" in mr:
            ax.axhline(mr["identity"]["mean_std"], color=COLORS["ref"], lw=1.0, ls="--")
        ax.set_xticks(np.arange(len(main_pts)))
        ax.set_xticklabels([PERTURBATION_STYLE[pt]["label"] for pt in main_pts])
        ax.set_ylabel("Mean |\u0394pred| (m/s)" if col == 0 else "")
        ax.set_ylim(0, ymax)
        ax.set_title(disp(mn))
        style_axes(ax, grid=True)
        add_panel_label(ax, "AB"[col])

        ax2 = axes[1, col]
        orig = mr["original"]
        x_all, y_all = [], []
        for pt in main_pts:
            xs, ys = [], []
            for mat in materials:
                if mat in orig and mat in mr[pt]["per_material"]:
                    xs.append(orig[mat])
                    ys.append(float(np.mean(mr[pt]["per_material"][mat])))
            x_all.extend(xs)
            y_all.extend(ys)
            ax2.scatter(
                xs,
                ys,
                s=20,
                alpha=0.75,
                color=PERTURBATION_STYLE[pt]["color"],
                marker=PERTURBATION_STYLE[pt]["marker"],
                edgecolors="white",
                linewidths=0.3,
                label=PERTURBATION_STYLE[pt]["label"],
            )
        lim_lo, lim_hi = rounded_limits(x_all + y_all, step=250.0, pad=0.03)
        ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=COLORS["ref"], lw=1.0, ls="--")
        ax2.set_xlim(lim_lo, lim_hi)
        ax2.set_ylim(lim_lo, lim_hi)
        ax2.set_xlabel("Original prediction (m/s)")
        ax2.set_ylabel("Perturbed prediction (m/s)" if col == 0 else "")
        ax2.set_title(disp(mn))
        style_axes(ax2, grid=False)
        add_panel_label(ax2, "CD"[col])

    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M1_composition_probe_main",
        supplementary=False,
        legacy_png_name="figure_m1_composition_probe.png",
    )
    plt.close(fig)

    # Supplementary full-analysis version with all destructive perturbations.
    full_pts = [pid for pid in perturbations.DESTRUCTIVE_IDS if pid in results.get("exp7a", {})]
    fig_full, axes_full = plt.subplots(2, 2, figsize=(7.2, 5.7))
    ymax_full = 0.0
    for mn in ["exp7a", "exp7c"]:
        for pt in full_pts:
            ymax_full = max(ymax_full, results[mn][pt]["delta_mean"] + results[mn][pt]["delta_std"])
    ymax_full *= 1.12

    for col, mn in enumerate(["exp7a", "exp7c"]):
        mr = results[mn]
        ax = axes_full[0, col]
        vals = [mr[pt]["delta_mean"] for pt in full_pts]
        errs = [mr[pt]["delta_std"] for pt in full_pts]
        colors = [PERTURBATION_STYLE[pt]["color"] for pt in full_pts]
        ax.bar(np.arange(len(full_pts)), vals, yerr=errs, capsize=1.8, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xticks(np.arange(len(full_pts)))
        ax.set_xticklabels([PERTURBATION_STYLE[pt]["label"] for pt in full_pts], rotation=25, ha="right")
        ax.set_ylim(0, ymax_full)
        ax.set_ylabel("Mean |\u0394pred| (m/s)" if col == 0 else "")
        ax.set_title(mn)
        style_axes(ax, grid=True)
        add_panel_label(ax, "AB"[col])

        ax2 = axes_full[1, col]
        orig = mr["original"]
        x_all, y_all = [], []
        for pt in full_pts:
            xs, ys = [], []
            for mat in materials:
                if mat in orig and mat in mr[pt]["per_material"]:
                    xs.append(orig[mat])
                    ys.append(float(np.mean(mr[pt]["per_material"][mat])))
            x_all.extend(xs)
            y_all.extend(ys)
            ax2.scatter(xs, ys, s=16, alpha=0.7, color=PERTURBATION_STYLE[pt]["color"], label=PERTURBATION_STYLE[pt]["label"])
        lim_lo, lim_hi = rounded_limits(x_all + y_all, step=250.0, pad=0.03)
        ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=COLORS["ref"], lw=1.0, ls="--")
        ax2.set_xlim(lim_lo, lim_hi)
        ax2.set_ylim(lim_lo, lim_hi)
        ax2.set_xlabel("Original prediction (m/s)")
        ax2.set_ylabel("Perturbed prediction (m/s)" if col == 0 else "")
        ax2.set_title(mn)
        style_axes(ax2, grid=False)
        add_panel_label(ax2, "CD"[col])

    handles, labels = axes_full[1, 1].get_legend_handles_labels()
    axes_full[1, 1].legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig_full.tight_layout()
    save_figure(fig_full, "Fig_M1_composition_probe_full", supplementary=True)
    plt.close(fig_full)
    print("Saved Figure M1 (destructive)")


# ---------------------------------------------------------------------------
# Template invariance -- placeholder (real implementation lands in S8).
# ---------------------------------------------------------------------------

def plot_m1_template(results: dict, output_dir: Path) -> None:
    """Supplementary 2x2 figure for DAP-4 template invariance.

    Top row: per-model scatter of ``v_orig`` (own cluster) vs ``v_template``
    (DAP-4 substituted cluster) with the y=x reference line; MAE / Spearman
    annotated in the corner.

    Bottom-right: bar chart of mean absolute delta per model, with per-fold
    Spearman tick marks overlaid for a robustness sense check.

    The unused bottom-left axis is left blank to preserve the 2x2 grid (so
    the bar chart sits at the same width as the scatter panels).
    """
    template_models = [
        m for m in ("exp7a", "exp7c", "exp7d")
        if "template_dap4" in results.get(m, {})
    ]
    if not template_models:
        print("  [M1 template] no template_dap4 data found, skipping plot")
        return

    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    scatter_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bar_ax = fig.add_subplot(gs[1, :])

    # Compute global axis limits across models.
    all_pts: list[float] = []
    panel_data: dict[str, dict] = {}
    for mn in template_models:
        td = results[mn]["template_dap4"]
        orig = results[mn]["original"]
        pm = td["per_material"]
        xs, ys = [], []
        for mat, preds in pm.items():
            if mat not in orig:
                continue
            xs.append(float(orig[mat]))
            ys.append(float(np.mean(preds)))
        panel_data[mn] = {"x": xs, "y": ys, "td": td}
        all_pts.extend(xs)
        all_pts.extend(ys)
    lim_lo, lim_hi = rounded_limits(all_pts, step=500.0, pad=0.04)

    for ax, mn in zip(scatter_axes, template_models):
        d = panel_data[mn]
        color = COLORS.get(mn, "#444444")
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=COLORS["ref"], lw=1.0, ls="--", zorder=2)
        ax.scatter(
            d["x"], d["y"],
            s=20, alpha=0.85, color=color, edgecolors="white", linewidths=0.4, zorder=4,
        )
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("V_det -- own cluster (m/s)")
        if ax is scatter_axes[0]:
            ax.set_ylabel("V_det -- DAP-4 template (m/s)")
        ax.set_title(disp(mn), loc="left", pad=4)
        td = d["td"]
        ann = (
            f"MAE = {td['delta_mean']:.0f} m/s\n"
            f"Spearman = {td['spearman']:.3f}\n"
            f"n = {td['n_materials']}"
        )
        ax.text(
            0.04, 0.96, ann,
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
        )
        style_axes(ax, grid=False)

    # Panel labels A/B/C across the scatter row.
    for ax, label in zip(scatter_axes, "ABC"):
        add_panel_label(ax, label)

    # Bottom row: mean abs delta + per-fold Spearman scatter overlay.
    x_pos = np.arange(len(template_models))
    means = [results[mn]["template_dap4"]["delta_mean"] for mn in template_models]
    stds = [results[mn]["template_dap4"]["delta_std"] for mn in template_models]
    bar_colors = [COLORS.get(mn, "#444444") for mn in template_models]
    bar_ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.85,
               edgecolor="white", linewidth=0.6, capsize=2.5)

    # Per-fold Spearman dots on the secondary y-axis.
    sp_ax = bar_ax.twinx()
    rng = np.random.default_rng(7)
    for i, mn in enumerate(template_models):
        per_fold_sp = results[mn]["template_dap4"].get("spearman_per_fold", []) or []
        if per_fold_sp:
            xj = i + rng.uniform(-0.18, 0.18, len(per_fold_sp))
            sp_ax.scatter(
                xj, per_fold_sp,
                s=22, color="#222222", marker="x", zorder=5, linewidths=1.0,
            )
        sp_pooled = results[mn]["template_dap4"].get("spearman", float("nan"))
        sp_ax.plot([i - 0.22, i + 0.22], [sp_pooled, sp_pooled],
                   color="#222222", lw=1.6, zorder=6)
    sp_ax.set_ylabel("Spearman (rank, per fold = x, pooled = bar)", labelpad=6)
    sp_ax.set_ylim(0.5, 1.02)
    sp_ax.spines["top"].set_visible(False)

    bar_ax.set_xticks(x_pos)
    bar_ax.set_xticklabels([disp(mn) for mn in template_models])
    bar_ax.set_ylabel("Mean |\u0394V_det| (m/s)")
    bar_ax.set_title("DAP-4 template -- delta and ranking summary",
                     loc="left", pad=4)
    style_axes(bar_ax, grid=True)
    bar_ax.grid(axis="x", visible=False)
    add_panel_label(bar_ax, "D")

    fig.suptitle("M1 -- DAP-4 template invariance (held-out fold)", fontsize=9, y=1.005)
    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M1_template_invariance_supp",
        supplementary=True,
        legacy_png_name="figure_m1_template_invariance.png",
    )
    plt.close(fig)
    print("Saved Figure M1 (template_dap4)")


def plot_m1_direct_comparison(results: dict, output_dir: Path) -> None:
    """Direct perturbation ladder comparison mirroring the Figure 4 logic.

    Delta-only view for the multi-task and single-task pretrained models.
    """
    model_order = [m for m in ("exp7a", "exp7c") if m in results]
    pert_ids = [pid for pid in LADDER_ORDER if any(pid in results.get(m, {}) for m in model_order)]
    if not model_order or not pert_ids:
        print("  [M1 ladder] no perturbation data found, skipping plot")
        return

    x = np.arange(len(pert_ids), dtype=float)
    width = 0.34
    offsets = np.linspace(-width, width, len(model_order)) if len(model_order) > 1 else np.array([0.0])

    fig, ax_delta = plt.subplots(1, 1, figsize=(8.8, 3.8))

    for off, mn in zip(offsets, model_order):
        vals, errs = [], []
        for pid in pert_ids:
            block = results.get(mn, {}).get(pid)
            if block is None:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(float(block.get("delta_mean", np.nan)))
                errs.append(float(block.get("delta_std", 0.0)))
        ax_delta.bar(
            x + off,
            vals,
            width=width,
            yerr=errs,
            capsize=2.0,
            color=COLORS.get(mn, "#666666"),
            alpha=0.86,
            edgecolor="white",
            linewidth=0.5,
            label=disp(mn),
        )

    for cut in (0.5, 2.5, 5.5, 8.5):
        ax_delta.axvline(cut, color="#DDDDDD", lw=0.8, zorder=0)

    ax_delta.set_ylabel("Mean |Δpred| (m/s)")
    ax_delta.set_title("M1 perturbation ladder -- direct comparison", loc="left", pad=4)
    ax_delta.set_yscale("log")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels([
        PERTURBATION_STYLE.get(pid, {}).get("label", pid) for pid in pert_ids
    ], rotation=30, ha="right")
    style_axes(ax_delta, grid=True)
    add_panel_label(ax_delta, "A")
    ax_delta.legend(frameon=False, ncol=len(model_order), loc="upper left")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M1_direct_comparison",
        supplementary=True,
        legacy_png_name="figure_m1_direct_comparison.png",
    )
    plt.close(fig)
    print("Saved Figure M1 (direct comparison)")
