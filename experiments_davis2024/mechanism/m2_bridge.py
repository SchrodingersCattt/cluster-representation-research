"""M2-bridge -- embedding-distance + scaled density probe + head-gradient alignment"""
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


def run_m2_bridge(output_dir: Path) -> None:
    """Bridge M2 (pred vs scale) and M3 (density in embedding): A/B/C from plan."""
    import torch
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score
    from deepmd.dpmodel.output_def import get_reduce_name
    from deepmd.pt.model.model.transform_output import fit_output_to_model_output
    from deepmd.pt.utils import env as pt_env

    print("\n" + "=" * 60 + "\nM2 bridge: embedding distance + scaled probe + head grad\n" + "=" * 60)
    materials = get_materials()
    densities = load_densities()
    device = pt_env.DEVICE
    dtype = pt_env.GLOBAL_PT_FLOAT_PRECISION

    results = {"scale_factors": SCALE_FACTORS, "checkpoint_step": runtime.CKPT_STEP, "checkpoint_fold_ids": list(runtime.ACTIVE_FOLD_IDS)}

    all_fold_embs = []
    density_probe_loo_r2 = {"exp7a": [], "exp7c": []}

    for fi in runtime.ACTIVE_FOLD_IDS:
        print(f"\n--- M2 bridge checkpoint fold {fi} ---")
        fold_mn = {}
        for mn in ["exp7a", "exp7c"]:
            print(f"  descriptors {mn}")
            dp = load_descriptor_model(mn, fi)
            per_scale = {}
            for sc in SCALE_FACTORS:
                ss = f"{sc:.2f}"
                per_scale[ss] = {}
                for mat in materials:
                    sd = paths.MECHANISM_DIR / f"scaled_{ss}" / mat
                    if not sd.exists():
                        continue
                    c, syms, _ = read_cluster_system(sd)
                    per_scale[ss][mat] = extract_descriptor(dp, c, syms)
            fold_mn[mn] = per_scale
            mats_ok = [m for m in materials if m in per_scale.get("1.00", {})]
            y_rho = np.array([densities[m] for m in mats_ok])
            X1 = np.array([per_scale["1.00"][m] for m in mats_ok])
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_predict
            rcv1 = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=5)
            rcv1.fit(X1, y_rho)
            yhat = cross_val_predict(Ridge(alpha=rcv1.alpha_), X1, y_rho, cv=LeaveOneOut())
            density_probe_loo_r2[mn].append(float(r2_score(y_rho, yhat)))
        all_fold_embs.append(fold_mn)

    emb_by_model = {}
    ridge_by_model = {}
    for mn in ["exp7a", "exp7c"]:
        emb_by_model[mn] = {}
        for sc in SCALE_FACTORS:
            ss = f"{sc:.2f}"
            emb_by_model[mn][ss] = {}
            for mat in materials:
                arrs = []
                for fd in all_fold_embs:
                    if mat in fd[mn].get(ss, {}):
                        arrs.append(fd[mn][ss][mat])
                if arrs:
                    emb_by_model[mn][ss][mat] = np.mean(np.stack(arrs, axis=0), axis=0)
        mats_ok = [m for m in materials if m in emb_by_model[mn].get("1.00", {})]
        y_rho = np.array([densities[m] for m in mats_ok])
        X1 = np.array([emb_by_model[mn]["1.00"][m] for m in mats_ok])
        ridge_by_model[mn] = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=LeaveOneOut()).fit(X1, y_rho)
        print(f"  {mn}: density_probe_loo_r2 mean over folds={float(np.mean(density_probe_loo_r2[mn])):.4f}")

    results["density_probe_loo_r2"] = {mn: float(np.mean(density_probe_loo_r2[mn])) for mn in ["exp7a", "exp7c"] if density_probe_loo_r2[mn]}
    results["density_probe_loo_r2_per_fold"] = {mn: density_probe_loo_r2[mn] for mn in ["exp7a", "exp7c"]}

    # A: embedding L2 distance to z(1.0)
    emb_dist: dict[str, dict[str, dict[str, float]]] = {m: {} for m in ["exp7a", "exp7c"]}
    emb_dist_norm: dict[str, dict[str, dict[str, float]]] = {m: {} for m in ["exp7a", "exp7c"]}
    for mn in ["exp7a", "exp7c"]:
        ref = emb_by_model[mn]["1.00"]
        for sc in SCALE_FACTORS:
            ss = f"{sc:.2f}"
            emb_dist[mn][ss] = {}
            emb_dist_norm[mn][ss] = {}
            for mat in materials:
                if mat not in ref or mat not in emb_by_model[mn].get(ss, {}):
                    continue
                delta = float(np.linalg.norm(emb_by_model[mn][ss][mat] - ref[mat]))
                emb_dist[mn][ss][mat] = delta
                emb_dist_norm[mn][ss][mat] = float(delta / (np.linalg.norm(ref[mat]) + 1e-12))
    results["embedding_l2_distance"] = emb_dist
    results["embedding_l2_distance_normalized"] = emb_dist_norm

    # B: probe-predicted density on scaled embeddings
    rho_probe: dict[str, dict[str, dict[str, float]]] = {m: {} for m in ["exp7a", "exp7c"]}
    rho_phys: dict[str, dict[str, float]] = {m: {} for m in ["exp7a", "exp7c"]}
    for mn in ["exp7a", "exp7c"]:
        ridge = ridge_by_model[mn]
        for sc in SCALE_FACTORS:
            ss = f"{sc:.2f}"
            rho_probe[mn][ss] = {}
            rho_phys[mn][ss] = {}
            for mat in materials:
                if mat not in emb_by_model[mn].get(ss, {}):
                    continue
                z = emb_by_model[mn][ss][mat].reshape(1, -1)
                rho_hat = float(ridge.predict(z)[0])
                rho_probe[mn][ss][mat] = rho_hat
                r0 = densities.get(mat)
                if r0 is not None:
                    rho_phys[mn][ss][mat] = float(r0 / (sc**3))
    results["rho_probe_scaled"] = rho_probe
    results["rho_physical_scaled"] = rho_phys

    # C: gradient alignment (pooled sensitivity to density direction)
    grad_align: dict[str, dict[str, float]] = {"exp7a": {}, "exp7c": {}}
    grad_norm: dict[str, dict[str, float]] = {"exp7a": {}, "exp7c": {}}
    grad_cos: dict[str, dict[str, float]] = {"exp7a": {}, "exp7c": {}}

    for mn in ["exp7a", "exp7c"]:
        print(f"\n--- head gradient {mn} ---")
        w = ridge_by_model[mn].coef_.ravel().astype(np.float64)
        w_n = np.linalg.norm(w)
        w_unit = w / (w_n + 1e-12)
        try:
            prop = load_property_model(mn, runtime.ACTIVE_FOLD_IDS[0], no_jit=True)
            pt_model = prop.deep_eval.get_model()
            pt_model.eval()
            atomic = pt_model.atomic_model
            fitting = atomic.fitting_net
            var_name = fitting.var_name
            redu_key = get_reduce_name(var_name)
            out_def = pt_model.atomic_output_def()
            dp_grad = load_descriptor_model(mn, runtime.ACTIVE_FOLD_IDS[0])
        except Exception as e:
            print(f"  skip grad: model load / introspection failed: {e}")
            results["gradient_alignment_error"] = str(e)
            continue

        for mat in materials:
            sd = paths.MECHANISM_DIR / "scaled_1.00" / mat
            if not sd.exists():
                continue
            c, syms, _ = read_cluster_system(sd)
            d_np = extract_descriptor_per_atom(dp_grad, c, syms)
            nloc = d_np.shape[0]
            at_list = [prop.get_type_map().index(s) for s in syms]
            atype_t = torch.tensor(at_list, dtype=torch.int64, device=device).view(1, nloc)

            desc_t = torch.tensor(d_np, dtype=dtype, device=device, requires_grad=True).view(1, nloc, -1)
            coord_dummy = torch.zeros(1, nloc, 3, dtype=dtype, device=device)

            try:
                fit_ret = fitting(desc_t, atype_t)
                if var_name not in fit_ret:
                    raise KeyError(f"missing {var_name} in fitting output keys={list(fit_ret.keys())}")
                stat_ret = {var_name: fit_ret[var_name]}
                stat_ret = atomic.apply_out_stat(stat_ret, atype_t)
                model_ret = fit_output_to_model_output(
                    stat_ret,
                    out_def,
                    coord_dummy,
                    do_atomic_virial=False,
                    create_graph=True,
                    mask=None,
                )
                y = model_ret[redu_key].squeeze()
                if y.ndim > 0:
                    y = y.sum()
                gy = torch.autograd.grad(y, desc_t, retain_graph=False, create_graph=False)[0]
                g_mean = gy[0].mean(dim=0).detach().cpu().numpy()
                gn = float(np.linalg.norm(g_mean))
                ga = float(abs(np.dot(g_mean, w_unit)))
                grad_align[mn][mat] = ga
                grad_norm[mn][mat] = gn
                grad_cos[mn][mat] = float(ga / (gn + 1e-12))
            except Exception as e:
                print(f"  {mat}: grad failed: {e}")
                continue

        if grad_align[mn]:
            print(
                f"  mean |g·ŵ|={np.mean(list(grad_align[mn].values())):.6f}, "
                f"mean ||ḡ||={np.mean(list(grad_norm[mn].values())):.6f}, "
                f"mean cos={np.mean(list(grad_cos[mn].values())):.4f}"
            )

    results["gradient_alignment_abs_dot"] = grad_align
    results["gradient_pooled_norm"] = grad_norm
    results["gradient_alignment_cosine"] = grad_cos

    out_json = output_dir / "mechanism_m2_bridge_results.json"

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, float):
            return obj
        return obj

    out_json.write_text(json.dumps(_json_safe(results), indent=2))
    print(f"Saved {out_json.name}")

    plot_m2_bridge_emb_dist(emb_dist, emb_dist_norm, output_dir)
    plot_m2_bridge_probe(rho_probe, rho_phys, materials, output_dir)
    plot_m2_bridge_grad(grad_align, grad_norm, grad_cos, output_dir)


def plot_m2_bridge_emb_dist(emb_dist: dict, emb_dist_norm: dict, output_dir: Path) -> None:
    materials = get_materials()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for col, mn in enumerate(["exp7a", "exp7c"]):
        ax = axes[col]
        mr = emb_dist[mn]
        for mat in materials:
            xs, ys = [], []
            for sc in sorted(mr.keys(), key=float):
                if mat in mr[sc]:
                    xs.append(float(sc))
                    ys.append(mr[sc][mat])
            if len(xs) > 1:
                ax.plot(xs, ys, color=COLORS["ref"], alpha=0.25, lw=0.7)
        all_x = sorted(set(float(s) for s in mr))
        my = []
        for sx in all_x:
            ss = f"{sx:.2f}"
            vals = [mr[ss][m] for m in materials if m in mr.get(ss, {})]
            my.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(all_x, my, color=COLORS[mn], lw=2.25, zorder=10)
        ax.axvline(1.0, color=COLORS["ref"], lw=1.0, ls=":")
        ax.set_xlabel("Scale factor")
        ax.set_ylabel(r"$\|z(s)-z(1)\|_2$" if col == 0 else "")
        ax.set_title(mn)
        style_axes(ax, grid=False)
        add_panel_label(ax, "AB"[col])
    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M2_bridgeA_embedding_shift_supp",
        supplementary=True,
        legacy_png_name="figure_m2_embedding_distance.png",
    )
    plt.close(fig)

    fig_n, axes_n = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for col, mn in enumerate(["exp7a", "exp7c"]):
        ax = axes_n[col]
        mr = emb_dist_norm[mn]
        for mat in materials:
            xs, ys = [], []
            for sc in sorted(mr.keys(), key=float):
                if mat in mr[sc]:
                    xs.append(float(sc))
                    ys.append(mr[sc][mat])
            if len(xs) > 1:
                ax.plot(xs, ys, color=COLORS["ref"], alpha=0.25, lw=0.7)
        all_x = sorted(set(float(s) for s in mr))
        my = []
        for sx in all_x:
            ss = f"{sx:.2f}"
            vals = [mr[ss][m] for m in materials if m in mr.get(ss, {})]
            my.append(float(np.mean(vals)) if vals else np.nan)
        ax.plot(all_x, my, color=COLORS[mn], lw=2.25, zorder=10)
        ax.axvline(1.0, color=COLORS["ref"], lw=1.0, ls=":")
        ax.set_xlabel("Scale factor")
        ax.set_ylabel(r"$\|z(s)-z(1)\|_2 / \|z(1)\|_2$" if col == 0 else "")
        ax.set_title(mn)
        style_axes(ax, grid=False)
        add_panel_label(ax, "AB"[col])
    fig_n.tight_layout()
    save_figure(fig_n, "Fig_M2_bridgeA_embedding_shift_normalized_supp", supplementary=True)
    plt.close(fig_n)
    print("Saved figure_m2_embedding_distance.png")


def plot_m2_bridge_probe(
    rho_probe: dict,
    rho_phys: dict,
    materials: list[str],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for col, mn in enumerate(["exp7a", "exp7c"]):
        ax = axes[col]
        pr, ph = rho_probe[mn], rho_phys[mn]
        for mat in materials:
            xs, yp, yh = [], [], []
            for ss in sorted(pr.keys(), key=float):
                if mat in pr.get(ss, {}) and mat in ph.get(ss, {}):
                    xs.append(float(ss))
                    yp.append(pr[ss][mat])
                    yh.append(ph[ss][mat])
            if len(xs) > 1:
                ax.plot(xs, yp, color=COLORS[mn], alpha=0.2, lw=0.7)
                ax.plot(xs, yh, color=COLORS["kj"], alpha=0.18, lw=0.7, ls="--")
        all_x = sorted(set(float(s) for s in pr))
        mp, mh = [], []
        for sx in all_x:
            ss = f"{sx:.2f}"
            vp = [pr[ss][m] for m in materials if m in pr.get(ss, {})]
            vh = [ph[ss][m] for m in materials if m in ph.get(ss, {})]
            mp.append(float(np.mean(vp)) if vp else np.nan)
            mh.append(float(np.mean(vh)) if vh else np.nan)
        ax.plot(all_x, mp, color=COLORS[mn], lw=2.25, label="Probe mean")
        ax.plot(all_x, mh, color=COLORS["kj"], lw=1.2, ls="--", label=r"Physical $ \rho_0 / s^3 $")
        ax.set_xlabel("Scale factor")
        ax.set_ylabel("Density (g/cm³)" if col == 0 else "")
        ax.set_title(mn)
        ax.legend(frameon=False, loc="upper left")
        ax.axvline(1.0, color=COLORS["ref"], lw=1.0, ls=":")
        style_axes(ax, grid=False)
        add_panel_label(ax, "AB"[col])
    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M2_bridgeB_density_probe_supp",
        supplementary=True,
        legacy_png_name="figure_m2_density_probe_scaled.png",
    )
    plt.close(fig)
    print("Saved figure_m2_density_probe_scaled.png")


def plot_m2_bridge_grad(grad_align: dict, grad_norm: dict, grad_cos: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.0))
    labs, means_c, scatter_x, scatter_y = [], [], [], []
    for mn in ["exp7a", "exp7c"]:
        if not grad_align.get(mn):
            continue
        vc = list(grad_cos[mn].values())
        labs.append(mn)
        means_c.append(float(np.mean(vc)))
        xi = len(labs) - 1
        scatter_x.extend([xi] * len(vc))
        scatter_y.extend(vc)
    x = np.arange(len(labs))
    ax.bar(x, means_c, 0.55, color=[COLORS["exp7a"], COLORS["exp7c"]][: len(labs)], edgecolor="white", linewidth=0.5, zorder=2)
    for xi, yi in zip(scatter_x, scatter_y):
        color = COLORS["exp7a"] if xi == 0 else COLORS["exp7c"]
        ax.scatter(xi, yi, s=12, color=color, alpha=0.25, edgecolors="none", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("Absolute alignment score")
    ax.set_title("Normalized projection on density direction")
    ax.set_ylim(0, max(means_c) * 1.2 if means_c else 1.0)
    style_axes(ax, grid=True)
    add_panel_label(ax, "A")
    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M2_bridgeC_alignment_supp",
        supplementary=True,
        legacy_png_name="figure_m2_head_gradient_alignment.png",
    )
    plt.close(fig)
    print("Saved figure_m2_head_gradient_alignment.png")
