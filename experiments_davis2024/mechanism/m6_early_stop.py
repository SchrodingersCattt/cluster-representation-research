"""M6 -- checkpoint early-stop diagnostics for MT/ST PEM models.

This module deliberately keeps the early-stop analysis separate from the main
PEM inference pipeline. It sweeps sparse checkpoints, caches descriptors and
predictions, and relates apparent accuracy to representation retention/drift.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from . import constants, paths, runtime
from .inference import extract_descriptor
from .io_data import (
    compute_composition_and_ob,
    get_family,
    get_materials,
    get_m1_heldout_mats,
    load_densities,
    load_gt_vdet,
    read_cluster_system,
)
from .plot_helpers import add_panel_label, save_figure, setup_nature_style, style_axes


DEFAULT_MODELS = constants.EARLY_STOP_DEFAULT_MODELS
OPTIONAL_MODELS = constants.EARLY_STOP_OPTIONAL_MODELS
DEFAULT_STEPS = constants.EARLY_STOP_DEFAULT_STEPS
VARIANTS = ["cluster_n1", "cluster_n2", "cluster_n3"]
MODEL_COLORS = {
    "exp7a": constants.COLORS["exp7a"],
    "exp7c": constants.COLORS["exp7c"],
    "exp7d": constants.COLORS["exp7d"],
    "exp7a_lr1e4": "#3D7FA3",
    "exp7c_lr1e4": "#B33A60",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    exp_root: Path
    property_head: str | None
    descriptor_head: str | None
    color: str


def _model_spec(model_name: str) -> ModelSpec:
    if model_name == "exp7a":
        return ModelSpec(model_name, constants.MODEL_DISPLAY_NAMES.get(model_name, model_name), paths.ROOT, "pems_vdet_kj", "deepems_vanilla", MODEL_COLORS[model_name])
    if model_name == "exp7c":
        return ModelSpec(model_name, constants.MODEL_DISPLAY_NAMES.get(model_name, model_name), paths.ROOT, None, None, MODEL_COLORS[model_name])
    if model_name == "exp7d":
        return ModelSpec(model_name, constants.MODEL_DISPLAY_NAMES.get(model_name, model_name), paths.ROOT, "pems_vdet_kj", None, MODEL_COLORS[model_name])
    if model_name == "exp7a_lr1e4":
        return ModelSpec(model_name, constants.MODEL_DISPLAY_NAMES.get(model_name, model_name), paths.ROOT / "ablation", "pems_vdet_kj", "deepems_vanilla", MODEL_COLORS[model_name])
    if model_name == "exp7c_lr1e4":
        return ModelSpec(model_name, constants.MODEL_DISPLAY_NAMES.get(model_name, model_name), paths.ROOT / "ablation", None, None, MODEL_COLORS[model_name])
    raise KeyError(f"Unknown early-stop model: {model_name}")


def _fold_dir(spec: ModelSpec, fold: int) -> Path:
    return spec.exp_root / f"{spec.name}_fold{fold}"


def _ckpt_path(spec: ModelSpec, fold: int, step: int) -> Path | None:
    p = _fold_dir(spec, fold) / f"model.ckpt-{step}.pt"
    return p if p.exists() else None


def _discover_steps(models: list[str], requested_steps: list[int], folds: list[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for model in models:
        spec = _model_spec(model)
        available = []
        for step in requested_steps:
            if any(_ckpt_path(spec, fold, step) is not None for fold in folds):
                available.append(step)
        out[model] = available
    return out


def _safe_mean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _safe_std(vals: list[float], *, ddof: int = 0) -> float | None:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    if ddof and len(vals) <= ddof:
        return 0.0
    return float(np.std(vals, ddof=ddof))


def _system_hash(names: list[str], symbols_by_name: dict[str, list[str]]) -> str:
    h = hashlib.sha1()
    for name in names:
        h.update(name.encode())
        h.update("\0".join(symbols_by_name[name]).encode())
    return h.hexdigest()[:12]


def _variant_dir(variant: str) -> Path:
    suffix = variant.replace("cluster_", "")
    return paths.DATA_ROOT / f"pems_cluster_{suffix}_systems"


def _load_train_variant_systems(materials: list[str]) -> dict[str, dict[str, tuple[np.ndarray, list[str], float]]]:
    systems: dict[str, dict[str, tuple[np.ndarray, list[str], float]]] = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        base = _variant_dir(variant)
        for mat in materials:
            sys_dir = base / mat
            if sys_dir.exists():
                systems[variant][mat] = read_cluster_system(sys_dir)
    return systems


def _composition_targets(materials: list[str]) -> dict[str, np.ndarray]:
    gt = load_gt_vdet()
    comp, ob_values = compute_composition_and_ob(materials)
    densities = load_densities()
    targets: dict[str, np.ndarray] = {
        "Vdet": np.array([gt.get(m, np.nan) for m in materials], dtype=float),
        "frac_N": np.array([comp[m]["N"] for m in materials], dtype=float),
        "frac_O": np.array([comp[m]["O"] for m in materials], dtype=float),
        "frac_C": np.array([comp[m]["C"] for m in materials], dtype=float),
        "frac_H": np.array([comp[m]["H"] for m in materials], dtype=float),
        "n_atoms": np.array([float(comp[m]["n_atoms"]) for m in materials], dtype=float),
        "density": np.array([densities.get(m, np.nan) for m in materials], dtype=float),
        "OB_signed": np.array([ob_values.get(m, np.nan) for m in materials], dtype=float),
        "OB_abs": np.array([abs(ob_values[m]) if m in ob_values else np.nan for m in materials], dtype=float),
    }
    return targets


def _load_exp_ood_values() -> dict[str, float]:
    p = paths.ROOT / "ood_experimental_values.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.get("values_m_s", {}).items()}


def _load_stress_clusters() -> dict[str, dict[str, Any]]:
    """Build direct A2BX5/ABX4 stress clusters from CIFs when available."""
    import predict_eap4_paph6 as ood_new
    from ase.io import read as ase_read
    from molcrys_kit.structures.crystal import MolecularCrystal

    cif_dir = paths.DATA_ROOT / "pems_cleaned_cifs"
    candidates = {
        "EAP-4": cif_dir / "EAP-4.cif",
        "EAP-8_old_ordered": cif_dir / "EAP-8.cif",
        "EAP-8_new_optimal": cif_dir / "EAP-8-new-ordered.cif",
        "EAP-8_random_best": cif_dir / "EAP-8-new-random-replicas" / "EAP-8-new-random-021.cif",
        "SY": cif_dir / "SY.cif",
        "PEP": cif_dir / "PEP.cif",
        "MPEP": cif_dir / "MPEP.cif",
        "HPEP": cif_dir / "HPEP.cif",
    }
    out: dict[str, dict[str, Any]] = {}
    for name, cif in candidates.items():
        if not cif.exists():
            continue
        try:
            crystal = MolecularCrystal.from_ase(ase_read(str(cif)), bond_thresholds=ood_new.PEM_BOND_THRESHOLDS)
            variants: dict[str, dict[str, Any]] = {}
            for variant in VARIANTS:
                seed = ood_new.CLUSTER_RANDOM_SEEDS[variant]
                cluster_crystal, seed_idx, sc_dims = ood_new.build_seeded_stoichiometric_cluster(crystal, dataset_name=variant, seed=seed)
                atoms = ood_new.crystal_to_minimum_image_atoms(cluster_crystal)
                atoms = ood_new.cluster_atoms_centered(atoms)
                variants[variant] = {
                    "coord": atoms.get_positions(),
                    "symbols": atoms.get_chemical_symbols(),
                    "seed_idx": int(seed_idx),
                    "supercell_dims": list(sc_dims),
                    "n_atoms": int(len(atoms)),
                }
            out[name] = {"cif": str(cif), "variants": variants}
        except Exception as exc:
            out[name] = {"cif": str(cif), "error": repr(exc), "variants": {}}

    # Zou2026 clusters are generated in isolated pems_zou2026_* directories by
    # the OOD_Zou2026 workflow. Reuse them if present; do not rebuild disordered
    # CIFs here because that belongs to the data-prep pipeline.
    zou_bases = {
        variant: paths.DATA_ROOT / f"pems_zou2026_{variant}_systems"
        for variant in VARIANTS
    }
    if any(base.exists() for base in zou_bases.values()):
        material_names = set()
        for base in zou_bases.values():
            if base.exists():
                material_names.update(p.name for p in base.iterdir() if p.is_dir())
        for mat in sorted(material_names):
            variants = out.setdefault(f"Zou2026:{mat}", {"cif": None, "variants": {}})["variants"]
            for variant, base in zou_bases.items():
                sys_dir = base / mat
                if sys_dir.exists():
                    coord, symbols, _prop = read_cluster_system(sys_dir)
                    variants[variant] = {
                        "coord": coord,
                        "symbols": symbols,
                        "seed_idx": None,
                        "supercell_dims": None,
                        "n_atoms": int(len(symbols)),
                    }
    return out


def _predict_coords(model: Any, type_map: list[str], coord: np.ndarray, symbols: list[str]) -> float:
    at = np.array([type_map.index(s) for s in symbols], dtype=np.int32)
    return float(model.eval(coords=np.asarray(coord, dtype=np.float64).reshape(1, -1, 3), atom_types=at, cells=None)[0].reshape(-1)[0])


def _prediction_cache_path(cache_dir: Path, model: str, fold: int, step: int) -> Path:
    return cache_dir / "predictions" / model / f"fold{fold}" / f"step{step}.json"


def _descriptor_cache_path(cache_dir: Path, model: str, fold: int, step: int) -> Path:
    return cache_dir / "descriptors" / model / f"fold{fold}" / f"step{step}.npz"


def _load_property_model_for_ckpt(spec: ModelSpec, ckpt: Path):
    from deepmd.pt.infer.deep_eval import DeepProperty
    kwargs = {"head": spec.property_head} if spec.property_head else {}
    return DeepProperty(str(ckpt), **kwargs)


def _load_descriptor_model_for_ckpt(spec: ModelSpec, ckpt: Path):
    from deepmd.infer import DeepPot
    kwargs = {"head": spec.descriptor_head} if spec.descriptor_head else {}
    return DeepPot(str(ckpt), **kwargs)


def _extract_descriptor_matrix(
    spec: ModelSpec,
    ckpt: Path,
    systems: dict[str, tuple[np.ndarray, list[str]]],
    cache_path: Path,
    *,
    refresh_cache: bool,
) -> tuple[list[str], np.ndarray]:
    names = sorted(systems)
    symbols_by_name = {name: systems[name][1] for name in names}
    sys_hash = _system_hash(names, symbols_by_name)
    if cache_path.exists() and not refresh_cache:
        data = np.load(cache_path, allow_pickle=False)
        if str(data["checkpoint"]) == str(ckpt) and str(data["system_hash"]) == sys_hash:
            return [str(x) for x in data["names"].tolist()], np.asarray(data["descriptors"], dtype=float)
    dp = _load_descriptor_model_for_ckpt(spec, ckpt)
    X = []
    for name in names:
        coord, symbols = systems[name]
        X.append(extract_descriptor(dp, coord, symbols))
    arr = np.asarray(X, dtype=float)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        names=np.array(names, dtype=str),
        descriptors=arr,
        checkpoint=str(ckpt),
        system_hash=sys_hash,
    )
    return names, arr


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    xx = np.linalg.norm(Xc.T @ Xc, ord="fro")
    yy = np.linalg.norm(Yc.T @ Yc, ord="fro")
    if xx <= 1e-12 or yy <= 1e-12:
        return float("nan")
    return float(xy / (xx * yy))


def _cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 1e-12)


def _ridge_probe_metrics(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    ok = np.isfinite(y)
    if int(ok.sum()) < 5:
        return {"r2": float("nan"), "mae": float("nan"), "alpha": float("nan")}
    Xv, yv = X[ok], y[ok]
    rcv = RidgeCV(alphas=np.array(constants.RIDGE_ALPHAS), cv=min(5, int(ok.sum())))
    rcv.fit(Xv, yv)
    pred = cross_val_predict(Ridge(alpha=rcv.alpha_), Xv, yv, cv=LeaveOneOut())
    return {
        "r2": float(r2_score(yv, pred)),
        "mae": float(mean_absolute_error(yv, pred)),
        "alpha": float(rcv.alpha_),
        "train_min": float(np.min(yv)),
        "train_max": float(np.max(yv)),
    }


def _fit_ridge_projection(X: np.ndarray, y: np.ndarray, X_ood: dict[str, np.ndarray]) -> dict[str, Any]:
    from sklearn.linear_model import Ridge, RidgeCV

    ok = np.isfinite(y)
    if int(ok.sum()) < 5:
        return {"ood_predictions": {}}
    Xv, yv = X[ok], y[ok]
    rcv = RidgeCV(alphas=np.array(constants.RIDGE_ALPHAS), cv=min(5, int(ok.sum())))
    rcv.fit(Xv, yv)
    model = Ridge(alpha=rcv.alpha_)
    model.fit(Xv, yv)
    return {
        "alpha": float(rcv.alpha_),
        "ood_predictions": {name: float(model.predict(vec.reshape(1, -1))[0]) for name, vec in X_ood.items()},
    }


def _family_silhouette(X: np.ndarray, materials: list[str]) -> float:
    try:
        from sklearn.metrics import silhouette_score
        labels = [get_family(m) for m in materials]
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            return float("nan")
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def _pool_prediction_errors(preds: dict[str, dict[str, float]], gt: dict[str, float]) -> dict[str, Any]:
    rows = []
    for mat, by_variant in preds.items():
        vals = [v for v in by_variant.values() if v is not None and np.isfinite(v)]
        if not vals:
            continue
        y_pred = float(np.mean(vals))
        y_true = gt.get(mat)
        row = {"material": mat, "pred_m_s": y_pred, "cluster_std_m_s": _safe_std(vals, ddof=0)}
        if y_true is not None and np.isfinite(y_true):
            row.update({"true_m_s": float(y_true), "error_m_s": y_pred - float(y_true), "abs_error_m_s": abs(y_pred - float(y_true))})
        rows.append(row)
    errors = [r["abs_error_m_s"] for r in rows if "abs_error_m_s" in r]
    return {
        "mae_m_s": _safe_mean(errors),
        "rmse_m_s": float(math.sqrt(np.mean([e * e for e in [r["error_m_s"] for r in rows if "error_m_s" in r]]))) if errors else None,
        "n_labeled": len(errors),
        "rows": rows,
    }


def _make_prediction_cache(
    spec: ModelSpec,
    ckpt: Path,
    fold: int,
    train_systems: dict[str, dict[str, tuple[np.ndarray, list[str], float]]],
    stress_clusters: dict[str, dict[str, Any]],
    gt: dict[str, float],
    cache_path: Path,
    *,
    refresh_cache: bool,
) -> dict[str, Any]:
    if cache_path.exists() and not refresh_cache:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("checkpoint") == str(ckpt):
            return data
    model = _load_property_model_for_ckpt(spec, ckpt)
    tm = model.get_type_map()

    heldout = set(get_m1_heldout_mats(fold))
    ind_preds: dict[str, dict[str, float]] = {}
    for variant, systems in train_systems.items():
        for mat in heldout:
            if mat not in systems:
                continue
            coord, symbols, _prop = systems[mat]
            ind_preds.setdefault(mat, {})[variant] = _predict_coords(model, tm, coord, symbols)

    stress_preds: dict[str, dict[str, float]] = {}
    for mat, payload in stress_clusters.items():
        for variant, cluster in payload.get("variants", {}).items():
            try:
                stress_preds.setdefault(mat, {})[variant] = _predict_coords(model, tm, cluster["coord"], cluster["symbols"])
            except Exception:
                continue

    data = {
        "model": spec.name,
        "fold": fold,
        "checkpoint": str(ckpt),
        "ind": _pool_prediction_errors(ind_preds, gt),
        "stress_predictions": stress_preds,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def _aggregate_prediction_metrics(fold_entries: list[dict[str, Any]], ood_values: dict[str, float]) -> dict[str, Any]:
    ind_maes = [entry["ind"].get("mae_m_s") for entry in fold_entries if entry.get("ind")]
    ind_rmses = [entry["ind"].get("rmse_m_s") for entry in fold_entries if entry.get("ind")]

    stress_by_mat: dict[str, dict[str, list[float]]] = {}
    for entry in fold_entries:
        for mat, by_variant in entry.get("stress_predictions", {}).items():
            vals = [float(v) for v in by_variant.values() if np.isfinite(v)]
            if vals:
                stress_by_mat.setdefault(mat, {}).setdefault("fold_means", []).append(float(np.mean(vals)))
                stress_by_mat[mat].setdefault("cluster_stds", []).append(float(np.std(vals)))

    stress_rows = []
    for mat, vals in sorted(stress_by_mat.items()):
        fold_means = vals.get("fold_means", [])
        pred = _safe_mean(fold_means)
        row = {
            "material": mat,
            "pred_mean_m_s": pred,
            "model_std_m_s": _safe_std(fold_means, ddof=1),
            "cluster_std_mean_m_s": _safe_mean(vals.get("cluster_stds", [])),
            "n_folds": len(fold_means),
        }
        if mat in ood_values and pred is not None:
            row["exp_m_s"] = ood_values[mat]
            row["error_m_s"] = pred - ood_values[mat]
            row["abs_error_m_s"] = abs(pred - ood_values[mat])
        stress_rows.append(row)

    labeled_errors = [r["abs_error_m_s"] for r in stress_rows if "abs_error_m_s" in r]
    stress_lookup = {r["material"]: r for r in stress_rows}
    eap_delta = None
    if "EAP-4" in stress_lookup and "EAP-8_old_ordered" in stress_lookup:
        if stress_lookup["EAP-4"].get("pred_mean_m_s") is not None and stress_lookup["EAP-8_old_ordered"].get("pred_mean_m_s") is not None:
            eap_delta = stress_lookup["EAP-8_old_ordered"]["pred_mean_m_s"] - stress_lookup["EAP-4"]["pred_mean_m_s"]

    return {
        "ind_mae_m_s": _safe_mean(ind_maes),
        "ind_rmse_m_s": _safe_mean(ind_rmses),
        "ind_mae_std_across_folds_m_s": _safe_std(ind_maes, ddof=1),
        "ood_labeled_mae_m_s": _safe_mean(labeled_errors),
        "stress_rows": stress_rows,
        "eap8_minus_eap4_m_s": eap_delta,
    }


def _aggregate_descriptors_by_fold(desc_by_fold: dict[int, tuple[list[str], np.ndarray]]) -> tuple[list[str], np.ndarray]:
    common = None
    by_name = {}
    for fold, (names, X) in desc_by_fold.items():
        s = set(names)
        common = s if common is None else common & s
        by_name[fold] = {name: X[i] for i, name in enumerate(names)}
    if not common:
        return [], np.zeros((0, 0), dtype=float)
    ordered = sorted(common)
    mats = []
    for name in ordered:
        mats.append(np.mean(np.stack([by_name[fold][name] for fold in by_name], axis=0), axis=0))
    return ordered, np.asarray(mats, dtype=float)


def _plot_accuracy(metrics: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for model, mdata in metrics.items():
        steps = sorted(int(s) for s in mdata)
        y_ind = [mdata[str(s)]["prediction_metrics"].get("ind_mae_m_s", np.nan) for s in steps]
        y_ood = [mdata[str(s)]["prediction_metrics"].get("ood_labeled_mae_m_s", np.nan) for s in steps]
        color = MODEL_COLORS.get(model, "#555555")
        label = _model_spec(model).label if model in MODEL_COLORS else model
        ax.plot(steps, y_ind, marker="o", color=color, lw=1.4, label=f"{label} IND")
        if any(np.isfinite(y_ood)):
            ax.plot(steps, y_ood, marker="s", color=color, lw=1.1, ls="--", label=f"{label} OOD")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("MAE (m s$^{-1}$)")
    ax.set_title("Checkpoint accuracy", loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    style_axes(ax, grid=True)
    fig.tight_layout()
    _save_m6_fig(fig, out_dir, "Fig_M6a_checkpoint_accuracy", "figure_m6a_checkpoint_accuracy.png")
    plt.close(fig)


def _plot_probe_retention(metrics: dict[str, Any], out_dir: Path) -> None:
    targets = ["Vdet", "density", "OB_signed", "OB_abs", "frac_O"]
    fig, axes = plt.subplots(1, len(targets), figsize=(2.2 * len(targets), 2.7), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        for model, mdata in metrics.items():
            steps = sorted(int(s) for s in mdata)
            y = [mdata[str(s)].get("probe_metrics", {}).get(target, {}).get("r2", np.nan) for s in steps]
            ax.plot(steps, y, marker="o", lw=1.2, color=MODEL_COLORS.get(model, "#555555"), label=model)
        ax.set_title(target, loc="left", fontsize=9)
        ax.set_xlabel("step")
        ax.axhline(0.0, color="#444444", lw=0.7, ls=":")
        style_axes(ax, grid=True)
    axes[0].set_ylabel("LOO $R^2$")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    _save_m6_fig(fig, out_dir, "Fig_M6b_probe_retention", "figure_m6b_probe_retention.png")
    plt.close(fig)


def _plot_embedding_drift(metrics: dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    for model, mdata in metrics.items():
        steps = sorted(int(s) for s in mdata)
        color = MODEL_COLORS.get(model, "#555555")
        axes[0].plot(steps, [mdata[str(s)].get("geometry", {}).get("family_silhouette", np.nan) for s in steps], marker="o", color=color, label=model)
        axes[1].plot(steps, [mdata[str(s)].get("geometry", {}).get("drift_cosine_to_initial", np.nan) for s in steps], marker="o", color=color)
        axes[2].plot(steps, [mdata[str(s)].get("geometry", {}).get("cross_fold_cosine", np.nan) for s in steps], marker="o", color=color)
    titles = ["Family silhouette", "Cosine to initial", "Cross-fold cosine"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("step")
        style_axes(ax, grid=True)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    _save_m6_fig(fig, out_dir, "Fig_M6c_embedding_drift", "figure_m6c_embedding_drift.png")
    plt.close(fig)


def _plot_eap_stress(metrics: dict[str, Any], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for model, mdata in metrics.items():
        steps = sorted(int(s) for s in mdata)
        y = [mdata[str(s)].get("prediction_metrics", {}).get("eap8_minus_eap4_m_s", np.nan) for s in steps]
        ax.plot(steps, y, marker="o", lw=1.3, color=MODEL_COLORS.get(model, "#555555"), label=model)
    ax.axhline(0.0, color="#444444", lw=0.8, ls=":")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Pred(EAP-8) - Pred(EAP-4) (m s$^{-1}$)")
    ax.set_title("EAP stress trajectory", loc="left")
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax, grid=True)
    fig.tight_layout()
    _save_m6_fig(fig, out_dir, "Fig_M6e_eap_stress", "figure_m6e_eap_stress.png")
    plt.close(fig)


def _plot_umap_snapshots(mean_descriptors: dict[str, dict[int, tuple[list[str], np.ndarray]]], gt: dict[str, float], out_dir: Path) -> None:
    snapshots = []
    labels = []
    values = []
    for model, by_step in mean_descriptors.items():
        if not by_step:
            continue
        chosen = sorted(by_step)[:2] + sorted(by_step)[-2:]
        seen = set()
        for step in chosen:
            if step in seen:
                continue
            seen.add(step)
            names, X = by_step[step]
            if len(names) == 0:
                continue
            snapshots.append((model, step, names, X))
            labels.extend([(model, step, name) for name in names])
            values.append(X)
    if not values:
        return
    Xall = np.vstack(values)
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=max(2, min(12, len(Xall) - 1)))
        X2all = reducer.fit_transform(Xall)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        X2all = PCA(n_components=2, random_state=42).fit_transform(Xall)
        method = "PCA"
    offset = 0
    n = len(snapshots)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)
    vdet_vals = np.array([gt.get(name, np.nan) for _model, _step, name in labels], dtype=float)
    vmin, vmax = float(np.nanmin(vdet_vals)), float(np.nanmax(vdet_vals))
    for ax in axes.ravel():
        ax.axis("off")
    for panel_idx, (model, step, names, X) in enumerate(snapshots):
        ax = axes.ravel()[panel_idx]
        Xi = X2all[offset: offset + len(names)]
        vi = np.array([gt.get(name, np.nan) for name in names], dtype=float)
        sc = ax.scatter(Xi[:, 0], Xi[:, 1], c=vi, cmap="viridis", vmin=vmin, vmax=vmax, s=20, edgecolors="white", linewidths=0.25)
        ax.set_title(f"{model} step {step}", loc="left", fontsize=8)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        style_axes(ax, grid=False)
        offset += len(names)
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Vdet (m/s)")
    _save_m6_fig(fig, out_dir, "Fig_M6d_umap_snapshots", "figure_m6d_umap_snapshots.png")
    plt.close(fig)


def _save_umap_inputs(mean_descriptors: dict[str, dict[int, tuple[list[str], np.ndarray]]], out_dir: Path) -> None:
    rows: list[tuple[str, int, str]] = []
    arrays: list[np.ndarray] = []
    for model, by_step in mean_descriptors.items():
        for step, (names, X) in by_step.items():
            for i, name in enumerate(names):
                rows.append((model, int(step), name))
                arrays.append(np.asarray(X[i], dtype=float))
    if not arrays:
        return
    target = out_dir / "umap_snapshot_inputs.npz"
    np.savez_compressed(
        target,
        descriptors=np.vstack(arrays),
        model=np.array([r[0] for r in rows], dtype=str),
        step=np.array([r[1] for r in rows], dtype=int),
        material=np.array([r[2] for r in rows], dtype=str),
    )


def _save_m6_fig(fig, out_dir: Path, paper_name: str, legacy_name: str) -> None:
    if runtime.PLOT_STYLE == "paper":
        target = out_dir / "figures"
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{paper_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(target / f"{paper_name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / legacy_name, dpi=300, bbox_inches="tight")


def _select_recommended_steps(metrics: dict[str, Any]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for model, by_step in metrics.items():
        candidates = []
        for step_s, cell in by_step.items():
            pm = cell.get("prediction_metrics", {})
            probes = cell.get("probe_metrics", {})
            geom = cell.get("geometry", {})
            score_parts = []
            if pm.get("ood_labeled_mae_m_s") is not None:
                score_parts.append(-pm["ood_labeled_mae_m_s"] / 100.0)
            elif pm.get("ind_mae_m_s") is not None:
                score_parts.append(-pm["ind_mae_m_s"] / 100.0)
            for t in ["Vdet", "density", "OB_signed", "frac_O"]:
                r2 = probes.get(t, {}).get("r2")
                if r2 is not None and np.isfinite(r2):
                    score_parts.append(float(r2))
            if geom.get("cross_fold_cosine") is not None and np.isfinite(geom["cross_fold_cosine"]):
                score_parts.append(float(geom["cross_fold_cosine"]))
            score = float(np.mean(score_parts)) if score_parts else float("nan")
            candidates.append((score, int(step_s), cell))
        candidates = [c for c in candidates if np.isfinite(c[0])]
        if not candidates:
            continue
        candidates.sort(reverse=True, key=lambda x: x[0])
        best = candidates[0]
        rec[model] = {
            "recommended_step": best[1],
            "score": best[0],
            "criterion": "max mean of normalized OOD/IND accuracy, physical-probe R2, and cross-fold descriptor stability",
            "ind_mae_m_s": best[2].get("prediction_metrics", {}).get("ind_mae_m_s"),
            "ood_labeled_mae_m_s": best[2].get("prediction_metrics", {}).get("ood_labeled_mae_m_s"),
            "eap8_minus_eap4_m_s": best[2].get("prediction_metrics", {}).get("eap8_minus_eap4_m_s"),
        }
    return rec


def _write_interpretation(summary: dict[str, Any], out_dir: Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# M6 Early-Stop Interpretation",
        "",
        "This diagnostic should be read as an early-stop screen, not as a replacement for external validation.",
        "A useful checkpoint is one that keeps OOD error low while retaining readable physical probes and stable cross-fold geometry.",
        "",
        "## Recommended Checkpoints",
        "",
    ]
    recommended = summary.get("recommended", {})
    if recommended:
        for model, rec in recommended.items():
            lines.append(
                f"- `{model}`: step `{rec.get('recommended_step')}` "
                f"(IND MAE={fmt(rec.get('ind_mae_m_s'))} m/s, "
                f"OOD MAE={fmt(rec.get('ood_labeled_mae_m_s'))} m/s, "
                f"EAP8-EAP4={fmt(rec.get('eap8_minus_eap4_m_s'))} m/s)."
            )
    else:
        lines.append("- No checkpoint recommendation was available because no finite metrics were produced.")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- EAP-4/EAP-8 are stress tests for extrapolating across the OB=0 boundary; EAP-4 agreement alone is not evidence that the model ranks EAP-8 correctly.",
            "- Training signed OB is entirely negative, so signed-OB and absOB probes are collinear on the training manifold.",
            "- Prefer a checkpoint before OOD MAE degrades, before density/OB/O-fraction probe R2 collapses, and before ST descriptors drift far from the early/pretrained geometry.",
            "",
        ]
    )
    (out_dir / "early_stop_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def run_m6(
    output_dir: Path,
    *,
    models: str | None = None,
    steps: str | None = None,
    no_umap: bool = False,
    refresh_cache: bool = False,
) -> None:
    """Run checkpoint early-stop diagnostics."""
    print("\n" + "=" * 60 + "\nM6: Early-stop checkpoint diagnostics\n" + "=" * 60)
    setup_nature_style()
    out_dir = output_dir / "early_stop"
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in (models.split(",") if models else DEFAULT_MODELS) if m.strip()]
    requested_steps = [int(s.strip()) for s in (steps.split(",") if steps else [str(s) for s in DEFAULT_STEPS]) if str(s).strip()]
    folds = list(runtime.ACTIVE_FOLD_IDS)
    gt = load_gt_vdet()
    materials = get_materials()
    train_systems = _load_train_variant_systems(materials)
    targets = _composition_targets(materials)
    ood_values = _load_exp_ood_values()
    stress_clusters = _load_stress_clusters()
    step_grid = _discover_steps(model_names, requested_steps, folds)

    # Fixed cluster_n1 systems for descriptors/probes.
    train_n1_systems = {
        mat: (train_systems["cluster_n1"][mat][0], train_systems["cluster_n1"][mat][1])
        for mat in materials
        if mat in train_systems["cluster_n1"]
    }
    train_names = sorted(train_n1_systems)
    y_by_target = {k: np.array([v[materials.index(m)] if m in materials else np.nan for m in train_names], dtype=float) for k, v in targets.items()}

    # Stress n1/mean systems for OOD linear-probe projections.
    stress_n1_systems: dict[str, tuple[np.ndarray, list[str]]] = {}
    stress_mean_vectors: dict[str, list[np.ndarray]] = {}
    for mat, payload in stress_clusters.items():
        variants = payload.get("variants", {})
        if "cluster_n1" in variants:
            stress_n1_systems[mat] = (variants["cluster_n1"]["coord"], variants["cluster_n1"]["symbols"])
        stress_mean_vectors[mat] = []

    metrics: dict[str, dict[str, Any]] = {}
    mean_descriptors: dict[str, dict[int, tuple[list[str], np.ndarray]]] = {}
    descriptor_folds: dict[str, dict[int, dict[int, tuple[list[str], np.ndarray]]]] = {}

    for model_name in model_names:
        spec = _model_spec(model_name)
        metrics[model_name] = {}
        mean_descriptors[model_name] = {}
        descriptor_folds[model_name] = {}
        for step in step_grid.get(model_name, []):
            print(f"\n--- {model_name} step {step} ---")
            fold_entries = []
            desc_by_fold: dict[int, tuple[list[str], np.ndarray]] = {}
            stress_desc_n1_by_fold: dict[int, tuple[list[str], np.ndarray]] = {}
            for fold in folds:
                ckpt = _ckpt_path(spec, fold, step)
                if ckpt is None:
                    continue
                pred_cache = _prediction_cache_path(cache_dir, model_name, fold, step)
                fold_entries.append(_make_prediction_cache(spec, ckpt, fold, train_systems, stress_clusters, gt, pred_cache, refresh_cache=refresh_cache))
                desc_cache = _descriptor_cache_path(cache_dir, model_name, fold, step)
                names, X = _extract_descriptor_matrix(spec, ckpt, train_n1_systems, desc_cache, refresh_cache=refresh_cache)
                desc_by_fold[fold] = (names, X)
                if stress_n1_systems:
                    stress_cache = desc_cache.with_name(f"step{step}_stress.npz")
                    snames, SX = _extract_descriptor_matrix(spec, ckpt, stress_n1_systems, stress_cache, refresh_cache=refresh_cache)
                    stress_desc_n1_by_fold[fold] = (snames, SX)
            if not fold_entries or not desc_by_fold:
                continue

            descriptor_folds[model_name][step] = desc_by_fold
            mean_names, Xmean = _aggregate_descriptors_by_fold(desc_by_fold)
            mean_descriptors[model_name][step] = (mean_names, Xmean)

            probe_metrics: dict[str, Any] = {}
            ood_probe_projection: dict[str, Any] = {}
            if mean_names:
                y_index = {name: i for i, name in enumerate(train_names)}
                aligned_y = {target: np.array([y_by_target[target][y_index[n]] if n in y_index else np.nan for n in mean_names], dtype=float) for target in y_by_target}
                for target, y in aligned_y.items():
                    probe_metrics[target] = _ridge_probe_metrics(Xmean, y)
                # Project stress descriptors from fold-averaged n1 descriptors.
                stress_names, SXmean = _aggregate_descriptors_by_fold(stress_desc_n1_by_fold)
                stress_vecs = {name: SXmean[i] for i, name in enumerate(stress_names)}
                for target, y in aligned_y.items():
                    ood_probe_projection[target] = _fit_ridge_projection(Xmean, y, stress_vecs)

            geometry: dict[str, Any] = {}
            if len(mean_names) >= 5 and Xmean.size:
                geometry["compactness"] = float(np.mean(np.linalg.norm(Xmean - Xmean.mean(axis=0, keepdims=True), axis=1)))
                geometry["family_silhouette"] = _family_silhouette(Xmean, mean_names)
            # Cross-fold descriptor stability.
            fold_mats = []
            if desc_by_fold:
                common = set.intersection(*(set(n) for n, _X in desc_by_fold.values()))
                cos_vals = []
                for mat in sorted(common):
                    vecs = []
                    for names, X in desc_by_fold.values():
                        idx = names.index(mat)
                        vecs.append(X[idx])
                    for i in range(len(vecs)):
                        for j in range(i + 1, len(vecs)):
                            den = np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])
                            if den > 1e-12:
                                cos_vals.append(float(np.dot(vecs[i], vecs[j]) / den))
                geometry["cross_fold_cosine"] = _safe_mean(cos_vals)
            # Drift from the earliest available checkpoint for the same model.
            initial_step = min(step_grid.get(model_name, [step]))
            if step != initial_step and initial_step in mean_descriptors[model_name] and Xmean.size:
                init_names, Xinit = mean_descriptors[model_name][initial_step]
                common = sorted(set(mean_names) & set(init_names))
                if common:
                    idx_now = [mean_names.index(n) for n in common]
                    idx_init = [init_names.index(n) for n in common]
                    cos = _cosine_rows(Xmean[idx_now], Xinit[idx_init])
                    geometry["drift_cosine_to_initial"] = float(np.nanmean(cos))
                    geometry["drift_l2_to_initial"] = float(np.mean(np.linalg.norm(Xmean[idx_now] - Xinit[idx_init], axis=1)))
            elif step == initial_step:
                geometry["drift_cosine_to_initial"] = 1.0
                geometry["drift_l2_to_initial"] = 0.0

            metrics[model_name][str(step)] = {
                "checkpoint_step": step,
                "n_folds": len(fold_entries),
                "prediction_metrics": _aggregate_prediction_metrics(fold_entries, ood_values),
                "probe_metrics": probe_metrics,
                "ood_probe_projection": ood_probe_projection,
                "geometry": geometry,
            }

    # Cross-model CKA against exp7a at matched steps.
    for model_name, by_step in metrics.items():
        for step_s, cell in by_step.items():
            step = int(step_s)
            if model_name == "exp7a" or "exp7a" not in mean_descriptors or step not in mean_descriptors["exp7a"]:
                continue
            names_a, Xa = mean_descriptors["exp7a"][step]
            names_b, Xb = mean_descriptors[model_name].get(step, ([], np.zeros((0, 0))))
            common = sorted(set(names_a) & set(names_b))
            if common:
                ia = [names_a.index(n) for n in common]
                ib = [names_b.index(n) for n in common]
                cell.setdefault("geometry", {})["cka_to_exp7a"] = _linear_cka(Xa[ia], Xb[ib])

    summary = {
        "models": model_names,
        "requested_steps": requested_steps,
        "available_steps": step_grid,
        "training_ob_signed_all_negative": bool(np.all(targets["OB_signed"][np.isfinite(targets["OB_signed"])] < 0)),
        "recommended": _select_recommended_steps(metrics),
        "interpretation": {
            "early_stop_use": "Prefer checkpoints before OOD MAE degrades and before density/OB/O-fraction probes or cross-fold descriptor stability collapse.",
            "eap_guardrail": "EAP-4/EAP-8 are stress tests for extrapolating across OB=0; EAP-4 agreement is not standalone proof of reliable EAP-8 ranking.",
            "ob_limitation": "Training OB is entirely negative, so signed OB and absOB are collinear in training and absOB probes cannot identify a true optimum at zero without extra positive-OB anchors or explicit features.",
        },
    }
    out_metrics = out_dir / "early_stop_metrics.json"
    out_summary = out_dir / "early_stop_summary.json"
    eap_traj = out_dir / "eap_stress_trajectory.json"
    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_interpretation(summary, out_dir)
    eap_payload = {
        model: {
            step: cell.get("prediction_metrics", {}).get("stress_rows", [])
            for step, cell in by_step.items()
        }
        for model, by_step in metrics.items()
    }
    eap_traj.write_text(json.dumps(eap_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_accuracy(metrics, out_dir)
    _plot_probe_retention(metrics, out_dir)
    _plot_embedding_drift(metrics, out_dir)
    _plot_eap_stress(metrics, out_dir)
    _save_umap_inputs(mean_descriptors, out_dir)
    if not no_umap:
        _plot_umap_snapshots(mean_descriptors, gt, out_dir)

    print(f"\nSaved early-stop metrics to {out_metrics}")
    print(f"Saved early-stop summary to {out_summary}")
