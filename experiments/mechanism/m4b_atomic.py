"""M4b -- Site-resolved atomic embedding analysis (A/B/X decomposition, ANOVA, coupling)"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
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


def _build_molecular_crystal(coord: np.ndarray, symbols: list[str]):
    """Build MolecularCrystal from cluster coordinates (nopbc, 100 Å box).

    PEM_BOND_THRESHOLDS is mandatory — without it MolecularCrystal.from_ase()
    uses the 3.5 Å default which merges ionic metal···O contacts into wrong
    molecules (AGENTS.md warning).
    """
    import sys
    if os.environ.get("MOLCRYSKIT_ROOT"):
        sys.path.insert(0, os.environ["MOLCRYSKIT_ROOT"])
    from ase import Atoms as _AseAtoms
    from molcrys_kit.structures.crystal import MolecularCrystal as _MC
    atoms = _AseAtoms(symbols=symbols, positions=coord, cell=np.eye(3) * 100.0, pbc=False)
    return _MC.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS)


def _classify_mol(mol) -> str:
    """Classify a CrystalMolecule as A / B / X site (same logic as predict_abx_grid.py)."""
    syms = mol.get_chemical_symbols()
    sym_set = set(syms)
    has_C = "C" in sym_set
    has_halogen_or_N = bool(sym_set & {"Cl", "I", "N"})
    has_metal = bool(sym_set & METAL_ELEMENTS)
    if has_C:
        return "A"
    if has_metal and "O" not in sym_set and not has_halogen_or_N:
        return "B"
    has_O = "O" in sym_set
    if has_O and (bool(sym_set & {"Cl", "I"}) or ("N" in sym_set and "H" not in sym_set)):
        return "X"
    if len(syms) <= 6 and not has_C:
        return "B"
    return "B"


def _get_abx_atom_indices(mc, coord: np.ndarray, symbols: list[str]) -> dict[str, list[int]]:
    """Map each molecule to A/B/X and return global atom indices per site.

    Uses position matching between MolecularCrystal molecules and the global
    coordinate array (CrystalMolecule has no .indices attribute).

    Special handling: when MolCrysKit merges a metal cation (e.g., K+) with
    an oxidizing anion (e.g., IO4-) into one molecule, we split the metal
    atoms to B-site and the remaining atoms to X-site.
    """
    site_atoms: dict[str, list[int]] = {"A": [], "B": [], "X": []}
    used = set()
    for mol in mc.molecules:
        mol_pos = mol.get_positions()
        mol_syms = mol.get_chemical_symbols()
        mol_sym_set = set(mol_syms)

        # Detect merged metal+anion molecules
        metal_atoms_in_mol = [i for i, s in enumerate(mol_syms) if s in METAL_ELEMENTS]
        has_anion_signature = ("O" in mol_sym_set and
                               (bool(mol_sym_set & {"Cl", "I"}) or
                                ("N" in mol_sym_set and "H" not in mol_sym_set)))
        if metal_atoms_in_mol and has_anion_signature and "C" not in mol_sym_set:
            # Split: metal atoms → B, rest → X
            for ai in range(len(mol_pos)):
                dists = np.linalg.norm(coord - mol_pos[ai], axis=1)
                best = int(np.argmin(dists))
                if dists[best] < 0.05 and symbols[best] == mol_syms[ai] and best not in used:
                    site = "B" if ai in metal_atoms_in_mol else "X"
                    site_atoms[site].append(best)
                    used.add(best)
        else:
            site = _classify_mol(mol)
            for ai in range(len(mol_pos)):
                dists = np.linalg.norm(coord - mol_pos[ai], axis=1)
                best = int(np.argmin(dists))
                if dists[best] < 0.05 and symbols[best] == mol_syms[ai] and best not in used:
                    site_atoms[site].append(best)
                    used.add(best)
    return site_atoms


def _load_pem_site_labels() -> dict[str, tuple[str, str, str]]:
    """Load (A_type, B_type, X_type) labels from pems.csv for each material."""
    import csv
    labels: dict[str, tuple[str, str, str]] = {}
    with paths.PEMS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mat = row.get("material", "").strip()
            if mat:
                labels[mat] = (
                    row.get("A_site", "").strip(),
                    row.get("B_site", "").strip(),
                    row.get("X_site", "").strip(),
                )
    return labels



def run_m4b(output_dir: Path) -> None:
    """M4b: Site-resolved embedding analysis.

    Parts 1-6 from plan 12:
      1. Per-atom descriptors → site-pooled z_A, z_B, z_X, z_all
      2. Site-based UMAP (main)
      3. Atomic-level UMAP (supplementary)
      4. Vdet ANOVA: Vdet ~ A_type + B_type + X_type
      5a. Material-level PCA + per-PC ANOVA
      5b. Site-specific PCA (sanity check)
      5c. Site-pooled → Vdet Ridge LOO R²
      6. A-B coupling: interaction plot + displacement cosine (X=ClO4- only)
    """
    print("\n" + "=" * 60 + "\nM4b: Site-resolved embedding analysis\n" + "=" * 60)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score

    materials = get_materials()
    gt = load_gt_vdet()
    site_labels = _load_pem_site_labels()

    # --- Step 0-1: Extract per-atom descriptors per checkpoint fold; merge by averaging ---
    per_ckpt = {}
    for fi in runtime.ACTIVE_FOLD_IDS:
        print(f"\n--- checkpoint fold {fi} ---")
        per_ckpt[str(fi)] = {}
        for mn in ["exp7a", "exp7c"]:
            print(f"  {mn}")
            dp = load_descriptor_model(mn, fi)
            mat_data = {}
            for mat in materials:
                coord, symbols, _ = read_cluster_system(paths.CLUSTER_N1_DIR / mat)
                desc = extract_descriptor_per_atom(dp, coord, symbols)
                mc = _build_molecular_crystal(coord, symbols)
                abx_idx = _get_abx_atom_indices(mc, coord, symbols)

                z_all = desc.mean(axis=0)
                site_emb = {}
                for site in ["A", "B", "X"]:
                    idx = abx_idx[site]
                    site_emb[site] = desc[idx].mean(axis=0) if idx else np.zeros(desc.shape[1])
                n_assigned = sum(len(v) for v in abx_idx.values())
                if n_assigned < len(symbols):
                    print(f"  WARN {mat}: only {n_assigned}/{len(symbols)} atoms assigned to A/B/X")

                mat_data[mat] = {
                    "z_all": z_all,
                    "z_A": site_emb["A"],
                    "z_B": site_emb["B"],
                    "z_X": site_emb["X"],
                    "desc_per_atom": desc,
                    "atom_sites": np.array(
                        ["A" if i in set(abx_idx["A"])
                         else "B" if i in set(abx_idx["B"])
                         else "X" if i in set(abx_idx["X"])
                         else "?" for i in range(len(symbols))]
                    ),
                    "symbols": symbols,
                    "abx_counts": {s: len(v) for s, v in abx_idx.items()},
                }
            per_ckpt[str(fi)][mn] = mat_data
            for mat in materials[:2]:
                d = mat_data[mat]
                print(f"    {mat}: A={d['abx_counts']['A']}, B={d['abx_counts']['B']}, X={d['abx_counts']['X']}")

    per_model = {}
    for mn in ["exp7a", "exp7c"]:
        merged = {}
        for mat in materials:
            z_all = np.mean([per_ckpt[fk][mn][mat]["z_all"] for fk in per_ckpt], axis=0)
            z_A = np.mean([per_ckpt[fk][mn][mat]["z_A"] for fk in per_ckpt], axis=0)
            z_B = np.mean([per_ckpt[fk][mn][mat]["z_B"] for fk in per_ckpt], axis=0)
            z_X = np.mean([per_ckpt[fk][mn][mat]["z_X"] for fk in per_ckpt], axis=0)
            desc_per_atom = np.mean(np.stack([per_ckpt[fk][mn][mat]["desc_per_atom"] for fk in per_ckpt], axis=0), axis=0)
            ref = per_ckpt[str(runtime.ACTIVE_FOLD_IDS[0])][mn][mat]
            merged[mat] = {
                "z_all": z_all,
                "z_A": z_A,
                "z_B": z_B,
                "z_X": z_X,
                "desc_per_atom": desc_per_atom,
                "atom_sites": ref["atom_sites"],
                "symbols": ref["symbols"],
                "abx_counts": ref["abx_counts"],
            }
        per_model[mn] = merged

    # --- Step 4: Vdet ANOVA ---
    print("\n--- Step 4: Vdet ANOVA ---")
    vdet_arr = np.array([gt.get(m, np.nan) for m in materials])
    valid_vdet = ~np.isnan(vdet_arr)
    a_types = [site_labels.get(m, ("?", "?", "?"))[0] for m in materials]
    b_types = [site_labels.get(m, ("?", "?", "?"))[1] for m in materials]
    x_types = [site_labels.get(m, ("?", "?", "?"))[2] for m in materials]

    # Compute both one-way η² (main text: intuitive, directly interpretable) and
    # Type II partial η² via OLS (SI: controls for A/B/X collinearity).
    #
    # One-way η² answers: "how much Vdet variance does X-site alone explain?"
    # Type II partial η² answers: "how much does X-site explain after controlling for A and B?"
    # Both are reported; one-way is used in the main figure, Type II goes to SI.
    from scipy.stats import f_oneway as _f_oneway
    from scipy.stats import f as _f_dist

    def _ols_ss_residual(y: np.ndarray, X_design: np.ndarray) -> float:
        """Fit OLS and return SS_residual."""
        coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ coef
        return float(np.dot(resid, resid))

    def _dummy_encode(factor_vals: list[str], valid_mask: np.ndarray) -> np.ndarray:
        """One-hot encode factor (drop first level for identifiability)."""
        vals = [factor_vals[i] for i in range(len(factor_vals)) if valid_mask[i]]
        levels = sorted(set(vals))
        if len(levels) <= 1:
            return np.zeros((len(vals), 0))
        mat = np.zeros((len(vals), len(levels) - 1))
        for i, v in enumerate(vals):
            j = levels.index(v) - 1
            if j >= 0:
                mat[i, j] = 1.0
        return mat

    # --- One-way η² (main text) ---
    anova_results: dict = {}
    for factor_name, factor_vals in [("A_type", a_types), ("B_type", b_types), ("X_type", x_types)]:
        groups: dict[str, list[float]] = {}
        for m, fv, v, ok in zip(materials, factor_vals, vdet_arr, valid_vdet):
            if ok and fv:
                groups.setdefault(fv, []).append(float(v))
        groups_filtered = {k: v for k, v in groups.items() if len(v) >= 1}
        group_arrays = [np.array(v) for v in groups_filtered.values() if len(v) >= 2]
        if len(group_arrays) >= 2:
            F_ow, p_ow = _f_oneway(*group_arrays)
            all_vals = np.concatenate(group_arrays)
            grand_mean = all_vals.mean()
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_arrays)
            eta2_ow = float(ss_between / ss_total) if ss_total > 0 else 0.0
            k_grp = len(group_arrays)
            n_tot = len(all_vals)
            df_w = n_tot - k_grp
            ss_within = ss_total - ss_between
            ms_within = float(ss_within / df_w) if df_w > 0 else 0.0
            omega2_ow = float(max(0.0, (ss_between - (k_grp - 1) * ms_within) / (ss_total + ms_within))) if ss_total > 0 else 0.0
        else:
            F_ow, p_ow, eta2_ow, omega2_ow = float("nan"), float("nan"), 0.0, 0.0
        anova_results[factor_name] = {
            "F_oneway": float(F_ow) if not np.isnan(F_ow) else None,
            "p_oneway": float(p_ow) if not np.isnan(p_ow) else None,
            "eta2": eta2_ow,          # one-way η² — used in main figure
            "omega2": omega2_ow,
            "n_levels": len(groups_filtered),
            "n_samples": sum(len(v) for v in groups_filtered.values()),
        }

    # --- Type II partial η² via OLS (SI robustness check) ---
    y_vdet = vdet_arr[valid_vdet]
    n_vdet = len(y_vdet)
    a_enc = _dummy_encode(a_types, valid_vdet)
    b_enc = _dummy_encode(b_types, valid_vdet)
    x_enc = _dummy_encode(x_types, valid_vdet)
    intercept = np.ones((n_vdet, 1))
    X_full = np.hstack([intercept, a_enc, b_enc, x_enc])
    ss_res_full = _ols_ss_residual(y_vdet, X_full)
    df_res_full = n_vdet - X_full.shape[1]

    factor_encs = [("A_type", a_enc, a_types), ("B_type", b_enc, b_types), ("X_type", x_enc, x_types)]
    for factor_name, fenc, fvals in factor_encs:
        other_encs = [e for (fn, e, _) in factor_encs if fn != factor_name]
        X_reduced = np.hstack([intercept] + other_encs)
        ss_res_reduced = _ols_ss_residual(y_vdet, X_reduced)
        ss_factor = max(0.0, ss_res_reduced - ss_res_full)
        df_factor = fenc.shape[1]
        denom = ss_factor + ss_res_full
        partial_eta2 = float(ss_factor / denom) if denom > 0 else 0.0
        if df_factor > 0 and df_res_full > 0:
            ms_factor = ss_factor / df_factor
            ms_res = ss_res_full / df_res_full
            F_t2 = float(ms_factor / ms_res) if ms_res > 0 else float("nan")
            p_t2 = float(1.0 - _f_dist.cdf(F_t2, df_factor, df_res_full)) if not np.isnan(F_t2) else float("nan")
        else:
            F_t2, p_t2 = float("nan"), float("nan")
        # Merge Type II results into existing dict entry
        anova_results[factor_name]["F_type2"] = F_t2 if not np.isnan(F_t2) else None
        anova_results[factor_name]["p_type2"] = p_t2 if not np.isnan(p_t2) else None
        anova_results[factor_name]["partial_eta2"] = partial_eta2
        anova_results[factor_name]["_note"] = (
            "eta2: one-way (main text); partial_eta2: Type II via OLS (SI robustness check)"
        )

    # Print both
    for factor_name in ["A_type", "B_type", "X_type"]:
        d = anova_results[factor_name]
        print(f"  {factor_name}: one-way η²={d['eta2']:.3f} (F={d['F_oneway']:.2f}, p={d['p_oneway']:.4f}), "
              f"partial η²={d['partial_eta2']:.3f} (F={d['F_type2']:.2f}, p={d['p_type2']:.4f}), "
              f"levels={d['n_levels']}")

    # --- Step 5a: Material-level PCA + per-PC ANOVA ---
    print("\n--- Step 5a: Material-level PCA + per-PC ANOVA ---")
    pca_results: dict = {}
    for mn in ["exp7a", "exp7c"]:
        Z_all = np.array([per_model[mn][m]["z_all"] for m in materials])
        pca = PCA(n_components=min(5, len(materials) - 1))
        PCs = pca.fit_transform(Z_all)
        evr = pca.explained_variance_ratio_.tolist()

        pc_anova: list[dict] = []
        for k in range(PCs.shape[1]):
            pc_vals = PCs[:, k]
            pc_info: dict = {"pc": k + 1, "evr": evr[k]}
            # Correlation with Vdet
            v_ok = valid_vdet
            if v_ok.sum() >= 5:
                corr = float(np.corrcoef(pc_vals[v_ok], vdet_arr[v_ok])[0, 1])
                pc_info["corr_vdet"] = corr
            # Per-factor partial η² on this PC (Type II via OLS, same as Step 4)
            pc_a_enc = _dummy_encode(a_types, np.ones(len(materials), dtype=bool))
            pc_b_enc = _dummy_encode(b_types, np.ones(len(materials), dtype=bool))
            pc_x_enc = _dummy_encode(x_types, np.ones(len(materials), dtype=bool))
            pc_intercept = np.ones((len(materials), 1))
            X_pc_full = np.hstack([pc_intercept, pc_a_enc, pc_b_enc, pc_x_enc])
            ss_pc_full = _ols_ss_residual(pc_vals, X_pc_full)
            pc_factor_encs = [("A_type", pc_a_enc), ("B_type", pc_b_enc), ("X_type", pc_x_enc)]
            for fname, fenc_pc in pc_factor_encs:
                other_pc = [e for (fn, e) in pc_factor_encs if fn != fname]
                X_pc_red = np.hstack([pc_intercept] + other_pc)
                ss_pc_red = _ols_ss_residual(pc_vals, X_pc_red)
                ss_f = max(0.0, ss_pc_red - ss_pc_full)
                denom_pc = ss_f + ss_pc_full
                pc_info[f"eta2_{fname}"] = float(ss_f / denom_pc) if denom_pc > 0 else 0.0
            pc_anova.append(pc_info)
        pca_results[mn] = {"evr": evr, "pc_anova": pc_anova}
        print(f"  {mn}: EVR={[f'{v:.3f}' for v in evr[:3]]}")
        for pi in pc_anova[:3]:
            print(f"    PC{pi['pc']}: corr(Vdet)={pi.get('corr_vdet', 'N/A'):.3f}, "
                  f"η²(A)={pi.get('eta2_A_type', 0):.3f}, η²(B)={pi.get('eta2_B_type', 0):.3f}, "
                  f"η²(X)={pi.get('eta2_X_type', 0):.3f}")

    # --- Step 5c: Site-pooled → Vdet RidgeCV LOO R² (per checkpoint fold, then mean ± std) ---
    print("\n--- Step 5c: Site-pooled → Vdet RidgeCV LOO R² ---")

    def _site_r2_loo(pm_local, mn_key: str) -> dict:
        """Return both R^2 and the per-material LOO predictions so the
        caller can build SI parity panels.  Each entry is a dict
        ``{"r2": float, "y_pred": list[float]}`` and we also stash
        ``y_true`` and the materials used (post-NaN filtering) once at
        the top level so the parity plotter knows which 25 materials
        each prediction lines up with.
        """
        sr_local = {}
        yv = vdet_arr.copy()
        v_ok = valid_vdet
        if v_ok.sum() < 5:
            return {}
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict
        for label, key in [("z_A", "z_A"), ("z_B", "z_B"), ("z_X", "z_X"), ("z_all", "z_all")]:
            X = np.array([pm_local[mn_key][m][key] for m in materials])[v_ok]
            y = yv[v_ok]
            rcv = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=5)
            rcv.fit(X, y)
            yp = cross_val_predict(Ridge(alpha=rcv.alpha_), X, y, cv=LeaveOneOut())
            sr_local[label] = {"r2": float(r2_score(y, yp)),
                                "y_pred": yp.tolist()}
        X_cat = np.array([
            np.concatenate([
                pm_local[mn_key][m]["z_A"],
                pm_local[mn_key][m]["z_B"],
                pm_local[mn_key][m]["z_X"],
            ])
            for m in materials
        ])[v_ok]
        y = yv[v_ok]
        rcv_cat = RidgeCV(alphas=np.array(RIDGE_ALPHAS), cv=5)
        rcv_cat.fit(X_cat, y)
        yp_cat = cross_val_predict(Ridge(alpha=rcv_cat.alpha_), X_cat, y, cv=LeaveOneOut())
        sr_local["z_ABX_concat"] = {"r2": float(r2_score(y, yp_cat)),
                                     "y_pred": yp_cat.tolist()}
        sr_local["__y_true"] = y.tolist()
        sr_local["__materials"] = [materials[i] for i in range(len(materials)) if v_ok[i]]
        return sr_local

    site_r2_list = {"exp7a": [], "exp7c": []}
    for fk in per_ckpt:
        for mn in ["exp7a", "exp7c"]:
            site_r2_list[mn].append(_site_r2_loo(per_ckpt[fk], mn))

    site_r2 = {}
    site_r2_std = {}
    site_y_pred_mean: dict[str, dict[str, list[float]]] = {}
    site_y_true: dict[str, list[float]] = {}
    site_materials: dict[str, list[str]] = {}
    for mn in ["exp7a", "exp7c"]:
        lst = site_r2_list[mn]
        if not lst or not lst[0]:
            site_r2[mn] = {}
            site_r2_std[mn] = {}
            site_y_pred_mean[mn] = {}
            continue
        site_keys = [k for k in lst[0].keys() if not k.startswith("__")]
        site_r2[mn] = {k: float(np.mean([d[k]["r2"] for d in lst])) for k in site_keys}
        site_r2_std[mn] = {k: float(np.std([d[k]["r2"] for d in lst])) if len(lst) > 1 else 0.0 for k in site_keys}
        # Mean LOO prediction across the five fold checkpoints, matching
        # how y_pred_emb is averaged in M3 -- gives a single number per
        # material for the parity plot.
        site_y_pred_mean[mn] = {
            k: np.mean(np.stack([np.array(d[k]["y_pred"]) for d in lst], axis=0), axis=0).tolist()
            for k in site_keys
        }
        # y_true and material list are identical across folds; copy from the first.
        site_y_true[mn] = lst[0]["__y_true"]
        site_materials[mn] = lst[0]["__materials"]
        print(f"  {mn}: " + ", ".join(f"{k}={site_r2[mn][k]:.3f}" for k in site_r2[mn]))

    # --- Step 6: A-B coupling (X=ClO4- only) ---
    print("\n--- Step 6: A-B coupling (X=ClO4- subset) ---")
    clop_mats = [m for m in materials if x_types[materials.index(m)] == "ClO4-"]
    clop_valid = [m for m in clop_mats if not np.isnan(gt.get(m, np.nan))]
    print(f"  ClO4- materials: {len(clop_mats)} total, {len(clop_valid)} with Vdet")

    # Interaction data for plotting
    interaction_data: dict = {}
    for m in clop_valid:
        a, b, x = site_labels.get(m, ("?", "?", "?"))
        interaction_data[m] = {"A": a, "B": b, "Vdet": gt[m]}

    # Displacement cosine analysis (using z_all)
    cosine_results: dict = {}
    for mn in ["exp7a", "exp7c"]:
        # Group ClO4- materials by (A, B)
        ab_map: dict[tuple[str, str], str] = {}  # (A, B) → material
        for m in clop_valid:
            a, b, _ = site_labels.get(m, ("?", "?", "?"))
            ab_map[(a, b)] = m

        # For each B with ≥2 A-types, compute displacement vectors
        b_to_a_mats: dict[str, list[tuple[str, str]]] = {}  # B → [(A, material)]
        for (a, b), m in ab_map.items():
            b_to_a_mats.setdefault(b, []).append((a, m))

        displacements: dict[str, list[tuple[str, str, np.ndarray]]] = {}
        for b_type, a_mats in b_to_a_mats.items():
            if len(a_mats) < 2:
                continue
            a_mats_sorted = sorted(a_mats, key=lambda x: x[0])
            # Compute pairwise displacement vectors
            for i in range(len(a_mats_sorted)):
                for j in range(i + 1, len(a_mats_sorted)):
                    a1, m1 = a_mats_sorted[i]
                    a2, m2 = a_mats_sorted[j]
                    dz = per_model[mn][m2]["z_all"] - per_model[mn][m1]["z_all"]
                    pair_key = f"{a1}→{a2}"
                    displacements.setdefault(pair_key, []).append((b_type, f"{m1}→{m2}", dz))

        # Compute cosine similarity between displacement vectors for the same A-pair across B-types
        cosines: list[dict] = []
        for pair_key, entries in displacements.items():
            if len(entries) < 2:
                continue
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    b1, label1, dz1 = entries[i]
                    b2, label2, dz2 = entries[j]
                    n1, n2 = np.linalg.norm(dz1), np.linalg.norm(dz2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos = float(np.dot(dz1, dz2) / (n1 * n2))
                    else:
                        cos = float("nan")
                    cosines.append({
                        "a_pair": pair_key,
                        "b1": b1, "b2": b2,
                        "cosine": cos,
                    })
        if cosines:
            mean_cos = np.nanmean([c["cosine"] for c in cosines])
            print(f"  {mn}: {len(cosines)} cosine pairs, mean={mean_cos:.3f}")
        else:
            mean_cos = float("nan")
            print(f"  {mn}: no cosine pairs (insufficient coverage)")
        cosine_results[mn] = {"cosines": cosines, "mean_cosine": float(mean_cos)}

    # --- Save results ---
    results = {
        "anova": anova_results,
        "pca": pca_results,
        "site_r2": site_r2,
        "site_r2_std_across_folds": site_r2_std,
        "site_y_pred_mean": site_y_pred_mean,
        "site_y_true": site_y_true,
        "site_materials": site_materials,
        "checkpoint_fold_ids": list(runtime.ACTIVE_FOLD_IDS),
        "cosine_analysis": cosine_results,
        "materials": materials,
        "site_labels": {m: list(site_labels.get(m, ("?", "?", "?"))) for m in materials},
        "abx_counts": {mn: {m: per_model[mn][m]["abx_counts"] for m in materials}
                       for mn in ["exp7a", "exp7c"]},
    }
    out_path = output_dir / "mechanism_m4b_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")

    # --- Figures ---
    plot_m4b_site_umap(per_model, materials, gt, site_labels, output_dir)
    plot_m4b_anova(anova_results, site_r2, interaction_data, output_dir)
    plot_m4b_atomic_umap(per_model, materials, output_dir)
    plot_m4b_pca_anova(pca_results, output_dir)
    plot_m4b_coupling(cosine_results, interaction_data, output_dir)


def plot_m4b_site_umap(per_model: dict, materials: list, gt: dict,
                        site_labels: dict, output_dir: Path) -> None:
    """Site-based UMAP: 3 panels per model (site type / Vdet / ion identity)."""
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP unavailable, skipping site UMAP")
        return

    setup_nature_style()
    vdet = np.array([gt.get(m, np.nan) for m in materials])
    valid = ~np.isnan(vdet)

    for mn in ["exp7a", "exp7c"]:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
        # Build 75-point matrix: 25 mats × 3 sites
        pts = []
        pt_site = []
        pt_mat = []
        pt_vdet = []
        pt_ion = []
        for m in materials:
            d = per_model[mn][m]
            sl = site_labels.get(m, ("?", "?", "?"))
            for si, (key, ion_label) in enumerate(zip(["z_A", "z_B", "z_X"], sl)):
                pts.append(d[key])
                pt_site.append(["A", "B", "X"][si])
                pt_mat.append(m)
                pt_vdet.append(gt.get(m, np.nan))
                pt_ion.append(ion_label)
        X = np.array(pts)
        pt_site = np.array(pt_site)
        pt_vdet = np.array(pt_vdet)

        um = UMAP(n_components=2, random_state=42, n_neighbors=max(2, min(15, len(X) - 1)))
        X2 = um.fit_transform(X)

        # Panel A: color by site type
        site_colors = {"A": COLORS["exp7a"], "B": COLORS["exp7c"], "X": "#009E73"}
        for s in ["A", "B", "X"]:
            mask = pt_site == s
            axes[0].scatter(X2[mask, 0], X2[mask, 1], c=site_colors[s],
                            s=24, alpha=0.8, edgecolors="white", linewidths=0.3, label=f"{s}-site")
        axes[0].legend(frameon=False, fontsize=7)
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
        style_axes(axes[0])
        add_panel_label(axes[0], "A")

        # Panel B: color by Vdet
        v_valid = ~np.isnan(pt_vdet)
        if v_valid.any():
            vmin, vmax = float(np.nanmin(pt_vdet)), float(np.nanmax(pt_vdet))
            sc = axes[1].scatter(X2[v_valid, 0], X2[v_valid, 1], c=pt_vdet[v_valid],
                                 cmap="viridis", vmin=vmin, vmax=vmax,
                                 s=24, edgecolors="white", linewidths=0.3)
            cbar = plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("Vdet (m/s)", fontsize=7)
            # Plot NaN points in gray
            if (~v_valid).any():
                axes[1].scatter(X2[~v_valid, 0], X2[~v_valid, 1], c="#CCCCCC",
                                s=16, alpha=0.5, edgecolors="white", linewidths=0.2)
        axes[1].set_xlabel("UMAP 1")
        style_axes(axes[1])
        add_panel_label(axes[1], "B")

        # Panel C: color by ion identity within each site
        unique_ions = list(dict.fromkeys(pt_ion))
        ion_cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique_ions))
        ion_color_map = {ion: ion_cmap(i) for i, ion in enumerate(unique_ions)}
        for ion in unique_ions:
            mask = np.array(pt_ion) == ion
            axes[2].scatter(X2[mask, 0], X2[mask, 1], c=[ion_color_map[ion]],
                            s=24, alpha=0.8, edgecolors="white", linewidths=0.3,
                            label=ion[:12])
        axes[2].legend(frameon=False, fontsize=5.5, ncol=2, loc="best",
                       handletextpad=0.3, columnspacing=0.5)
        axes[2].set_xlabel("UMAP 1")
        style_axes(axes[2])
        add_panel_label(axes[2], "C")

        fig.suptitle(f"M4b site-pooled UMAP — {mn}", fontsize=9, y=1.01)
        fig.tight_layout()
        save_figure(
            fig,
            f"Fig_M4b_site_umap_{mn}_main",
            supplementary=False,
            legacy_png_name=f"figure_m4b_site_umap_{mn}.png",
        )
        plt.close(fig)
    print("Saved Figure M4b site UMAP")


def plot_m4b_anova(anova_results: dict, site_r2: dict,
                    interaction_data: dict, output_dir: Path) -> None:
    """ANOVA η² bar chart + site R² comparison."""
    setup_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # Panel A: ANOVA η²
    factors = ["A_type", "B_type", "X_type"]
    flabels = ["A-site", "B-site", "X-site"]
    fcolors = [COLORS["exp7a"], COLORS["exp7c"], "#009E73"]
    eta2_vals = [anova_results[f]["eta2"] for f in factors]
    x = np.arange(len(factors))
    ax1.bar(x, eta2_vals, color=fcolors, edgecolor="white", linewidth=0.5, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(flabels)
    ax1.set_ylabel("η² (effect size)")
    ax1.set_ylim(0, max(eta2_vals) * 1.3 if max(eta2_vals) > 0 else 1.0)
    ax1.set_title("Vdet ANOVA", fontsize=8)
    for i, v in enumerate(eta2_vals):
        ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    style_axes(ax1, grid=True)
    add_panel_label(ax1, "A")

    # Panel B: Site-pooled → Vdet R²
    site_keys = ["z_A", "z_B", "z_X", "z_all", "z_ABX_concat"]
    site_key_labels = ["z_A", "z_B", "z_X", "z_all", "z_A+B+X"]
    x2 = np.arange(len(site_keys))
    w = 0.30
    for i, (mn, color) in enumerate([("exp7a", COLORS["exp7a"]), ("exp7c", COLORS["exp7c"])]):
        vals = [site_r2.get(mn, {}).get(k, float("nan")) for k in site_keys]
        ax2.bar(x2 + (i - 0.5) * w, vals, w * 0.9, color=color,
                edgecolor="white", linewidth=0.5, label=mn)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(site_key_labels, fontsize=7)
    ax2.set_ylabel("R² (LOO)")
    ax2.set_ylim(-0.5, 1.05)
    ax2.axhline(0, color=COLORS["ref"], lw=1.0, ls="--")
    ax2.set_title("Site embedding → Vdet", fontsize=8)
    ax2.legend(frameon=False, fontsize=7, loc="upper left")
    style_axes(ax2, grid=True)
    add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M4b_anova_main",
        supplementary=False,
        legacy_png_name="figure_m4b_anova.png",
    )
    plt.close(fig)
    print("Saved Figure M4b ANOVA")


def plot_m4b_atomic_umap(per_model: dict, materials: list, output_dir: Path) -> None:
    """Atomic-level UMAP: single combined 2×2 figure (rows=model, cols=coloring) for supplementary."""
    try:
        from umap import UMAP
    except ImportError:
        print("  UMAP unavailable, skipping atomic UMAP")
        return

    setup_nature_style()
    panel_labels = ["A", "B", "C", "D"]
    site_colors = {"A": COLORS["exp7a"], "B": COLORS["exp7c"], "X": "#009E73", "?": "#CCCCCC"}

    # Pre-compute UMAP for each model
    umap_data = {}
    for mn in ["exp7a", "exp7c"]:
        all_desc, all_sites, all_elems = [], [], []
        for mat in materials:
            d = per_model[mn][mat]
            all_desc.append(d["desc_per_atom"])
            all_sites.extend(d["atom_sites"].tolist())
            all_elems.extend(d["symbols"])
        X = np.vstack(all_desc)
        all_sites = np.array(all_sites)
        all_elems = np.array(all_elems)
        print(f"  {mn}: {len(X)} atoms → UMAP")
        um = UMAP(n_components=2, random_state=42, n_neighbors=max(2, min(15, len(X) - 1)))
        X2 = um.fit_transform(X)
        umap_data[mn] = (X2, all_sites, all_elems)

    # Build combined 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))

    for row, mn in enumerate(["exp7a", "exp7c"]):
        X2, all_sites, all_elems = umap_data[mn]
        label_idx = row * 2

        # Left panel: color by site
        ax_site = axes[row, 0]
        for s in ["A", "B", "X", "?"]:
            mask = all_sites == s
            if mask.any():
                ax_site.scatter(X2[mask, 0], X2[mask, 1], c=site_colors[s],
                                s=3, alpha=0.4, label=f"{s}-site", rasterized=True)
        ax_site.legend(frameon=False, fontsize=7, markerscale=3)
        ax_site.set_xlabel("UMAP 1")
        ax_site.set_ylabel("UMAP 2")
        ax_site.set_title(f"{disp(mn)} — by site", fontsize=8)
        style_axes(ax_site)
        add_panel_label(ax_site, panel_labels[label_idx])

        # Right panel: color by element
        ax_elem = axes[row, 1]
        unique_elems = sorted(set(all_elems))
        ecmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique_elems))
        for ei, elem in enumerate(unique_elems):
            mask = all_elems == elem
            ax_elem.scatter(X2[mask, 0], X2[mask, 1], c=[ecmap(ei)],
                            s=3, alpha=0.4, label=elem, rasterized=True)
        ax_elem.legend(frameon=False, fontsize=5.5, ncol=3, markerscale=3,
                       handletextpad=0.2, columnspacing=0.3)
        ax_elem.set_xlabel("UMAP 1")
        ax_elem.set_title(f"{disp(mn)} — by element", fontsize=8)
        style_axes(ax_elem)
        add_panel_label(ax_elem, panel_labels[label_idx + 1])

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M4b_atomic_umap_combined_supp",
        supplementary=True,
        legacy_png_name="figure_m4b_atomic_umap_combined.png",
    )
    plt.close(fig)
    print("Saved Figure M4b atomic UMAP (combined)")


def plot_m4b_pca_anova(pca_results: dict, output_dir: Path) -> None:
    """PCA variance + per-PC η² heatmap (supplementary)."""
    setup_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    for col, (mn, ax) in enumerate(zip(["exp7a", "exp7c"], axes)):
        pr = pca_results[mn]
        pc_data = pr["pc_anova"]
        n_pcs = len(pc_data)
        x = np.arange(n_pcs)
        w = 0.18

        # Bars for η² per factor
        for fi, (fname, color, label) in enumerate([
            ("eta2_A_type", COLORS["exp7a"], "A-site"),
            ("eta2_B_type", COLORS["exp7c"], "B-site"),
            ("eta2_X_type", "#009E73", "X-site"),
        ]):
            vals = [pc.get(fname, 0) for pc in pc_data]
            ax.bar(x + (fi - 1) * w, vals, w * 0.9, color=color,
                   edgecolor="white", linewidth=0.5, label=label, alpha=0.85)

        # Overlay EVR as line
        evr = [pc["evr"] for pc in pc_data]
        ax2 = ax.twinx()
        ax2.plot(x, evr, "k-o", ms=4, lw=1.2, label="EVR")
        ax2.set_ylabel("Explained var. ratio", fontsize=7)
        ax2.set_ylim(0, max(evr) * 1.5 if evr else 1.0)
        ax2.tick_params(labelsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"PC{pc['pc']}\nr={pc.get('corr_vdet', 0):.2f}" for pc in pc_data],
                           fontsize=6.5)
        ax.set_ylabel("η²")
        ax.set_ylim(0, 1.05)
        ax.set_title(mn, fontsize=8)
        if col == 0:
            ax.legend(frameon=False, fontsize=6.5, loc="upper right")
        style_axes(ax, grid=True)
        add_panel_label(ax, "AB"[col])

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M4b_pca_anova_supp",
        supplementary=True,
        legacy_png_name="figure_m4b_pca_anova.png",
    )
    plt.close(fig)
    print("Saved Figure M4b PCA ANOVA")


def plot_m4b_coupling(cosine_results: dict, interaction_data: dict, output_dir: Path) -> None:
    """Interaction plot (X=ClO4-) + displacement cosine summary (supplementary)."""
    setup_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Panel A: Interaction plot (Vdet vs B-type, lines = A-type)
    # Only B-types with ≥2 A-types
    by_ab: dict[str, dict[str, list[float]]] = {}  # B → {A → [Vdet]}
    for m, info in interaction_data.items():
        by_ab.setdefault(info["B"], {}).setdefault(info["A"], []).append(info["Vdet"])
    # Filter B-types with ≥2 distinct A-types
    b_types_ok = [b for b, adict in by_ab.items() if len(adict) >= 2]
    if b_types_ok:
        all_a_types = sorted(set(a for b in b_types_ok for a in by_ab[b]))
        a_cmap = matplotlib.colormaps.get_cmap("Set1").resampled(max(len(all_a_types), 3))
        a_colors = {a: a_cmap(i) for i, a in enumerate(all_a_types)}
        b_labels = sorted(b_types_ok)
        x_pos = {b: i for i, b in enumerate(b_labels)}
        for a_type in all_a_types:
            xs, ys = [], []
            for b_type in b_labels:
                vals = by_ab.get(b_type, {}).get(a_type, [])
                if vals:
                    xs.append(x_pos[b_type])
                    ys.append(np.mean(vals))
            if len(xs) >= 2:
                ax1.plot(xs, ys, "-o", color=a_colors[a_type], ms=5, lw=1.2,
                         label=a_type[:15], alpha=0.85)
            elif xs:
                ax1.scatter(xs, ys, c=[a_colors[a_type]], s=25, zorder=5)
        ax1.set_xticks(list(range(len(b_labels))))
        ax1.set_xticklabels([b[:8] for b in b_labels], fontsize=6.5, rotation=30, ha="right")
        ax1.set_ylabel("Vdet (m/s)")
        ax1.set_title("Interaction plot (X=ClO4-)", fontsize=8)
        ax1.legend(frameon=False, fontsize=5.5, loc="best", handletextpad=0.3)
    else:
        ax1.text(0.5, 0.5, "Insufficient B-type\ncoverage", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=9)
    style_axes(ax1, grid=True)
    add_panel_label(ax1, "A")

    # Panel B: Displacement cosine summary
    mn_list = ["exp7a", "exp7c"]
    mn_means = [cosine_results.get(mn, {}).get("mean_cosine", float("nan")) for mn in mn_list]
    x = np.arange(len(mn_list))
    bar_colors = [COLORS["exp7a"], COLORS["exp7c"]]
    ax2.bar(x, mn_means, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mn_list)
    ax2.set_ylabel("Mean cosine similarity")
    ax2.set_ylim(-1, 1)
    ax2.axhline(0, color=COLORS["ref"], lw=1.0, ls="--")
    ax2.axhline(1, color=COLORS["ref"], lw=0.8, ls=":", alpha=0.5)
    ax2.set_title("A-swap displacement consistency", fontsize=8)
    for i, v in enumerate(mn_means):
        if not np.isnan(v):
            ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    style_axes(ax2, grid=True)
    add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(
        fig,
        "Fig_M4b_coupling_supp",
        supplementary=True,
        legacy_png_name="figure_m4b_coupling.png",
    )
    plt.close(fig)
    print("Saved Figure M4b coupling")


# ===========================================================================
# M5a — Cross-fold descriptor stability
