#!/usr/bin/env python3
"""Focused OOD-eap8 prediction via the EAP-4 template.

Builds OOD-eap8 by replacing the ammonium B-site in the EAP-4 A2BX5
structure with methylammonium, then predicts Vdet with selected PEM models.

This is intentionally separate from predict_eap4_paph6.py so the OOD-eap8
experiment can be rerun without regenerating the full OOD-new series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write
from molcrys_kit.structures.crystal import MolecularCrystal

import predict_eap4_paph6 as ood_new

EXP_DIR = Path(__file__).resolve().parent
EAP4_CIF = EXP_DIR / "00_data_prep" / "pems_cleaned_cifs" / "EAP-4.cif"
METHYLAMMONIUM_SOURCE = EXP_DIR / "ood_cluster_xyz" / "EAP-8_cluster_n1.xyz"
XYZ_OUT_DIR = EXP_DIR / "ood_eap8_cluster_xyz"
OUT_PATH = EXP_DIR / "ood_eap8_predictions.json"


def _extract_methylammonium_from_ood_new_xyz():
    """Return the CH6N molecule from the existing OOD-new direct EAP-8 cluster."""
    if not METHYLAMMONIUM_SOURCE.exists():
        raise FileNotFoundError(
            f"Missing methylammonium source cluster: {METHYLAMMONIUM_SOURCE}"
        )
    mc = MolecularCrystal.from_ase(ase_read(str(METHYLAMMONIUM_SOURCE)))
    candidates = []
    for mol in mc.molecules:
        symbols = mol.get_chemical_symbols()
        counts = {sym: symbols.count(sym) for sym in set(symbols)}
        if counts == {"C": 1, "H": 6, "N": 1}:
            candidates.append(mol)
        elif "C" in symbols and len(symbols) <= 8:
            candidates.append(mol)
    if not candidates:
        raise RuntimeError("Could not find CH3NH3+ / CH6N in OOD-new EAP-8 cluster")
    candidates.sort(key=lambda mol: len(mol.get_chemical_symbols()))
    mol = candidates[0].copy()
    print(
        "Methylammonium source:",
        METHYLAMMONIUM_SOURCE.name,
        "CH6N",
        f"({len(mol.get_chemical_symbols())} atoms)",
    )
    return mol


def _build_ood_eap8_clusters(ch3nh3_mol) -> list[tuple[str, object]]:
    """Build EAP-4-template OOD-eap8 clusters and save XYZ snapshots."""
    print(f"Template CIF: {EAP4_CIF}")
    eap4_crystal = MolecularCrystal.from_ase(
        ase_read(str(EAP4_CIF)),
        bond_thresholds=ood_new.PEM_BOND_THRESHOLDS,
    )
    print(f"EAP-4 template molecules: {len(eap4_crystal.molecules)}")

    XYZ_OUT_DIR.mkdir(parents=True, exist_ok=True)
    clusters: list[tuple[str, object]] = []
    for dataset_name, seed in ood_new.CLUSTER_RANDOM_SEEDS.items():
        cluster_crystal, seed_idx, sc_dims = ood_new.build_seeded_stoichiometric_cluster(
            eap4_crystal,
            dataset_name=dataset_name,
            seed=seed,
        )
        cluster_atoms = ood_new.crystal_to_minimum_image_atoms(cluster_crystal)
        cluster_atoms = ood_new.cluster_atoms_centered(cluster_atoms)

        template_mc = MolecularCrystal.from_ase(cluster_atoms)
        template_abx = ood_new.get_abx_indices(template_mc)
        substituted_mc = ood_new.build_substituted_structure(
            template_mc,
            template_abx,
            a_mol=None,
            b_mol=ch3nh3_mol,
            x_mol=None,
        )
        if substituted_mc is None:
            print(f"  [{dataset_name}] substitution failed")
            continue

        sub_atoms = ood_new.mc_to_atoms(substituted_mc)
        xyz_path = XYZ_OUT_DIR / f"OOD-eap8_{dataset_name}.xyz"
        ase_write(str(xyz_path), sub_atoms)
        clusters.append((dataset_name, sub_atoms))
        print(
            f"  [{dataset_name}] {len(sub_atoms)} atoms, "
            f"seed_idx={seed_idx}, sc={sc_dims}, saved={xyz_path.name}"
        )
    if not clusters:
        raise RuntimeError("No OOD-eap8 clusters were built")
    return clusters


def _latest_checkpoint(exp_name: str) -> Path:
    exp_dir = EXP_DIR / exp_name
    ckpts = sorted(
        exp_dir.glob("model.ckpt-*.pt"),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {exp_dir}")
    return ckpts[-1]


def _predict_single_model(exp_name: str, clusters: list[tuple[str, object]]) -> dict[str, object]:
    from deepmd.pt.infer.deep_eval import DeepProperty

    ckpt = _latest_checkpoint(exp_name)
    print(f"\nModel {exp_name}: {ckpt.name}")
    model = DeepProperty(str(ckpt), head=ood_new.HEAD)
    type_map = model.get_type_map()
    preds = {}
    for dataset_name, atoms in clusters:
        pred = ood_new.run_inference(model, type_map, atoms)
        preds[f"pred_{dataset_name}_m_s"] = pred
        print(f"  [{dataset_name}] {pred:.1f} m/s")
    vals = np.array(list(preds.values()), dtype=float)
    return {
        "model": exp_name,
        "checkpoint": str(ckpt),
        "pred_mean_m_s": float(vals.mean()),
        "pred_std_m_s": float(vals.std()),
        "n_clusters": int(len(vals)),
        **preds,
    }


def _predict_exp7a(clusters: list[tuple[str, object]]) -> dict[str, object]:
    print("\nModel exp7a: fold0-fold4")
    fold_models = ood_new._load_fold_models(EXP_DIR, ood_new.EXP7A_BASE_FOLDS)
    all_preds: list[float] = []
    fold_means: list[float] = []
    by_fold: dict[str, list[float]] = {}
    for fi, (model, type_map) in enumerate(fold_models):
        fold_preds = []
        for dataset_name, atoms in clusters:
            pred = ood_new.run_inference(model, type_map, atoms)
            fold_preds.append(pred)
            all_preds.append(pred)
        by_fold[f"fold{fi}"] = fold_preds
        fold_mean = float(np.mean(fold_preds))
        fold_means.append(fold_mean)
        print(f"  [fold{fi}] mean={fold_mean:.1f} m/s ({len(fold_preds)} clusters)")
    vals = np.array(all_preds, dtype=float)
    return {
        "model": "exp7a",
        "folds": ood_new.EXP7A_BASE_FOLDS,
        "pred_mean_m_s": float(vals.mean()),
        "pred_std_m_s": float(vals.std()),
        "fold_means_m_s": fold_means,
        "n_preds": int(len(vals)),
        "cluster_preds_by_fold_m_s": by_fold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-models",
        nargs="+",
        default=["exp6v1_allpems", "exp6_allpems"],
        help="Single-model experiment directories to evaluate.",
    )
    parser.add_argument("--skip-exp7a", action="store_true", help="Skip exp7a 5-fold ensemble.")
    args = parser.parse_args()

    ch3nh3_mol = _extract_methylammonium_from_ood_new_xyz()
    clusters = _build_ood_eap8_clusters(ch3nh3_mol)

    results: dict[str, object] = {
        "material": "OOD-eap8",
        "composition": "(H2en2+)2(CH3NH3+)(ClO4-)5",
        "template": "EAP-4",
        "stoichiometry": "A2BX5",
        "methylammonium_source": str(METHYLAMMONIUM_SOURCE),
        "cluster_xyz_dir": str(XYZ_OUT_DIR),
    }
    model_results = {}
    for exp_name in args.single_models:
        model_results[exp_name] = _predict_single_model(exp_name, clusters)
    if not args.skip_exp7a:
        model_results["exp7a"] = _predict_exp7a(clusters)
    results["predictions"] = model_results

    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUT_PATH}")

    print("\nSummary")
    for name, res in model_results.items():
        n = res.get("n_preds", res.get("n_clusters", "?"))
        print(f"  {name:<15} {res['pred_mean_m_s']:.1f} ± {res['pred_std_m_s']:.1f} m/s  (N={n})")


if __name__ == "__main__":
    main()
