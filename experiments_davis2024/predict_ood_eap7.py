#!/usr/bin/env python3
"""Focused OOD EAP-7 prediction via the EAP-4 template.

Builds EAP-7 by replacing the ammonium B site in the EAP-4 A2BX5 template
with hydrazinium (N2H5+) from DAP-7, then predicts Vdet with exp7a folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read as ase_read, write as ase_write
from molcrys_kit.structures.crystal import MolecularCrystal

import predict_eap4_paph6 as ood_new

EXP_DIR = Path(__file__).resolve().parent
EAP4_CIF = EXP_DIR / "00_data_prep" / "pems_cleaned_cifs" / "EAP-4.cif"
HYDRAZINIUM_SOURCE_CIF = EXP_DIR / "00_data_prep" / "pems_cleaned_cifs" / "DAP-7.cif"
XYZ_OUT_DIR = EXP_DIR / "ood_eap7_cluster_xyz"
OUT_PATH = EXP_DIR / "ood_eap7_predictions.json"


def _extract_hydrazinium_from_dap7():
    """Return the H5N2 molecule from DAP-7 as the N2H5+ B-site replacement."""
    mc = MolecularCrystal.from_ase(
        ase_read(str(HYDRAZINIUM_SOURCE_CIF)),
        bond_thresholds=ood_new.PEM_BOND_THRESHOLDS,
    )
    candidates = []
    for mol in mc.molecules:
        symbols = mol.get_chemical_symbols()
        counts = {sym: symbols.count(sym) for sym in set(symbols)}
        if counts == {"H": 5, "N": 2}:
            candidates.append(mol)
    if not candidates:
        raise RuntimeError(f"Could not find H5N2 hydrazinium in {HYDRAZINIUM_SOURCE_CIF}")
    mol = candidates[0].copy()
    print(
        "Hydrazinium source:",
        HYDRAZINIUM_SOURCE_CIF.name,
        "H5N2",
        f"({len(mol.get_chemical_symbols())} atoms)",
    )
    return mol


def _build_ood_eap7_clusters(hydrazinium_mol) -> list[tuple[str, object]]:
    """Build EAP-4-template EAP-7 clusters and save XYZ snapshots."""
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
            b_mol=hydrazinium_mol,
            x_mol=None,
        )
        if substituted_mc is None:
            print(f"  [{dataset_name}] substitution failed")
            continue

        sub_atoms = ood_new.mc_to_atoms(substituted_mc)
        xyz_path = XYZ_OUT_DIR / f"EAP-7_{dataset_name}.xyz"
        ase_write(str(xyz_path), sub_atoms)
        clusters.append((dataset_name, sub_atoms))
        print(
            f"  [{dataset_name}] {len(sub_atoms)} atoms, "
            f"seed_idx={seed_idx}, sc={sc_dims}, saved={xyz_path.name}"
        )
    if not clusters:
        raise RuntimeError("No EAP-7 clusters were built")
    return clusters


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
        print(
            f"  [fold{fi}] mean={fold_mean:.1f} m/s "
            f"(cluster std={np.std(fold_preds, ddof=0):.1f}, {len(fold_preds)} clusters)"
        )

    vals = np.array(all_preds, dtype=float)
    fold_arr = np.array(fold_means, dtype=float)
    return {
        "model": "exp7a",
        "folds": ood_new.EXP7A_BASE_FOLDS,
        "grand_mean_m_s": float(fold_arr.mean()),
        "model_deviation_m_s": float(fold_arr.std(ddof=1)),
        "all_prediction_mean_m_s": float(vals.mean()),
        "all_prediction_std_m_s": float(vals.std(ddof=0)),
        "fold_means_m_s": fold_means,
        "n_folds": int(len(fold_means)),
        "n_clusters_per_fold": int(len(clusters)),
        "n_prediction_values": int(len(vals)),
        "cluster_preds_by_fold_m_s": by_fold,
    }


def main() -> None:
    hydrazinium_mol = _extract_hydrazinium_from_dap7()
    clusters = _build_ood_eap7_clusters(hydrazinium_mol)
    pred = _predict_exp7a(clusters)
    results = {
        "material": "EAP-7",
        "composition": "(H2en2+)2(N2H5+)(ClO4-)5",
        "template": "EAP-4",
        "stoichiometry": "A2BX5",
        "hydrazinium_source": str(HYDRAZINIUM_SOURCE_CIF),
        "cluster_xyz_dir": str(XYZ_OUT_DIR),
        "predictions": {"exp7a": pred},
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUT_PATH}")
    print("\nSummary")
    print(
        f"  EAP-7 exp7a: {pred['grand_mean_m_s']:.1f} ± "
        f"{pred['model_deviation_m_s']:.1f} m/s "
        f"(model deviation across {pred['n_folds']} fold means; "
        f"{pred['n_clusters_per_fold']} clusters/fold)"
    )


if __name__ == "__main__":
    main()
