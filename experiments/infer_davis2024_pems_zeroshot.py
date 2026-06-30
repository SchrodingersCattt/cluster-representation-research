#!/usr/bin/env python3
"""Recompute David2024-trained zero-shot predictions on PEMs without touching shared outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "00_data_prep"
PRED_OUT = ROOT / "davis2024_pems_zeroshot_predictions.json"
SUMMARY_OUT = ROOT / "davis2024_pems_zeroshot_summary.json"

DATASETS = {
    "pems_crystal": DATA_ROOT / "pems_crystal_systems",
    "pems_cluster_n1": DATA_ROOT / "pems_cluster_n1_systems",
}

BASELINES: list[dict[str, Any]] = [
    {
        "id": "exp1a_crystal_dpa32",
        "short_id": "exp1a",
        "label": "CHNO-only DPA crystal",
        "exp_dir": ROOT / "exp1a_crystal_dpa32",
        "checkpoint": "model.ckpt-200000.pt",
        "head": None,
        "configuration": "DPA3.2-5M, CHNO labels, crystal training input",
    },
    {
        "id": "exp1b_molecule_dpa32",
        "short_id": "exp1b",
        "label": "CHNO-only DPA molecule",
        "exp_dir": ROOT / "exp1b_molecule_dpa32",
        "checkpoint": "model.ckpt-200000.pt",
        "head": None,
        "configuration": "DPA3.2-5M, CHNO labels, molecular training input",
    },
    {
        "id": "exp3a_crystal_deepems",
        "short_id": "exp3a",
        "label": "CHNO-only DeepEMs crystal",
        "exp_dir": ROOT / "exp3a_crystal_deepems",
        "checkpoint": "model.ckpt-200000.pt",
        "head": None,
        "configuration": "DeepEMs-LAM, CHNO labels, crystal training input",
    },
    {
        "id": "exp3b_molecule_deepems",
        "short_id": "exp3b",
        "label": "CHNO-only DeepEMs molecule",
        "exp_dir": ROOT / "exp3b_molecule_deepems",
        "checkpoint": "model.ckpt-200000.pt",
        "head": None,
        "configuration": "DeepEMs-LAM, CHNO labels, molecular training input",
    },
    {
        "id": "exp4c_multitask_deepems_mol_kj",
        "short_id": "exp4c_kj",
        "label": "CHNO+DFT+experiment transfer",
        "exp_dir": ROOT / "exp4c_multitask_deepems_mol",
        "checkpoint": "model.ckpt-400000.pt",
        "head": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
        "configuration": "DeepEMs-LAM, CHNO labels + DFT + experimental-property head",
    },
    {
        "id": "exp4d_multitask_deepems_mol_kj",
        "short_id": "exp4d_kj",
        "label": "CHNO+DFT transfer",
        "exp_dir": ROOT / "exp4d_multitask_deepems_mol",
        "checkpoint": "model.ckpt-400000.pt",
        "head": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
        "configuration": "DeepEMs-LAM, CHNO labels + DFT",
    },
]


def _systems(dataset: str) -> list[Path]:
    root = DATASETS[dataset]
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _infer(model_path: Path, systems: list[Path], head: str | None) -> list[dict[str, Any]]:
    from deepmd.pt.infer.deep_eval import DeepProperty
    import dpdata

    kwargs = {"head": head} if head is not None else {}
    model = DeepProperty(str(model_path), **kwargs)
    model_type_map = np.array(model.get_type_map())
    rows: list[dict[str, Any]] = []
    for sys_path in systems:
        system = dpdata.LabeledSystem(str(sys_path), fmt="deepmd/npy")
        data_type_map = system.data["atom_names"]
        atom_types = np.array([
            np.where(model_type_map == data_type_map[t])[0][0]
            for t in system.data["atom_types"]
        ], dtype=np.int32)
        cells = None if (sys_path / "nopbc").exists() else system.data["cells"]
        truth = np.asarray(system.data["energies"], dtype=float).reshape(-1)
        pred = np.asarray(
            model.eval(coords=system.data["coords"], atom_types=atom_types, cells=cells)[0],
            dtype=float,
        ).reshape(-1)
        for y_true, y_pred in zip(truth, pred):
            rows.append({
                "system": sys_path.name,
                "ground_truth_m_s": float(y_true),
                "predicted_m_s": float(y_pred),
                "abs_error_m_s": float(abs(y_true - y_pred)),
            })
    return rows


def main() -> None:
    predictions: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"baselines": {}, "datasets": list(DATASETS)}
    for baseline in BASELINES:
        ckpt = baseline["exp_dir"] / baseline["checkpoint"]
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        summary["baselines"][baseline["label"]] = {
            "internal_id": baseline["short_id"],
            "run_id": baseline["id"],
            "configuration": baseline["configuration"],
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "head": baseline["head"],
            "datasets": {},
        }
        for dataset in DATASETS:
            print(f"=== {baseline['label']} on {dataset} ({baseline['checkpoint']}) ===")
            rows = _infer(ckpt, _systems(dataset), baseline["head"])
            errors = np.array([r["abs_error_m_s"] for r in rows], dtype=float)
            preds = np.array([r["predicted_m_s"] for r in rows], dtype=float)
            truth = np.array([r["ground_truth_m_s"] for r in rows], dtype=float)
            summary["baselines"][baseline["label"]]["datasets"][dataset] = {
                "n_samples": int(len(rows)),
                "mae_m_s": round(float(errors.mean()), 1),
                "median_ae_m_s": round(float(np.median(errors)), 1),
                "rmse_m_s": round(float(np.sqrt(np.mean((preds - truth) ** 2))), 1),
            }
            for row in rows:
                predictions.append({
                    "baseline": baseline["label"],
                    "internal_id": baseline["short_id"],
                    "run_id": baseline["id"],
                    "dataset": dataset,
                    "checkpoint": baseline["checkpoint"],
                    "head": baseline["head"],
                    **row,
                })
    PRED_OUT.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {PRED_OUT}")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
