#!/usr/bin/env python3
"""Check that the code-availability package has the assets it advertises.

The checker is intentionally conservative: lightweight figure assets should be
present in this repository, while DeepMD checkpoints and DeepMD npy systems are
reported as external requirements unless the user has placed the external data
archive into the documented locations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RequiredFile:
    path: str
    tier: str
    severity: str
    note: str = ""


LIGHTWEIGHT_REQUIRED = [
    RequiredFile("README.md", "base", "error"),
    RequiredFile("MANIFEST.md", "base", "error"),
    RequiredFile("AGENTS.md", "base", "error"),
    RequiredFile("pyproject.toml", "base", "error", "uv dependency metadata"),
    RequiredFile("environment.yml", "base", "warning", "DeepMD/MolCrysKit conda recipe"),
    RequiredFile("data/pems/mix.csv", "figures", "error"),
    RequiredFile("data/pems/pems.csv", "figures", "error", "historical alias used by scripts"),
    RequiredFile("experiments/paper_plot_style.py", "figures", "error"),
    RequiredFile("experiments/branch_meta.json", "figures", "error"),
    RequiredFile("experiments/abx_grid_predictions_exp6v1_allpems_400k.json", "figures", "error"),
    RequiredFile("experiments/pems_ood_5fold_exp7a.json", "figures", "error"),
    RequiredFile("experiments/pems_ood_heldout_exp7_all.json", "figures", "error"),
    RequiredFile("experiments/pems_uq_calibration.json", "figures", "error"),
    RequiredFile("experiments/pems_ood_model_deviation_exp7a.json", "figures", "error"),
    RequiredFile("experiments/cross_infer_rep.json", "figures", "error"),
    RequiredFile("experiments/ood_experimental_values.json", "figures", "warning"),
    RequiredFile("experiments/_stats_bootstrap/bootstrap_results.json", "figures", "error"),
    RequiredFile("experiments/mechanism_results/mechanism_m1_results.json", "figures", "error"),
    RequiredFile("experiments/mechanism_results/mechanism_m3_results.json", "figures", "error"),
    RequiredFile("experiments/mechanism_results/mechanism_m4a_results.json", "figures", "error"),
    RequiredFile("experiments/mechanism_results/mechanism_m4b_results.json", "figures", "error"),
    RequiredFile("experiments/mechanism_results/mechanism_m5a_results.json", "figures", "error"),
    RequiredFile("experiments/00_data_prep/pems_5fold_splits_v2.json", "figures", "error"),
    RequiredFile("experiments/00_data_prep/pems_5fold_splits_seed7.json", "figures", "warning"),
    RequiredFile("experiments/00_data_prep/pems_5fold_splits_seed13.json", "figures", "warning"),
    RequiredFile("manuscript/figures/_qa_check.py", "figures", "error"),
    RequiredFile("manuscript/figures/_plot_fig2d_polyhedra.py", "figures", "error"),
    RequiredFile("manuscript/figures/figure_style.py", "figures", "error"),
    RequiredFile("manuscript/figures/plot_fig3.py", "figures", "error"),
    RequiredFile("manuscript/figures/plot_fig4.py", "figures", "error"),
    RequiredFile("manuscript/figures/plot_fig5.py", "figures", "error"),
    RequiredFile("manuscript/figures/fig4_cache.npz", "figures", "error", "needed for plot_fig4.py without checkpoints"),
    RequiredFile("manuscript/figures/fig4_counterfactual_abx3_abx4.json", "figures", "warning"),
    RequiredFile("manuscript/figures/figure4-hypergraph.pdf", "figures", "warning", "vector panel-a overlay"),
    RequiredFile("manuscript/figures/figure4-hypergraph.png", "figures", "warning", "raster fallback"),
    RequiredFile("manuscript/figures/_cluster_umap_cache.npz", "figures", "error", "needed for plot_fig5.py without recomputing descriptors"),
    RequiredFile("manuscript/figures/_material_pooled_umap_cache.npz", "figures", "error", "needed for plot_fig5.py without recomputing descriptors"),
    RequiredFile("src/crystal_viewer/scene.py", "figures", "error", "vendored crystal viewer package"),
    RequiredFile("src/crystal_viewer/renderer.py", "figures", "error", "vendored crystal viewer package"),
    RequiredFile("src/crystal_viewer/presets.py", "figures", "error", "vendored crystal viewer package"),
    RequiredFile("src/crystal_viewer/legacy/plot_crystal.py", "figures", "error", "CIF parsing backend"),
    RequiredFile("src/stoich_cluster_learning/viz/topology_projection.py", "figures", "error"),
    RequiredFile("src/stoich_cluster_learning/viz/coordination.py", "figures", "error"),
    RequiredFile("src/stoich_cluster_learning/viz/polyhedra.py", "figures", "error"),
    RequiredFile("src/stoich_cluster_learning/data/fig5.py", "figures", "error"),
    RequiredFile("data/abx4/properties.csv", "figures", "error"),
    RequiredFile("data/abx4/cifs/PEP.cif", "figures", "error"),
    RequiredFile("data/abx4/cifs/MPEP.cif", "figures", "error"),
    RequiredFile("data/abx4/cifs/HPEP.cif", "figures", "error"),
    RequiredFile("data/abx4/cifs/SY.cif", "figures", "warning"),
    RequiredFile("data/abx4/pxrd/manifest.json", "figures", "error"),
    RequiredFile("data/abx4/pxrd/PEP_measured.csv", "figures", "error"),
    RequiredFile("data/abx4/pxrd/PEP_simulated.csv", "figures", "error"),
    RequiredFile("data/abx4/pxrd/MPEP_measured.csv", "figures", "error"),
    RequiredFile("data/abx4/pxrd/MPEP_simulated.csv", "figures", "error"),
    RequiredFile("data/abx4/pxrd/HPEP_measured.csv", "figures", "error"),
    RequiredFile("data/abx4/pxrd/HPEP_simulated.csv", "figures", "error"),
    RequiredFile("experiments/00_data_prep/pems_cluster_cifs/cluster_n1/DAP-4.cif", "figures", "warning", "training cluster rendering"),
    RequiredFile("experiments/00_data_prep/pems_cluster_cifs_ood/PEP/cluster_n1.cif", "figures", "warning", "OOD cluster rendering"),
]

EXTERNAL_REQUIRED = [
    RequiredFile("experiments/exp6v1_allpems/model.ckpt-400000.pt", "inference", "external", "canonical full-data model"),
    RequiredFile("experiments/exp7a_fold0/model.ckpt-400000.pt", "inference", "external", "5-fold ensemble checkpoint"),
    RequiredFile("experiments/exp7a_fold1/model.ckpt-400000.pt", "inference", "external", "5-fold ensemble checkpoint"),
    RequiredFile("experiments/exp7a_fold2/model.ckpt-400000.pt", "inference", "external", "5-fold ensemble checkpoint"),
    RequiredFile("experiments/exp7a_fold3/model.ckpt-400000.pt", "inference", "external", "5-fold ensemble checkpoint"),
    RequiredFile("experiments/exp7a_fold4/model.ckpt-400000.pt", "inference", "external", "5-fold ensemble checkpoint"),
    RequiredFile("experiments/00_data_prep/pems_cluster_n1_systems", "inference", "external", "DeepMD npy cluster systems"),
    RequiredFile("experiments/00_data_prep/pems_cluster_n2_systems", "inference", "external", "DeepMD npy cluster systems"),
    RequiredFile("experiments/00_data_prep/pems_cluster_n3_systems", "inference", "external", "DeepMD npy cluster systems"),
    RequiredFile("experiments/00_data_prep/pems_crystal_systems", "inference", "external", "DeepMD npy crystal systems"),
    RequiredFile("pretrained_models/deepems-lam.pt", "training", "external", "pretrained backbone for finetuning"),
    RequiredFile("training_data/davis2024_training_data", "training", "external", "Davis2024 DeepMD systems"),
]

TIERS = {
    "base": {"base"},
    "figures": {"base", "figures"},
    "inference": {"base", "figures", "inference"},
    "training": {"base", "figures", "inference", "training"},
    "all": {"base", "figures", "inference", "training"},
}


def iter_requirements(tier: str) -> Iterable[RequiredFile]:
    selected = TIERS[tier]
    yield from (item for item in LIGHTWEIGHT_REQUIRED if item.tier in selected)
    if tier in {"inference", "training", "all"}:
        yield from (item for item in EXTERNAL_REQUIRED if item.tier in selected)


def exists(path_text: str) -> bool:
    return (ROOT / path_text).exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=sorted(TIERS), default="figures")
    parser.add_argument(
        "--strict-external",
        action="store_true",
        help="Count missing external assets as errors instead of informational requirements.",
    )
    args = parser.parse_args()

    errors: list[RequiredFile] = []
    warnings: list[RequiredFile] = []
    externals: list[RequiredFile] = []

    for item in iter_requirements(args.tier):
        if exists(item.path):
            continue
        if item.severity == "error":
            errors.append(item)
        elif item.severity == "warning":
            warnings.append(item)
        elif item.severity == "external":
            if args.strict_external:
                errors.append(item)
            else:
                externals.append(item)

    def print_group(title: str, items: list[RequiredFile]) -> None:
        if not items:
            return
        print(f"\n{title} ({len(items)}):")
        for item in items:
            suffix = f" — {item.note}" if item.note else ""
            print(f"  - {item.path}{suffix}")

    print(f"Checked release assets in {ROOT} [tier={args.tier}]")
    print_group("Missing required files", errors)
    print_group("Missing recommended files", warnings)
    print_group("External assets not present", externals)

    if errors:
        print("\nFAIL: required release assets are missing.")
        return 1
    if warnings:
        print("\nPASS with warnings: lightweight required assets are present, but recommended files are missing.")
        return 0
    if externals:
        print("\nPASS: lightweight assets are present. External assets are needed for inference/training workflows.")
        return 0
    print("\nPASS: all checked assets are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
