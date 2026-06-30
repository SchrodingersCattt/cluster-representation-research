#!/usr/bin/env python3
"""Build the atomistic cluster-UMAP cache used by Fig. 5f and Extended Data.

The displayed marker for each material is the centroid of that material's
atoms after fitting one joint UMAP to fold-averaged multi-task descriptors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
EXP_DIR = ROOT / "experiments"
DATA_PREP = EXP_DIR / "00_data_prep"
CLUSTER_N1_DIR = DATA_PREP / "pems_cluster_n1_systems"
CLUSTER_N1_OOD_DIR = DATA_PREP / "pems_cluster_n1_systems_ood"
CLEANED_CIF_DIR = DATA_PREP / "pems_cleaned_cifs"
CACHE_PATH = THIS_DIR / "_cluster_umap_cache.npz"

sys.path.insert(0, str(THIS_DIR))
import plot_fig4 as fig4  # noqa: E402

sys.path.insert(0, str(EXP_DIR))
from predict_sy_series import (  # noqa: E402
    CLUSTER_RANDOM_SEEDS,
    PEM_BOND_THRESHOLDS,
    build_seeded_stoichiometric_cluster,
    cluster_atoms_centered,
    crystal_to_minimum_image_atoms,
)
from molcrys_kit.structures.crystal import MolecularCrystal  # noqa: E402


IND25_MATERIALS = tuple(sorted(path.name for path in CLUSTER_N1_DIR.iterdir() if path.is_dir()))
OOD_NEW_MATERIALS = ("PEP", "MPEP", "HPEP")
OOD_HELDOUT_MATERIALS = ("DAC-4", "TAP-2", "DPPE-1", "EAP-4", "SY")
FOLD_IDS = (0, 1, 2, 3, 4)


def _read_npy_system(system_dir: Path) -> tuple[np.ndarray, list[str]]:
    coord, symbols = fig4._read_cluster_system(system_dir)
    return np.asarray(coord, dtype=np.float64), list(symbols)


def _build_cluster_n1_from_cif(cif_path: Path) -> tuple[np.ndarray, list[str]]:
    if not cif_path.exists():
        raise FileNotFoundError(f"Missing CIF for UMAP cluster build: {cif_path}")
    crystal = MolecularCrystal.from_ase(
        ase_read(str(cif_path)),
        bond_thresholds=PEM_BOND_THRESHOLDS,
    )
    cluster_crystal, seed_idx, sc_dims = build_seeded_stoichiometric_cluster(
        crystal,
        dataset_name="cluster_n1",
        seed=CLUSTER_RANDOM_SEEDS["cluster_n1"],
    )
    atoms = crystal_to_minimum_image_atoms(cluster_crystal)
    atoms = cluster_atoms_centered(atoms)
    print(
        f"  built cluster_n1 from {cif_path.name}: "
        f"{len(atoms)} atoms, seed_idx={seed_idx}, sc={sc_dims}"
    )
    return np.asarray(atoms.get_positions(), dtype=np.float64), atoms.get_chemical_symbols()


def _load_material_system(material: str, material_set: str) -> tuple[np.ndarray, list[str]]:
    if material_set == "ind25":
        return _read_npy_system(CLUSTER_N1_DIR / material)

    if material_set == "ood_heldout" and material != "EAP-4":
        return _read_npy_system(CLUSTER_N1_OOD_DIR / material)

    return _build_cluster_n1_from_cif(CLEANED_CIF_DIR / f"{material}.cif")


def _fold_averaged_atom_embedding(coord: np.ndarray, symbols: list[str]) -> np.ndarray:
    fold_embeddings: list[np.ndarray] = []
    for fold_id in FOLD_IDS:
        print(f"    descriptor fold {fold_id}")
        model = fig4._load_descriptor_model("exp7a", fold_id)
        fold_embeddings.append(fig4._extract_per_atom_embedding(model, coord, symbols))
    shapes = {embedding.shape for embedding in fold_embeddings}
    if len(shapes) != 1:
        raise RuntimeError(f"Fold descriptor shapes do not match: {sorted(shapes)}")
    return np.mean(np.stack(fold_embeddings, axis=0), axis=0)


def _material_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows.extend((name, "ind25") for name in IND25_MATERIALS)
    rows.extend((name, "ood_new") for name in OOD_NEW_MATERIALS)
    rows.extend((name, "ood_heldout") for name in OOD_HELDOUT_MATERIALS)
    return rows


def main() -> None:
    material_names: list[str] = []
    material_sets: list[str] = []
    atom_materials: list[str] = []
    atom_symbols: list[str] = []
    descriptor_blocks: list[np.ndarray] = []

    for material, material_set in _material_rows():
        print(f"[{material_set}] {material}")
        coord, symbols = _load_material_system(material, material_set)
        emb = _fold_averaged_atom_embedding(coord, symbols)
        if emb.shape[0] != len(symbols):
            raise RuntimeError(
                f"{material}: descriptor atom count {emb.shape[0]} "
                f"!= symbol count {len(symbols)}"
            )
        material_names.append(material)
        material_sets.append(material_set)
        atom_materials.extend([material] * len(symbols))
        atom_symbols.extend(symbols)
        descriptor_blocks.append(emb)

    atom_embedding = np.concatenate(descriptor_blocks, axis=0)
    print(
        f"Fitting joint atomistic UMAP on {atom_embedding.shape[0]} atoms "
        f"from {len(material_names)} materials"
    )
    atom_umap = fig4._atomic_umap_2d(atom_embedding)

    atom_materials_arr = np.asarray(atom_materials)
    centroids = []
    for material in material_names:
        mask = atom_materials_arr == material
        centroids.append(atom_umap[mask].mean(axis=0))
    material_centroid = np.asarray(centroids, dtype=np.float64)

    np.savez_compressed(
        CACHE_PATH,
        materials=np.asarray(material_names),
        material_set=np.asarray(material_sets),
        atom_umap=np.asarray(atom_umap, dtype=np.float64),
        atom_materials=atom_materials_arr,
        atom_symbols=np.asarray(atom_symbols),
        material_centroid=material_centroid,
    )
    print(f"Saved {CACHE_PATH}")


if __name__ == "__main__":
    main()
