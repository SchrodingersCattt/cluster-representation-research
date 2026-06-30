#!/usr/bin/env python3
"""
Convert filtered David 2024 SMILES into isolated-molecule DeepMD `deepmd/npy` systems.

Input:
- [`experiments/00_data_prep/filtered_index.json`](experiments/00_data_prep/filtered_index.json)

Output layout:
- [`experiments/00_data_prep/molecule_systems/<refcode>/`](experiments/00_data_prep/molecule_systems)
  - `nopbc`
  - `type.raw`
  - `type_map.raw`
  - `set.000/coord.npy`
  - `set.000/box.npy`
  - `set.000/energy.npy`
  - `set.000/force.npy`
  - `set.000/property.npy`

Implementation notes:
- Molecules are generated from SMILES with RDKit ETKDGv3 + MMFF/UFF optimization.
- Non-periodic systems use a large 100×100×100 Å box.
- [`nopbc`](experiments/00_data_prep/molecule_systems) marker file is written explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[2]
FILTERED_INDEX = ROOT / "experiments/00_data_prep/filtered_index.json"
OUT_DIR = ROOT / "experiments/00_data_prep/molecule_systems"
BOX = np.eye(3, dtype=np.float64) * 100.0
TYPE_MAP = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]
TYPE_TO_ID = {sym: i for i, sym in enumerate(TYPE_MAP)}


def build_coords_from_smiles(smiles: str) -> tuple[list[str], np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("MolFromSmiles failed")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError(f"EmbedMolecule failed with code {status}")

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_props is not None:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=50)

    conf = mol.GetConformer()
    coords = []
    species = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        species.append(atom.GetSymbol())
    coords = np.array(coords, dtype=np.float64)

    # Center in the large box.
    center = coords.mean(axis=0, keepdims=True)
    coords = coords - center + np.array([[50.0, 50.0, 50.0]], dtype=np.float64)
    return species, coords


def write_single_system(system_dir: Path, coords_cart: np.ndarray, atom_types: np.ndarray, prop: float) -> None:
    set_dir = system_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)

    coord_flat = coords_cart.reshape(1, -1).astype(np.float64)
    box_flat = BOX.reshape(1, 9).astype(np.float64)
    energy = np.array([prop], dtype=np.float64)
    force = np.zeros((1, coords_cart.shape[0], 3), dtype=np.float64)

    (system_dir / "nopbc").write_text("", encoding="utf-8")
    np.savetxt(system_dir / "type.raw", atom_types.astype(np.int32), fmt="%d")
    (system_dir / "type_map.raw").write_text("\n".join(TYPE_MAP) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", coord_flat)
    np.save(set_dir / "box.npy", box_flat)
    np.save(set_dir / "energy.npy", energy)
    np.save(set_dir / "force.npy", force)
    np.save(set_dir / "property.npy", energy.copy())


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered = json.loads(FILTERED_INDEX.read_text(encoding="utf-8"))["kept"]

    for rec in filtered:
        refcode = rec["refcode"]
        smiles = rec["chiral_smiles"]
        prop = float(rec["calc_det_velocity_m_s"])

        species, coords_cart = build_coords_from_smiles(smiles)
        atom_types = np.array([TYPE_TO_ID[s] for s in species], dtype=np.int32)
        write_single_system(OUT_DIR / refcode, coords_cart, atom_types, prop)

    print(f"Wrote molecule systems to {OUT_DIR}")


if __name__ == "__main__":
    main()
