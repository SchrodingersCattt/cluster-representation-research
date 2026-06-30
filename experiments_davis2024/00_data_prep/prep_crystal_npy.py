#!/usr/bin/env python3
"""
Convert filtered David 2024 crystal structures into DeepMD `deepmd/npy` systems.

Input:
- [`data/davis2024/energetic_crystals_dataset/energetic_crystals.npz`](data/davis2024/energetic_crystals_dataset/energetic_crystals.npz)
- [`experiments_davis2024/00_data_prep/filtered_index.json`](experiments_davis2024/00_data_prep/filtered_index.json)

Output layout:
- [`experiments_davis2024/00_data_prep/crystal_systems/<refcode>/`](experiments_davis2024/00_data_prep/crystal_systems)
  - `type.raw`
  - `type_map.raw`
  - `set.000/coord.npy`
  - `set.000/box.npy`
  - `set.000/energy.npy`
  - `set.000/force.npy`
  - `set.000/property.npy`

Notes:
- Property target is stored as fake energy and then copied to [`property.npy`](experiments_davis2024/00_data_prep/crystal_systems).
- Forces are zero-filled placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure


ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = ROOT / "data/davis2024/energetic_crystals_dataset/energetic_crystals.npz"
FILTERED_INDEX = ROOT / "experiments_davis2024/00_data_prep/filtered_index.json"
OUT_DIR = ROOT / "experiments_davis2024/00_data_prep/crystal_systems"
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


def write_single_system(system_dir: Path, coords_cart: np.ndarray, box: np.ndarray, atom_types: np.ndarray, prop: float) -> None:
    set_dir = system_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)

    coord_flat = coords_cart.reshape(1, -1).astype(np.float64)
    box_flat = box.reshape(1, 9).astype(np.float64)
    energy = np.array([prop], dtype=np.float64)
    force = np.zeros((1, coords_cart.shape[0], 3), dtype=np.float64)

    np.savetxt(system_dir / "type.raw", atom_types.astype(np.int32), fmt="%d")
    (system_dir / "type_map.raw").write_text("\n".join(TYPE_MAP) + "\n", encoding="utf-8")
    np.save(set_dir / "coord.npy", coord_flat)
    np.save(set_dir / "box.npy", box_flat)
    np.save(set_dir / "energy.npy", energy)
    np.save(set_dir / "force.npy", force)
    np.save(set_dir / "property.npy", energy.copy())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = np.load(NPZ_PATH, allow_pickle=True)
    filtered = json.loads(FILTERED_INDEX.read_text(encoding="utf-8"))["kept"]

    for rec in filtered:
        idx = rec["index"]
        refcode = rec["refcode"]
        prop = float(rec["calc_det_velocity_m_s"])

        species = [str(x) for x in bundle["species"][idx]]
        coords_frac = np.array(bundle["coords_frac"][idx], dtype=np.float64)
        lattice_mat = np.array(bundle["lattice"][idx], dtype=np.float64)

        # Validate through pymatgen, then use Cartesian coordinates for DeepMD.
        structure = Structure(Lattice(lattice_mat), species, coords_frac, coords_are_cartesian=False)
        coords_cart = np.array(structure.cart_coords, dtype=np.float64)
        box = np.array(structure.lattice.matrix, dtype=np.float64)
        atom_types = np.array([TYPE_TO_ID[s] for s in species], dtype=np.int32)

        write_single_system(OUT_DIR / refcode, coords_cart, box, atom_types, prop)

    print(f"Wrote crystal systems to {OUT_DIR}")


if __name__ == "__main__":
    main()
