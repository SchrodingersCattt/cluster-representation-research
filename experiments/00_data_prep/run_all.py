#!/usr/bin/env python3
"""
Run the David 2024 experiment preparation pipeline in order.

Steps:
1. [`inspect_dataset_and_models.py`](experiments/00_data_prep/inspect_dataset_and_models.py)
2. [`build_filtered_index.py`](experiments/00_data_prep/build_filtered_index.py)
3. [`run_cleaning.py`](experiments/00_data_prep/run_cleaning.py) — removes >500 atoms, close contacts, missing H
4. [`prep_crystal_npy.py`](experiments/00_data_prep/prep_crystal_npy.py)
5. [`prep_molecule_npy.py`](experiments/00_data_prep/prep_molecule_npy.py)
6. [`prep_exp_val.py`](experiments/00_data_prep/prep_exp_val.py)
7. [`make_splits.py`](experiments/00_data_prep/make_splits.py)
8. [`gen_inputs.py`](experiments/00_data_prep/gen_inputs.py)
9. [`gen_submit_yaml.py`](experiments/00_data_prep/gen_submit_yaml.py)

Preflight checks:
- [`rdkit`](https://www.rdkit.org/) is required for molecule filtering/generation.
- [`pymatgen`](https://pymatgen.org/) is required for crystal conversion.
"""

from __future__ import annotations

import importlib
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "inspect_dataset_and_models.py",
    "build_filtered_index.py",
    "run_cleaning.py",
    "prep_crystal_npy.py",
    "prep_molecule_npy.py",
    "prep_exp_val.py",
    "make_splits.py",
    "gen_inputs.py",
    "gen_submit_yaml.py",
]


def _require_module(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Required dependency '{module_name}' is not available in the active environment. "
            f"Please install it or run this pipeline in an environment that contains it."
        )


def main() -> None:
    _require_module("torch")
    _require_module("numpy")
    _require_module("pymatgen")
    _require_module("rdkit")

    for script_name in SCRIPTS:
        script_path = ROOT / script_name
        print(f"\n===== Running {script_path.name} =====")
        runpy.run_path(str(script_path), run_name="__main__")
    print("\nAll David 2024 preparation steps completed.")


if __name__ == "__main__":
    main()
