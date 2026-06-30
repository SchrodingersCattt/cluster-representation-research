"""Per-system property and descriptor inference helpers."""
from __future__ import annotations

import numpy as np


def predict_single(model, coord: np.ndarray, symbols: list[str]) -> float:
    tm = model.get_type_map()
    at = np.array([tm.index(s) for s in symbols], dtype=np.int32)
    return float(model.eval(coords=coord.reshape(1, -1, 3), atom_types=at, cells=None)[0].flatten()[0])


def extract_descriptor(dp_model, coord: np.ndarray, symbols: list[str]) -> np.ndarray:
    return extract_descriptor_per_atom(dp_model, coord, symbols).mean(axis=0)


def extract_descriptor_per_atom(dp_model, coord: np.ndarray, symbols: list[str]) -> np.ndarray:
    """Per-atom descriptors, shape (n_atoms, dim)."""
    tm = dp_model.get_type_map()
    at = np.array([tm.index(s) for s in symbols], dtype=np.int32)
    coords_f = np.asarray(coord, dtype=np.float64).reshape(1, -1, 3)
    desc = dp_model.eval_descriptor(coords_f, None, at)
    if isinstance(desc, (list, tuple)):
        desc = desc[0]
    arr = np.asarray(desc)
    if arr.ndim == 3:
        arr = arr[0]
    return arr.reshape(-1, arr.shape[-1])
