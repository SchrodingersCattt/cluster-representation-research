"""Lightweight statistics helpers shared by M-series experiments."""
from __future__ import annotations

import numpy as np


def bootstrap_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for R^2 (resample rows with replacement).

    Samples where y_true has near-zero variance are discarded to avoid
    degenerate R^2 values (+/- inf or extreme negatives) from constant draws.
    """
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 3:
        return float("nan"), float("nan")
    r2s: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.var(y_true[idx]) < 1e-6:
            continue
        r2s.append(float(r2_score(y_true[idx], y_pred[idx])))
    if len(r2s) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(r2s, [(100.0 - ci) / 2, 100.0 - (100.0 - ci) / 2])
    return float(lo), float(hi)
