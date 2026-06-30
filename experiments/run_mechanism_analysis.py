#!/usr/bin/env python3
"""M-series mechanism analysis CLI dispatcher.

Central question: How does DFT energy regularization change representation
quality and robustness?

The actual experiment logic lives in the ``mechanism/`` subpackage; this file
is a thin command-line dispatcher that wires runtime configuration
(``--folds``/``--ckpt-step``/``--plot-style``) into ``mechanism.runtime`` and
then calls each ``run_mX(output_dir)`` entrypoint.

Experiments:
  M0  -- Perturbation sensitivity from existing data (no new inference)
  M1  -- Composition probe + DAP-4 template invariance (exp7a, exp7c[, exp7d])
         (M5b "DAP-4 template invariance" was merged into M1 in 2026-04
         under the ``template_dap4`` perturbation; see
         ``mechanism/perturbations.py``)
  M2  -- Distance scaling (exp7a vs exp7c)
  M2 bridge (m2bridge / m2emb / m2probe / m2grad) -- embedding distance vs
         scale, Ridge density probe on scaled embeddings, head gradient vs
         probe direction
  M3  -- Linear probe (embedding content analysis)
  M3b -- Nonlinear probe: Ridge vs MLP on embeddings (5-fold CV)
  M4a -- Embedding compactness (exp7a vs exp7c)
  M4b -- Site-resolved embedding analysis (A/B/X decomposition, ANOVA)
  M5a -- Cross-fold descriptor stability (exp7a vs exp7c vs exp7d)
  M6  -- Early-stop checkpoint diagnostics (accuracy, probes, drift, OOD stress)

Usage:
    conda activate dpa3
    python experiments/run_mechanism_analysis.py --experiments m0,m1,m2,m3,m4a
    python experiments/run_mechanism_analysis.py --ckpt-step 400000   # default; -1 = latest
    python experiments/run_mechanism_analysis.py --experiments m1 --skip-inference
"""
from __future__ import annotations

import argparse
import matplotlib

matplotlib.use("Agg")

from mechanism import paths, runtime
from mechanism.m0_perturbation import run_m0
from mechanism.m1_composition import run_m1
from mechanism.m2_bridge import run_m2_bridge
from mechanism.m2_density import run_m2
from mechanism.m3_linear_probe import run_m3
from mechanism.m3b_nonlinear_probe import run_m3b
from mechanism.m4a_embedding import run_m4a
from mechanism.m4b_atomic import run_m4b
from mechanism.m5a_stability import run_m5a
from mechanism.m6_early_stop import run_m6
from mechanism.plot_helpers import setup_nature_style

# Dispatch table -- "alias -> (callable, kwargs accepted)"
# Aliases that share a callable (e.g. m2bridge/m2emb/m2probe/m2grad all map
# to run_m2_bridge) are folded into a single entry below.
_DISPATCH: dict[str, tuple[callable, set[str]]] = {
    "m0":  (run_m0, set()),
    "m1":  (run_m1, {"skip_inference"}),
    "m2":  (run_m2, {"skip_inference"}),
    "m3":  (run_m3, set()),
    "m3b": (run_m3b, set()),
    "m4a": (run_m4a, set()),
    "m4b": (run_m4b, set()),
    "m5a": (run_m5a, set()),
    "m6":  (run_m6, {"models", "steps", "no_umap", "refresh_cache"}),
    "early_stop": (run_m6, {"models", "steps", "no_umap", "refresh_cache"}),
}
_M2_BRIDGE_ALIASES = {"m2bridge", "m2emb", "m2probe", "m2grad"}


def _dispatch(name: str, *, skip_inference: bool) -> bool:
    if name in _M2_BRIDGE_ALIASES:
        run_m2_bridge(paths.OUTPUT_DIR)
        return True
    entry = _DISPATCH.get(name)
    if entry is None:
        return False
    fn, accepted = entry
    kwargs: dict = {}
    if "skip_inference" in accepted:
        kwargs["skip_inference"] = skip_inference
    if "models" in accepted:
        kwargs["models"] = getattr(_dispatch, "m6_models", None)
    if "steps" in accepted:
        kwargs["steps"] = getattr(_dispatch, "m6_steps", None)
    if "no_umap" in accepted:
        kwargs["no_umap"] = getattr(_dispatch, "m6_no_umap", False)
    if "refresh_cache" in accepted:
        kwargs["refresh_cache"] = getattr(_dispatch, "m6_refresh_cache", False)
    fn(paths.OUTPUT_DIR, **kwargs)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="M-series mechanism analysis")
    parser.add_argument(
        "--experiments",
        default="m0,m1,m2,m3,m4a",
        help="Comma-separated subset of m0,m1,m2,m2bridge,m3,m3b,m4a,m4b,m5a,m6,early_stop.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Reuse cached predictions where supported (m1, m2).",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated checkpoint fold indices (e.g. 0,1,2,3,4). Default: all 5.",
    )
    parser.add_argument(
        "--plot-style",
        choices=["paper", "analysis"],
        default="paper",
        help="Plotting mode. 'paper' saves Nature-style figures to paper_figures/ and supplementary/.",
    )
    parser.add_argument(
        "--ckpt-step",
        type=int,
        default=400000,
        help="Use model.ckpt-N.pt for all model loads (default 400000). Use -1 for latest checkpoint.",
    )
    parser.add_argument(
        "--m6-models",
        type=str,
        default=None,
        help="Comma-separated M6 early-stop models (default: exp7a,exp7c,exp7d).",
    )
    parser.add_argument(
        "--m6-steps",
        type=str,
        default=None,
        help="Comma-separated M6 checkpoint steps (default: sparse early/mid/late grid).",
    )
    parser.add_argument(
        "--m6-no-umap",
        action="store_true",
        help="Skip M6 UMAP/PCA snapshot plotting for a faster metric-only run.",
    )
    parser.add_argument(
        "--m6-refresh-cache",
        action="store_true",
        help="Refresh M6 descriptor/prediction caches.",
    )
    args = parser.parse_args()

    fold_ids = [int(x.strip()) for x in args.folds.split(",") if x.strip() != ""]
    if not fold_ids:
        fold_ids = [0, 1, 2, 3, 4]
    runtime.configure(
        ckpt_step=None if args.ckpt_step < 0 else args.ckpt_step,
        plot_style=args.plot_style,
        fold_ids=fold_ids,
    )
    setup_nature_style()
    _dispatch.m6_models = args.m6_models
    _dispatch.m6_steps = args.m6_steps
    _dispatch.m6_no_umap = args.m6_no_umap
    _dispatch.m6_refresh_cache = args.m6_refresh_cache

    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths.SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [e.strip().lower() for e in args.experiments.split(",") if e.strip()]
    unknown = [
        e for e in experiments
        if e not in _DISPATCH and e not in _M2_BRIDGE_ALIASES
    ]
    if unknown:
        parser.error(
            f"Unknown experiment alias(es): {unknown}. "
            f"Available: {sorted(set(_DISPATCH) | _M2_BRIDGE_ALIASES)}"
        )

    for name in experiments:
        if not _dispatch(name, skip_inference=args.skip_inference):
            print(f"  [skip] no dispatcher for '{name}'")

    print("\n" + "=" * 60 + f"\nAll done. Results in: {paths.OUTPUT_DIR}\n" + "=" * 60)


if __name__ == "__main__":
    main()
