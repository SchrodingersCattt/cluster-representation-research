"""M-series mechanism experiments package.

Public API: each experiment module exposes ``run_mX(output_dir)`` and a
plotting helper. The CLI dispatcher lives at
``experiments/run_mechanism_analysis.py``.

Layout:
- runtime         -- mutable runtime state (CKPT_STEP, PLOT_STYLE, ACTIVE_FOLD_IDS)
- paths           -- path constants
- constants       -- static configuration (TYPE_MAP, MODEL_CONFIGS, COLORS, ...)
- io_models       -- checkpoint resolution + DeepProperty/DeepPot loaders
- io_data         -- cluster-system reader, materials, density, OB, fold splits
- inference       -- predict_single + descriptor extractors
- stats           -- bootstrap CI helpers
- plot_helpers    -- thin wrappers around ``paper_plot_style`` for mechanism plots
- perturbations   -- registry of M1 perturbations (destructive + template)
- m{0,1,2,2_bridge,3,3b,4a,4b,5a}_*  -- one file per experiment
"""
"""M-series mechanism experiments package.

Public API: each experiment module exposes ``run_mX(output_dir)`` and a
plotting helper. The CLI dispatcher lives at
``experiments/run_mechanism_analysis.py``.

Layout:
- runtime         -- mutable runtime state (CKPT_STEP, PLOT_STYLE, ACTIVE_FOLD_IDS)
- paths           -- path constants
- constants       -- static configuration (TYPE_MAP, MODEL_CONFIGS, COLORS, ...)
- io_models       -- checkpoint resolution + DeepProperty/DeepPot loaders
- io_data         -- cluster-system reader, materials, density, OB, fold splits
- inference       -- predict_single + descriptor extractors
- stats           -- bootstrap CI helpers
- plot_helpers    -- thin wrappers around ``paper_plot_style`` for mechanism plots
- perturbations   -- registry of M1 perturbations (destructive + template)
- m{0,1,2,2_bridge,3,3b,4a,4b,5a}_*  -- one file per experiment
"""
