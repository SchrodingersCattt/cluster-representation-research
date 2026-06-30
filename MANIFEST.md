# MANIFEST.md

Mapping between internal experiment codes used in this repository and the names
used in the manuscript.

## 1. Experiment Code → Manuscript Name

### Main experiments (5-fold cross-validation)

| Code | Manuscript Name | Heads | Representation | Folds | Best Ckpt |
|------|----------------|-------|----------------|-------|-----------|
| `exp7a` | MT-FT (Multi-task fine-tuned) | energy + Vdet | cluster | 5 | 400k |
| `exp7b` | MT-FT-aux (Multi-task auxiliary) | energy + Davis2024 + Vdet | cluster | 5 | 600k |
| `exp7c` | ST-FT (Single-task fine-tuned) | Vdet only | cluster | 5 | 200k |
| `exp7d` | ST-TFS (Single-task from scratch) | Vdet only (random init) | cluster | 5 | 200k |
| `exp8a` | Crystal-representation variant | energy + Vdet | crystal (pbc) | 5 | 400k |
| `exp9a` | DD-ranked CV variant | energy + Vdet | cluster | 5 | 400k |

### Full-data model (no CV split)

| Code | Manuscript Name | Description |
|------|----------------|-------------|
| `exp6v1_allpems` | Full-data multi-task model | Canonical 2-head MT model, all 25 MIX materials, cluster input |

### Sub-experiments

| Code | Manuscript Name | Description |
|------|----------------|-------------|
| `exp_ood_loso` | Leave-one-species-out | 11 species × 2 variants (MT/ST) = 22 models |
| `exp_ood_pretrained_domain` | Pretrained-domain OOD | 3 models (MT/ST/TFS) |
| `ablation/*` | Hyperparameter ablations | 14 families × 5 folds = 70 models |

### Ablation families

| Code | Ablation variable | Relative to |
|------|-------------------|-------------|
| `exp7a_200k` | Training steps (200k vs 400k) | exp7a |
| `exp7a_800k` | Training steps (800k vs 400k) | exp7a |
| `exp7a_decay200` | Decay ratio (200 vs default) | exp7a |
| `exp7a_lr1e4` | Learning rate (1e-4 vs 2e-5) | exp7a |
| `exp7a_lr5e6` | Learning rate (5e-6 vs 2e-5) | exp7a |
| `exp7a_seed7` | CV split seed (7 vs 42) | exp7a |
| `exp7a_seed13` | CV split seed (13 vs 42) | exp7a |
| `exp7b_lr1e4` | Learning rate | exp7b |
| `exp7b_lr5e6` | Learning rate | exp7b |
| `exp7c_decay200` | Decay ratio | exp7c |
| `exp7c_lr1e4` | Learning rate | exp7c |
| `exp7c_seed7` | CV split seed | exp7c |
| `exp7c_seed13` | CV split seed | exp7c |
| `exp7d_lr1e4` | Learning rate | exp7d |

---

## 2. Mechanism Analysis Code → Manuscript Name

| Code | Manuscript Name | Description |
|------|----------------|-------------|
| `M0` | Perturbation robustness analysis | Random coordinate perturbation sensitivity |
| `M1` | Geometry-scrambling rank-preservation | Composition probe via molecular replacement |
| `M2` | Uniform scaling sensitivity | Density dependence via isotropic scaling |
| `M2-bridge` | Density-in-embedding bridge | Embedding distance vs density change |
| `M3` | Linear probe analysis | Ridge regression on descriptor embeddings |
| `M3b` | Nonlinear probe analysis | MLP probe on descriptor embeddings |
| `M4a` | Embedding compactness | UMAP + family clustering of descriptors |
| `M4b` | Site-resolved atomic embedding | Per-atom embedding ANOVA by A/B/X site |
| `M5a` | Embedding stability | Training step convergence of embeddings |

---

## 3. Result JSON → Figure Mapping

Each JSON file is consumed by one or more figure scripts.

### Main figures

| JSON file | Figure script | Panel |
|-----------|--------------|-------|
| `pems_ood_heldout_exp7_all.json` | `plot_fig3.py` | Panels b, c |
| `cross_infer_rep.json` | `plot_fig3.py` | Panel d |
| `pems_sensitivity_summary.json` | `plot_fig3.py` | Panel e |
| `pems_uq_calibration.json` | `plot_fig3.py` | Panel f |
| `pems_ood_model_deviation_exp7a.json` | `plot_fig3.py` | UQ overlay |
| `00_data_prep/pems_5fold_splits_v2.json` | `plot_fig3.py` | Panel a |
| `abx_grid_predictions_exp6v1_allpems_400k.json` | `plot_fig4.py` | Panel a |
| `mechanism_results/mechanism_m1_results.json` | `plot_fig4.py` | Panel b |
| `mechanism_results/mechanism_m3_results.json` | `plot_fig4.py` | Panel c |
| `mechanism_results/mechanism_m4a_results.json` | `plot_fig4.py` | Panel c |
| `mechanism_results/mechanism_m4b_results.json` | `plot_fig4.py` | Panel d |
| `mechanism_results/mechanism_m5a_results.json` | `plot_fig4.py` | Panel d |
| `_stats_bootstrap/bootstrap_results.json` | `plot_fig4.py` | Error bars |
| `pems_ood_5fold_exp7a.json` | `plot_fig5.py` | All panels |

### Extended Data / SI figures

| JSON file | Figure script |
|-----------|--------------|
| `exp_ood_loso/loso_results_summary.json` | `plot_ed_loso_hierarchy.py` |
| `cross_infer_rep.json` | `plot_ed_periodic_control.py` |
| `ablation_full_eval_summary.json` | `plot_si_ood_heldout.py`, `plot_si_uq_ood_heldout.py`, `_emit_si_ablation_tables.py` |
| `pems_ood_5fold_exp7a.json` | `plot_si_abx4_ood.py` |
| `pems_ood_5fold_exp7c.json` | `plot_si_abx4_ood.py` |
| `pems_ood_5fold_exp7d.json` | `plot_si_abx4_ood.py` |
| `davis2024_pems_zeroshot_predictions.json` | `plot_si_davis2024_pems_parity.py` |
| `davis2024_pems_zeroshot_summary.json` | `plot_si_davis2024_pems_parity.py` |
| `exp_ood_pretrained_domain/pretrained_domain_results.json` | `plot_fig3.py`, `plot_si_uq_pretrained_domain.py` |
| `exp_ood_pretrained_domain/_descriptor_distances/descriptor_distances.json` | `plot_fig3.py` |

---

## 4. Model Head Names (Historical)

The following head names are embedded in model checkpoint files and **cannot be
renamed** without retraining:

| Head name in checkpoint | Correct reference |
|------------------------|-------------------|
| `david2024_vdet_kj` | Davis et al. 2024 calculated Vdet (kJ/cc → m/s) |
| `david2024_vdet_exp` | Davis et al. 2024 experimental Vdet |
| `pems_vdet_kj` | MIX experimental Vdet (m/s) |

The `david2024` prefix is a historical misspelling of "Davis". Code references
to these head names include an explanatory comment.
