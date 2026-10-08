# REPRODUCTION.md

From a number in the manuscript to the command, files, and value in this
repository. Experiment codes such as `exp7a` are defined in [MANIFEST.md](MANIFEST.md).
Release tiers are defined in [README_RELEASE.md](README_RELEASE.md).

Do not guess a directory from an experiment code. If a claim is not listed here,
it is not yet mapped.

JSON keys use the file names `PEP`, `MPEP`, and `HPEP`. The manuscript prints
those as PEP, PEP-M, and PEP-H.

## How to read an entry

| Field | Meaning |
| --- | --- |
| Manuscript | Where the rounded number appears |
| Tier | 1 = read a file already in git; 2 = rerun inference; 3 = retrain |
| Command | Run from the repository root |
| Inputs | Repository-relative paths |
| Output | File and JSON field that hold the value |
| Expected | Value stored in that field, then the rounded manuscript number |
| Missing | External files this tier needs. No download URL is invented |

`families.exp7a` is MT-FT (energy + detonation velocity, cluster input).
`families.exp7c` is ST-FT (detonation velocity only, same backbone).

The summary file stores Kamlet–Jacobs references rounded to the nearest
m/s (`9090`, `8729`, `8764`). Absolute errors in that file are against those
rounded references. The manuscript text quotes the same MAE after rounding.

## Headline claims

### Crystal-derived ABX4 MAE, 92 m/s

- Manuscript: abstract and the synthesis section. Three new ABX4 compounds, clusters cut from their own crystals, MT-FT five-fold ensemble.
- Tier: 1 to read the cached summary. Tier 2 regenerates `experiments/pems_ood_5fold_exp7a.json` and the summary must be rebuilt from it. Tier 3 retrains the five MT-FT folds first.
- Command (tier 1): read the field below. The same predictions are plotted by `python manuscript/figures/plot_fig5.py`.
- Command (tier 2): `python experiments/infer_pems.py ood --series exp7a`
- Inputs: `data/abx4/cifs/` for PEP, MPEP, and HPEP; five MT-FT checkpoints `experiments/exp7a_fold{0,1,2,3,4}/model.ckpt-400000.pt` (not in git).
- Output: `experiments/ablation_full_eval_summary.json` → `families.exp7a.OOD_new`
- Expected: `mae_m_s` = 92.20011206741037, printed as 92. Per material, `pred` / `ae`: PEP 8870.843948273841 / 219.1560517261587, MPEP 8698.45701997022 / 30.542980029780665, HPEP 8790.901304446292 / 26.90130444629176. The manuscript rounds the predictions to 8871, 8699, and 8791 and the absolute errors to 219, 30, and 27.
- Missing for tier 2–3: the five `exp7a` checkpoints, the DeepMD `.npy` systems under `experiments/00_data_prep/pems_cluster_n{1,2,3}_systems/`, and for tier 3 the backbone `pretrained_models/deepems-lam.pt`. The data-availability URL in `README.md` is still a placeholder.

### Template-built ABX4 MAE, 91 m/s

- Manuscript: abstract and the template-cluster paragraph. Same three compounds and the same MT-FT ensemble, with ions placed on the DAP-4 cluster instead of the new crystals.
- Tier: 1 once the cache file below is present. It is not in this git repository, so tier 1 cannot be run from a fresh clone. Tier 2 is the inference that writes that cache. Tier 3 retrains `exp7a` first.
- Command (tier 1): `python manuscript/figures/plot_si_abx4_ood.py` after the cache exists.
- Inputs: `manuscript/figures/_si_abx4_ood_predictions.json` (absent from git). Own-crystal panels of the same script read `experiments/pems_ood_5fold_exp7{a,c,d}.json`, which are present.
- Output: `aggregated.exp7a.<material>.abx3_template.mean_m_s` for `PEP`, `MPEP`, and `HPEP`.
- Expected: means 8922.953948734217, 8756.969249847518, and 8840.534785735557 (15 samples = 5 folds × 3 cluster realizations). Against the unrounded Kamlet–Jacobs references 9090.284, 8728.664, and 8763.850, the absolute errors are 167.33005126578246, 28.30524984751719, and 76.68478573555694. Their mean is 90.77336228295219, printed as 91.
- Missing: `manuscript/figures/_si_abx4_ood_predictions.json`, plus the same checkpoints and backbone as the crystal-derived entry.

### Five-fold in-distribution MAE, 273 m/s

- Manuscript: results text for MT-FT on the 25 labelled MIX materials. Each material is scored only by the fold that held it out.
- Tier: 1 to read the summary. Tier 2 reruns cross-validation inference into `experiments/pems_predictions.json`. Tier 3 retrains the folds.
- Command (tier 1): read the field below. Figure 3 is `python manuscript/figures/plot_fig3.py`.
- Command (tier 2): `python experiments/infer_pems.py cv --series exp7a`
- Inputs: `data/pems/mix.csv`; split file `experiments/00_data_prep/pems_5fold_splits_v2.json`; cluster systems and `exp7a` checkpoints (not in git).
- Output: `experiments/ablation_full_eval_summary.json` → `families.exp7a.IND_25.mae_m_s`
- Expected: 273.2291890068937, printed as 273. `IND_25.n` is 25. Per-material `pred`, `exp`, and `ae` sit under `families.exp7a.IND_25.per_material`.
- Missing for tier 2–3: the same checkpoints, `.npy` systems, and tier-3 backbone as above.

### Single-task fine-tuning ABX4 MAE, 584 m/s

- Manuscript: synthesis section. ST-FT, the same backbone without the energy and force loss, on the same three crystal-derived ABX4 clusters.
- Tier: 1 to read the summary. Tier 2 regenerates `experiments/pems_ood_5fold_exp7c.json`. Tier 3 retrains `exp7c`.
- Command (tier 2): `python experiments/infer_pems.py ood --series exp7c`
- Inputs: `data/abx4/cifs/`; checkpoints `experiments/exp7c_fold{0,1,2,3,4}/model.ckpt-200000.pt` (not in git; `MANIFEST.md` lists the `exp7c` best checkpoint as 200k).
- Output: `experiments/ablation_full_eval_summary.json` → `families.exp7c.OOD_new.mae_m_s`
- Expected: 584.109100772288, printed as 584.
- Missing for tier 2–3: the five `exp7c` checkpoints and, for tier 3, `pretrained_models/deepems-lam.pt`.

## Figures

Tier 1 for every figure below is the plotting command. It only reads files already named here. Tier 2 and tier 3 are the inference and training commands in the headline section, or the dispatcher named in [MANIFEST.md](MANIFEST.md). Checkpoints, `.npy` systems, and `pretrained_models/deepems-lam.pt` stay outside git.

### Figure 3

- Command: `python manuscript/figures/plot_fig3.py`
- Inputs and the manuscript numbers they support:
  - `experiments/00_data_prep/pems_5fold_splits_v2.json` — five-fold membership.
  - `experiments/ablation_full_eval_summary.json` → `families.exp7a.IND_25.mae_m_s` = 273.2291890068937 (printed 273). `families.exp7a.OOD_heldout.mae_m_s` = 34.46599664726273 (printed 35; the four-material set, with `DAI-1_0.5 4_0.5` excluded). `families.exp7c.IND_25.mae_m_s` is the ST-FT in-distribution error; `families.exp7d.IND_25.mae_m_s` = 297.24107807303386 (printed 297).
  - `experiments/pems_ood_heldout_exp7_all.json` — per-material held-out predictions plotted with that summary.
  - `experiments/cross_infer_rep.json` → `exp7a.cluster_n1.mean_mae` = 270.5686104066625 and `exp7a.crystal.mean_mae` = 405.4248682427654 (cluster-trained model on crystals, printed 405). `exp8a.crystal.mean_mae` = 201.45655902027966 (crystal-trained in-distribution, printed 201). `exp8a.cluster_n1.mean_mae` = 1314.317887025996 (crystal-trained model on clusters, printed 1314).
  - `experiments/exp_ood_pretrained_domain/pretrained_domain_results.json` → `pretrained_domain_mt.dap_mae_m_s` = 143.50336283171976 (printed 144), `pretrained_domain_st.dap_mae_m_s` = 162.51008119057974 (printed 163), `pretrained_domain_sd.dap_mae_m_s` = 296.791533559781 (printed 297).
  - `experiments/pems_sensitivity_summary.json`, `experiments/pems_uq_calibration.json`, and `experiments/pems_ood_model_deviation_exp7a.json` — sensitivity and uncertainty panels. Panel-to-file names are in [MANIFEST.md](MANIFEST.md).
- The manuscript's 1624 m/s figure for the crystal-trained model on the three new ABX4 clusters is not a field in these JSON files, so a fresh clone cannot recompute it at tier 1.

### Figure 4

- Command: `python manuscript/figures/plot_fig4.py`
- Inputs: `experiments/abx_grid_predictions_exp6v1_allpems_400k.json`, `experiments/mechanism_results/mechanism_m{1,3,4a,4b,5a}_results.json`, and `experiments/_stats_bootstrap/bootstrap_results.json`.
- Mechanism probes M0–M5a and the hyperparameter ablation grid are not given separate entries. Their codes, scripts, and result files are the tables in [MANIFEST.md](MANIFEST.md).

### Figure 5

- Command: `python manuscript/figures/plot_fig5.py`
- Inputs: `experiments/pems_ood_5fold_exp7a.json` (crystal-derived PEP / MPEP / HPEP predictions; the 92 m/s MAE is the headline entry above) and the curated assets under `data/abx4/` (`cifs/`, `pxrd/`, `properties.csv`).
- Tier 2 that rewrites the prediction file: `python experiments/infer_pems.py ood --series exp7a`.

### Supplementary and extended-data figures

Run the matching script from the repository root. Each script reads the files in the table. Further panel maps are in [MANIFEST.md](MANIFEST.md).

| Command | Primary inputs |
| --- | --- |
| `python manuscript/figures/plot_si_abx4_ood.py` | `experiments/pems_ood_5fold_exp7{a,c,d}.json`; template panel also needs `manuscript/figures/_si_abx4_ood_predictions.json`, which is absent |
| `python manuscript/figures/plot_si_ood_heldout.py` | `experiments/ablation_full_eval_summary.json` |
| `python manuscript/figures/plot_si_uq_ood_heldout.py` | `experiments/ablation_full_eval_summary.json` |
| `python manuscript/figures/plot_si_davis2024_pems_parity.py` | `experiments/davis2024_pems_zeroshot_predictions.json`, `experiments/davis2024_pems_zeroshot_summary.json` |
| `python manuscript/figures/plot_si_uq_pretrained_domain.py` | `experiments/exp_ood_pretrained_domain/pretrained_domain_results.json` |
| `python manuscript/figures/plot_ed_loso_hierarchy.py` | `experiments/exp_ood_loso/loso_results_summary.json` |
| `python manuscript/figures/plot_ed_periodic_control.py` | `experiments/cross_infer_rep.json` |
| `python manuscript/figures/plot_ed_ood_heldout_umap.py` | `manuscript/figures/_cluster_umap_cache.npz` via `plot_fig5.plot_cluster_umap_panel` |
