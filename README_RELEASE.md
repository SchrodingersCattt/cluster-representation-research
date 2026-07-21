# Release completeness notes

This repository is organized in three reproducibility tiers.

## Tier 1: lightweight code + cached figure/results reproduction

Included in this Git repository:

- manuscript figure scripts in `manuscript/figures/`
- reusable visualization/source modules in `src/`, including:
  - `src/crystal_viewer/` for structure rendering
  - `src/stoich_cluster_learning/viz/` for topology, coordination, and polyhedra helpers
- precomputed JSON result files in `experiments/`
- curated Figure 5 ABX4 assets in `data/abx4/`:
  - CIF files under `data/abx4/cifs/`
  - converted PXRD traces under `data/abx4/pxrd/`
  - material constants under `data/abx4/properties.csv`
- cached figure assets needed to avoid rerunning descriptor extraction, including:
  - `manuscript/figures/fig4_cache.npz`
  - `manuscript/figures/_cluster_umap_cache.npz`
  - `manuscript/figures/_material_pooled_umap_cache.npz`
- `data/pems/pems.csv` as a compatibility alias for the MIX table

The original working `ABX4_expdata/` tree is **not** part of the release. Its
code has been promoted into `src/`, and only the small data subset required by
the public figure scripts is kept under `data/abx4/`.

Sanity check:

```bash
python scripts/check_release_assets.py --tier figures
```

Figure scripts should be run from the repository root or their own directory;
they resolve all release-local paths relative to this repository.

## Tier 2: inference reproduction

Inference requires large external assets that are not meant to be stored in Git:

- fine-tuned DeepMD checkpoints (`model.ckpt-*.pt`)
- DeepMD-kit `.npy` systems under `experiments/00_data_prep/pems_cluster_*_systems/`
- optional crystal-system `.npy` directories for periodic controls

Expected placement after unpacking the data-availability archive:

```text
experiments/exp6v1_allpems/model.ckpt-400000.pt
experiments/exp7a_fold0/model.ckpt-400000.pt
experiments/exp7a_fold1/model.ckpt-400000.pt
experiments/exp7a_fold2/model.ckpt-400000.pt
experiments/exp7a_fold3/model.ckpt-400000.pt
experiments/exp7a_fold4/model.ckpt-400000.pt
experiments/00_data_prep/pems_cluster_n1_systems/
experiments/00_data_prep/pems_cluster_n2_systems/
experiments/00_data_prep/pems_cluster_n3_systems/
```

Check external inference assets strictly:

```bash
python scripts/check_release_assets.py --tier inference --strict-external
```

## Tier 3: full training reproduction

Full training additionally requires:

- the pretrained backbone (`pretrained_models/deepems-lam.pt`)
- Davis2024 DeepMD training systems reconstructed from Davis et al. (2024)
- the training configs in `experiments/training_configs/`

The Davis2024 raw dataset is not redistributed here; see the Data Availability notes.

## Dependency policy

`uv` is used for the lightweight Python analysis/figure environment. The source
layout is declared in `pyproject.toml`:

```bash
uv sync --extra figures --extra viewer
```

DeepMD/PyTorch/CUDA execution is environment-sensitive, so it is documented separately in `environment.yml` rather than treated as fully solved by `uv.lock`.

If MolCrysKit is installed from source instead of a package, set:

```bash
export MOLCRYSKIT_ROOT=/path/to/MolCrysKit
```
