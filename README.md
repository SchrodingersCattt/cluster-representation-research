# Stoichiometric Cluster Learning for Multi-ionic Energetic Salts

Code and data for:

> **Stoichiometric cluster learning for few-shot property prediction of multi-ionic energetic salts**
>
> Ming-Yu Guo†, Wei-Jia Zou†, Yu Shang\*, Wei-Xiong Zhang\*
>
> Sun Yat-sen University, Guangzhou

## What this repository contains

| Directory | Contents |
|-----------|----------|
| `experiments/` | Core inference, data preparation, and mechanism analysis scripts |
| `experiments/mechanism/` | Mechanism analysis subpackage (M0–M5a probes) |
| `experiments/00_data_prep/` | Data preparation pipeline + cleaned CIFs + CV splits |
| `experiments/*.json` | Pre-computed result files consumed by figure scripts |
| `manuscript/figures/` | All publication figure scripts (Figures 3–5, SI, ED) |
| `data/pems/mix.csv` | Full MIX dataset (39 materials, 25 with experimental Vdet) |
| `data/pems/confs/` | Original CIF files for MIX materials |
| `AGENTS.md` | AI collaboration document (see note below) |
| `MANIFEST.md` | **Experiment code ↔ manuscript name mapping** |

## Important notes

### Experiment naming convention

Throughout the codebase, experiments are referred to by internal codes such as
`exp7a`, `exp7c`, `exp8a`, etc. These codes appear in filenames, directory names,
JSON keys, and script variables. **See [`MANIFEST.md`](MANIFEST.md) for a
complete mapping between these codes and the names used in the manuscript**
(e.g., `exp7a` = "MT-FT", the multi-task fine-tuned baseline).

### Naming: `davis2024` and `pems`

- The directory `experiments/` refers to the Davis et al. (2024)
  energetic crystal dataset. The codebase historically used the misspelling
  `david2024`; this has been corrected to `davis2024` in this release. However,
  **model head names embedded in checkpoint files** (e.g., `david2024_vdet_kj`)
  retain the historical spelling and cannot be renamed without retraining.
  These are annotated with comments in the source code.

- `mix.csv` and the `pems_*` prefix both refer to the same MIX (multi-ionic
  crystalline) dataset. The `pems_` prefix is a historical artifact used
  consistently in directory names and variable names (e.g.,
  `pems_cluster_n1_systems/`, `pems_5fold_splits.json`).

### About `AGENTS.md`

`AGENTS.md` is a machine-readable guidance document used during AI-assisted
development. It encodes project constraints, key algorithms (e.g., the
stoichiometric cluster pipeline), and coding conventions. It is provided for
transparency and reproducibility of the AI-assisted research workflow — you
do not need to read it to use this code.

## Data Availability

Training data (DeepMD-kit npy format), pretrained backbone, and all fine-tuned
model checkpoints are available at:

> **[TODO: Insert HuggingFace/Figshare URL]**

See the Data Availability repository's `MANIFEST.md` for contents.

The DPA-3.2-5M pretrained model is available at:
[AISSquare](https://www.aissquare.com/models/detail?pageType=models&name=DPA-3.2-5M&id=392).

The Davis et al. (2024) energetic crystal dataset is described in:
Davis, J. V. et al. *Machine Learning Models for High Explosive Crystal Density
and Performance.* Propellants Explos. Pyrotech. **49**, e202400060 (2024).

## Environment setup

```bash
# Create conda environment with deepmd-kit >= 3.0
conda create -n cluster-rep python=3.10
conda activate cluster-rep
pip install deepmd-kit[torch] ase pymatgen scikit-learn matplotlib

# MolCrysKit (required for cluster pipeline)
pip install molcrys-kit  # or install from source
```

## Reproduction workflow

### 1. Data preparation

```bash
cd experiments/00_data_prep
python prep_pems_npy.py          # Build cluster + crystal training systems
python prep_crystal_npy.py       # Build Davis2024 crystal systems
```

### 2. Training (requires DeepMD-kit)

```bash
cd experiments/exp7a_fold0  # Example: multi-task baseline, fold 0
dp --pt train input.json --finetune deepems-lam.pt
```

Training configs (`input.json`) are provided in the Data Availability
repository alongside each model checkpoint.

### 3. Inference

```bash
cd experiments
python infer_pems.py cv           # 5-fold cross-validation
python infer_pems.py ood          # OOD ensemble predictions
python predict_abx_grid.py       # ABX combinatorial grid
python run_mechanism_analysis.py  # Mechanism probes (M0–M5a)
```

### 4. Figures

```bash
cd manuscript/figures
python plot_fig3.py               # Figure 3: CV + representation comparison
python plot_fig4.py               # Figure 4: ABX grid + mechanism
python plot_fig5.py               # Figure 5: OOD predictions + synthesis
```

## Repository structure

```
.
├── README.md
├── AGENTS.md                      # AI agent guidance document
├── MANIFEST.md                    # Experiment code mapping
├── data/pems/
│   ├── mix.csv                    # 39 materials (ground truth)
│   └── confs/                     # Original CIF files
├── experiments/
│   ├── infer_pems.py              # Unified PEMs inference
│   ├── predict_abx_grid.py        # ABX combinatorial grid
│   ├── run_mechanism_analysis.py   # Mechanism analysis dispatcher
│   ├── mechanism/                  # M0–M5a analysis modules
│   ├── 00_data_prep/               # Data pipeline + cleaned CIFs
│   ├── paper_plot_style.py         # Publication figure style
│   └── *.json                      # Pre-computed results
└── manuscript/figures/
    ├── plot_fig3.py                # Main figures
    ├── plot_fig4.py
    ├── plot_fig5.py
    └── plot_si_*.py                # SI figures
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite:

```bibtex
@article{guo2026cluster,
  title={Stoichiometric cluster learning for few-shot property prediction
         of multi-ionic energetic salts},
  author={Guo, Ming-Yu and Zou, Wei-Jia and Shang, Yu and Zhang, Wei-Xiong},
  journal={Nature Computational Science},
  year={2026}
}
```
