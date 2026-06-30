# AGENTS.md

This file provides guidance to AI coding agents when working with this repository.
It is **not** a human-readable README — see [README.md](README.md) for that.

## Project Overview

Fine-tune DPA-3 / DeepEMs-LAM foundation models to predict detonation velocity
(m/s) for multi-ionic energetic salts (MIX perovskite-type materials).
Key finding: stoichiometric vacancy-cluster representation (nopbc) outperforms
crystal representation for transfer to unseen stoichiometries.

Two datasets:
- **Davis et al. 2024**: 14,244 CHNO energetic crystals with calculated Vdet.
- **MIX (this work)**: 39 materials (25 with experimental Vdet), covering ABX₃,
  A₂BX₅, and ABX₄ stoichiometries.

See [MANIFEST.md](MANIFEST.md) for experiment code ↔ manuscript name mapping.

## Environment

Requires: `deepmd-kit >= 3.0`, `MolCrysKit`, `ASE`, `pymatgen`, `numpy`,
`matplotlib`, `scikit-learn`.

## Critical: MolCrysKit Cluster Pipeline

The canonical model (`exp6v1_allpems`) was trained on **clusters**, not crystals.

```python
from molcrys_kit.structures.crystal import MolecularCrystal
from molcrys_kit.analysis.stoichiometry import StoichiometryAnalyzer
from molcrys_kit.operations.defects import VacancyGenerator

# 1. Parse CIF → MolecularCrystal
atoms = ase_read(str(cif_path))
crystal = MolecularCrystal.from_ase(atoms, bond_thresholds=PEM_BOND_THRESHOLDS)

# 2. Build supercell
supercell = crystal.get_supercell(*sc_dims)  # e.g. (2,2,2)

# 3. Identify stoichiometric unit
analyzer = StoichiometryAnalyzer(supercell)
simplest_unit = analyzer.get_simplest_unit()

# 4. Select seed molecule (spread across supercell)
seed_candidates = analyzer.species_map.get(seed_species_id, [])

# 5. Extract cluster via vacancy generation
generator = VacancyGenerator(supercell)
_, cluster_crystal = generator.generate_vacancy(
    target_spec=simplest_unit,
    seed_index=seed_index,
    return_removed_cluster=True,
    random_seed=seed,
)

# 6. Minimum-image wrap + center in 100 Å box
cluster_atoms = crystal_to_minimum_image_atoms(cluster_crystal)
cluster_atoms.positions = (
    cluster_atoms.get_positions()
    - cluster_atoms.get_positions().mean(axis=0, keepdims=True)
    + np.array([[50.0, 50.0, 50.0]])
)
cluster_atoms.set_cell(np.eye(3) * 100.0)
cluster_atoms.set_pbc(False)
```

**`PEM_BOND_THRESHOLDS` is mandatory** — without it, `MolecularCrystal.from_ase()`
uses a 3.5 Å default cutoff that merges ionic metal···O contacts into wrong
molecules. Defined in `prep_pems_npy.py` and `infer_pems.py`.

**Supercell schedule**: `[(2,2,2), (3,3,3), (2,3,3), (3,3,4), (4,4,4)]` — use
the smallest that yields ≥ 3 distinct seed candidates.

**Three variants per material**: `cluster_n1` (seed=101), `cluster_n2` (seed=202),
`cluster_n3` (seed=303).

## DeepMD Inference

```python
from deepmd.pt.infer.deep_eval import DeepProperty

model = DeepProperty(str(ckpt_path), head="pems_vdet_kj")
# NOTE: the Davis2024 head is named "david2024_vdet_kj" inside checkpoints
# (historical misspelling; correct surname is "Davis").
model_type_map = model.get_type_map()
atom_types_model = np.array([
    np.where(np.array(model_type_map) == t)[0][0] for t in unique_types
], dtype=np.int32)
pred = model.eval(coords=coords, atom_types=atom_types_for_model, cells=None)
```

`cells=None` for nopbc clusters. The head `pems_vdet_kj` predicts in m/s.

## Key Files

| File | Purpose |
|------|---------|
| `experiments/00_data_prep/prep_pems_npy.py` | Cluster build pipeline |
| `experiments/infer_pems.py` | Unified inference (IND CV, OOD, UQ) |
| `experiments/cross_infer_rep.py` | Cluster vs crystal MAE |
| `experiments/predict_abx_grid.py` | ABX combinatorial grid |
| `experiments/run_mechanism_analysis.py` | M-series dispatcher |
| `experiments/mechanism/` | Mechanism analysis subpackage |
| `experiments/paper_plot_style.py` | Publication figure style |
| `manuscript/figures/plot_fig{3,4,5}.py` | Main figure scripts |
| `data/pems/mix.csv` | Ground truth (39 materials) |

## Ion Substitution Rules

- **K → NH₄**: remove K, add N at K position, add 4H at tetrahedral offsets
  (±1,±1,±1)/√3 × 1.03 Å.
- **NH₄ → K**: identify NH₄ nitrogen by absence of C neighbors within 1.7 Å.
  Remove N + all H within 1.3 Å, add K.

## Figure Style

All publication figures use `paper_plot_style.py` as single source of truth.

| Function | Purpose |
|----------|---------|
| `setup_nature_style()` | Set rcParams (Arial, Nature font sizes, PDF fonttype 42) |
| `style_axes(ax)` | Remove top/right spines |
| `add_panel_label(ax, label)` | Bold uppercase at (-0.12, 1.03) |
| `save_png_pdf(fig, path)` | Save PNG + PDF at 300 dpi |

### Color palette

| Key | Hex | Usage |
|-----|-----|-------|
| `exp7a` | `#205C77` | Multi-task baseline |
| `exp7b` | `#657217` | Multi-task auxiliary-head |
| `exp7c` | `#931143` | Single-task pretrained |
| `exp7d` | `#474747` | Single-task scratch |

## Constraints

- Never use "corpus" for labelled datasets; say "dataset".
- Prefer descriptive manuscript names over internal codes in prose.
- All comparisons must be quantified (MAE in m/s, not "better").
- Zero-trust: verify facts against raw data files, not cached summaries.
