"""Perturbation registry for M1 composition probe + template invariance.

Two kinds of perturbations live here:

    destructive    -- geometry is destroyed (scrambled / random / sphere / ...).
                      Probes the model's reliance on geometric arrangement.
    polyhedron_preserve
                   -- local polyhedral shape is preserved while selected
                      A-X, B-X, or A-B COM distances are scaled.
    polyhedron_break
                   -- fragment-level COM swaps that preserve each molecule
                      internally but disrupt the local polyhedral tiling.
    template       -- chemistry preserved, geometry transplanted onto a
                      universal local-polyhedra template (currently only DAP-4).
                      Probes the model's invariance to the choice of geometric
                      framework. Merged from M5b in the 2026-04 refactor.

To add a new local-polyhedra template (e.g. ``template_pap6``):
    1. Build the template dataset under ``00_data_prep/`` (mirror
       ``pems_dap4_template_systems``).
    2. Add a ``Perturbation`` entry below with kind="template" and source_dir
       pointing to the new dataset.
    3. Re-run ``run_mechanism_analysis.py --experiments m1``.

The new key (e.g. ``template_pap6``) will appear automatically under
``aggregated.<model>`` and ``per_fold.<fid>.<model>`` in
``mechanism_m1_results.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Perturbation:
    """Single perturbation entry.

    Attributes:
        id:         unique key, e.g. "scrambled_swap", "template_dap4".
        kind:       "destructive", "polyhedron_preserve",
                    "polyhedron_break", or "template".
        n_seeds:    number of stochastic realisations per material.
                    Use 1 for deterministic perturbations.
        models:     model keys participating in this perturbation.
        source_dir: base directory containing one subfolder per material.
                    For seeded perturbations the suffix _s{i} is appended to
                    source_dir.name (matches build_mechanism_perturbations.py).
        seeded:     when True, append _s{i} to source_dir.name for each seed.
    """
    id: str
    kind: str
    n_seeds: int
    models: tuple[str, ...]
    source_dir: Path
    seeded: bool = False

    def system_path(self, mat: str, seed: int = 0) -> Path:
        """Return the cluster-system directory for one material under this perturbation."""
        if self.seeded:
            return self.source_dir.with_name(f"{self.source_dir.name}_s{seed}") / mat
        return self.source_dir / mat


PERTURBATIONS: tuple[Perturbation, ...] = (
    # --- Destructive (preserve M1 behaviour) -----------------------------
    Perturbation(
        id="scrambled_swap",
        kind="destructive",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "scrambled_swap",
        seeded=True,
    ),
    Perturbation(
        id="scrambled_random",
        kind="destructive",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "scrambled_random",
        seeded=True,
    ),
    Perturbation(
        id="scrambled_random_compact",
        kind="destructive",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "scrambled_random_compact",
        seeded=True,
    ),
    Perturbation(
        id="random_sphere",
        kind="destructive",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "random_sphere",
        seeded=True,
    ),
    Perturbation(
        id="sorted_line",
        kind="destructive",
        n_seeds=1,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "sorted_line",
        seeded=False,
    ),
    Perturbation(
        id="swapped_bsite",
        kind="destructive",
        n_seeds=1,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "swapped_bsite",
        seeded=False,
    ),
    # Rigid per-molecule transforms (chemistry preserved, geometry shifted).
    # Each material has a single perturbed system in
    # ``pems_mod_rotation_systems/cluster_n1/<mat>`` and
    # ``pems_mod_translation_systems/cluster_n1/<mat>`` (built by
    # ``00_data_prep/build_pems_perturbations.py``).  Coverage is 17/25
    # materials -- materials whose B-site is a single atom or whose
    # MoleculeManipulator pipeline failed are skipped.
    Perturbation(
        id="rotation",
        kind="destructive",
        n_seeds=1,
        models=("exp7a", "exp7c"),
        source_dir=paths.DATA_ROOT / "pems_mod_rotation_systems" / "cluster_n1",
        seeded=False,
    ),
    Perturbation(
        id="translation",
        kind="destructive",
        n_seeds=1,
        models=("exp7a", "exp7c"),
        source_dir=paths.DATA_ROOT / "pems_mod_translation_systems" / "cluster_n1",
        seeded=False,
    ),
    # --- Local-polyhedron tests (added 2026-05) -------------------------
    # Stretch operations rigidly translate whole molecular fragments along
    # a selected A-X, B-X, or A-B COM axis.  The six "seed" slots map to
    # scale factors [0.85, 0.90, 0.95, 1.05, 1.10, 1.15].
    Perturbation(
        id="stretch_bx",
        kind="polyhedron_preserve",
        n_seeds=6,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "stretch_bx",
        seeded=True,
    ),
    Perturbation(
        id="stretch_ax",
        kind="polyhedron_preserve",
        n_seeds=6,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "stretch_ax",
        seeded=True,
    ),
    Perturbation(
        id="stretch_ab",
        kind="polyhedron_preserve",
        n_seeds=6,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "stretch_ab",
        seeded=True,
    ),
    # Fragment COM swaps preserve each molecular fragment internally but
    # move it into the other role's local environment.
    Perturbation(
        id="swap_a_b",
        kind="polyhedron_break",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "swap_a_b",
        seeded=True,
    ),
    Perturbation(
        id="swap_b_x",
        kind="polyhedron_break",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "swap_b_x",
        seeded=True,
    ),
    Perturbation(
        id="swap_a_x",
        kind="polyhedron_break",
        n_seeds=5,
        models=("exp7a", "exp7c"),
        source_dir=paths.MECHANISM_DIR / "swap_a_x",
        seeded=True,
    ),
    # --- Template (merged from M5b 2026-04) ------------------------------
    Perturbation(
        id="template_dap4",
        kind="template",
        n_seeds=1,
        models=("exp7a", "exp7c", "exp7d"),
        source_dir=paths.TEMPLATE_N1_DIR,
        seeded=False,
    ),
    # Single-task pretrained model on the same perturbation ladder. Keep this
    # separate from manuscript-facing outputs until the figure is ready.
    Perturbation(
        id="template_dap4_st",
        kind="template",
        n_seeds=1,
        models=("exp7c",),
        source_dir=paths.TEMPLATE_N1_DIR,
        seeded=False,
    ),
)

DESTRUCTIVE_IDS: tuple[str, ...] = tuple(p.id for p in PERTURBATIONS if p.kind == "destructive")
POLYHEDRON_PRESERVE_IDS: tuple[str, ...] = tuple(p.id for p in PERTURBATIONS if p.kind == "polyhedron_preserve")
POLYHEDRON_BREAK_IDS: tuple[str, ...] = tuple(p.id for p in PERTURBATIONS if p.kind == "polyhedron_break")
TEMPLATE_IDS: tuple[str, ...] = tuple(p.id for p in PERTURBATIONS if p.kind == "template")
ALL_IDS: tuple[str, ...] = tuple(p.id for p in PERTURBATIONS)


def by_id(pid: str) -> Perturbation:
    for p in PERTURBATIONS:
        if p.id == pid:
            return p
    raise KeyError(pid)


def for_model(model_name: str) -> tuple[Perturbation, ...]:
    """Subset of perturbations applicable to ``model_name``."""
    return tuple(p for p in PERTURBATIONS if model_name in p.models)


def models_for(pert_id: str) -> tuple[str, ...]:
    return by_id(pert_id).models
