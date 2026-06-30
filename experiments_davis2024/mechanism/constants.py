"""Static configuration shared by all M-series experiments.

Includes:
- TYPE_MAP, MODEL_CONFIGS, MODEL_DISPLAY_NAMES, COLORS, PERTURBATION_STYLE
- N_SEEDS, SCALE_FACTORS, RIDGE_ALPHAS, KJ_RHO_COEF, FOLD_IDS_DEFAULT
- M3/M3b composition + OB constants (ATOMIC_MASS, METAL_O_DEMAND, COMPOSITION_ELEMENTS)
- M4b bonding constants (_METAL_ELEMENTS, _HEAVY*, PEM_BOND_THRESHOLDS)
"""
from __future__ import annotations

TYPE_MAP = [
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
    'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce',
    'Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir',
    'Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm',
    'Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc',
    'Lv','Ts','Og',
]

MODEL_CONFIGS = {
    "exp7a": {
        "exp_dir": "exp7a_fold0",
        "property_head": "pems_vdet_kj",
        "descriptor_head": "deepems_vanilla",
        "label": "exp7a (DFT + PEMs)",
    },
    "exp7c": {
        "exp_dir": "exp7c_fold0",
        "property_head": None,
        "descriptor_head": None,
        "label": "exp7c (PEMs only)",
    },
    "exp7d": {
        "exp_dir": "exp7d_fold0",
        "property_head": "pems_vdet_kj",
        "descriptor_head": None,
        "label": "exp7d (ST scratch)",
    },
}

# Legacy ordering kept for backward-compatible iteration in code that has
# not yet migrated to perturbations.PERTURBATIONS.
PERTURBATION_TYPES_M1 = [
    "scrambled_swap", "scrambled_random", "scrambled_random_compact",
    "random_sphere", "sorted_line", "swapped_bsite",
]

SCALE_FACTORS = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30]
N_SEEDS = 5
FOLD_IDS_DEFAULT = [0, 1, 2, 3, 4]

# Ridge / RidgeCV grid (small n -> may need strong regularization)
RIDGE_ALPHAS = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]

# M6 early-stop defaults. Keep the grid sparse so the first diagnostic pass is
# tractable while still covering initial, middle, and late checkpoints.
EARLY_STOP_DEFAULT_MODELS = ["exp7a", "exp7c", "exp7d"]
EARLY_STOP_OPTIONAL_MODELS = ["exp7a_lr1e4", "exp7c_lr1e4"]
EARLY_STOP_DEFAULT_STEPS = [2000, 10000, 20000, 50000, 100000, 150000, 200000, 300000, 400000]

# Kamlet-Jacobs density term coefficient (g/cm^3) in
#   D = A * (1 + 1.30*rho_0) * sqrt(1 + B*rho_0)
# ratio uses (1 + 1.30*rho) / (1 + 1.30*rho_0)
KJ_RHO_COEF = 1.30

# Canonical colors -- must match paper_plot_style.EXP_COLORS exactly.
# Failure condition: exp7a != #205C77, exp7c != #931143, exp7b != #657217 (AGENTS.md)
COLORS = {
    "exp7a": "#205C77",   # dark teal  -- multi-task baseline
    "exp7b": "#657217",   # olive      -- multi-task auxiliary-head variant
    "exp7c": "#931143",   # crimson    -- single-task pretrained variant
    "exp7d": "#474747",   # dark gray  -- single-task scratch baseline
    "ref": "#474747",
    "kj": "#555555",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "exp7a":           "MT-DFT",
    "exp7a_lr1e4":     "MT-DFT-fast",
    "exp7b":           "MT-3head",
    "exp7c":           "ST-pretrained",
    "exp7d":           "ST-scratch",
    "exp6v2_allpems":  "MT-full",
}

PERTURBATION_STYLE = {
    "scrambled_swap":           {"label": "swap",     "color": "#5C7DA6", "marker": "o"},
    "scrambled_random":         {"label": "random",   "color": "#9C9C9C", "marker": "s"},
    "random_sphere":            {"label": "sphere",   "color": "#C7A252", "marker": "^"},
    "sorted_line":              {"label": "line",     "color": "#8E6C9F", "marker": "D"},
    "swapped_bsite":            {"label": "B-swap",   "color": "#4F9D69", "marker": "P"},
    "scrambled_random_compact": {"label": "compact",  "color": "#C44E52", "marker": "X"},
    # Template invariance (added 2026-04 when M5b was merged into M1).
    "template_dap4":            {"label": "DAP-4 tmpl", "color": "#3F7C58", "marker": "*"},
    # Rigid per-molecule transforms (added 2026-04 when rotation/translation
    # were promoted from M0-only sensitivity into the M1 perturbation
    # registry).  Each material has a single perturbed system per
    # ``pems_mod_{rotation,translation}_systems/cluster_n1/<mat>``.
    "rotation":                 {"label": "rotation",    "color": "#A85E32", "marker": "p"},
    "translation":              {"label": "translation", "color": "#5B7BBF", "marker": "h"},
    # Polyhedron-tiling perturbations (added 2026-05).
    "stretch_bx":               {"label": "B-X stretch", "color": "#4A8C86", "marker": "o"},
    "stretch_ax":               {"label": "A-X stretch", "color": "#6FA6A0", "marker": "s"},
    "stretch_ab":               {"label": "A-B stretch", "color": "#8EBDB8", "marker": "^"},
    "swap_a_b":                 {"label": "A-B swap",    "color": "#B07A3C", "marker": "D"},
    "swap_b_x":                 {"label": "B-X swap",    "color": "#C6924F", "marker": "P"},
    "swap_a_x":                 {"label": "A-X swap",    "color": "#D6A866", "marker": "X"},
}

# ---------------------------------------------------------------------------
# M3 / M3b shared composition + OB constants
# ---------------------------------------------------------------------------

# Atomic masses for OB calculation (g/mol)
ATOMIC_MASS = {
    "C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999,
    "Cl": 35.453, "I": 126.904, "Na": 22.990, "K": 39.098,
    "Rb": 85.468, "Ba": 137.327, "Ag": 107.868, "S": 32.06,
    "P": 30.974, "F": 18.998,
}

# Metal oxide stoichiometry: O atoms consumed per metal atom
# Na -> Na2O (0.5), K -> K2O (0.5), Rb -> Rb2O (0.5), Ag -> Ag2O (0.5), Ba -> BaO (1.0)
METAL_O_DEMAND = {"Na": 0.5, "K": 0.5, "Rb": 0.5, "Ag": 0.5, "Ba": 1.0}

COMPOSITION_ELEMENTS = ["C", "H", "N", "O", "Cl", "I"]

# ---------------------------------------------------------------------------
# M4b bonding constants
# ---------------------------------------------------------------------------

# Metal elements for MolCrysKit classification (reused from predict_abx_grid.py)
METAL_ELEMENTS = {
    'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Ga','Ge','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
    'Sb','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
    'Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
}

# PEM_BOND_THRESHOLDS -- mandatory for MolecularCrystal.from_ase() (AGENTS.md).
# Without this, the default 3.5 A cutoff merges ionic metal...O contacts into
# wrong molecules (e.g. K+ merged with ClO4-).
# Mirrors the definition in 00_data_prep/prep_pems_npy.py exactly.
_HEAVY = ["I", "Na", "K", "Rb", "Ba", "Ag"]
_HEAVY_NONO_LIMITS: dict[str, float] = {
    "I": 2.10, "Na": 2.30, "K": 2.50, "Rb": 2.60, "Ba": 2.60, "Ag": 2.30,
}
_HEAVY_O_LIMITS: dict[str, float] = {
    "I": 2.05, "Na": 2.20, "K": 2.30, "Rb": 2.40, "Ba": 2.40, "Ag": 2.20,
}
PEM_BOND_THRESHOLDS: dict[tuple[str, str], float] = {}
for _metal, _lim in _HEAVY_NONO_LIMITS.items():
    for _org in ["C", "H", "N", "Cl"]:
        PEM_BOND_THRESHOLDS[(_metal, _org)] = _lim
        PEM_BOND_THRESHOLDS[(_org, _metal)] = _lim
for _metal, _lim in _HEAVY_O_LIMITS.items():
    PEM_BOND_THRESHOLDS[(_metal, "O")] = _lim
    PEM_BOND_THRESHOLDS[("O", _metal)] = _lim
for _i, _m1 in enumerate(_HEAVY):
    for _m2 in _HEAVY[_i:]:
        PEM_BOND_THRESHOLDS[(_m1, _m2)] = 3.2
        if _m1 != _m2:
            PEM_BOND_THRESHOLDS[(_m2, _m1)] = 3.2
del _metal, _lim, _org, _i, _m1, _m2  # clean up loop variables


def disp(model_name: str) -> str:
    """Return display name for a model key, falling back to the raw key."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)
