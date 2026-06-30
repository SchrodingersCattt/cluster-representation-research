#!/usr/bin/env python3
"""
Cross-representation inference: cluster vs crystal.

For each model family (exp7a/b/c/d, exp8a), evaluate each fold's checkpoint
on BOTH cluster (n1) and crystal held-out val systems.

This reveals whether a cluster-trained model can generalise to crystal inputs
and vice versa.

Output
------
  cross_infer_rep.json   — raw per-fold MAE matrix
  cross_infer_rep_heatmap.png/.pdf  — heatmap figure

Usage
-----
  python cross_infer_rep.py
  python cross_infer_rep.py --series exp7a exp8a   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_plot_style import EXP_COLORS, save_png_pdf, setup_nature_style, style_axes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "00_data_prep"
SPLITS_FILE = DATA_ROOT / "pems_5fold_splits_v2.json"
CLUSTER_N1_DIR = DATA_ROOT / "pems_cluster_n1_systems"
CRYSTAL_DIR = DATA_ROOT / "pems_crystal_systems"
CSV_PATH = ROOT.parent / "data" / "pems" / "pems.csv"

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# head=None → single-task model (no head kwarg)
FAMILIES: dict[str, dict[str, Any]] = {
    "exp7a": {"head": "pems_vdet_kj"},
    "exp7b": {"head": "pems_vdet_kj"},
    "exp7c": {"head": None},
    "exp7d": {"head": None},
    "exp8a": {"head": "pems_vdet_kj"},
}

# Datasets to evaluate on (name → directory, pbc flag)
DATASETS: dict[str, dict[str, Any]] = {
    "cluster_n1": {"dir": CLUSTER_N1_DIR, "pbc": False},
    "crystal":    {"dir": CRYSTAL_DIR,    "pbc": True},
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_splits() -> dict[str, list[str]]:
    data = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))
    return data.get("folds", {})


def resolve_checkpoint(exp_dir: Path) -> Path | None:
    symlink = exp_dir / "model.ckpt.pt"
    if symlink.exists():
        return symlink
    ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"),
                   key=lambda p: int(p.stem.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def run_inference_on_dir(
    ckpt: Path,
    head: str | None,
    sys_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt_array, pred_array) for a list of npy system dirs."""
    import dpdata
    from deepmd.pt.infer.deep_eval import DeepProperty

    kwargs: dict = {}
    if head is not None:
        kwargs["head"] = head
    model = DeepProperty(str(ckpt), **kwargs)
    model_tm = np.array(model.get_type_map())

    gt_list: list[float] = []
    pred_list: list[float] = []

    for sys_path in sys_dirs:
        try:
            vs = dpdata.LabeledSystem(str(sys_path), fmt="deepmd/npy")
            data_tm = vs.data["atom_names"]
            coords = vs.data["coords"]
            nopbc = (sys_path / "nopbc").exists()
            cells = None if nopbc else vs.data["cells"]
            atom_types = np.array([
                int(np.where(model_tm == data_tm[t])[0][0])
                for t in vs.data["atom_types"]
            ], dtype=np.int32)
            gt = vs.data["energies"]
            pred = model.eval(coords=coords, atom_types=atom_types, cells=cells)[0].reshape(-1)
            gt_list.extend(gt.tolist())
            pred_list.extend(pred.tolist())
        except Exception as e:
            print(f"    WARNING {sys_path.name}: {e}")

    del model
    return np.array(gt_list), np.array(pred_list)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_cross_infer(families: list[str]) -> dict:
    """
    Returns nested dict:
      results[family][dataset_name] = {
          "fold_maes": [mae_fold0, ...],
          "mean_mae": float,
          "std_mae": float,
      }
    """
    splits = load_splits()
    results: dict[str, dict[str, Any]] = {}

    for family in families:
        cfg = FAMILIES[family]
        head = cfg["head"]
        results[family] = {}

        for ds_name, ds_cfg in DATASETS.items():
            ds_dir: Path = ds_cfg["dir"]
            fold_maes: list[float] = []

            for fold_idx in range(5):
                exp_dir = ROOT / f"{family}_fold{fold_idx}"
                if not exp_dir.exists():
                    print(f"  SKIP {family}_fold{fold_idx}: dir not found")
                    continue
                ckpt = resolve_checkpoint(exp_dir)
                if ckpt is None:
                    print(f"  SKIP {family}_fold{fold_idx}: no checkpoint")
                    continue

                val_mats: list[str] = splits.get(str(fold_idx), [])
                sys_dirs = [ds_dir / m for m in val_mats if (ds_dir / m).exists()]
                if not sys_dirs:
                    print(f"  SKIP {family}_fold{fold_idx} on {ds_name}: no systems")
                    continue

                print(f"  {family}_fold{fold_idx} × {ds_name}: {ckpt.name}, {len(sys_dirs)} systems")
                gt, pred = run_inference_on_dir(ckpt, head, sys_dirs)
                if len(gt) == 0:
                    continue
                mae = float(np.mean(np.abs(gt - pred)))
                fold_maes.append(mae)
                print(f"    MAE = {mae:.1f} m/s")

            if fold_maes:
                results[family][ds_name] = {
                    "fold_maes": fold_maes,
                    "mean_mae": float(np.mean(fold_maes)),
                    "std_mae": float(np.std(fold_maes)),
                    "n_folds": len(fold_maes),
                }
            else:
                results[family][ds_name] = {
                    "fold_maes": [],
                    "mean_mae": float("nan"),
                    "std_mae": float("nan"),
                    "n_folds": 0,
                }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(results: dict, families: list[str], out_path: Path) -> None:
    """Heatmap: rows = model families, cols = datasets."""
    setup_nature_style()

    ds_names = list(DATASETS.keys())
    n_rows = len(families)
    n_cols = len(ds_names)

    mat = np.full((n_rows, n_cols), np.nan)
    for i, fam in enumerate(families):
        for j, ds in enumerate(ds_names):
            v = results.get(fam, {}).get(ds, {}).get("mean_mae", float("nan"))
            mat[i, j] = v

    # Colour scale: white=low, red=high; cap at 1500 for visibility
    vmax = min(float(np.nanmax(mat)), 1500.0)
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(4.5, 0.55 * n_rows + 1.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}"
                # white text on dark cells
                color = "white" if v > 0.6 * vmax else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color=color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7,
                        color="#888888")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(["Cluster (n1)", "Crystal"], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_title("Cross-representation MAE (m/s)\n5-fold mean", fontsize=10)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE (m/s)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Draw grid lines between cells
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", lw=0.8)

    style_axes(ax, grid=False)
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    save_png_pdf(fig, out_path, dpi=300)
    plt.close(fig)
    print(f"Saved heatmap: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", nargs="+", default=None,
                        help="Families to run (default: all). E.g. exp7a exp8a")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    families = args.series if args.series else list(FAMILIES.keys())
    # Validate
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        print(f"Unknown families: {unknown}. Available: {list(FAMILIES.keys())}")
        sys.exit(1)

    print(f"Running cross-representation inference for: {families}")
    results = run_cross_infer(families)

    # Save JSON
    out_json = ROOT / "cross_infer_rep.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    # Print summary table
    ds_names = list(DATASETS.keys())
    header = f"{'Family':<12}" + "".join(f"  {d:>14}" for d in ds_names)
    print("\n" + header)
    print("-" * len(header))
    for fam in families:
        row = f"{fam:<12}"
        for ds in ds_names:
            v = results.get(fam, {}).get(ds, {}).get("mean_mae", float("nan"))
            row += f"  {v:>14.1f}" if not np.isnan(v) else f"  {'N/A':>14}"
        print(row)

    if not args.no_plot:
        plot_heatmap(results, families, ROOT / "cross_infer_rep_heatmap.png")


if __name__ == "__main__":
    main()
