#!/usr/bin/env python3
"""
Inference + parity plot for David2024 experiments.

Uses DeepProperty API to run inference on validation systems, then plots
predicted vs ground-truth detonation velocity with MAE/RMSE annotations.

Supports both single-task models (exp1–3) and multi-task models (exp4)
which require specifying a head name via the ``head`` kwarg to DeepProperty.

Usage:
    python infer_and_plot.py                    # auto-detect latest checkpoints for all exps
    python infer_and_plot.py --exp exp1a_crystal_dpa32 --ckpt model.ckpt-6000.pt
    python infer_and_plot.py --exp exp4a_multitask_deepems_kj
    python infer_and_plot.py --wait 300         # poll every 300s for new ckpts, then infer
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_plot_style import EXP_COLORS, save_png_pdf, setup_nature_style, style_axes

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
# Each entry maps to:
#   exp_dir        : subdirectory under ROOT containing checkpoints
#   val_sys_type   : which npy system directory to read
#   split_key      : key in splits.json for validation refcodes
#   head           : (optional) multi-task head name for DeepProperty
#
# For multi-task models (exp4*), a single training directory may expose
# multiple property heads.  We create separate "logical" experiment names
# for each head so they appear on their own parity panel.
# ---------------------------------------------------------------------------
EXP_CONFIG: dict[str, dict] = {
    "exp1a_crystal_dpa32": {
        "exp_dir": "exp1a_crystal_dpa32",
        "val_sys_type": "crystal_systems",
        "split_key": "shared_85_15",
    },
    "exp1b_molecule_dpa32": {
        "exp_dir": "exp1b_molecule_dpa32",
        "val_sys_type": "molecule_systems",
        "split_key": "shared_85_15",
    },
    "exp2_theory2exp_dpa32": {
        "exp_dir": "exp2_theory2exp_dpa32",
        "val_sys_type": "exp_val_systems",
        "split_key": "theory_to_experiment",
    },
    "exp3a_crystal_deepems": {
        "exp_dir": "exp3a_crystal_deepems",
        "val_sys_type": "crystal_systems",
        "split_key": "shared_85_15",
    },
    "exp3b_molecule_deepems": {
        "exp_dir": "exp3b_molecule_deepems",
        "val_sys_type": "molecule_systems",
        "split_key": "shared_85_15",
    },
    # --- Multi-task: exp4a (3 heads) ---
    "exp4a_multitask_deepems_kj": {
        "exp_dir": "exp4a_multitask_deepems",
        "val_sys_type": "crystal_systems",
        "split_key": "shared_85_15",
        "head": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    },
    "exp4a_multitask_deepems_exp": {
        "exp_dir": "exp4a_multitask_deepems",
        "val_sys_type": "exp_val_systems",
        "split_key": "theory_to_experiment",
        "head": "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis",
    },
    # --- Multi-task: exp4b (2 heads) ---
    "exp4b_multitask_deepems_kj": {
        "exp_dir": "exp4b_multitask_deepems",
        "val_sys_type": "crystal_systems",
        "split_key": "shared_85_15",
        "head": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    },
    # --- Multi-task: exp4c (3 heads, molecule) ---
    "exp4c_multitask_deepems_mol_kj": {
        "exp_dir": "exp4c_multitask_deepems_mol",
        "val_sys_type": "molecule_systems",
        "split_key": "shared_85_15",
        "head": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    },
    "exp4c_multitask_deepems_mol_exp": {
        "exp_dir": "exp4c_multitask_deepems_mol",
        "val_sys_type": "exp_val_systems",
        "split_key": "theory_to_experiment",
        "head": "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis",
    },
    # --- Multi-task: exp4d (2 heads, molecule) ---
    "exp4d_multitask_deepems_mol_kj": {
        "exp_dir": "exp4d_multitask_deepems_mol",
        "val_sys_type": "molecule_systems",
        "split_key": "shared_85_15",
        "head": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    },
}

EXP_LABELS = {
    "exp1a_crystal_dpa32": "Exp1a: Crystal + DPA-3.2",
    "exp1b_molecule_dpa32": "Exp1b: Molecule + DPA-3.2",
    "exp2_theory2exp_dpa32": "Exp2: Theory→Exp + DPA-3.2",
    "exp3a_crystal_deepems": "Exp3a: Crystal + deepems-lam",
    "exp3b_molecule_deepems": "Exp3b: Molecule + deepems-lam",
    "exp4a_multitask_deepems_kj": "Exp4a-KJ: MT deepems (KJ val)",
    "exp4a_multitask_deepems_exp": "Exp4a-Exp: MT deepems (Exp val)",
    "exp4b_multitask_deepems_kj": "Exp4b-KJ: MT deepems (KJ val)",
    "exp4c_multitask_deepems_mol_kj": "Exp4c-KJ: MT deepems mol (KJ val)",
    "exp4c_multitask_deepems_mol_exp": "Exp4c-Exp: MT deepems mol (Exp val)",
    "exp4d_multitask_deepems_mol_kj": "Exp4d-KJ: MT deepems mol (KJ val)",
}

HEAD_LABELS = {
    None: "single-task property",
    "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis": "david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis",
    "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis": "david2024_vdet_exp"  # Historical name in checkpoint; correct spelling is "Davis",
}

VAL_SYS_LABELS = {
    "crystal_systems": "crystal_systems",
    "molecule_systems": "molecule_systems",
    "exp_val_systems": "exp_val_systems",
}

SPLIT_LABELS = {
    "shared_85_15": "shared_85_15 holdout",
    "theory_to_experiment": "theory_to_experiment holdout",
}


def get_latest_ckpt(exp_dir: Path) -> Path | None:
    """Find the latest model checkpoint in an experiment directory."""
    ckpts = sorted(exp_dir.glob("model.ckpt-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    if ckpts:
        return ckpts[-1]
    final = exp_dir / "model.ckpt.pt"
    if final.exists():
        return final
    return None


def get_val_systems(val_sys_type: str, split_key: str) -> list[Path]:
    """Get validation system paths for an experiment."""
    splits = json.loads((ROOT / "00_data_prep" / "splits.json").read_text())
    val_refcodes = splits[split_key]["val_refcodes"]
    sys_dir = ROOT / "00_data_prep" / val_sys_type
    paths = []
    for rc in val_refcodes:
        p = sys_dir / rc
        if p.exists():
            paths.append(p)
    return paths


def run_inference(
    model_path: Path,
    val_systems: list[Path],
    head: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference using DeepProperty API.

    Parameters
    ----------
    model_path : Path
        Path to a ``.pt`` checkpoint file.
    val_systems : list[Path]
        List of deepmd/npy system directories.
    head : str or None
        For multi-task models, the name of the head to use (e.g.
        ``"david2024_vdet_kj"  # Historical name in checkpoint; correct spelling is "Davis"``).  ``None`` for single-task models.

    Returns
    -------
    (gt, pred) : tuple[np.ndarray, np.ndarray]
    """
    from deepmd.pt.infer.deep_eval import DeepProperty
    import dpdata

    kwargs: dict = {}
    if head is not None:
        kwargs["head"] = head

    model = DeepProperty(str(model_path), **kwargs)
    model_type_map = model.get_type_map()

    gt_list = []
    pred_list = []

    for i, sys_path in enumerate(val_systems):
        if (i + 1) % 200 == 0:
            print(f"  Inference {i+1}/{len(val_systems)}...")
        try:
            vs = dpdata.LabeledSystem(str(sys_path), fmt="deepmd/npy")
            data_type_map = vs.data["atom_names"]
            coords = vs.data["coords"]  # shape: (nframes, natoms, 3)

            # Detect nopbc (molecule) systems → pass cells=None
            nopbc_file = sys_path / "nopbc"
            if nopbc_file.exists():
                cells = None
            else:
                cells = vs.data["cells"]  # shape: (nframes, 3, 3)

            # Remap atom types to model type map
            atom_types = np.array(
                [
                    np.where(np.array(model_type_map) == data_type_map[t])[0][0]
                    for t in vs.data["atom_types"]
                ],
                dtype=np.int32,
            )

            gt = vs.data["energies"]  # property stored as energy
            pred = model.eval(coords=coords, atom_types=atom_types, cells=cells)[0]
            pred = pred.reshape(-1)

            gt_list.extend(gt)
            pred_list.extend(pred)
        except Exception as e:
            print(f"  WARNING: Failed on {sys_path.name}: {e}")
            continue

    return np.array(gt_list), np.array(pred_list)


def _describe_eval_target(exp_name: str) -> tuple[str, str]:
    """Return (head_label, val_label) for a logical experiment panel."""
    cfg = EXP_CONFIG[exp_name]
    head = HEAD_LABELS.get(cfg.get("head"), cfg.get("head") or "single-task property")
    val_sys = VAL_SYS_LABELS.get(cfg["val_sys_type"], cfg["val_sys_type"])
    split = SPLIT_LABELS.get(cfg["split_key"], cfg["split_key"])
    return head, f"{val_sys} | {split}"


def make_parity_plot(results: dict, out_path: Path) -> None:
    """Create a tiled parity plot for all experiments."""
    setup_nature_style()
    n = len(results)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.3 * nrows), squeeze=False)

    for idx, (exp_name, (gt, pred, ckpt_name)) in enumerate(results.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        errors = np.abs(gt - pred)
        mae = np.mean(errors)
        median_ae = np.median(errors)
        rmse = np.sqrt(np.mean((gt - pred) ** 2))
        mape = 100 * np.mean(np.abs((gt - pred) / gt))
        n_outliers = int(np.sum(np.abs(pred) > 1e6))

        # Clip predictions to ±3× GT range for plot (don't distort axes with outliers)
        gt_range = gt.max() - gt.min()
        pred_clipped = np.clip(pred, gt.min() - gt_range, gt.max() + gt_range)

        ax.scatter(
            gt,
            pred_clipped,
            s=18,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
            color=EXP_COLORS["blue"],
            zorder=3,
        )

        vmin = gt.min() * 0.95
        vmax = gt.max() * 1.05
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.0, color=EXP_COLORS["gray"], zorder=2)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal")

        ax.set_xlabel("Ground Truth (m/s)")
        ax.set_ylabel("Predicted (m/s)")

        label = EXP_LABELS.get(exp_name, exp_name)
        head_label, val_label = _describe_eval_target(exp_name)
        ax.set_title(f"{label}\n{head_label} | {val_label}\n{ckpt_name}")
        outlier_note = f"\n[{n_outliers} outliers clipped]" if n_outliers > 0 else ""
        ax.text(
            0.05, 0.95,
            f"MAE = {mae:.0f} m/s\nMedian AE = {median_ae:.0f} m/s\nRMSE = {rmse:.0f} m/s\nMAPE = {mape:.1f}%{outlier_note}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#CFCFCF", alpha=0.95),
        )
        style_axes(ax, grid=True)

    # Hide unused
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("David2024 parity evaluation", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_png_pdf(fig, out_path, dpi=300)
    print(f"Saved parity plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="Specific experiment name (may be a logical name like exp4a_multitask_deepems_kj)")
    parser.add_argument("--ckpt", type=str, default=None, help="Specific checkpoint filename")
    parser.add_argument("--wait", type=int, default=0,
                        help="If >0, poll every WAIT seconds for new checkpoints before running inference (useful for exp4 still training)")
    args = parser.parse_args()

    if args.exp:
        exp_names = [args.exp]
    else:
        exp_names = list(EXP_CONFIG.keys())

    # ------------------------------------------------------------------
    # Optional: wait/poll for checkpoints to appear
    # ------------------------------------------------------------------
    if args.wait > 0:
        print(f"Polling mode: checking every {args.wait}s for checkpoints...")
        while True:
            found_any = False
            for exp_name in exp_names:
                cfg = EXP_CONFIG[exp_name]
                exp_dir = ROOT / cfg["exp_dir"]
                ckpt = get_latest_ckpt(exp_dir) if args.ckpt is None else exp_dir / args.ckpt
                if ckpt is not None and ckpt.exists():
                    found_any = True
                    break
            if found_any:
                print(f"Found checkpoint(s). Proceeding with inference.")
                break
            print(f"  No checkpoints yet. Sleeping {args.wait}s... [{time.strftime('%H:%M:%S')}]")
            time.sleep(args.wait)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    results = {}
    for exp_name in exp_names:
        cfg = EXP_CONFIG[exp_name]
        exp_dir = ROOT / cfg["exp_dir"]
        if not exp_dir.exists():
            print(f"Skipping {exp_name}: directory not found")
            continue

        if args.ckpt:
            ckpt = exp_dir / args.ckpt
        else:
            ckpt = get_latest_ckpt(exp_dir)

        if ckpt is None or not ckpt.exists():
            print(f"Skipping {exp_name}: no checkpoint found")
            continue

        head = cfg.get("head", None)
        head_msg = f" (head={head})" if head else ""
        print(f"\n=== {exp_name}{head_msg} ({ckpt.name}) ===")

        val_systems = get_val_systems(cfg["val_sys_type"], cfg["split_key"])
        print(f"  Val systems: {len(val_systems)}")

        if not val_systems:
            print(f"  No val systems found, skipping")
            continue

        gt, pred = run_inference(ckpt, val_systems, head=head)
        print(f"  Inferred {len(gt)} samples")
        if len(gt) > 0:
            errors = np.abs(gt - pred)
            mae = float(np.mean(errors))
            median_ae = float(np.median(errors))
            n_outliers = int(np.sum(np.abs(pred) > 1e6))
            print(f"  MAE = {mae:.0f} m/s  |  Median AE = {median_ae:.0f} m/s  |  Outliers(>1e6): {n_outliers}")
            results[exp_name] = (gt, pred, ckpt.name)

    if results:
        out_path = ROOT / "parity_plots.png"
        make_parity_plot(results, out_path)

        # Also save numeric results
        summary = {}
        for exp_name, (gt, pred, ckpt_name) in results.items():
            errors = np.abs(gt - pred)
            mae = float(np.mean(errors))
            median_ae = float(np.median(errors))
            rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
            n_outliers = int(np.sum(np.abs(pred) > 1e6))
            summary[exp_name] = {
                "checkpoint": ckpt_name,
                "n_val": len(gt),
                "mae_m_s": round(mae, 1),
                "median_ae_m_s": round(median_ae, 1),
                "rmse_m_s": round(rmse, 1),
                "n_outliers_1e6": n_outliers,
            }
        summary_path = ROOT / "inference_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved inference summary to {summary_path}")
    else:
        print("\nNo results to plot.")


if __name__ == "__main__":
    main()
