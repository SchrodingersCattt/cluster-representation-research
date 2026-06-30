#!/usr/bin/env python3
"""Emit the LaTeX bodies for the SI hyperparameter / robustness ablation tables.

Reads `experiments/ablation_full_eval_summary.json` and prints two
tabular bodies that are pasted (not auto-generated) into `manuscript/SI.tex`
under \\section{Training-variant ablations}, \\subsection*{Hyperparameter and
split-robustness ablations}:

    1. Aggregate MAE table (family x {IND_25, OOD_heldout, OOD_new})
    2. Per-material AE table (family x 8 OOD materials)

Run after refreshing `ablation_full_eval_summary.json` to verify the SI numbers
match the source JSON exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments" / "ablation_full_eval_summary.json"

ROW_ORDER: list[tuple[str, list[tuple[str, str]]]] = [
    ("Default learning-rate baselines", [
        ("exp7a", "MT-FT"),
        ("exp7b", "MT-FT-aux"),
        ("exp7c", "ST-FT"),
        ("exp7d", "ST-TFS"),
    ]),
    ("Learning-rate schedule", [
        ("exp7a_lr1e4", "MT-FT, $1{\\times}10^{-4}$"),
        ("exp7a_lr5e6", "MT-FT, $5{\\times}10^{-6}$"),
        ("exp7b_lr1e4", "MT-FT-aux, $1{\\times}10^{-4}$"),
        ("exp7b_lr5e6", "MT-FT-aux, $5{\\times}10^{-6}$"),
        ("exp7c_lr1e4", "ST-FT, $1{\\times}10^{-4}$"),
        ("exp7d_lr1e4", "ST-TFS, $1{\\times}10^{-4}$"),
    ]),
    ("Total training length", [
        ("exp7a_200k", "MT-FT, 200\\,k steps"),
        ("exp7a_800k", "MT-FT, 800\\,k steps"),
    ]),
    ("Cross-validation split seed", [
        ("exp7a_seed7",  "MT-FT, split seed 7"),
        ("exp7a_seed13", "MT-FT, split seed 13"),
        ("exp7c_seed7",  "ST-FT, split seed 7"),
        ("exp7c_seed13", "ST-FT, split seed 13"),
    ]),
    ("Learning-rate decay ratio (decay\\_steps / numb\\_steps)", [
        ("exp7a_decay200", "MT-FT, ratio 1/200"),
        ("exp7c_decay200", "ST-FT, ratio 1/200"),
    ]),
]

# Order matches the SI per-material table; the double-perovskite fail case is shown last in the
# OOD-holdout block because it is excluded from the OOD-holdout aggregate MAE
# (see ablation_full_eval_summary.json:metadata.OOD_heldout_eval_materials)
# and is discussed separately as a representation-level fail case.
OOD_HELDOUT = ["DAC-4", "TAP-2", "EAP-4", "SY", "DAI-1_0.5 4_0.5"]
OOD_NEW = ["PEP", "MPEP", "HPEP"]


def fmt(v: float | None, precision: int = 1) -> str:
    return "---" if v is None else f"{v:.{precision}f}"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    families = data["families"]

    print("% --- Table 1: aggregate MAE per family ----------------------------")
    print("% Columns: family description | IND_25 MAE | OOD-holdout MAE | OOD-new MAE")
    for group_label, fam_pairs in ROW_ORDER:
        print(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{group_label}}}}} \\\\")
        for fam, label in fam_pairs:
            r = families.get(fam)
            if r is None:
                print(f"\\quad {label} & --- & --- & --- \\\\")
                continue
            print(
                f"\\quad {label} & "
                f"{fmt(r['IND_25']['mae_m_s'])} & "
                f"{fmt(r['OOD_heldout']['mae_m_s'])} & "
                f"{fmt(r['OOD_new']['mae_m_s'])} \\\\"
            )
        print("\\addlinespace")

    print()
    print("% --- Table 2: per-material AE for the 8 OOD materials -------------")
    print("% Columns: family | DAC-4 | TAP-2 | DAI-1$_{0.5}$4$_{0.5}$ | EAP-4 | DEP | PEP | PEP-M | PEP-H")
    for group_label, fam_pairs in ROW_ORDER:
        print(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{group_label}}}}} \\\\")
        for fam, label in fam_pairs:
            r = families.get(fam)
            if r is None:
                print(f"\\quad {label} & " + " & ".join(["---"] * 8) + " \\\\")
                continue
            held = r["OOD_heldout"]["per_material"] or {}
            new = r["OOD_new"]["per_material"] or {}
            cells = []
            for m in OOD_HELDOUT:
                v = held.get(m, {}).get("ae")
                cells.append(fmt(v, 0 if (v is not None and v >= 100) else 1))
            for m in OOD_NEW:
                v = new.get(m, {}).get("ae")
                cells.append(fmt(v, 0 if (v is not None and v >= 100) else 1))
            print(f"\\quad {label} & " + " & ".join(cells) + " \\\\")
        print("\\addlinespace")


if __name__ == "__main__":
    main()
