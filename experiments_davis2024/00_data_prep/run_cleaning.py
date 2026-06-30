#!/usr/bin/env python3
"""
David2024 dataset cleaning pipeline (v2).

Steps:
1. Load NPZ + filtered_index
2. Remove systems with >500 atoms
3. Remove systems with heavy-atom pair distance < 1 Å
4. Remove systems missing H (zero hydrogen atoms)
5. Overwrite filtered_index.json with cleaned entries
6. Write statistics report

Output:
- filtered_index.json: overwritten with cleaned entries (downstream scripts unchanged)
- cleaned_index.json: detailed cleaning log
- cleaning_report.md: human-readable statistics
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── paths ──
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "davis2024" / "energetic_crystals_dataset"
NPZ_PATH = DATA_DIR / "energetic_crystals.npz"
PREP_DIR = ROOT / "experiments_davis2024" / "00_data_prep"
FILTERED_INDEX = PREP_DIR / "filtered_index.json"
OUTPUT_DIR = PREP_DIR

ATOM_COUNT_CAP = 500
HEAVY_DISTANCE_THRESHOLD = 1.0  # Å — pairs closer than this are suspicious


def load_npz() -> dict[str, np.ndarray]:
    """Load NPZ dataset."""
    return dict(np.load(NPZ_PATH, allow_pickle=True))


def load_filtered_index() -> dict[str, Any]:
    """Load the existing filtered index."""
    return json.loads(FILTERED_INDEX.read_text(encoding="utf-8"))


def get_heavy_atom_min_distance(coords_cart: np.ndarray, species: list[str]) -> float:
    """
    Compute minimum pairwise distance among heavy (non-H) atoms.
    Returns inf if fewer than 2 heavy atoms.
    """
    heavy_mask = np.array([s != "H" for s in species])
    heavy_coords = coords_cart[heavy_mask]
    n = len(heavy_coords)
    if n < 2:
        return float("inf")
    if n > 2000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, min(2000, n), replace=False)
        heavy_coords = heavy_coords[idx]
        n = len(heavy_coords)
    diff = heavy_coords[:, None, :] - heavy_coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def check_missing_h(species: list[str]) -> bool:
    """Return True if the system has zero H atoms."""
    return "H" not in species


def main() -> None:
    print("Loading NPZ dataset...")
    npz = load_npz()
    n_total = len(npz["refcode"])
    refcodes = npz["refcode"]
    num_atoms_arr = npz["num_atoms"]
    species_arr = npz["species"]
    coords_cart_arr = npz["coords_cart"]

    print(f"Total entries in NPZ: {n_total}")

    # Load filtered index to know which entries are in our working set
    filtered_idx = load_filtered_index()
    kept_entries = filtered_idx["kept"]
    kept_refcodes = set(entry["refcode"] for entry in kept_entries)
    print(f"Entries in filtered index (kept): {len(kept_refcodes)}")

    # Build refcode→index mapping
    refcode_to_idx = {str(rc): i for i, rc in enumerate(refcodes)}
    # Build refcode→filtered entry mapping
    refcode_to_entry = {entry["refcode"]: entry for entry in kept_entries}

    # ── Step 1: Flag large atom count ──
    print("\n=== Step 1: Remove systems with >500 atoms ===")
    large_systems = []
    for rc in kept_refcodes:
        idx = refcode_to_idx.get(rc)
        if idx is None:
            continue
        natoms = int(num_atoms_arr[idx])
        if natoms > ATOM_COUNT_CAP:
            large_systems.append((rc, natoms))
    large_systems.sort(key=lambda x: -x[1])
    large_set = {rc for rc, _ in large_systems}
    print(f"  Systems with >{ATOM_COUNT_CAP} atoms: {len(large_systems)}")

    # ── Step 2: Flag heavy-atom close contacts ──
    print("\n=== Step 2: Remove systems with heavy-atom distance < 1 Å ===")
    close_contact_systems = []
    working_refcodes = kept_refcodes - large_set
    total_to_check = len(working_refcodes)
    for i, rc in enumerate(sorted(working_refcodes)):
        if (i + 1) % 2000 == 0:
            print(f"  Checked {i+1}/{total_to_check}...")
        idx = refcode_to_idx.get(rc)
        if idx is None:
            continue
        species = list(species_arr[idx])
        coords_cart = np.array(coords_cart_arr[idx], dtype=np.float64)
        min_dist = get_heavy_atom_min_distance(coords_cart, species)
        if min_dist < HEAVY_DISTANCE_THRESHOLD:
            close_contact_systems.append((rc, min_dist))
    close_contact_systems.sort(key=lambda x: x[1])
    close_set = {rc for rc, _ in close_contact_systems}
    print(f"  Systems with heavy-atom dist < {HEAVY_DISTANCE_THRESHOLD} Å: {len(close_contact_systems)}")

    # ── Step 3: Remove missing-H systems ──
    print("\n=== Step 3: Remove systems missing hydrogen ===")
    remove_so_far = large_set | close_set
    surviving_refcodes = kept_refcodes - remove_so_far
    missing_h_systems = []
    for rc in sorted(surviving_refcodes):
        idx = refcode_to_idx.get(rc)
        if idx is None:
            continue
        species = list(species_arr[idx])
        if check_missing_h(species):
            missing_h_systems.append(rc)
    missing_h_set = set(missing_h_systems)
    print(f"  Systems with zero H (in surviving set): {len(missing_h_systems)}")

    # ── Final kept set ──
    all_removed = large_set | close_set | missing_h_set
    final_kept = kept_refcodes - all_removed
    print(f"\n  Final kept: {len(final_kept)} (removed {len(all_removed)} total)")

    # ── Overwrite filtered_index.json with cleaned entries ──
    print("\n=== Overwriting filtered_index.json with cleaned entries ===")
    cleaned_kept = [refcode_to_entry[rc] for rc in sorted(final_kept) if rc in refcode_to_entry]
    new_filtered = dict(filtered_idx)  # preserve other keys like discarded etc.
    new_filtered["kept"] = cleaned_kept
    new_filtered["_cleaning_applied"] = True
    new_filtered["_cleaning_removed_count"] = len(all_removed)
    FILTERED_INDEX.write_text(json.dumps(new_filtered, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"  Wrote {len(cleaned_kept)} entries to {FILTERED_INDEX}")

    # ── Write detailed cleaning log ──
    cleaned_index = {
        "total_in_npz": n_total,
        "originally_filtered_kept": len(kept_refcodes),
        "removed_large_atoms": sorted(list(large_set)),
        "removed_large_atoms_count": len(large_set),
        "removed_close_contact": sorted(list(close_set)),
        "removed_close_contact_count": len(close_set),
        "removed_missing_h": sorted(list(missing_h_set)),
        "removed_missing_h_count": len(missing_h_set),
        "total_removed": len(all_removed),
        "final_kept_count": len(final_kept),
        "final_kept_refcodes": sorted(list(final_kept)),
        "atom_count_cap": ATOM_COUNT_CAP,
        "heavy_distance_threshold": HEAVY_DISTANCE_THRESHOLD,
    }
    out_path = OUTPUT_DIR / "cleaned_index.json"
    out_path.write_text(json.dumps(cleaned_index, indent=2, default=str), encoding="utf-8")
    print(f"  Cleaned index log written to {out_path}")

    # ── Write statistics report ──
    report_lines = [
        "# David2024 Dataset Cleaning Report (v2)",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total entries in NPZ | {n_total} |",
        f"| After initial filtering (SMILES failures etc.) | {len(kept_refcodes)} |",
        f"| Removed: >500 atoms | {len(large_set)} |",
        f"| Removed: heavy-atom dist < 1 Å | {len(close_set)} |",
        f"| Removed: missing hydrogen | {len(missing_h_set)} |",
        f"| **Total removed** | **{len(all_removed)}** |",
        f"| **Final kept** | **{len(final_kept)}** |",
        "",
        "## Large Atom Count Systems Removed",
        "",
        f"Cap: {ATOM_COUNT_CAP} atoms. {len(large_systems)} systems removed.",
        "",
        "| Refcode | Atoms |",
        "|---------|-------|",
    ]
    for rc, n in large_systems[:30]:
        report_lines.append(f"| {rc} | {n} |")
    if len(large_systems) > 30:
        report_lines.append(f"| ... | ({len(large_systems) - 30} more) |")

    report_lines += [
        "",
        "## Close Contact Systems Removed",
        "",
        f"Threshold: heavy-atom pairwise distance < {HEAVY_DISTANCE_THRESHOLD} Å. {len(close_contact_systems)} systems removed.",
        "",
        "| Refcode | Min Distance (Å) |",
        "|---------|------------------|",
    ]
    for rc, d in close_contact_systems[:30]:
        report_lines.append(f"| {rc} | {d:.4f} |")
    if len(close_contact_systems) > 30:
        report_lines.append(f"| ... | ({len(close_contact_systems) - 30} more) |")

    report_lines += [
        "",
        "## Missing Hydrogen Systems Removed",
        "",
        f"Systems with zero H atoms: **{len(missing_h_set)}** — all removed.",
        "",
    ]

    # Atom count distribution of final set
    final_atoms = []
    for rc in sorted(final_kept):
        idx = refcode_to_idx.get(rc)
        if idx is not None:
            final_atoms.append(int(num_atoms_arr[idx]))

    if final_atoms:
        arr = np.array(final_atoms)
        report_lines += [
            "## Final Dataset Atom Count Distribution",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Count | {len(arr)} |",
            f"| Min | {arr.min()} |",
            f"| Max | {arr.max()} |",
            f"| Mean | {arr.mean():.1f} |",
            f"| Median | {np.median(arr):.0f} |",
            f"| p95 | {np.percentile(arr, 95):.0f} |",
            f"| p99 | {np.percentile(arr, 99):.0f} |",
            "",
        ]

    report_path = OUTPUT_DIR / "cleaning_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Cleaning report written to {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
