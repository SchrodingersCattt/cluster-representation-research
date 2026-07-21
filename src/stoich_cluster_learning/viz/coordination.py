"""PEM/MIX fragment classification helpers for coordination analyses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

METAL_ELEMS = {"Na", "K", "Rb", "Cs", "Ag", "Li"}


def infer_x_family(name: str) -> str:
    """Infer the anion family from a MIX material name.

    Naming convention: the third character of the prefix encodes X:
    ``*AP*`` -> ``ClO4``, ``*AN*`` -> ``NO3``, ``*AI*`` -> ``IO4``.
    """
    base = name.split("-")[0].upper()
    third = base[2] if len(base) >= 3 else ""
    return {"P": "ClO4", "N": "NO3", "I": "IO4"}.get(third, "ClO4")


def classify_pem_fragment(elem_set: set[str], x_family: str) -> str:
    """Map a fragment's element set to A/B/X chemistry labels."""
    has_c = "C" in elem_set
    has_h = "H" in elem_set
    has_n = "N" in elem_set
    has_o = "O" in elem_set
    has_cl = "Cl" in elem_set
    has_i = "I" in elem_set
    is_pure_metal = bool(elem_set) and elem_set <= METAL_ELEMS

    if has_c:
        return "ORG"
    if has_cl and x_family == "ClO4":
        return "X"
    if has_i and x_family == "IO4":
        return "X"
    if has_n and has_o and x_family == "NO3":
        return "X"
    if is_pure_metal:
        return "B_metal"
    if has_n and has_h:
        return "B_NH4"
    return "?"


def reclassify_pem_fragments(bundle: Any, x_family: str) -> None:
    """Relabel ``bundle.topology_fragment_table`` in-place for MIX chemistry.

    The generic crystal-viewer classifier only reliably recognizes perchlorate
    anions. This function applies the project-specific A/B/X rules used for
    MIX perovskite-type energetic salts.
    """
    table = bundle.topology_fragment_table
    if not table:
        return

    elem_sets: list[set[str]] = []
    for fragment in table:
        elems = {bundle.raw_atoms[i]["elem"] for i in fragment.get("site_indices", [])}
        elem_sets.append(elems)
    raw_types = [classify_pem_fragment(elem_set, x_family) for elem_set in elem_sets]

    has_inorganic_b = any(label in {"B_metal", "B_NH4"} for label in raw_types)
    organic_indices = [idx for idx, label in enumerate(raw_types) if label == "ORG"]

    if organic_indices:
        sizes = [(idx, table[idx]["heavy_atom_count"]) for idx in organic_indices]
        if has_inorganic_b:
            for idx in organic_indices:
                raw_types[idx] = "A"
        else:
            max_h = max(size for _, size in sizes)
            min_h = min(size for _, size in sizes)
            for idx, heavy_count in sizes:
                raw_types[idx] = "B" if heavy_count == min_h and heavy_count != max_h else "A"

    counters: dict[str, int] = defaultdict(int)
    for idx, label in enumerate(raw_types):
        if label in {"B_metal", "B_NH4"}:
            label = "B"
        table[idx]["type"] = label
        label_index = counters[label]
        counters[label] += 1
        table[idx]["label"] = f"{label}{label_index}"


# Backwards-compatible aliases for migrated figure helpers.
_infer_x_family = infer_x_family
_classify_pem_fragment = classify_pem_fragment
