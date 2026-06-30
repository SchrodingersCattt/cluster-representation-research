#!/usr/bin/env python3
"""Validate OB formula — test iodine O demand to match DAI references."""

ATOMIC_MASS = {
    "C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999,
    "Cl": 35.453, "I": 126.904, "K": 39.098
}

def mw(f):
    return sum(f.get(e, 0) * ATOMIC_MASS[e] for e in f)

DAI_4 = {"C": 6, "H": 18, "N": 3, "I": 3, "O": 12}
DAI_2 = {"C": 6, "H": 14, "N": 2, "K": 1, "I": 3, "O": 12}

print("Testing iodine O demand to match DAI-4 (ref=-17.0%) and DAI-2 (ref=-13.2%)")
print("=" * 80)
for I_demand in [0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    for name, f, ref in [("DAI-4", DAI_4, -17.0), ("DAI-2", DAI_2, -13.2)]:
        n_C = f.get("C", 0)
        n_H = f.get("H", 0)
        n_O = f.get("O", 0)
        n_Cl = f.get("Cl", 0)
        n_I = f.get("I", 0)
        n_K = f.get("K", 0)
        metal = n_K * 0.5
        M = mw(f)
        numerator = n_O - 2*n_C - (n_H - n_Cl)/2 - metal - n_I * I_demand
        ob = (1600 / M) * numerator
        delta = ob - ref
        flag = " <-- match!" if abs(delta) < 0.2 else ""
        print(f"  I_demand={I_demand:.1f}  {name}: calc={ob:>7.2f}  ref={ref:>6.1f}  Δ={delta:>+6.2f}{flag}")
    print()

# Backward-solve: what I_demand makes DAI-4 = -17.0?
for name, f, ref in [("DAI-4", DAI_4, -17.0), ("DAI-2", DAI_2, -13.2)]:
    n_C = f.get("C", 0)
    n_H = f.get("H", 0)
    n_O = f.get("O", 0)
    n_Cl = f.get("Cl", 0)
    n_I = f.get("I", 0)
    n_K = f.get("K", 0)
    metal = n_K * 0.5
    M = mw(f)
    # ref = (1600/M) * (n_O - 2*n_C - (n_H - n_Cl)/2 - metal - n_I * x)
    # ref * M / 1600 = n_O - 2*n_C - (n_H - n_Cl)/2 - metal - n_I * x
    rhs = ref * M / 1600
    lhs_no_I = n_O - 2*n_C - (n_H - n_Cl)/2 - metal
    x = (lhs_no_I - rhs) / n_I
    print(f"Solved I_demand for {name}: {x:.4f} O per I atom")
