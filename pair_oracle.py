#!/usr/bin/env python3
"""
pair_oracle.py - identify the baseline columns by their published medians, then
recompute the paired B1-oracle comparison on all 60 held-out scenes.

The ablation table reports B1 contrast lookup 0.253 nm, B2 fixed nuisances
1.883 nm, B3 gain-only 0.516 nm and the full method 0.099-0.101 nm. Matching
each column's median against those values identifies the columns without
relying on an assumed ordering.
"""
import math
import numpy as np

B = np.load("experiments/res_baselines.npy")
O = np.load("experiments/res_b1_oracle_60.npy")
truth = B[:, 0]
assert np.allclose(truth, O[:, 0]), "oracle and baseline scene sets differ"
oracle = O[:, 1]

PUB = {"B1 contrast lookup": 0.253, "B2 fixed nuisances": 1.883,
       "B3 gain-only": 0.516, "Ours (full joint)": 0.101}

print(f"{'col':>4}{'median |err|':>14}   best match")
med = {}
for c in range(1, B.shape[1]):
    m = float(np.median(np.abs(B[:, c] - truth)))
    med[c] = m
    name, gap = min(((k, abs(m - v)) for k, v in PUB.items()), key=lambda t: t[1])
    print(f"{c:>4}{m:>14.4f}   {name}  (published {PUB[name]}, off {gap:.4f})")

ours_col = min(med, key=lambda c: abs(med[c] - PUB["Ours (full joint)"]))
ours = B[:, ours_col]
print(f"\n-> treating column {ours_col} as the full method "
      f"(median {med[ours_col]:.4f} vs published {PUB['Ours (full joint)']})")

e_o = np.abs(oracle - truth)
e_u = np.abs(ours - truth)
n = len(truth)
wins = int((e_o < e_u).sum())
print(f"\nPAIRED, all {n} held-out scenes")
print(f"  B1-oracle better on {wins} of {n}   (archived claim was 37 of 45)")
print(f"  median |error|: oracle {np.median(e_o):.4f} nm, ours {np.median(e_u):.4f} nm"
      f"  -> factor {np.median(e_u)/np.median(e_o):.2f}")
b = int(((e_u >= 5.8) & (e_o < 5.8)).sum())
c = int(((e_o >= 5.8) & (e_u < 5.8)).sum())
k = b + c
p = 1.0 if k == 0 else min(1.0, 2*sum(math.comb(k, i) for i in range(min(b, c)+1))/2**k)
print(f"  failures (>5.8 nm): ours {int((e_u>=5.8).sum())}, oracle {int((e_o>=5.8).sum())}"
      f"   discordant {k}, exact McNemar p = {p:.4f}")

sd = 0.6745
print(f"\nCRB CHECK\n  oracle median {np.median(e_o):.4f} nm -> sigma "
      f"{np.median(e_o)/sd:.4f} nm against the 0.0289 nm known-nuisance bound"
      f"  -> {(np.median(e_o)/sd)/0.0289:.2f}x")
unc, orc = 0.253, float(np.median(e_o))
print(f"\nGAP RECOMPUTED ON ONE POPULATION\n  uncalibrated {unc}, ours "
      f"{med[ours_col]:.4f}, oracle {orc:.4f}"
      f"  -> self-calibration recovers {100*(unc-med[ours_col])/(unc-orc):.1f}% of the gap")
