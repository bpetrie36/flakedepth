#!/usr/bin/env python3
"""
stability.py - how much do the reported numbers depend on how many flakes you measured?

Re-running at different n confounds two things: genuine sample-size dependence and
which particular flakes happened to be included. Bootstrap resampling separates them
using the run you already have, and gives confidence intervals rather than point
estimates. This is the analysis a referee will ask for.

USAGE
    python stability.py out_L/thickness.csv --material graphene
    python stability.py out_L/thickness.csv out_M/thickness.csv out_H/thickness.csv --material graphene

REPORTS
  * bootstrap 95% CI on recovered layer height and exact agreement at the full n
  * how those CIs shrink with n (subsample curve) -- tells you when you have enough
  * whether datasets differ by more than sampling noise (the cross-dataset question)
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "core"), _os.path.join(_os.path.dirname(_HERE), "core")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, csv
import numpy as np

LAYER = {"graphene": 0.335, "hbn": 0.333, "mos2": 0.615,
         "mose2": 0.646, "ws2": 0.616, "wse2": 0.649}
B = 4000   # bootstrap replicates

def load(path):
    rows = list(csv.DictReader(open(path)))
    lab = np.array([float(r["mask_class"]) for r in rows])
    d = np.array([float(r["thickness_nm"]) for r in rows])
    return lab, d

def stats_of(lab, d, layer):
    """slope (recovered layer height), exact agreement, correlation."""
    A = np.vstack([lab, np.ones_like(lab)]).T
    sl, ic = np.linalg.lstsq(A, d, rcond=None)[0]
    ex = float(np.mean(np.round(d / layer) == lab))
    r = float(np.corrcoef(lab, d)[0, 1]) if len(set(lab)) > 1 else np.nan
    return float(sl), ex, r

def boot(lab, d, layer, n=None, reps=B, seed=0):
    rng = np.random.default_rng(seed)
    N = len(lab); n = n or N
    out = []
    for _ in range(reps):
        i = rng.integers(0, N, n)
        if len(set(lab[i])) < 2:
            continue
        out.append(stats_of(lab[i], d[i], layer))
    a = np.array(out)
    return a  # columns: slope, exact, r

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--material", required=True, choices=sorted(LAYER))
    p.add_argument("--subsample", default="25,50,100,150,200")
    a = p.parse_args()
    layer = LAYER[a.material]
    print(f"stability analysis  |  {a.material}, true layer height {layer:.3f} nm")
    print(f"{B} bootstrap replicates\n")

    keep = {}
    for path in a.csvs:
        lab, d = load(path)
        name = _os.path.basename(_os.path.dirname(path)) or path
        keep[name] = (lab, d)
        sl, ex, r = stats_of(lab, d, layer)
        bs = boot(lab, d, layer)
        slo, shi = np.percentile(bs[:, 0], [2.5, 97.5])
        elo, ehi = np.percentile(bs[:, 1], [2.5, 97.5])
        print(f"=== {name}  (n={len(lab)}) ===")
        print(f"  recovered layer height  {sl:.4f} nm   95% CI [{slo:.4f}, {shi:.4f}]"
              f"   error {100*(sl-layer)/layer:+.1f}% (CI [{100*(slo-layer)/layer:+.1f}, "
              f"{100*(shi-layer)/layer:+.1f}]%)")
        print(f"  exact agreement         {100*ex:.1f}%      95% CI [{100*elo:.1f}, {100*ehi:.1f}]%")
        print(f"  correlation r           {r:.4f}")
        print(f"  true layer height inside CI? {'YES' if slo <= layer <= shi else 'NO'}")
        print()

    print("SUBSAMPLE CURVE - width of the 95% CI on layer height vs. number of flakes")
    print(f"{'dataset':<12}" + "".join(f"{('n='+x):>16}" for x in a.subsample.split(",")))
    for name, (lab, d) in keep.items():
        cells = []
        for x in a.subsample.split(","):
            n = int(x)
            if n > len(lab):
                cells.append(f"{'--':>16}"); continue
            bs = boot(lab, d, layer, n=n, reps=1500)
            lo, hi = np.percentile(bs[:, 0], [2.5, 97.5])
            cells.append(f"{f'+/-{100*(hi-lo)/2/layer:.1f}%':>16}")
        print(f"{name:<12}" + "".join(cells))
    print("  (half-width of the CI, as a percentage of the true layer height)")
    print("  Halving the CI needs 4x the flakes: if the curve has flattened, more data")
    print("  will not help and the residual error is systematic, not statistical.")

    if len(keep) > 1:
        print("\nPAIRWISE: do datasets differ by more than sampling noise?")
        names = list(keep)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                (l1, d1), (l2, d2) = keep[names[i]], keep[names[j]]
                b1 = boot(l1, d1, layer, seed=1)[:, 0]
                b2 = boot(l2, d2, layer, seed=2)[:, 0]
                diff = b1 - b2
                lo, hi = np.percentile(diff, [2.5, 97.5])
                sig = "DIFFERENT (CI excludes 0)" if lo > 0 or hi < 0 else "consistent"
                print(f"  {names[i]} vs {names[j]}: layer-height difference "
                      f"{np.median(diff):+.4f} nm, 95% CI [{lo:+.4f}, {hi:+.4f}] -> {sig}")

if __name__ == "__main__":
    main()
