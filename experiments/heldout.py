"""FROZEN-CONFIG held-out evaluation. No hyperparameter touched after this file was written.
Config frozen: window=40, top-7 candidates, dedupe 0.08, k_basis=5, positivity weight 400,
grid 140, dual warm/cold LM starts, bidirectional sweep.
Seeds 200+ were never used during development (dev used 0,3,7,11,12,13,21,22,31-34,55,77,99)."""
import sys, time
import numpy as np
from scene import sample_scene
from estimator_v2 import estimate_v2
from dtmm import n_graphene, n_mos2, MOS2_LAYER_NM

mat = sys.argv[1]; seed = int(sys.argv[2]); n = int(sys.argv[3]); out = sys.argv[4]
n_fn = n_graphene if mat == "graphene" else n_mos2
dmin = 0.335 if mat == "graphene" else MOS2_LAYER_NM
rng = np.random.default_rng(seed)
rows, t0 = [], time.time()
for i in range(n):
    d, yf, yb, nf, nb = sample_scene(rng, n_fn=n_fn, d_range=(dmin, 120.0))
    r = estimate_v2(yf, yb, nf, nb, n_fn=n_fn, k_basis=5)
    rows.append((d, r["d"]))
    np.save(out, np.array(rows))
    if (i + 1) % 10 == 0:
        a = np.array(rows); e = np.abs(a[:,1]-a[:,0])
        print(f"{i+1}/{n} median={np.median(e):.3f} MAE={e.mean():.3f} fail={int(np.sum(e>5.8))} "
              f"({(time.time()-t0)/(i+1):.2f}s/scene)", flush=True)
