"""M3 coverage: alternate standard substrate (90 nm oxide) and hBN (transparent, k=0)."""
import sys, numpy as np, jax, jax.numpy as jnp
TOX = float(sys.argv[1]); MAT = sys.argv[2]; N = int(sys.argv[3]); OUT = sys.argv[4]
import scene, estimator_v2
scene.TOX0 = TOX; estimator_v2.TOX0 = TOX          # set BEFORE any jit tracing
from scene import sample_scene
from estimator_v2 import estimate_v2
from dtmm import n_graphene, n_hbn, n_mos2, GRAPHENE_LAYER_NM, HBN_LAYER_NM, MOS2_LAYER_NM
MATS = {"graphene": (n_graphene, GRAPHENE_LAYER_NM), "hbn": (n_hbn, HBN_LAYER_NM), "mos2": (n_mos2, MOS2_LAYER_NM)}
n_fn, layer = MATS[MAT]
rng = np.random.default_rng(210)
rows = []
for i in range(N):
    d, yf, yb, nf, nb = sample_scene(rng, n_fn=n_fn, d_range=(layer, 120.0))
    rows.append((d, estimate_v2(yf, yb, nf, nb, n_fn=n_fn)["d"]))
    np.save(OUT, np.array(rows))
r = np.array(rows); e = np.abs(r[:,1]-r[:,0]); sub = e[r[:,0]<25]
print("%-9s tox=%5.0f n=%3d  median=%7.3f  MAE=%7.3f  fail=%2d (%.0f%%)  sub25_med=%6.3f sub25_fail=%d/%d"
      % (MAT, TOX, len(e), np.median(e), e.mean(), np.sum(e>5.8), 100*np.mean(e>5.8), np.median(sub), np.sum(sub>5.8), len(sub)))
