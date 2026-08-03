"""Truth rendered at NA; estimator either normal-incidence or NA-aware."""
import sys, time
import numpy as np, jax.numpy as jnp, jax
import estimator_v2 as E
from dtmm import reflectance_NA, n_graphene
from scene import FAMILY, _CMF, TOX0, SIG_TOX, SIG_LOGG, SIGMA
from dtmm import LAM_NM, _XYZ2RGB

NA = 0.55
est_mode = sys.argv[1]          # "normal" | "na"
n_scenes, out = int(sys.argv[2]), sys.argv[3]

if est_mode == "na":            # patch the estimator's forward physics BEFORE first jit
    E.reflectance = lambda d, t, l, n_fn: reflectance_NA(d, t, l, n_fn, NA=NA, n_nodes=4)

def sample_scene_na(rng, n_f=100, n_bg=300):
    d = float(np.exp(rng.uniform(np.log(0.335), np.log(120.0))))
    t_ox = float(rng.normal(TOX0, SIG_TOX))
    g = np.exp(rng.normal(0, SIG_LOGG, 3))
    I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))])
    nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    def render(dd):
        R = jax.vmap(lambda l: reflectance_NA(dd, t_ox, l, n_graphene, NA=NA, n_nodes=8))(LAM_NM)
        return g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm))
    y_f = render(d) + rng.normal(0, SIGMA / np.sqrt(n_f), 3)
    y_b = render(0.0) + rng.normal(0, SIGMA / np.sqrt(n_bg), 3)
    return d, jnp.asarray(y_f), jnp.asarray(y_b), n_f, n_bg

rng = np.random.default_rng(77)
rows, t0 = [], time.time()
for i in range(n_scenes):
    d, yf, yb, nf, nb = sample_scene_na(rng)
    r = E.estimate_v2(yf, yb, nf, nb, k_basis=5)
    rows.append((d, r["d"]))
    np.save(out, np.array(rows))
    if (i + 1) % 5 == 0:
        a = np.array(rows); e = a[:, 1] - a[:, 0]
        print(f"{i+1}/{n_scenes} median|e|={np.median(np.abs(e)):.3f} median_signed={np.median(e):+.3f} "
              f"fail={np.sum(np.abs(e)>5.8)} ({(time.time()-t0)/(i+1):.1f}s/scene)", flush=True)
np.save(out, np.array(rows))
