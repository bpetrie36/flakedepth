"""Three-region mode-lattice inference, hard-regime stress test (all d in 28-120 nm)."""
import sys, time
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import estimator_v2 as E
from estimator_v2 import BASIS, _rgb81, _REF81, _lm, SIGMA, SIG_TOX, SIG_LOGG, TOX0
from dtmm import n_graphene, LAM_NM, _XYZ2RGB, reflectance
from scene import FAMILY, _CMF, SIGMA as SC_SIGMA

B5, SIGC = BASIS[5]

def _resid3(x, ys, yb, nf, nb, n_fn):
    ds = jnp.exp(x[:3]); tox = x[3]; g = jnp.exp(x[4:7]); c = x[7:12]
    ps = [g * _rgb81(ds[i], tox, c, n_fn, B5) for i in range(3)]
    pb = g * _rgb81(0.0, tox, c, n_fn, B5)
    Ipos = (_REF81 + c @ B5)[::4]
    return jnp.concatenate(
        [jnp.sqrt(nf) * (ys[i] - ps[i]) / SIGMA for i in range(3)] +
        [jnp.sqrt(nb) * (yb - pb) / SIGMA,
         jnp.array([(tox - TOX0) / SIG_TOX]), x[4:7] / SIG_LOGG, c / SIGC,
         jnp.maximum(-Ipos, 0.0) * 400.0])

@partial(jax.jit, static_argnums=(5,))
def polish3_batch(x0s, ys, yb, nf, nb, n_fn):
    rfn = lambda x: _resid3(x, ys, yb, nf, nb, n_fn)
    return jax.vmap(lambda x0: _lm(rfn, x0, iters=18, lam0=1e-3))(x0s)

def estimate_tri(ys, yb, nf, nb, n_fn=n_graphene):
    mode_lists = []
    for y in ys:
        r = E.estimate_v2(y, yb, nf, nb, n_fn=n_fn, k_basis=5)
        mode_lists.append(r["modes"][:3])
    x0s = []
    for a in mode_lists[0]:
        for b in mode_lists[1]:
            for cM in mode_lists[2]:
                x0s.append(np.concatenate([[np.log(a["d"]), np.log(b["d"]), np.log(cM["d"])],
                                           a["theta"][1:]]))
    while len(x0s) < 27:
        x0s.append(x0s[0])
    xs, Ls = polish3_batch(jnp.asarray(np.stack(x0s[:27])), tuple(ys), yb,
                           float(nf), float(nb), n_fn)
    b = int(np.argmin(np.array(Ls)))
    dhat = np.exp(np.array(xs[b][:3]))
    singles = [m[0]["d"] for m in mode_lists]
    return dhat, singles

def sample3(rng, n_f=100, n_bg=300):
    ds = np.exp(rng.uniform(np.log(28.0), np.log(120.0), 3))
    t_ox = float(rng.normal(TOX0, SIG_TOX))
    g = np.exp(rng.normal(0, SIG_LOGG, 3))
    I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))])
    nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    def render(dd):
        R = jax.vmap(lambda l: reflectance(float(dd), t_ox, l, n_graphene))(LAM_NM)
        return g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm))
    ys = [jnp.asarray(render(d) + rng.normal(0, SC_SIGMA / np.sqrt(n_f), 3)) for d in ds]
    yb = jnp.asarray(render(0.0) + rng.normal(0, SC_SIGMA / np.sqrt(n_bg), 3))
    return ds, ys, yb, n_f, n_bg

if __name__ == "__main__":
    n_scenes, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(99)
    rows, t0 = [], time.time()
    for i in range(n_scenes):
        ds, ys, yb, nf, nb = sample3(rng)
        dhat, singles = estimate_tri(ys, yb, nf, nb)
        rows.append(np.concatenate([ds, dhat, singles]))
        np.save(out, np.array(rows))
        if (i + 1) % 5 == 0:
            a = np.array(rows)
            eT = np.abs(a[:, 3:6] - a[:, :3]).ravel()
            eS = np.abs(a[:, 6:9] - a[:, :3]).ravel()
            print(f"{i+1}/{n_scenes} single: fail={np.sum(eS>5.8)}/{len(eS)} | "
                  f"tri: fail={np.sum(eT>5.8)}/{len(eT)} med={np.median(eT):.2f} "
                  f"({(time.time()-t0)/(i+1):.1f}s/scene)", flush=True)
