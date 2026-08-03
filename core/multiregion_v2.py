"""
Multi-region estimator v2: mode-lattice joint inference.

v1-multi searched the 2D thickness space with 64 blind Adam starts (9.5 s/scene,
41.7% hard-regime failures). v2 exploits structure instead:
  1. Run the single-region v2 profile for each flake region (cheap, exhaustive
     per-region mode enumeration).
  2. Form the lattice of joint hypotheses (mode_i of region 1) x (mode_j of region 2)
     -- typically 2-4 modes each, so <= ~12 hypotheses.
  3. Polish every lattice point with full joint LM under ONE shared nuisance set
     (t_ox, gains, 5-comp illuminant + positivity). A nuisance vector that rescues
     the wrong order for one region must survive the other region's data.
  4. Select the best joint posterior; report alternates.
"""
import jax, jax.numpy as jnp, numpy as np
from functools import partial
import estimator_v2 as E
from estimator_v2 import BASIS, _rgb81, _REF81, _lm, SIGMA, SIG_TOX, SIG_LOGG, TOX0
from dtmm import n_graphene

def _resid_joint(x, y1, y2, yb, n1, n2, nb, n_fn, B, sigc):
    d1, d2 = jnp.exp(x[0]), jnp.exp(x[1])
    tox, g, c = x[2], jnp.exp(x[3:6]), x[6:]
    p1 = g * _rgb81(d1, tox, c, n_fn, B)
    p2 = g * _rgb81(d2, tox, c, n_fn, B)
    pb = g * _rgb81(0.0, tox, c, n_fn, B)
    I_sub = (_REF81 + c @ B)[::4]
    return jnp.concatenate([
        jnp.sqrt(n1) * (y1 - p1) / SIGMA,
        jnp.sqrt(n2) * (y2 - p2) / SIGMA,
        jnp.sqrt(nb) * (yb - pb) / SIGMA,
        jnp.array([(tox - TOX0) / SIG_TOX]),
        x[3:6] / SIG_LOGG,
        c / sigc,
        jnp.maximum(-I_sub, 0.0) * 400.0])

@partial(jax.jit, static_argnums=(7, 8))
def _joint_polish_batch(x0s, y1, y2, yb, n1, n2, nb, n_fn, k_basis):
    B, sigc = BASIS[k_basis]
    rfn = lambda x: _resid_joint(x, y1, y2, yb, n1, n2, nb, n_fn, B, sigc)
    def one(x0):
        x, L = _lm(rfn, x0, iters=18, lam0=1e-3)
        return x, L
    return jax.vmap(one)(x0s)

def estimate_v2_multi(y1, y2, yb, n1, n2, nb, n_fn=n_graphene, k_basis=5, max_modes=4):
    # per-region mode enumeration (each region vs the same background)
    r1 = E.estimate_v2(y1, yb, n1, nb, n_fn=n_fn, k_basis=k_basis)
    r2 = E.estimate_v2(y2, yb, n2, nb, n_fn=n_fn, k_basis=k_basis)
    m1 = r1["modes"][:max_modes]
    m2 = r2["modes"][:max_modes]
    # lattice of joint hypotheses; nuisances initialized from region-1's mode fit
    x0s = []
    for a in m1:
        for b in m2:
            x0s.append(np.concatenate([[np.log(a["d"]), np.log(b["d"])], a["theta"][1:]]))
    # pad to a fixed batch size so the jitted polish compiles once
    while len(x0s) < 8:
        x0s.append(x0s[0])
    x0s = jnp.asarray(np.stack(x0s[:8]))
    xs, Ls = _joint_polish_batch(x0s, y1, y2, yb, float(n1), float(n2), float(nb), n_fn, k_basis)
    xs, Ls = np.array(xs), np.array(Ls)
    order = np.argsort(Ls)
    best = xs[order[0]]
    d1, d2 = float(np.exp(best[0])), float(np.exp(best[1]))
    alts = []
    for o in order[1:]:
        da, db = float(np.exp(xs[o][0])), float(np.exp(xs[o][1]))
        if abs(np.log(da / d1)) > 0.05 or abs(np.log(db / d2)) > 0.05:
            alts.append(((da, db), float(Ls[o] - Ls[order[0]])))
        if len(alts) >= 2:
            break
    return {"d1": d1, "d2": d2, "L": float(Ls[order[0]]), "alts": alts,
            "singles": (r1["d"], r2["d"])}

if __name__ == "__main__":
    import sys, time
    from multiregion import sample_scene3
    seed, n_scenes, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for i in range(n_scenes):
        d1, d2, (y1, y2, yb), nf, nb = sample_scene3(rng)
        r = estimate_v2_multi(y1, y2, yb, nf, nf, nb)
        rows.append((d1, r["d1"], d2, r["d2"], r["singles"][0], r["singles"][1]))
        if (i + 1) % 5 == 0:
            a = np.array(rows)
            e = np.abs(np.concatenate([a[:, 1] - a[:, 0], a[:, 3] - a[:, 2]]))
            print(f"{i+1}/{n_scenes} median={np.median(e):.3f} fail={np.sum(e>5.8)}/{len(e)} "
                  f"({(time.time()-t0)/(i+1):.1f}s/scene)", flush=True)
    np.save(out, np.array(rows))
