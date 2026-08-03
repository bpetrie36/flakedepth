"""
Known-filter dual-frame protocol: photograph the same flake twice under two different
illuminants (turn the lamp dial / switch LED channel). Thickness and oxide are shared;
gains and illuminant coefficients are per-frame. Metamer partners are illuminant-dependent,
so the ambiguity sets of the two frames should intersect only near the truth.
"""
import sys, time
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import estimator_v2 as E
from estimator_v2 import BASIS, _rgb81, _REF81, _lm, SIGMA, SIG_TOX, SIG_LOGG, TOX0
from dtmm import n_graphene, LAM_NM as LAMFULL
from scene import FAMILY, _CMF, SIGMA as SC_SIGMA
from dtmm import _XYZ2RGB

K = 5
B5, SIGC = BASIS[K]
from estimator_v2 import LAM81
T_FILT = jnp.asarray(0.15 + 0.85 / (1.0 + np.exp(-(np.array(LAM81) - 570.0) / 25.0)))
from scene import FAMILY as _FAM
from dtmm import LAM_NM as _LAMF
T_FULL = jnp.asarray(0.15 + 0.85 / (1.0 + np.exp(-(np.array(_LAMF) - 570.0) / 25.0)))

def _rgb81_filt(d, tox, c, n_fn, B, T):
    from estimator_v2 import _CMF81, LAM81
    from dtmm import reflectance
    I = (_REF81 + c @ B) * T
    R = jax.vmap(lambda l: reflectance(d, tox, l, n_fn))(LAM81)
    XYZ = jnp.trapezoid(I * R * _CMF81, LAM81, axis=-1) / jnp.trapezoid(I * _CMF81[1], LAM81)
    return _XYZ2RGB @ XYZ

def _resid_dual(x, yfA, ybA, yfB, ybB, nf, nb, n_fn):
    # x = [logd, tox, loggA(3), c_shared(5), loggB(3)] -> 13
    d, tox = jnp.exp(x[0]), x[1]
    gA, c = jnp.exp(x[2:5]), x[5:10]
    gB = jnp.exp(x[10:13])
    pfA, pbA = gA * _rgb81(d, tox, c, n_fn, B5), gA * _rgb81(0.0, tox, c, n_fn, B5)
    pfB = gB * _rgb81_filt(d, tox, c, n_fn, B5, T_FILT)
    pbB = gB * _rgb81_filt(0.0, tox, c, n_fn, B5, T_FILT)
    Ipos = (_REF81 + c @ B5)[::4]
    return jnp.concatenate([
        jnp.sqrt(nf) * (yfA - pfA) / SIGMA, jnp.sqrt(nb) * (ybA - pbA) / SIGMA,
        jnp.sqrt(nf) * (yfB - pfB) / SIGMA, jnp.sqrt(nb) * (ybB - pbB) / SIGMA,
        jnp.array([(tox - TOX0) / SIG_TOX]),
        x[2:5] / SIG_LOGG, c / SIGC,
        x[10:13] / SIG_LOGG,
        jnp.maximum(-Ipos, 0.0) * 400.0])

@partial(jax.jit, static_argnums=(6,))
def profile_dual(yfA, ybA, yfB, ybB, nf, nb, n_fn, n_grid=120):
    logd = jnp.linspace(jnp.log(0.3), jnp.log(130.0), n_grid)
    nuis0 = jnp.concatenate([jnp.array([TOX0]), jnp.zeros(11)])
    def rfn_at(ld):
        return lambda nu: _resid_dual(jnp.concatenate([jnp.array([ld]), nu]),
                                      yfA, ybA, yfB, ybB, nf, nb, n_fn)
    nuis0, _ = _lm(rfn_at(logd[0]), nuis0, iters=15)
    cold = jnp.concatenate([jnp.array([TOX0]), jnp.zeros(11)])
    def step(nuis, ld):
        rfn = rfn_at(ld)
        nw, Lw = _lm(rfn, nuis, iters=5, lam0=1e-3)
        nc, Lc = _lm(rfn, cold, iters=5, lam0=1e-2)
        take = Lw <= Lc
        nuis = jnp.where(take, nw, nc)
        return nuis, (jnp.where(take, Lw, Lc), nuis)
    _, (Lu, tu) = jax.lax.scan(step, nuis0, logd)
    _, (Ld_r, td_r) = jax.lax.scan(step, nuis0, logd[::-1])
    Ld, td = Ld_r[::-1], td_r[::-1]
    take = Lu <= Ld
    return logd, jnp.where(take, Lu, Ld), jnp.where(take[:, None], tu, td)

@partial(jax.jit, static_argnums=(7,))
def polish_dual(x0, yfA, ybA, yfB, ybB, nf, nb_, n_fn=n_graphene):
    rfn = lambda x: _resid_dual(x, yfA, ybA, yfB, ybB, nf, nb_, n_fn)
    return _lm(rfn, x0, iters=15, lam0=1e-3)

def estimate_dual(yfA, ybA, yfB, ybB, nf, nb, n_fn=n_graphene):
    logd, Ls, tr = profile_dual(yfA, ybA, yfB, ybB, float(nf), float(nb), n_fn)
    Ls = np.array(Ls)
    idx = [i for i in range(1, len(Ls) - 1) if Ls[i] <= Ls[i - 1] and Ls[i] <= Ls[i + 1]]
    idx = sorted(idx, key=lambda i: Ls[i])[:4] or [int(np.argmin(Ls))]
    best = None
    for i in idx:
        x0 = jnp.concatenate([jnp.array([logd[i]]), tr[i]])
        x, L = polish_dual(x0, yfA, ybA, yfB, ybB, float(nf), float(nb), n_fn)
        if best is None or float(L) < best[1]:
            best = (float(np.exp(np.array(x)[0])), float(L))
    return best[0]

def sample_dual(rng, n_f=100, n_bg=300, d_range=(28.0, 120.0)):
    d = float(np.exp(rng.uniform(np.log(d_range[0]), np.log(d_range[1]))))
    t_ox = float(rng.normal(TOX0, SIG_TOX))
    idx0 = rng.integers(len(FAMILY))
    frames = []
    for fi in range(2):
        g = np.exp(rng.normal(0, SIG_LOGG, 3))
        I = jnp.asarray(FAMILY[idx0]) * (T_FULL if fi == 1 else 1.0)
        nrm = jnp.trapezoid(I * _CMF[1], LAMFULL)
        def render(dd):
            from dtmm import reflectance
            R = jax.vmap(lambda l: reflectance(dd, t_ox, l, n_graphene))(LAMFULL)
            return g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAMFULL, axis=-1) / nrm))
        yf = render(d) + rng.normal(0, SC_SIGMA / np.sqrt(n_f), 3)
        yb = render(0.0) + rng.normal(0, SC_SIGMA / np.sqrt(n_bg), 3)
        frames.append((jnp.asarray(yf), jnp.asarray(yb)))
    return d, frames, n_f, n_bg

if __name__ == "__main__":
    n_scenes, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(55)
    rows, t0 = [], time.time()
    for i in range(n_scenes):
        d, ((yfA, ybA), (yfB, ybB)), nf, nb = sample_dual(rng)
        d_single = E.estimate_v2(yfA, ybA, nf, nb, k_basis=5)["d"]       # frame A alone
        d_dual = estimate_dual(yfA, ybA, yfB, ybB, nf, nb)
        rows.append((d, d_single, d_dual))
        np.save(out, np.array(rows))
        if (i + 1) % 5 == 0:
            a = np.array(rows)
            e1, e2 = np.abs(a[:, 1] - a[:, 0]), np.abs(a[:, 2] - a[:, 0])
            print(f"{i+1}/{n_scenes} single: fail={np.sum(e1>5.8)} med={np.median(e1):.2f} | "
                  f"dual: fail={np.sum(e2>5.8)} med={np.median(e2):.2f} ({(time.time()-t0)/(i+1):.1f}s/scene)",
                  flush=True)
