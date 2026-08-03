"""
Adaptive spectral acquisition ("active interference metrology").

After frame 1, the estimator inspects its own posterior modes. If unimodal: stop,
one frame suffices. If multimodal: for every filter in a bank of realizable gels,
predict each mode's GAIN-INVARIANT flake/background color ratio under that filter
(using the mode's own fitted oxide + illuminant), and acquire frame 2 through the
filter that maximizes the minimum whitened separation between mode predictions.
Joint fit over both frames (shared illuminant and oxide, per-frame gains) resolves
the order. The inference engine designs its own next measurement.
"""
import sys, time
import numpy as np, jax, jax.numpy as jnp
from functools import partial
import estimator_v2 as E
from estimator_v2 import (BASIS, _rgb81, _REF81, _lm, SIGMA, SIG_TOX, SIG_LOGG,
                          TOX0, LAM81, _CMF81)
from dtmm import n_graphene, reflectance, LAM_NM, _XYZ2RGB
from scene import FAMILY, _CMF, SIGMA as SC_SIGMA

K = 5
B5, SIGC = BASIS[K]
L81 = np.array(LAM81)
LFULL = np.array(LAM_NM)

def _sig(x): return 1.0 / (1.0 + np.exp(-x))
def _lp(l0, lam): return 0.12 + 0.88 * _sig((lam - l0) / 22.0)
def _sp(l0, lam): return 0.12 + 0.88 * _sig((l0 - lam) / 22.0)

FILTER_DEFS = [
    ("neutral", lambda lam: np.ones_like(lam)),
    ("LP500", lambda lam: _lp(500, lam)), ("LP560", lambda lam: _lp(560, lam)),
    ("LP620", lambda lam: _lp(620, lam)),
    ("SP560", lambda lam: _sp(560, lam)), ("SP620", lambda lam: _sp(620, lam)),
    ("BP520-590", lambda lam: _lp(520, lam) * _sp(590, lam)),
    ("BP460-560", lambda lam: _lp(460, lam) * _sp(560, lam)),
]
BANK81 = jnp.asarray(np.stack([f(L81) for _, f in FILTER_DEFS]))
BANKFULL = np.stack([f(LFULL) for _, f in FILTER_DEFS])

def _rgbT(d, tox, c, T, n_fn):
    I = (_REF81 + c @ B5) * T
    R = jax.vmap(lambda l: reflectance(d, tox, l, n_fn))(LAM81)
    XYZ = jnp.trapezoid(I * R * _CMF81, LAM81, axis=-1) / jnp.trapezoid(I * _CMF81[1], LAM81)
    return _XYZ2RGB @ XYZ

@partial(jax.jit, static_argnums=(3,))
def _mode_ratios(theta, T, _unused, n_fn):
    """Gain-invariant flake/bg RGB ratio predicted by a fitted mode under filter T."""
    d, tox, c = jnp.exp(theta[0]), theta[1], theta[5:]
    return _rgbT(d, tox, c, T, n_fn) / _rgbT(0.0, tox, c, T, n_fn)

def choose_filter(modes, n_fn=n_graphene):
    """Max-min whitened separation between top modes over the bank; returns index, score."""
    thetas = [jnp.asarray(m["theta"]) for m in modes[:3]]
    best = (-1.0, 0)
    scores = []
    for fi in range(len(FILTER_DEFS)):
        rat = [np.array(_mode_ratios(t, BANK81[fi], 0, n_fn)) for t in thetas]
        sep = min(np.linalg.norm(rat[i] - rat[j])
                  for i in range(len(rat)) for j in range(i + 1, len(rat)))
        scores.append(sep)
        if sep > best[0]:
            best = (sep, fi)
    return best[1], np.array(scores)

# ---- joint two-frame fit with arbitrary known filter T (shared c, tox; per-frame gains) ----
def _resid_T(x, yfA, ybA, yfB, ybB, T, nf, nb, n_fn):
    d, tox = jnp.exp(x[0]), x[1]
    gA, c, gB = jnp.exp(x[2:5]), x[5:10], jnp.exp(x[10:13])
    pfA, pbA = gA * _rgb81(d, tox, c, n_fn, B5), gA * _rgb81(0.0, tox, c, n_fn, B5)
    pfB, pbB = gB * _rgbT(d, tox, c, T, n_fn), gB * _rgbT(0.0, tox, c, T, n_fn)
    Ipos = (_REF81 + c @ B5)[::4]
    return jnp.concatenate([
        jnp.sqrt(nf) * (yfA - pfA) / SIGMA, jnp.sqrt(nb) * (ybA - pbA) / SIGMA,
        jnp.sqrt(nf) * (yfB - pfB) / SIGMA, jnp.sqrt(nb) * (ybB - pbB) / SIGMA,
        jnp.array([(tox - TOX0) / SIG_TOX]),
        x[2:5] / SIG_LOGG, c / SIGC, x[10:13] / SIG_LOGG,
        jnp.maximum(-Ipos, 0.0) * 400.0])

@partial(jax.jit, static_argnums=(8,))
def _polish_T_batch(x0s, yfA, ybA, yfB, ybB, T, nf, nb, n_fn):
    rfn = lambda x: _resid_T(x, yfA, ybA, yfB, ybB, T, nf, nb, n_fn)
    return jax.vmap(lambda x0: _lm(rfn, x0, iters=18, lam0=1e-3))(x0s)

def fit_two_frames(modes, yfA, ybA, yfB, ybB, T, nf, nb, n_fn=n_graphene):
    """Initialize from frame-1 modes (each with fresh frame-B gains) and joint-polish."""
    x0s = []
    for m in modes[:4]:
        th = m["theta"]
        x0s.append(np.concatenate([th, np.zeros(3)]))  # [logd,tox,gA,c] + gB=0
    while len(x0s) < 4:
        x0s.append(x0s[0])
    xs, Ls = _polish_T_batch(jnp.asarray(np.stack(x0s[:4])), yfA, ybA, yfB, ybB,
                             T, float(nf), float(nb), n_fn)
    b = int(np.argmin(np.array(Ls)))
    return float(np.exp(np.array(xs[b])[0]))

# ---- scene machinery ----
def sample_hard(rng, n_f=100, n_bg=300):
    d = float(np.exp(rng.uniform(np.log(28.0), np.log(120.0))))
    t_ox = float(rng.normal(TOX0, SIG_TOX))
    ill = rng.integers(len(FAMILY))
    def frame(Tfull):
        g = np.exp(rng.normal(0, SIG_LOGG, 3))
        I = jnp.asarray(FAMILY[ill] * Tfull)
        nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
        def render(dd):
            R = jax.vmap(lambda l: reflectance(dd, t_ox, l, n_graphene))(LAM_NM)
            return g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm))
        yf = jnp.asarray(render(d) + rng.normal(0, SC_SIGMA / np.sqrt(n_f), 3))
        yb = jnp.asarray(render(0.0) + rng.normal(0, SC_SIGMA / np.sqrt(n_bg), 3))
        return yf, yb
    return d, t_ox, ill, frame, n_f, n_bg


D_RIVAL_GRID = jnp.exp(jnp.linspace(jnp.log(0.5), jnp.log(128.0), 220))

@partial(jax.jit, static_argnums=(1,))
def _ratio_curve(theta, n_fn):
    tox, c = theta[1], theta[5:]
    def rho(d):
        return _rgbT(d, tox, c, jnp.ones_like(LAM81), n_fn) / _rgbT(0.0, tox, c, jnp.ones_like(LAM81), n_fn)
    return jax.vmap(rho)(D_RIVAL_GRID)

def rival_set(theta, d_hat, n_fn=n_graphene, tau=0.05, max_rivals=3):
    """Metamer partners of d_hat implied by the physics at the fitted nuisances."""
    curve = np.array(_ratio_curve(jnp.asarray(theta), n_fn))
    d0 = np.array(_mode_ratios(jnp.asarray(theta), BANK81[0], 0, n_fn))
    dist = np.linalg.norm(curve - d0, axis=1)
    dg = np.array(D_RIVAL_GRID)
    riv = []
    for i in range(1, len(dg) - 1):
        if dist[i] <= dist[i-1] and dist[i] <= dist[i+1] and dist[i] < tau:
            if abs(np.log(dg[i] / d_hat)) > 0.08:
                riv.append((dist[i], dg[i]))
    riv.sort()
    return [d for _, d in riv[:max_rivals]]

AMBIG_GAP = 20.0  # retained for reference; trigger is now physics-derived

if __name__ == "__main__":
    n_scenes, out = int(sys.argv[1]), sys.argv[2]
    rng = np.random.default_rng(55)
    rows, t0 = [], time.time()
    n_second = 0
    TRIGGER = 40.0   # profile window for acquisition triggering (wide on purpose:
                     # a false trigger costs one extra frame, a miss costs a failure)
    for i in range(n_scenes):
        d, t_ox, ill, frame, nf, nb = sample_hard(rng)
        yfA, ybA = frame(np.ones_like(LFULL))
        # one sweep, reused for both answer modes and acquisition rivals
        logd, Ls, tracks = E.profile_sweep(yfA, ybA, float(nf), float(nb), n_graphene, K)
        Ls = np.array(Ls)
        loc = [j for j in range(1, len(Ls) - 1) if Ls[j] <= Ls[j-1] and Ls[j] <= Ls[j+1]]
        loc = sorted(loc, key=lambda j: Ls[j])
        cands = []
        for j in loc:
            if Ls[j] > Ls[loc[0]] + TRIGGER: break
            x, L, sd, hld = E.polish(logd[j], tracks[j], yfA, ybA, float(nf), float(nb), n_graphene, K)
            dd = float(np.exp(x[0]))
            if all(abs(np.log(dd / q["d"])) > 0.08 for q in cands):
                cands.append({"d": dd, "L": float(L), "theta": np.array(x)})
        cands.sort(key=lambda m: m["L"])
        d_single = cands[0]["d"]
        if len(cands) < 2:
            d_adapt, fi, worst_d = d_single, -1, d_single
        else:
            n_second += 1
            fi, scores = choose_filter(cands)
            wi = int(np.argmin(scores[1:]) + 1)
            yfB, ybB = frame(BANKFULL[fi])
            d_adapt = fit_two_frames(cands, yfA, ybA, yfB, ybB, BANK81[fi], nf, nb)
            yfW, ybW = frame(BANKFULL[wi])
            worst_d = fit_two_frames(cands, yfA, ybA, yfW, ybW, BANK81[wi], nf, nb)
        rows.append((d, d_single, d_adapt, worst_d, fi))
        np.save(out, np.array(rows))
        if (i + 1) % 5 == 0:
            a = np.array(rows)
            e1, ea, ew = (np.abs(a[:, k] - a[:, 0]) for k in (1, 2, 3))
            print(f"{i+1}/{n_scenes} single_fail={np.sum(e1>5.8)} adaptive_fail={np.sum(ea>5.8)} "
                  f"worstfilter_fail={np.sum(ew>5.8)} second_frames={n_second} "
                  f"({(time.time()-t0)/(i+1):.1f}s/scene)", flush=True)
