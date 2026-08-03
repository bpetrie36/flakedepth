"""
Estimator v2: profile-likelihood continuation + Levenberg-Marquardt.

v1 (scene.py): 32 independent Adam runs x 700 steps from log-spaced thickness inits.
Wasteful (most starts fall into the same basins) and not exhaustive (a mode between
two inits can be missed).

v2 idea: the loss is only multi-modal in d; the nuisance subproblem at fixed d is
a benign nonlinear least-squares. So:
  1. Sweep d over a dense log grid; at each d, solve for nuisances with LM,
     warm-started from the previous grid point (continuation). This traces the
     exact profile posterior L*(d) and enumerates every mode by construction.
  2. Extract all local minima of L*(d) within a posterior window of the best.
  3. Polish each candidate with full joint LM over (d, nuisances).
  4. Report best mode + alternates with posterior gaps.

Also: 81-wavelength grid (validated: 7.5e-5 max RGB error, sigma is 5e-3) and an
optional 5-component illuminant basis (99.0% family variance vs 92.8% for 3).
"""
import os as _os

import jax, jax.numpy as jnp, numpy as np
from functools import partial
import dtmm
from dtmm import reflectance, cie_cmfs, d65 as d65_fn, _XYZ2RGB, n_graphene, n_mos2

jax.config.update("jax_enable_x64", True)

LAM81 = jnp.linspace(380.0, 780.0, 81)
_CMF81 = cie_cmfs(LAM81)
_IB = np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "illum_basis.npz"))
_REF81 = jnp.asarray(np.interp(np.array(LAM81), _IB["lam"], _IB["ref"]))
BASIS = {3: (jnp.asarray(np.stack([np.interp(np.array(LAM81), _IB["lam"], b) for b in _IB["B"]])),
             jnp.asarray(_IB["sig_c"])),
         5: (jnp.asarray(np.stack([np.interp(np.array(LAM81), _IB["lam"], b) for b in _IB["B5"]])),
             jnp.asarray(_IB["sig_c5"]))}
SIGMA, SIG_TOX, SIG_LOGG, TOX0 = 0.005, 5.0, 0.1, 285.0

def _rgb81(d, tox, c, n_fn, B):
    I = _REF81 + c @ B
    R = jax.vmap(lambda l: reflectance(d, tox, l, n_fn))(LAM81)
    XYZ = jnp.trapezoid(I * R * _CMF81, LAM81, axis=-1) / jnp.trapezoid(I * _CMF81[1], LAM81)
    return _XYZ2RGB @ XYZ

def _resid(logd, nuis, yf, yb, nf, nb, n_fn, B, sigc):
    """Weighted residual vector: 6 data + (4+k) prior pseudo-residuals. L = 0.5*||r||^2."""
    d, tox, g, c = jnp.exp(logd), nuis[0], jnp.exp(nuis[1:4]), nuis[4:]
    pf, pb = g * _rgb81(d, tox, c, n_fn, B), g * _rgb81(0.0, tox, c, n_fn, B)
    # physical constraint: spectral power is nonnegative. Wrong interference orders
    # were found cheating with negative-illuminant fits; penalize below zero.
    I_sub = (_REF81 + c @ B)[::4]
    r_pos = jnp.maximum(-I_sub, 0.0) * 400.0
    return jnp.concatenate([
        jnp.sqrt(nf) * (yf - pf) / SIGMA,
        jnp.sqrt(nb) * (yb - pb) / SIGMA,
        jnp.array([(tox - TOX0) / SIG_TOX]),
        nuis[1:4] / SIG_LOGG,
        c / sigc,
        r_pos])

def _lm(resid_fn, x0, iters, lam0=1e-2):
    """Levenberg-Marquardt with multiplicative damping, fixed iteration count (jit-safe)."""
    def step(carry, _):
        x, lam = carry
        r = resid_fn(x)
        J = jax.jacfwd(resid_fn)(x)
        A = J.T @ J
        g = J.T @ r
        delta = jnp.linalg.solve(A + lam * (jnp.diag(jnp.diag(A)) + 1e-8 * jnp.eye(A.shape[0])), -g)
        x_new = x + delta
        better = jnp.sum(resid_fn(x_new) ** 2) < jnp.sum(r ** 2)
        x = jnp.where(better, x_new, x)
        lam = jnp.where(better, jnp.maximum(lam * 0.4, 1e-8), jnp.minimum(lam * 5.0, 1e4))
        return (x, lam), None
    (x, _), _ = jax.lax.scan(step, (x0, lam0), None, length=iters)
    return x, 0.5 * jnp.sum(resid_fn(x) ** 2)

@partial(jax.jit, static_argnums=(4, 5))
def profile_sweep(yf, yb, nf, nb, n_fn, k_basis, n_grid=140):
    """Continuation over the d grid; returns per-d profiled loss and nuisance tracks."""
    B, sigc = BASIS[k_basis]
    logd_grid = jnp.linspace(jnp.log(0.3), jnp.log(130.0), n_grid)
    nuis0 = jnp.concatenate([jnp.array([TOX0]), jnp.zeros(3 + k_basis)])
    # burn in the first grid point harder, then warm-start the rest
    rfn0 = lambda nu: _resid(logd_grid[0], nu, yf, yb, nf, nb, n_fn, B, sigc)
    nuis0, _ = _lm(rfn0, nuis0, iters=15)
    cold = jnp.concatenate([jnp.array([TOX0]), jnp.zeros(3 + B.shape[0])])
    def scan_step(nuis, logd):
        rfn = lambda nu: _resid(logd, nu, yf, yb, nf, nb, n_fn, B, sigc)
        nu_w, L_w = _lm(rfn, nuis, iters=5, lam0=1e-3)
        nu_c, L_c = _lm(rfn, cold, iters=5, lam0=1e-2)
        take_w = L_w <= L_c
        nuis = jnp.where(take_w, nu_w, nu_c)
        return nuis, (jnp.where(take_w, L_w, L_c), nuis)
    # bidirectional continuation: warm-start hysteresis drags each one-way sweep
    # onto a branch, so sweep up and down and keep the pointwise better solve
    _, (Ls_up, tr_up) = jax.lax.scan(scan_step, nuis0, logd_grid)
    _, (Ls_dn_r, tr_dn_r) = jax.lax.scan(scan_step, nuis0, logd_grid[::-1])
    Ls_dn, tr_dn = Ls_dn_r[::-1], tr_dn_r[::-1]
    take_up = Ls_up <= Ls_dn
    Ls = jnp.where(take_up, Ls_up, Ls_dn)
    tracks = jnp.where(take_up[:, None], tr_up, tr_dn)
    return logd_grid, Ls, tracks

@partial(jax.jit, static_argnums=(6, 7))
def polish(logd0, nuis0, yf, yb, nf, nb, n_fn, k_basis):
    B, sigc = BASIS[k_basis]
    x0 = jnp.concatenate([jnp.array([logd0]), nuis0])
    rfn = lambda x: _resid(x[0], x[1:], yf, yb, nf, nb, n_fn, B, sigc)
    x, L = _lm(rfn, x0, iters=15, lam0=1e-3)
    # Gauss-Newton curvature -> sd on log d
    J = jax.jacfwd(rfn)(x)
    H = J.T @ J + 1e-9 * jnp.eye(x.shape[0])
    cov = jnp.linalg.inv(H)
    return x, L, jnp.sqrt(cov[0, 0]), -0.5 * jnp.linalg.slogdet(H)[1]

def estimate_v2(yf, yb, nf, nb, n_fn=n_graphene, k_basis=5, window=40.0):
    """Full v2 estimate. Returns dict with best mode, sd, alternates, profile."""
    logd, Ls, tracks = profile_sweep(yf, yb, float(nf), float(nb), n_fn, k_basis)
    Ls = np.array(Ls)
    # local minima of the profile within `window` of the best
    idx = [i for i in range(1, len(Ls) - 1) if Ls[i] <= Ls[i - 1] and Ls[i] <= Ls[i + 1]]
    idx = sorted(idx, key=lambda i: Ls[i])
    idx = [i for i in idx if Ls[i] < Ls[idx[0]] + window][:7] or [int(np.argmin(Ls))]
    modes = []
    for i in idx:
        x, L, sd_logd, half_logdet = polish(logd[i], tracks[i], yf, yb, float(nf), float(nb), n_fn, k_basis)
        d = float(np.exp(x[0]))
        modes.append({"d": d, "L": float(L), "sd": d * float(sd_logd), "theta": np.array(x), "lap": float(-L + half_logdet)})
    # dedupe polished modes that merged
    modes.sort(key=lambda m: m["L"])
    dedup = []
    for m in modes:
        if all(abs(np.log(m["d"] / q["d"])) > 0.08 for q in dedup):
            dedup.append(m)
    best = dedup[0]
    return {"d": best["d"], "sd": best["sd"], "L": best["L"], "theta": best["theta"],
            "alts": [(q["d"], q["L"] - best["L"]) for q in dedup[1:]],
            "modes": dedup,
            "profile": (np.exp(np.array(logd)), Ls)}

if __name__ == "__main__":
    import time, sys
    from scene import sample_scene
    seed, n_scenes, k_basis, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for i in range(n_scenes):
        d, yf, yb, nf, nbg = sample_scene(rng)
        r = estimate_v2(yf, yb, nf, nbg, k_basis=k_basis)
        rows.append((d, r["d"]))
        if (i + 1) % 10 == 0:
            e = np.abs(np.diff(np.array(rows), axis=1))
            print(f"{i+1}/{n_scenes} median={np.median(e):.3f} MAE={e.mean():.3f} "
                  f"fail={np.sum(e>5.8)} ({(time.time()-t0)/(i+1):.2f}s/scene)", flush=True)
    np.save(out, np.array(rows))
