"""Independent baselines isolating what each component of the method contributes.

B1 contrast-lookup : the field-standard approach. Precompute gain-invariant flake/substrate
                     RGB ratio vs thickness at NOMINAL nuisances (D65, spec oxide); answer by
                     nearest neighbour on the observed ratio. No optimization, no calibration.
B2 fixed-nuisance  : optimize d only, nuisances pinned at prior means. Isolates the value of
                     estimating nuisances at all.
B3 gain-only       : optimize d and the three gains (standard white-balance correction), but
                     not oxide or illuminant. Isolates the value of full self-calibration.
Ours               : full joint estimation (estimator_v2).
"""
import numpy as np, jax, jax.numpy as jnp
from functools import partial
from estimator_v2 import BASIS, _rgb81, _REF81, _lm, SIGMA, SIG_TOX, SIG_LOGG, TOX0
from dtmm import n_graphene

B5, SIGC = BASIS[5]
_DGRID = np.exp(np.linspace(np.log(0.3), np.log(130.0), 600))

@partial(jax.jit, static_argnums=(1,))
def _ratio_table(tox0, n_fn):
    z = jnp.zeros(5)
    def rho(d):
        return _rgb81(d, tox0, z, n_fn, B5) / _rgb81(0.0, tox0, z, n_fn, B5)
    return jax.vmap(rho)(jnp.asarray(_DGRID))

def b1_contrast_lookup(yf, yb, n_fn=n_graphene, tox0=TOX0):
    tab = np.array(_ratio_table(tox0, n_fn))
    obs = np.array(yf) / np.array(yb)
    return float(_DGRID[np.argmin(np.linalg.norm(tab - obs, axis=1))])

def _resid_restricted(x, yf, yb, nf, nb, n_fn, tox0, mode):
    d = jnp.exp(x[0])
    g = jnp.exp(x[1:4]) if mode == "gain" else jnp.ones(3)
    z = jnp.zeros(5)
    pf, pb = g * _rgb81(d, tox0, z, n_fn, B5), g * _rgb81(0.0, tox0, z, n_fn, B5)
    r = [jnp.sqrt(nf) * (yf - pf) / SIGMA, jnp.sqrt(nb) * (yb - pb) / SIGMA]
    if mode == "gain":
        r.append(x[1:4] / SIG_LOGG)
    return jnp.concatenate(r)

@partial(jax.jit, static_argnums=(5, 7))
def _fit_restricted(x0s, yf, yb, nf, nb, n_fn, tox0, mode):
    rfn = lambda x: _resid_restricted(x, yf, yb, nf, nb, n_fn, tox0, mode)
    return jax.vmap(lambda x0: _lm(rfn, x0, iters=25, lam0=1e-2))(x0s)

def b2_fixed_nuisance(yf, yb, nf, nb, n_fn=n_graphene, tox0=TOX0):
    x0s = jnp.asarray(np.log(np.exp(np.linspace(np.log(0.3), np.log(130.0), 24)))[:, None])
    xs, Ls = _fit_restricted(x0s, yf, yb, float(nf), float(nb), n_fn, tox0, "fixed")
    return float(np.exp(np.array(xs)[int(np.argmin(np.array(Ls))), 0]))

def b3_gain_only(yf, yb, nf, nb, n_fn=n_graphene, tox0=TOX0):
    d0 = np.linspace(np.log(0.3), np.log(130.0), 24)
    x0s = jnp.asarray(np.stack([np.concatenate([[a], np.zeros(3)]) for a in d0]))
    xs, Ls = _fit_restricted(x0s, yf, yb, float(nf), float(nb), n_fn, tox0, "gain")
    return float(np.exp(np.array(xs)[int(np.argmin(np.array(Ls))), 0]))
