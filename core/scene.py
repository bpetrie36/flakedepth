"""
Scene-level forward model + inference for flake thickness from one RGB microscope image.

Unknowns theta = [log d, t_ox, log g_R, log g_G, log g_B, c1, c2, c3]:
  d      flake thickness (nm)
  t_ox   oxide thickness (nm), wafer-spec prior N(285, 5)
  g      per-channel gains (white balance / exposure), log-prior N(0, 0.1)
  c      illuminant spectral-shape coefficients on an empirical 3-dim basis
         (PCA over halogen/daylight/LED/fluorescent family), prior N(0, sig_c)

Observations: mean RGB of flake region (N_f pixels) + mean RGB of bare-substrate
region (N_bg pixels), additive Gaussian pixel noise sigma.

Inference: MAP with multi-start over thickness hypotheses (breaks interference-order
multimodality), Adam refinement through the differentiable TMM -> color pipeline.
"""
import os as _os

import jax, jax.numpy as jnp, numpy as np
from functools import partial
import dtmm
from dtmm import LAM_NM, reflectance, cie_cmfs, _XYZ2RGB, n_graphene, n_mos2

jax.config.update("jax_enable_x64", True)

_IB = np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "illum_basis.npz"))
ILLUM_REF = jnp.asarray(np.interp(np.array(LAM_NM), _IB["lam"], _IB["ref"]))
ILLUM_B = jnp.asarray(np.stack([np.interp(np.array(LAM_NM), _IB["lam"], b) for b in _IB["B"]]))
SIG_C = jnp.asarray(_IB["sig_c"])
SIG_TOX, SIG_LOGG, SIGMA = 5.0, 0.1, 0.005
TOX0 = 285.0

_CMF = cie_cmfs(LAM_NM)

def _rgb(d, t_ox, c, n_fn):
    I = ILLUM_REF + c @ ILLUM_B
    R = jax.vmap(lambda l: reflectance(d, t_ox, l, n_fn))(LAM_NM)
    XYZ = jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / jnp.trapezoid(I * _CMF[1], LAM_NM)
    return _XYZ2RGB @ XYZ

def scene_obs(theta, n_fn):
    """Predicted (flake_mean_rgb, bg_mean_rgb)."""
    d, t_ox, g, c = jnp.exp(theta[0]), theta[1], jnp.exp(theta[2:5]), theta[5:8]
    return g * _rgb(d, t_ox, c, n_fn), g * _rgb(0.0, t_ox, c, n_fn)

def neg_log_post(theta, y_f, y_b, n_f, n_bg, n_fn):
    p_f, p_b = scene_obs(theta, n_fn)
    L = n_f * jnp.sum((y_f - p_f) ** 2) / (2 * SIGMA**2) \
      + n_bg * jnp.sum((y_b - p_b) ** 2) / (2 * SIGMA**2)
    L += (theta[1] - TOX0) ** 2 / (2 * SIG_TOX**2)
    L += jnp.sum(theta[2:5] ** 2) / (2 * SIG_LOGG**2)
    L += jnp.sum((theta[5:8] / SIG_C) ** 2) / 2
    return L

# ---------------- Bayesian CRB with the full nuisance set ----------------
def bayes_crb(d, n_bg=300.0, n_f=100.0, n_fn=n_graphene, t_ox=TOX0):
    theta0 = jnp.concatenate([jnp.array([jnp.log(d), t_ox]), jnp.zeros(6)])
    Jf = jax.jacfwd(lambda th: scene_obs(th, n_fn)[0])(theta0)
    Jb = jax.jacfwd(lambda th: scene_obs(th, n_fn)[1])(theta0)
    F = (n_f * Jf.T @ Jf + n_bg * Jb.T @ Jb) / SIGMA**2
    F_prior = jnp.diag(jnp.concatenate([jnp.array([0.0, 1 / SIG_TOX**2]),
                                        jnp.full(3, 1 / SIG_LOGG**2), 1 / SIG_C**2]))
    cov = jnp.linalg.inv(F + F_prior)
    return jnp.sqrt(cov[0, 0]) * d  # delta(log d) -> delta d

# ---------------- MAP estimator: multi-start + Adam through the physics ----------------
def _adam_scan(loss_fn, theta0, steps=700, lr=0.02):
    def step(carry, _):
        th, m, v, t = carry
        g = jax.grad(loss_fn)(th)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g**2
        t = t + 1
        mh, vh = m / (1 - 0.9**t), v / (1 - 0.999**t)
        th = th - lr * mh / (jnp.sqrt(vh) + 1e-9)
        return (th, m, v, t), None
    (th, *_), _ = jax.lax.scan(step, (theta0, jnp.zeros_like(theta0),
                                      jnp.zeros_like(theta0), 0.0), None, length=steps)
    return th, loss_fn(th)

@partial(jax.jit, static_argnums=(4,))
def map_estimate(y_f, y_b, n_f, n_bg, n_fn, d_min=0.3, d_max=130.0, K=32):
    loss = lambda th: neg_log_post(th, y_f, y_b, n_f, n_bg, n_fn)
    d_inits = jnp.exp(jnp.linspace(jnp.log(d_min), jnp.log(d_max), K))
    th_inits = jnp.stack([jnp.concatenate([jnp.array([jnp.log(d0), TOX0]), jnp.zeros(6)])
                          for d0 in d_inits])
    ths, losses = jax.vmap(lambda t0: _adam_scan(loss, t0))(th_inits)
    best = jnp.argmin(losses)
    return ths[best], losses[best]

# ---------------- synthetic scene generator (truth uses FULL SPDs, not the basis) ----------------
FAMILY = np.stack([np.interp(np.array(LAM_NM), _IB["lam"], s) for s in _IB["family"]])

def sample_scene(rng, n_fn=n_graphene, d_range=(0.335, 120.0), n_f=100, n_bg=300):
    d = float(np.exp(rng.uniform(np.log(d_range[0]), np.log(d_range[1]))))
    t_ox = float(rng.normal(TOX0, SIG_TOX))
    g = np.exp(rng.normal(0, SIG_LOGG, 3))
    I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))])  # real SPD: misspecified vs 3-dim basis
    R_f = jax.vmap(lambda l: reflectance(d, t_ox, l, n_fn))(LAM_NM)
    R_b = jax.vmap(lambda l: reflectance(0.0, t_ox, l, n_fn))(LAM_NM)
    nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    y_f = g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R_f * _CMF, LAM_NM, axis=-1) / nrm))
    y_b = g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R_b * _CMF, LAM_NM, axis=-1) / nrm))
    y_f = y_f + rng.normal(0, SIGMA / np.sqrt(n_f), 3)
    y_b = y_b + rng.normal(0, SIGMA / np.sqrt(n_bg), 3)
    return d, jnp.asarray(y_f), jnp.asarray(y_b), n_f, n_bg


def sample_scene_hard(rng, n_fn=n_graphene, d_range=(0.335, 120.0), n_f=100, n_bg=300,
                      tox_sd=15.0, logg_sd=0.25, noise_mult=2.0):
    """Adversarial variant: oxide and gains drawn from distributions 3x and 2.5x wider
    than the estimator's priors, noise 2x the assumed level, real SPDs."""
    d = float(np.exp(rng.uniform(np.log(d_range[0]), np.log(d_range[1]))))
    t_ox = float(rng.normal(TOX0, tox_sd))
    g = np.exp(rng.normal(0, logg_sd, 3))
    I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))])
    R_f = jax.vmap(lambda l: reflectance(d, t_ox, l, n_fn))(LAM_NM)
    R_b = jax.vmap(lambda l: reflectance(0.0, t_ox, l, n_fn))(LAM_NM)
    nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    y_f = g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R_f * _CMF, LAM_NM, axis=-1) / nrm))
    y_b = g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R_b * _CMF, LAM_NM, axis=-1) / nrm))
    sg = SIGMA * noise_mult
    y_f = y_f + rng.normal(0, sg / np.sqrt(n_f), 3)
    y_b = y_b + rng.normal(0, sg / np.sqrt(n_bg), 3)
    return d, jnp.asarray(y_f), jnp.asarray(y_b), n_f, n_bg
