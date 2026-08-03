"""
The honest identifiability test. Parameters theta = [d, t_ox, gR, gG, gB]:
flake thickness + oxide thickness + per-channel white-balance/illumination gains,
all unknown. Measurements: one flake pixel + N_bg bare-substrate pixels.
Fisher matrix F = (1/sigma^2) (J_f^T J_f + N_bg * J_b^T J_b);
marginalized CRB on d = sqrt([F^-1]_{00}).
This is the quantitative test of the self-calibration claim: bare substrate
in-frame supplies the calibration that phi-Adapt learns with black-box modules.
"""
import jax, jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dtmm import rgb_pixel, n_graphene, n_mos2, GRAPHENE_LAYER_NM, MOS2_LAYER_NM

SIGMA = 0.005

def flake_obs(theta, n_fn):
    d, tox, g = theta[0], theta[1], theta[2:]
    return g * rgb_pixel(d, tox) if n_fn is n_graphene else g * _rgb_mat(d, tox, n_fn)

def _rgb_mat(d, tox, n_fn):
    from dtmm import reflectance, d65, cie_cmfs, _XYZ2RGB, LAM_NM
    R = jax.vmap(lambda l: reflectance(d, tox, l, n_fn))(LAM_NM)
    I, cmf = d65(LAM_NM), cie_cmfs(LAM_NM)
    XYZ = jnp.trapezoid(I * R * cmf, LAM_NM, axis=-1) / jnp.trapezoid(I * cmf[1], LAM_NM)
    return _XYZ2RGB @ XYZ

def bg_obs(theta):
    return theta[2:] * rgb_pixel(0.0, theta[1])

def crb_marginal(d, tox, n_bg, n_fn=n_graphene):
    theta0 = jnp.array([d, tox, 1.0, 1.0, 1.0])
    Jf = jax.jacfwd(lambda th: flake_obs(th, n_fn))(theta0)   # (3,5)
    Jb = jax.jacfwd(bg_obs)(theta0)                            # (3,5), col 0 = 0
    F = (Jf.T @ Jf + n_bg * (Jb.T @ Jb)) / SIGMA**2
    return jnp.sqrt(jnp.linalg.inv(F)[0, 0])

def crb_known(d, tox, n_fn=n_graphene):
    J = jax.jacfwd(lambda dd: flake_obs(jnp.array([dd, tox, 1., 1., 1.]), n_fn))(d)
    return SIGMA / jnp.linalg.norm(J)

d_grid = jnp.linspace(GRAPHENE_LAYER_NM, 120.0, 160)
TOX = 285.0
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

# Panel A: graphene, marginalized CRB(d) for increasing background evidence
ax = axes[0]
known = np.array(jax.vmap(lambda d: crb_known(d, TOX))(d_grid))
for n_bg, c in [(3, "#fdae61"), (30, "#f46d43"), (300, "#d73027"), (3000, "#7f0000")]:
    marg = np.array(jax.vmap(lambda d: crb_marginal(d, TOX, n_bg))(d_grid))
    ax.semilogy(np.array(d_grid), marg, color=c, lw=1.8, label=f"marginalized, $N_{{bg}}$={n_bg}")
ax.semilogy(np.array(d_grid), known, "k--", lw=2, label="oracle (nuisances known)")
ax.axhline(5.8, color="#4575b4", lw=1.5, ls=":", label="$\\varphi$-Adapt error (5.8 nm)")
ax.set_xlabel("graphene thickness $d$ (nm)"); ax.set_ylabel("CRB on $d$ (nm)")
ax.set_title("Self-calibration: joint estimation of\n$(d, t_{ox}, g_{RGB})$, $t_{ox}$=285 nm")
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")

# Panel B: convergence vs background pixel count at fixed thicknesses
ax = axes[1]
Ns = np.array([1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
for d0, lab, c in [(GRAPHENE_LAYER_NM, "monolayer (0.335 nm)", "#2166ac"),
                   (10.0, "10 nm", "#66bd63"), (50.0, "50 nm", "#5e3c99")]:
    v = np.array([crb_marginal(d0, TOX, float(n)) for n in Ns])
    ax.loglog(Ns, v, "o-", color=c, lw=1.8, ms=4, label=f"$d$ = {lab}")
    ax.axhline(float(crb_known(d0, TOX)), color=c, ls="--", lw=1, alpha=0.6)
ax.axhline(5.8, color="#4575b4", lw=1.5, ls=":")
ax.set_xlabel("background pixels $N_{bg}$"); ax.set_ylabel("marginalized CRB on $d$ (nm)")
ax.set_title("Convergence to oracle bound\n(dashed = known-nuisance floor)")
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")

# Panel C: material generality — MoS2 with tabulated Song dispersion
ax = axes[2]
d_grid_m = jnp.linspace(MOS2_LAYER_NM, 120.0, 120)
known_m = np.array(jax.vmap(lambda d: crb_known(d, TOX, n_mos2))(d_grid_m))
marg_m = np.array(jax.vmap(lambda d: crb_marginal(d, TOX, 300, n_mos2))(d_grid_m))
ax.semilogy(np.array(d_grid_m), marg_m, color="#d73027", lw=1.8, label="marginalized, $N_{bg}$=300")
ax.semilogy(np.array(d_grid_m), known_m, "k--", lw=2, label="oracle")
ax.axhline(5.8, color="#4575b4", lw=1.5, ls=":", label="$\\varphi$-Adapt error")
ax.set_xlabel("MoS$_2$ thickness $d$ (nm)"); ax.set_ylabel("CRB on $d$ (nm)")
ax.set_title("Zero-shot material swap: MoS$_2$\n(Song et al. tabulated dispersion, no retraining)")
ax.legend(fontsize=8); ax.grid(alpha=0.25, which="both")

plt.tight_layout()
plt.savefig("nuisance_crb.png", dpi=160)

m300 = np.array(jax.vmap(lambda d: crb_marginal(d, TOX, 300))(d_grid))
print("graphene @ N_bg=300: median marginalized CRB = %.3f nm | oracle median = %.3f nm | ratio = %.2fx"
      % (np.median(m300), np.median(known), np.median(m300 / known)))
print("fraction of thickness range below 5.8 nm (marginalized, N_bg=300): %.1f%%" % (100 * np.mean(m300 < 5.8)))
print("MoS2 @ N_bg=300: median marginalized CRB = %.3f nm" % np.median(marg_m))
