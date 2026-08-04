"""
Differentiable TMM forward model for 2D-flake thickness estimation from RGB microscopy.
v0.2: authoritative data (Aspnes-Studna Si, Song MoS2, exact CIE 1931/D65): validates contrast physics + computes Cramer-Rao bounds on
thickness identifiability via JAX autograd through the full optics->color pipeline.

NOTE: Si n,k and D65 tables are coarse placeholders from standard references
(Aspnes-Studna-like values); swap for authoritative tabulated data before any
publication-grade run. SiO2 uses Malitson Sellmeier. Graphene uses the Blake
et al. constant-index approximation n = 2.6 - 1.3i, d_layer = 0.335 nm.
"""
import os as _os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---------------- wavelength grid ----------------
LAM_NM = jnp.linspace(380.0, 780.0, 161)  # 2.5 nm steps

# ---------------- materials ----------------
def n_sio2(lam_nm):
    """Malitson (1965) Sellmeier, lambda in nm -> real index."""
    L = (lam_nm / 1000.0) ** 2  # um^2
    n2 = 1 + 0.6961663 * L / (L - 0.0684043**2) \
           + 0.4079426 * L / (L - 0.1162414**2) \
           + 0.8974794 * L / (L - 9.896161**2)
    return jnp.sqrt(n2) + 0.0j

_DAT = np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "optical_data.npz"))
_SI_NM, _SI_N, _SI_K = (jnp.asarray(_DAT[k]) for k in ("si_nm","si_n","si_k"))
_MOS2_NM = jnp.asarray(_DAT["mos2_bulk_nm"]); _MOS2_N = jnp.asarray(_DAT["mos2_bulk_n"]); _MOS2_K = jnp.asarray(_DAT["mos2_bulk_k"])

def n_si(lam_nm):
    n = jnp.interp(lam_nm, _SI_NM, _SI_N)
    k = jnp.interp(lam_nm, _SI_NM, _SI_K)
    return n + 1j * k

def n_graphene(lam_nm):
    return jnp.full_like(lam_nm, 2.6) + 1j * 1.3

def n_mos2(lam_nm):
    """Song et al. 2019 bulk MoS2 (swap for layer-dependent Song-1L..13L in v0.3)."""
    return jnp.interp(lam_nm, _MOS2_NM, _MOS2_N) + 1j * jnp.interp(lam_nm, _MOS2_NM, _MOS2_K)

MOS2_LAYER_NM = 0.615

GRAPHENE_LAYER_NM = 0.335

# ---------------- differentiable TMM (normal incidence) ----------------
def _layer_matrix(n, d_nm, lam_nm):
    delta = 2.0 * jnp.pi * n * d_nm / lam_nm
    c, s = jnp.cos(delta), jnp.sin(delta)
    return jnp.array([[c, 1j * s / n], [1j * n * s, c]])

def reflectance(d_flake_nm, t_ox_nm, lam_nm, n_flake_fn=n_graphene):
    """R(lambda) for air / flake(d) / SiO2(t_ox) / Si stack. Scalar lambda."""
    # convention: materials supply n + ik (absorbing k>0); this characteristic-matrix
    # form assumes exp(-iwt) with n - ik, so conjugate at the boundary. Validated to
    # 4 decimals against the reference `tmm` package (Byrnes).
    n0, ns = 1.0 + 0.0j, jnp.conj(n_si(lam_nm))
    M = jnp.eye(2, dtype=jnp.complex128)
    M = M @ _layer_matrix(jnp.conj(n_flake_fn(lam_nm)), d_flake_nm, lam_nm)
    M = M @ _layer_matrix(jnp.conj(n_sio2(lam_nm)), t_ox_nm, lam_nm)
    num = n0 * M[0, 0] + n0 * ns * M[0, 1] - M[1, 0] - ns * M[1, 1]
    den = n0 * M[0, 0] + n0 * ns * M[0, 1] + M[1, 0] + ns * M[1, 1]
    r = num / den
    return jnp.abs(r) ** 2

reflectance_spectrum = jax.vmap(reflectance, in_axes=(None, None, 0))

# ---------------- illuminant + observer + camera ----------------
def _gpw(x, mu, s1, s2):
    sig = jnp.where(x < mu, s1, s2)
    return jnp.exp(-0.5 * ((x - mu) / sig) ** 2)

_CMF_TAB = jnp.asarray(_DAT["cmf"]); _LAM_TAB = jnp.asarray(_DAT["lam"])
def cie_cmfs(lam_nm):
    """Exact CIE 1931 2-deg observer (tabulated, via color-science)."""
    return jnp.stack([jnp.interp(lam_nm, _LAM_TAB, _CMF_TAB[i]) for i in range(3)])

_D65_TAB = jnp.asarray(_DAT["d65"])
def d65(lam_nm):
    """Exact CIE D65 SPD (tabulated, via color-science)."""
    return jnp.interp(lam_nm, _LAM_TAB, _D65_TAB)

_XYZ2RGB = jnp.array([[ 3.2406, -1.5372, -0.4986],
                      [-0.9689,  1.8758,  0.0415],
                      [ 0.0557, -0.2040,  1.0570]])

def rgb_pixel(d_flake_nm, t_ox_nm, lam_nm=LAM_NM):
    """Linear-sRGB pixel value for the stack under D65 (relative to perfect reflector)."""
    R = reflectance_spectrum(d_flake_nm, t_ox_nm, lam_nm)
    I = d65(lam_nm)
    cmf = cie_cmfs(lam_nm)                      # (3, L)
    norm = jnp.trapezoid(I * cmf[1], lam_nm)
    XYZ = jnp.trapezoid(I * R * cmf, lam_nm, axis=-1) / norm
    return _XYZ2RGB @ XYZ

# ---------------- Fisher information / Cramer-Rao bound ----------------
def crb_std_nm(d_flake_nm, t_ox_nm, sigma=0.005):
    """CRB on thickness (nm): additive Gaussian noise sigma per linear RGB channel.
    sigma=0.005 ~ shot/read noise of a decent 8-12 bit microscope camera."""
    J = jax.jacfwd(rgb_pixel, argnums=0)(d_flake_nm, t_ox_nm)  # dRGB/dd, shape (3,)
    fisher = jnp.sum(J ** 2) / sigma ** 2
    return 1.0 / jnp.sqrt(fisher)

crb_grid = jax.jit(jax.vmap(jax.vmap(crb_std_nm, in_axes=(0, None)), in_axes=(None, 0)))

# ---------------- self-test ----------------
if __name__ == "__main__":
    # bare 300nm-oxide wafer should look violet/purple-ish; 90nm greenish-yellow? sanity RGB
    for tox in (90.0, 285.0):
        print(f"t_ox={tox:5.0f}nm  bare RGB={np.array(rgb_pixel(0.0, tox))}")
    # monolayer graphene contrast at 550nm vs oxide: expect maxima near ~90 and ~285 nm
    tox = jnp.linspace(20, 400, 381)
    r0 = jax.vmap(lambda t: reflectance(0.0, t, 550.0))(tox)
    r1 = jax.vmap(lambda t: reflectance(GRAPHENE_LAYER_NM, t, 550.0))(tox)
    C = (r0 - r1) / r0
    i = int(jnp.argmax(C[:150])); j = 150 + int(jnp.argmax(C[150:]))
    print(f"contrast peaks at t_ox = {float(tox[i]):.0f} nm and {float(tox[j]):.0f} nm "
          f"(literature: ~90 and ~285 nm); C_max = {float(C[j]):.3f}")
    print(f"CRB(monolayer, 285nm oxide) = {float(crb_std_nm(GRAPHENE_LAYER_NM, 285.0)):.3f} nm")


# ---------------- oblique incidence + numerical-aperture averaging (v0.3) ----------------
def reflectance_oblique(d_flake_nm, t_ox_nm, lam_nm, s2, n_fn=n_graphene):
    """Unpolarized reflectance at incidence angle theta with s2 = sin^2(theta).
    Characteristic-matrix TMM with modified phase and admittances; conjugate convention."""
    n0c = 1.0 + 0.0j
    nsc = jnp.conj(n_si(lam_nm))
    nfc = jnp.conj(n_fn(lam_nm))
    noxc = jnp.conj(n_sio2(lam_nm))
    q0 = jnp.sqrt(n0c * n0c - s2)
    qs = jnp.sqrt(nsc * nsc - s2)
    def stack_R(eta_of):
        M = jnp.eye(2, dtype=jnp.complex128)
        for nc, dd in ((nfc, d_flake_nm), (noxc, t_ox_nm)):
            q = jnp.sqrt(nc * nc - s2)
            delta = 2.0 * jnp.pi * q * dd / lam_nm
            eta = eta_of(nc, q)
            c, s = jnp.cos(delta), jnp.sin(delta)
            M = M @ jnp.array([[c, 1j * s / eta], [1j * eta * s, c]])
        eta0, etas = eta_of(n0c, q0), eta_of(nsc, qs)
        B = M[0, 0] + etas * M[0, 1]
        C = M[1, 0] + etas * M[1, 1]
        r = (eta0 * B - C) / (eta0 * B + C)
        return jnp.abs(r) ** 2
    Rs = stack_R(lambda nc, q: q)            # s-polarization
    Rp = stack_R(lambda nc, q: nc * nc / q)  # p-polarization
    return 0.5 * (Rs + Rp)

def reflectance_NA(d_flake_nm, t_ox_nm, lam_nm, n_fn=n_graphene, NA=0.55, n_nodes=6):
    """Pupil-averaged reflectance for epi-illumination with uniform pupil fill:
    uniform in u = sin^2(theta) over [0, NA^2] (midpoint rule)."""
    u = (jnp.arange(n_nodes) + 0.5) / n_nodes * NA**2
    return jnp.mean(jax.vmap(lambda s2: reflectance_oblique(d_flake_nm, t_ox_nm, lam_nm, s2, n_fn))(u))


# ---------------- hBN (Grudinin ordinary ray; transparent in visible, k=0) ----------------
_HBN_NM = jnp.linspace(400.0, 780.0, 20)
_HBN_N = jnp.array([2.286,2.271,2.259,2.250,2.243,2.237,2.232,2.227,2.224,2.221,
                    2.219,2.216,2.215,2.213,2.212,2.211,2.210,2.209,2.208,2.207])
def n_hbn(lam_nm):
    return jnp.interp(lam_nm, _HBN_NM, _HBN_N) + 0.0j
HBN_LAYER_NM = 0.333


# ---------------- TMD in-plane (ordinary ray) dispersion, Munkhbat et al. ----------------
_TMD_NM = jnp.linspace(400.0, 780.0, 20)
_MOSE2_N = jnp.array([3.552,4.023,4.425,4.617,4.720,4.837,4.979,5.099,5.148,5.111,
                      5.049,5.036,5.002,4.909,4.947,5.162,5.138,5.013,4.922,4.865])
_MOSE2_K = jnp.array([2.986,3.013,2.791,2.515,2.327,2.189,2.033,1.825,1.589,1.385,
                      1.268,1.184,1.068,1.035,1.126,1.020,0.7649,0.6695,0.6642,0.7427])
_WS2_N = jnp.array([3.781,4.317,4.519,5.073,5.403,5.000,4.989,4.965,4.696,4.516,
                    4.359,4.113,5.264,4.808,4.582,4.441,4.343,4.270,4.213,4.167])
_WS2_K = jnp.array([2.526,2.527,2.226,2.313,1.376,0.9867,1.035,0.6045,0.4672,0.4565,
                    0.4908,0.9093,0.5371,0.1027,0.02939,0.00655,0.001037,0.0,0.0,0.0])
_WSE2_N = jnp.array([4.045,4.272,4.417,4.539,4.755,4.755,4.634,4.628,4.745,5.030,
                     5.131,5.000,4.852,4.726,4.616,4.511,4.406,4.326,4.653,5.259])
_WSE2_K = jnp.array([2.466,2.226,2.117,1.963,1.839,1.468,1.402,1.414,1.459,1.343,
                     0.9681,0.7029,0.5724,0.5043,0.4715,0.4726,0.5276,0.7177,1.125,0.5426])

def n_mose2(lam_nm):
    return jnp.interp(lam_nm,_TMD_NM,_MOSE2_N) + 1j*jnp.interp(lam_nm,_TMD_NM,_MOSE2_K)
def n_ws2(lam_nm):
    return jnp.interp(lam_nm,_TMD_NM,_WS2_N) + 1j*jnp.interp(lam_nm,_TMD_NM,_WS2_K)
def n_wse2(lam_nm):
    return jnp.interp(lam_nm,_TMD_NM,_WSE2_N) + 1j*jnp.interp(lam_nm,_TMD_NM,_WSE2_K)

MOSE2_LAYER_NM, WS2_LAYER_NM, WSE2_LAYER_NM = 0.646, 0.616, 0.649
