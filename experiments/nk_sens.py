"""Sensitivity to error in the assumed optical constants -- the question a materials
reviewer asks first. Truth is rendered with n,k scaled by (1+eps); the estimator keeps
the nominal tabulated values."""
import sys, numpy as np, jax, jax.numpy as jnp
eps_n, eps_k, N, OUT = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
import dtmm
from dtmm import LAM_NM, _XYZ2RGB, reflectance
from scene import _CMF, TOX0, SIG_TOX, SIG_LOGG, SIGMA, FAMILY
from estimator_v2 import estimate_v2
from dtmm import n_graphene

def n_true(lam):                       # perturbed material used only for RENDERING
    return jnp.full_like(lam, 2.6*(1+eps_n)) + 1j*1.3*(1+eps_k)

rng = np.random.default_rng(220); rows=[]
for i in range(N):
    d = float(np.exp(rng.uniform(np.log(0.335), np.log(120.))))
    tox = float(rng.normal(TOX0, SIG_TOX)); g = np.exp(rng.normal(0, SIG_LOGG, 3))
    I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))]); nrm = jnp.trapezoid(I*_CMF[1], LAM_NM)
    def render(dd):
        R = jax.vmap(lambda l: reflectance(dd, tox, l, n_true))(LAM_NM)
        return g*np.array(_XYZ2RGB@(jnp.trapezoid(I*R*_CMF, LAM_NM, axis=-1)/nrm))
    yf = jnp.asarray(render(d)+rng.normal(0,SIGMA/10,3)); yb = jnp.asarray(render(0.)+rng.normal(0,SIGMA/np.sqrt(300),3))
    rows.append((d, estimate_v2(yf, yb, 100, 300, n_fn=n_graphene)['d']))   # estimator uses NOMINAL n,k
    np.save(OUT, np.array(rows))
r=np.array(rows); e=r[:,1]-r[:,0]; sub=np.abs(e)[r[:,0]<25]
print('dn=%+.0f%% dk=%+.0f%%: median|e|=%.3f  signed_median=%+.3f  fail=%d/%d  sub25_med=%.3f'
      % (100*eps_n, 100*eps_k, np.median(np.abs(e)), np.median(e), np.sum(np.abs(e)>5.8), N, np.median(sub)))
