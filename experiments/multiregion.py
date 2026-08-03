"""Two flake regions + background, shared nuisances. The metamer-breaking experiment."""
import jax, jax.numpy as jnp, numpy as np
from functools import partial
from scene import (_rgb, TOX0, SIG_TOX, SIG_LOGG, SIG_C, SIGMA, FAMILY, _CMF)
from dtmm import LAM_NM, reflectance, _XYZ2RGB, n_graphene

def obs3(theta, n_fn):
    d1, d2 = jnp.exp(theta[0]), jnp.exp(theta[1])
    t_ox, g, c = theta[2], jnp.exp(theta[3:6]), theta[6:9]
    return (g * _rgb(d1, t_ox, c, n_fn), g * _rgb(d2, t_ox, c, n_fn), g * _rgb(0.0, t_ox, c, n_fn))

def nlp3(theta, y1, y2, yb, n1, n2, nb, n_fn):
    p1, p2, pb = obs3(theta, n_fn)
    L = (n1 * jnp.sum((y1 - p1) ** 2) + n2 * jnp.sum((y2 - p2) ** 2)
         + nb * jnp.sum((yb - pb) ** 2)) / (2 * SIGMA**2)
    L += (theta[2] - TOX0) ** 2 / (2 * SIG_TOX**2)
    L += jnp.sum(theta[3:6] ** 2) / (2 * SIG_LOGG**2)
    L += jnp.sum((theta[6:9] / SIG_C) ** 2) / 2
    return L

def _adam(loss_fn, theta0, steps=400, lr=0.02):
    def step(carry, _):
        th, m, v, t = carry
        g = jax.grad(loss_fn)(th)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g**2; t = t + 1
        th = th - lr * (m / (1 - 0.9**t)) / (jnp.sqrt(v / (1 - 0.999**t)) + 1e-9)
        return (th, m, v, t), None
    (th, *_), _ = jax.lax.scan(step, (theta0, jnp.zeros_like(theta0),
                                      jnp.zeros_like(theta0), 0.0), None, length=steps)
    return th, loss_fn(th)

@partial(jax.jit, static_argnums=(6,))
def map3(y1, y2, yb, n1, n2, nb, n_fn, K=8):
    loss = lambda th: nlp3(th, y1, y2, yb, n1, n2, nb, n_fn)
    dg = jnp.exp(jnp.linspace(jnp.log(0.3), jnp.log(130.0), K))
    inits = jnp.stack([jnp.concatenate([jnp.array([jnp.log(a), jnp.log(b), TOX0]), jnp.zeros(6)])
                       for a in dg for b in dg])
    ths, Ls = jax.vmap(lambda t0: _adam(loss, t0))(inits)
    b = jnp.argmin(Ls)
    return ths[b], Ls[b]

def sample_scene3(rng, n_fn=n_graphene, n_f=100, n_bg=300):
    d1 = float(np.exp(rng.uniform(np.log(0.335), np.log(120.0))))
    d2 = float(np.exp(rng.uniform(np.log(0.335), np.log(120.0))))
    t_ox = float(rng.normal(TOX0, SIG_TOX))
    g = np.exp(rng.normal(0, SIG_LOGG, 3))
    I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))])
    nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    def render(d):
        R = jax.vmap(lambda l: reflectance(d, t_ox, l, n_fn))(LAM_NM)
        return g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm))
    y1 = render(d1) + rng.normal(0, SIGMA / np.sqrt(n_f), 3)
    y2 = render(d2) + rng.normal(0, SIGMA / np.sqrt(n_f), 3)
    yb = render(0.0) + rng.normal(0, SIGMA / np.sqrt(n_bg), 3)
    return d1, d2, map(jnp.asarray, (y1, y2, yb)), n_f, n_bg

if __name__ == "__main__":
    import sys, time
    seed, n_scenes, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for i in range(n_scenes):
        d1, d2, (y1, y2, yb), nf, nb = sample_scene3(rng)
        th, L = map3(y1, y2, yb, float(nf), float(nf), float(nb), n_graphene)
        rows.append((d1, float(jnp.exp(th[0])), d2, float(jnp.exp(th[1]))))
        if (i + 1) % 5 == 0:
            r = np.array(rows); e = np.abs(np.concatenate([r[:,1]-r[:,0], r[:,3]-r[:,2]]))
            print(f"{i+1}/{n_scenes} median={np.median(e):.3f} MAE={e.mean():.3f} "
                  f"fail={np.sum(e>5.8)}/{len(e)} ({(time.time()-t0)/(i+1):.1f}s/scene)", flush=True)
    np.save(out, np.array(rows))
