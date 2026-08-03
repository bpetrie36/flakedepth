import numpy as np, jax.numpy as jnp, time, sys
from scene import sample_scene, map_estimate, bayes_crb
from dtmm import n_graphene, n_mos2, MOS2_LAYER_NM

def run(n_fn, n_scenes, d_min, seed, tag):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for i in range(n_scenes):
        d, yf, yb, nf, nbg = sample_scene(rng, n_fn=n_fn, d_range=(d_min, 120.0))
        th, L = map_estimate(yf, yb, float(nf), float(nbg), n_fn)
        d_hat = float(jnp.exp(th[0]))
        crb = float(bayes_crb(d, n_fn=n_fn))
        rows.append((d, d_hat, crb))
        if (i + 1) % 10 == 0:
            e = np.abs(np.array(rows)[:, 1] - np.array(rows)[:, 0])
            print(f"[{tag}] {i+1}/{n_scenes}  running MAE={e.mean():.3f} median={np.median(e):.3f} nm "
                  f"({(time.time()-t0)/(i+1):.1f}s/scene)", flush=True)
    return np.array(rows)

if __name__ == "__main__":
    g = run(n_graphene, 150, 0.335, 1, "graphene")
    np.save("results_graphene.npy", g)
    m = run(n_mos2, 75, MOS2_LAYER_NM, 2, "mos2")
    np.save("results_mos2.npy", m)
    for tag, r in (("graphene", g), ("mos2", m)):
        e = np.abs(r[:, 1] - r[:, 0])
        print(f"FINAL [{tag}]: MAE={e.mean():.3f} nm  median={np.median(e):.3f} nm  "
              f"p90={np.percentile(e,90):.3f} nm  max={e.max():.3f} nm  "
              f"%<5.8nm={100*np.mean(e<5.8):.1f}%", flush=True)
