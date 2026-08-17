#!/usr/bin/env python3
"""
regen_b1_oracle.py - regenerate the B1-oracle row of the ablation table.

WHY THIS EXISTS
  experiments/res_b1_oracle.npy is shape (45, 3) while res_baselines.npy is
  (60, 5), and the script that produced the former is not in the repository. The
  45 truths are an exact subset of the 60, so the scene set and seed match, but
  the selection rule is unknown, and the 15 excluded scenes are systematically
  EASIER (B1 median error 0.219 nm on excluded vs 0.374 nm on included). A table
  row captioned "identical held-out scenes (n=60)" cannot rest on an unexplained
  subset. This regenerates the row on all 60 with a documented procedure.

WHAT B1-ORACLE IS
  The B1 contrast lookup of baselines.py, given each scene's TRUE oxide
  thickness and TRUE illuminant spectrum instead of nominal ones. Channel gains
  need not be supplied: the lookup is on the flake/substrate RGB ratio, in which
  gains cancel exactly.

  baselines.b1_contrast_lookup already accepts tox0, but builds its table
  through the 5-component illuminant basis at c=0. The true SPDs are drawn from
  an 11-spectrum family and are deliberately NOT representable in that basis, so
  the oracle table is built directly from the scene's own spectrum here, by the
  same integral scene.py uses to render it.

VERIFICATION BUILT IN
  Replaying the sampler with the same seed must reproduce the true thicknesses
  stored in res_baselines.npy. If it does not, the scene set is wrong and the
  script says so and stops.

USAGE
  python regen_b1_oracle.py --seed 201 --n 60
  python regen_b1_oracle.py --seed 201 --n 60 --out experiments/res_b1_oracle_60.npy
"""
import argparse, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "core")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=201)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dmin", type=float, default=0.335)
    ap.add_argument("--dmax", type=float, default=120.0)
    ap.add_argument("--fail-nm", dest="fail_nm", type=float, default=5.8)
    ap.add_argument("--baselines", default="experiments/res_baselines.npy")
    ap.add_argument("--archived", default="experiments/res_b1_oracle.npy")
    ap.add_argument("--out", default="experiments/res_b1_oracle_60.npy")
    a = ap.parse_args()

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dtmm import LAM_NM, reflectance, n_graphene, _XYZ2RGB
    from scene import _CMF, FAMILY, TOX0, SIG_TOX, SIG_LOGG, SIGMA
    from baselines import _DGRID

    def render(d, t_ox, I):
        """RGB of a region, exactly as scene.sample_scene renders it (gains aside)."""
        R = jax.vmap(lambda l: reflectance(d, t_ox, l, n_graphene))(LAM_NM)
        nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
        return np.array(_XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm))

    # --- replay the scene sampler, capturing the nuisances it does not return ---
    rng = np.random.default_rng(a.seed)
    scenes = []
    for _ in range(a.n):
        d = float(np.exp(rng.uniform(np.log(a.dmin), np.log(a.dmax))))
        t_ox = float(rng.normal(TOX0, SIG_TOX))
        g = np.exp(rng.normal(0, SIG_LOGG, 3))
        idx = int(rng.integers(len(FAMILY)))
        I = jnp.asarray(FAMILY[idx])
        y_f = g * render(d, t_ox, I) + rng.normal(0, SIGMA / np.sqrt(100.0), 3)
        y_b = g * render(0.0, t_ox, I) + rng.normal(0, SIGMA / np.sqrt(300.0), 3)
        scenes.append(dict(d=d, t_ox=t_ox, I=I, idx=idx, y_f=y_f, y_b=y_b))

    d_true = np.array([s["d"] for s in scenes])

    # --- verification: the replayed scenes must be the archived ones ---
    if os.path.isfile(a.baselines):
        B = np.load(a.baselines)
        ref = B[:, 0]
        if len(ref) != a.n or not np.allclose(np.sort(ref), np.sort(d_true), rtol=1e-9):
            print("MISMATCH: replayed scenes are not the archived baseline scenes.")
            print(f"  replayed d range {d_true.min():.4f}-{d_true.max():.4f}, n={len(d_true)}")
            print(f"  archived d range {ref.min():.4f}-{ref.max():.4f}, n={len(ref)}")
            print("  Do not use this output. Find the seed and n that reproduce the archive.")
            sys.exit(1)
        print(f"scene set verified against {a.baselines}: {a.n} scenes, seed {a.seed}")
    else:
        print(f"WARNING: {a.baselines} not found; scene set NOT verified")

    # --- the oracle lookup, per scene, at true oxide and true illuminant ---
    est = np.empty(a.n)
    for k, s in enumerate(scenes):
        base = render(0.0, s["t_ox"], s["I"])
        tab = np.stack([render(float(dg), s["t_ox"], s["I"]) / base for dg in _DGRID])
        obs = np.asarray(s["y_f"]) / np.asarray(s["y_b"])
        est[k] = float(_DGRID[int(np.argmin(np.linalg.norm(tab - obs, axis=1)))])
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{a.n}", flush=True)

    err = np.abs(est - d_true)
    thin = d_true < 25.0
    print(f"\nB1-ORACLE ON ALL {a.n} SCENES (seed {a.seed})")
    print(f"  median |error|   {np.median(err):.4f} nm")
    print(f"  mean   |error|   {err.mean():.4f} nm")
    print(f"  failures >{a.fail_nm:g} nm  {int((err > a.fail_nm).sum())}/{a.n}")
    print(f"  sub-25 nm median {np.median(err[thin]):.4f} nm  (n={int(thin.sum())})")

    # --- what the archived 45-row subset was, and how the two differ ---
    if os.path.isfile(a.archived):
        A = np.load(a.archived)
        keep = np.isin(np.round(d_true, 9), np.round(A[:, 0], 9))
        print(f"\nARCHIVED SUBSET: {A.shape[0]} of {a.n} scenes")
        print(f"  archived median {np.median(np.abs(A[:,1]-A[:,0])):.4f} nm")
        print(f"  same scenes, regenerated: {np.median(err[keep]):.4f} nm")
        print(f"  the {int((~keep).sum())} excluded scenes: {np.median(err[~keep]):.4f} nm")
        print(f"  excluded d range {d_true[~keep].min():.3f}-{d_true[~keep].max():.3f} nm")
        agree = np.allclose(np.sort(A[:, 1]), np.sort(est[keep]), rtol=1e-6, atol=1e-6)
        print(f"  regeneration reproduces the archived estimates: {agree}")
        if not agree:
            print("  -> the archived row was produced by a DIFFERENT procedure, not just a subset")

    np.save(a.out, np.stack([d_true, est], axis=1))
    print(f"\nwrote {a.out}  (columns: true_d, oracle_estimate)")
    print("\nCommit this script with the .npy so the row is reproducible.")


if __name__ == "__main__":
    main()
