#!/usr/bin/env python3
"""
gate_synthetic.py - is the oxide gate a property of the method or of graphene?

WHY
  On real micrographs, rejecting flakes whose fitted oxide departs from their
  acquisition's median by more than 10 nm caught 194 of 194 ridge failures across
  19 acquisitions and left 408 estimates with none wrong. That rests entirely on
  graphene, and on labels that are themselves optically derived.

  The obvious referee question is whether it generalizes. MaskTerial's other
  materials cannot answer it: WSe2 and MoSe2 are refused by the colour-fit
  criterion, and hBN's class labels are the thing under dispute, so "failure" is
  not well defined on any of them.

  Synthetic scenes can, and better: ground truth is exact, so a failure is a
  failure rather than a disagreement with an annotation.

WHAT IT DOES
  The paper's own sampler draws a fresh oxide for every scene, so its benchmark
  has no shared wafer and the gate is not testable on it. This samples an
  ACQUISITION instead: one fixed wafer oxide, with thickness, per-channel gains
  and illuminant SPD varying scene to scene exactly as they do in a real run.
  Then it applies the same gate and scores it against exact truth.

  Repeat per material. If recall stays near 100% with a clean kept set across
  materials with different layer heights and dispersions, the gate follows from
  the ridge geometry rather than from anything particular to graphene.

USAGE
  python gate_synthetic.py --material graphene --n 100 --oxide 285 --seed 301
  python gate_synthetic.py --material all --n 80 --oxide 285 --out gate_synth/

NOTE
  Normal incidence by default, matching the paper's synthetic benchmark. Pass
  --na to use the NA-averaged forward model instead.
"""
import argparse, csv, math, os, statistics as st, sys, time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "core"), os.path.join(os.path.dirname(_HERE), "core")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def med_mad(v):
    v = [x for x in v if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = st.median(v)
    return m, 1.4826 * st.median([abs(x - m) for x in v])


def run_material(mat, n_scenes, tox_true, seed, na, fail_nm, prior_sd, nk_error=0.0):
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dtmm import (LAM_NM, reflectance, reflectance_NA, _XYZ2RGB,
                      n_graphene, n_mos2, n_hbn, n_mose2, n_ws2, n_wse2,
                      GRAPHENE_LAYER_NM, MOS2_LAYER_NM, HBN_LAYER_NM,
                      MOSE2_LAYER_NM, WS2_LAYER_NM, WSE2_LAYER_NM)
    from scene import _CMF, FAMILY, SIGMA, SIG_LOGG
    import estimator_v2 as E

    MAT = {"graphene": (n_graphene, GRAPHENE_LAYER_NM), "mos2": (n_mos2, MOS2_LAYER_NM),
           "hbn": (n_hbn, HBN_LAYER_NM), "mose2": (n_mose2, MOSE2_LAYER_NM),
           "ws2": (n_ws2, WS2_LAYER_NM), "wse2": (n_wse2, WSE2_LAYER_NM)}
    n_fn, layer = MAT[mat]

    # Model misspecification. The scene is RENDERED with dispersion scaled by
    # (1+eps); the estimator keeps the nominal table, exactly as in the paper's
    # dispersion-sensitivity study. This is the only way the synthetic setting can
    # produce the kind of failure real images produce: with a correct forward
    # model the estimator's only failure mode is a genuine two-thickness
    # ambiguity, which is multimodal and leaves the nuisances alone.
    if nk_error:
        _base = n_fn
        n_fn_render = lambda lam: _base(lam) * (1.0 + nk_error)
    else:
        n_fn_render = n_fn

    if na and na > 0.05:
        refl = lambda d, t, l: reflectance_NA(d, t, l, n_fn_render, NA=na, n_nodes=6)
        E.reflectance = lambda d, t, l, f: reflectance_NA(d, t, l, f, NA=na, n_nodes=6)
    else:
        refl = lambda d, t, l: reflectance(d, t, l, n_fn_render)

    # The prior is the nominal wafer, as it would be in a real run.
    E.TOX0 = float(tox_true)
    E.SIG_TOX = float(prior_sd)
    jax.clear_caches()

    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for i in range(n_scenes):
        # ONE wafer. Thickness, gains and illuminant vary, as within an acquisition.
        d = float(np.exp(rng.uniform(np.log(layer), np.log(120.0))))
        g = np.exp(rng.normal(0, SIG_LOGG, 3))
        I = jnp.asarray(FAMILY[rng.integers(len(FAMILY))])
        R_f = jax.vmap(lambda l: refl(d, tox_true, l))(LAM_NM)
        R_b = jax.vmap(lambda l: refl(0.0, tox_true, l))(LAM_NM)
        nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
        y_f = g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R_f * _CMF, LAM_NM, axis=-1) / nrm))
        y_b = g * np.array(_XYZ2RGB @ (jnp.trapezoid(I * R_b * _CMF, LAM_NM, axis=-1) / nrm))
        y_f = y_f + rng.normal(0, SIGMA / np.sqrt(100.0), 3)
        y_b = y_b + rng.normal(0, SIGMA / np.sqrt(300.0), 3)
        try:
            r = E.estimate_v2(jnp.asarray(y_f), jnp.asarray(y_b), 100.0, 300.0,
                              n_fn=n_fn, k_basis=5)
        except Exception as e:
            print(f"  scene {i}: {e}", flush=True)
            continue
        rows.append(dict(material=mat, d_true=round(d, 5), d_hat=round(float(r["d"]), 5),
                         err=round(float(r["d"]) - d, 5),
                         oxide_hat=round(float(r["theta"][1]), 3),
                         oxide_true=tox_true, sd_nm=round(float(r["sd"]), 5),
                         n_modes=len(r["modes"])))
        if (i + 1) % 20 == 0:
            print(f"  {mat}: {i+1}/{n_scenes}  {(time.time()-t0)/(i+1):.1f}s/scene", flush=True)
    return rows


def score(rows, gate_nm, fail_nm):
    ref, _ = med_mad([r["oxide_hat"] for r in rows])
    F = lambda r: abs(r["err"]) >= fail_nm
    keep = [r for r in rows if abs(r["oxide_hat"] - ref) <= gate_nm]
    drop = [r for r in rows if abs(r["oxide_hat"] - ref) > gate_nm]
    nf = sum(1 for r in rows if F(r))
    kf = sum(1 for r in keep if F(r))
    df = sum(1 for r in drop if F(r))
    return dict(n=len(rows), ref=ref, fails=nf, kept=len(keep), keptfail=kf,
                dropped=len(drop), dropfail=df,
                keptrate=100 * kf / len(keep) if keep else float("nan"),
                prec=100 * df / len(drop) if drop else float("nan"),
                recall=100 * df / nf if nf else float("nan"),
                amb=sum(1 for r in rows if r["n_modes"] > 1),
                ambcatch=sum(1 for r in rows if r["n_modes"] > 1 and F(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="graphene",
                    help="graphene | mos2 | hbn | mose2 | ws2 | wse2 | all")
    ap.add_argument("--n", type=int, default=80, help="scenes per material")
    ap.add_argument("--oxide", type=float, default=285.0, help="the wafer, shared by all scenes")
    ap.add_argument("--prior-sd", dest="prior_sd", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--na", type=float, default=0.0)
    ap.add_argument("--gate-nm", dest="gate_nm", type=float, default=10.0)
    ap.add_argument("--fail-nm", dest="fail_nm", type=float, default=5.8,
                    help="the paper's failure threshold, matched to published SOTA error")
    ap.add_argument("--nk-error", dest="nk_error", type=float, default=0.0,
                    help="render dispersion scaled by (1+eps), estimate at nominal; "
                         "e.g. 0.05 for 5%%. 0 = well-specified control")
    ap.add_argument("--out", default="gate_synth")
    a = ap.parse_args()

    mats = (["graphene", "mos2", "hbn", "wse2"] if a.material == "all" else [a.material])
    os.makedirs(a.out, exist_ok=True)
    allrows, results = [], {}
    for k, m in enumerate(mats):
        print(f"[{m}] one wafer at {a.oxide:g} nm, {a.n} scenes, "
              f"dispersion error {100*a.nk_error:g}%", flush=True)
        rows = run_material(m, a.n, a.oxide, a.seed + 17 * k, a.na, a.fail_nm,
                            a.prior_sd, a.nk_error)
        if not rows:
            continue
        allrows += rows
        results[m] = score(rows, a.gate_nm, a.fail_nm)

    L = []
    A = L.append
    A("SYNTHETIC GATE TEST - exact ground truth, one wafer per material")
    A(f"wafer {a.oxide:g} nm, prior sd {a.prior_sd:g}, "
      f"{'NA-averaged NA=%g' % a.na if a.na else 'normal incidence'}, "
      f"gate {a.gate_nm:g} nm, failure > {a.fail_nm:g} nm, "
      f"dispersion error {100*a.nk_error:g}%")
    A("")
    A(f"  {'material':<10}{'n':>4}{'fails':>7}{'ref ox':>9}{'kept':>6}{'kept fail':>11}"
      f"{'rejected':>10}{'precision':>11}{'recall':>9}{'modeflag':>10}")
    for m, s in results.items():
        A(f"  {m:<10}{s['n']:>4}{s['fails']:>7}{s['ref']:>9.1f}{s['kept']:>6}"
          f"{s['keptfail']:>11}{s['dropped']:>10}{s['prec']:>10.1f}%{s['recall']:>8.1f}%"
          f"{s['ambcatch']:>5}/{s['amb']:<4}")
    if len(results) > 1:
        pooled = score(allrows, a.gate_nm, a.fail_nm)
        A("")
        A(f"  POOLED n={pooled['n']}  failures {pooled['fails']}  "
          f"kept {pooled['kept']} of which {pooled['keptfail']} fail  "
          f"recall {pooled['recall']:.1f}%  precision {pooled['prec']:.1f}%")
        A(f"  mode-count flag over the same scenes: fires {pooled['amb']}, "
          f"catches {pooled['ambcatch']} of {pooled['fails']}")
    A("")
    A("READING")
    A("  With a CORRECT forward model (--nk-error 0) the estimator's only failure")
    A("  mode is a genuine two-thickness ambiguity: multimodal, and the nuisances")
    A("  stay put. The mode-count flag catches those and the oxide gate has nothing")
    A("  to see. On real micrographs the reverse holds - failures are unimodal and")
    A("  the oxide is displaced - which says the real failures come from model")
    A("  misspecification rather than from ambiguity. Raising --nk-error should")
    A("  therefore move the synthetic signature toward the real one: mode-flag")
    A("  recall falling, oxide-gate recall rising. If it does, the detector's")
    A("  mechanism is established. If it does not, the real-data result stands as")
    A("  an empirical regularity with its mechanism open, and should be reported")
    A("  that way.")

    s = "\n".join(L)
    print("\n" + s)
    with open(os.path.join(a.out, "per_scene.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader(); w.writerows(allrows)
    open(os.path.join(a.out, "summary.txt"), "w").write(s + "\n")
    print(f"\nwrote {a.out}/per_scene.csv, summary.txt")


if __name__ == "__main__":
    main()
