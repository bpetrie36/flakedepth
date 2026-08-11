#!/usr/bin/env python3
"""
multiregion_eval.py - the mode-lattice protocol on REAL micrographs.

WHY THIS EXISTS
  Section 5.8 reports that multi-region inference cuts hard-regime failures from
  41.7% to 5.6%. That is measured on synthetic scenes. On real MaskTerial images
  the single-frame estimator puts 10-40% of flakes per acquisition into a
  spurious basin near 35-41 nm oxide, almost all of them unimodal, none caught by
  the ambiguity flag. The protocol has never been run against that.

  Many MaskTerial images contain several annotated flakes of different classes in
  one field, imaged under one illuminant, one exposure and one wafer. That is
  exactly the structure the mode lattice needs: a nuisance vector that rescues the
  wrong interference order for one region must also survive the other region's
  data.

WHAT IT MEASURES
  For every within-image pair of regions of DIFFERENT annotated class, it runs
  estimate_v2_multi, which internally also produces the two single-region
  estimates. So every comparison is PAIRED on identical flakes -- the standard
  §5.11 sets for design decisions -- and the script reports exact McNemar on the
  failure outcomes rather than two independent rates.

  Failure is |estimate - label x layer height| >= --fail-nm. That uses labels, and
  is reported as an outcome, never as a filter on anything else.

USAGE
  python multiregion_eval.py --dataset .\\by_acquisition\\<ACQ> --material graphene \\
         --na 0.45 --oxide 93.5 --limit 40 --out mr_<ACQ>

  python multiregion_eval.py --dataset ... --list-only     # pairs found, no fitting

TRANSFER FUNCTION
  Defaults to linear, like the rest of the repo. FLAKEDEPTH_GAMMA=srgb overrides.
"""
import argparse
import csv
import glob
import math
import os
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "core"), os.path.join(os.path.dirname(_HERE), "core")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_GAMMA = os.environ.get("FLAKEDEPTH_GAMMA", "linear")

BG_ERODE = 4
CORE_ERODE = 3
BG_SUBSAMPLE = 7
MIN_BG_PX = 2000
MIN_CORE_PX = 60


def decode(x):
    if _GAMMA == "linear":
        return x
    if _GAMMA == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x ** float(_GAMMA)


def erode(mask, iters):
    from scipy import ndimage
    return ndimage.binary_erosion(mask, iterations=iters, border_value=0)


def components(mask, min_px):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    return [lab == k for k in range(1, n + 1) if (lab == k).sum() >= min_px]


def harvest_pairs(ds, split, min_px, limit, max_pairs_per_image):
    """Yield (stem, (cls_a, y_a, n_a), (cls_b, y_b, n_b), y_bg, n_bg) for
    within-image pairs of DIFFERENT annotated class."""
    from PIL import Image
    idir = os.path.join(ds, f"{split}_images")
    mdir = os.path.join(ds, f"{split}_semantic_masks")
    if not os.path.isdir(idir):
        sys.exit(f"missing {idir}")
    made = 0
    for ip in sorted(glob.glob(os.path.join(idir, "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = glob.glob(os.path.join(mdir, stem + ".*"))
        if not mp:
            continue
        img = decode(np.asarray(Image.open(ip).convert("RGB")).astype(np.float64) / 255.0)
        msk = np.asarray(Image.open(mp[0]))
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        bg = erode(msk == 0, BG_ERODE)
        if bg.sum() < MIN_BG_PX:
            continue
        y_bg = np.median(img[bg][::BG_SUBSAMPLE], axis=0)
        if np.min(y_bg) < 1e-4:
            continue

        regions = []
        for cls in sorted(set(np.unique(msk)) - {0}):
            for comp in components(msk == cls, min_px):
                core = erode(comp, CORE_ERODE)
                if core.sum() < MIN_CORE_PX:
                    continue
                regions.append((int(cls), np.median(img[core], axis=0), int(core.sum())))
        if len(regions) < 2:
            continue

        # Largest region per class, then pair distinct classes. Pairing the two
        # biggest of each class keeps the observation noise low; pairing DIFFERENT
        # classes is what makes the shared nuisance set discriminating.
        best = {}
        for cls, y, n in regions:
            if cls not in best or n > best[cls][2]:
                best[cls] = (cls, y, n)
        keys = sorted(best)
        made_here = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                yield stem, best[keys[i]], best[keys[j]], y_bg, int(bg.sum())
                made += 1
                made_here += 1
                if limit and made >= limit:
                    return
                if made_here >= max_pairs_per_image:
                    break
            if made_here >= max_pairs_per_image:
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--material", default="graphene")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--oxide", type=float, default=90.0)
    p.add_argument("--oxide-sd", dest="oxide_sd", type=float, default=10.0)
    p.add_argument("--na", type=float, default=0.45)
    p.add_argument("--min-px", dest="min_px", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="max pairs; 0 = all")
    p.add_argument("--max-pairs-per-image", dest="mppi", type=int, default=2)
    p.add_argument("--fail-nm", dest="fail_nm", type=float, default=1.0)
    p.add_argument("--out", default="results_multiregion")
    p.add_argument("--list-only", action="store_true",
                   help="report the pairs found and exit, without fitting")
    a = p.parse_args()

    print(f"[multiregion_eval] transfer function: {_GAMMA}", flush=True)

    if a.list_only:
        n, by_combo = 0, {}
        for stem, ra, rb, ybg, nbg in harvest_pairs(a.dataset, a.split, a.min_px,
                                                    a.limit, a.mppi):
            n += 1
            by_combo[(ra[0], rb[0])] = by_combo.get((ra[0], rb[0]), 0) + 1
        print(f"pairs available: {n}")
        for k in sorted(by_combo):
            print(f"  classes {k[0]}+{k[1]}: {by_combo[k]}")
        return

    import jax
    jax.config.update("jax_enable_x64", True)
    from dtmm import (n_graphene, n_mos2, n_hbn, n_mose2, n_ws2, n_wse2, reflectance_NA,
                      GRAPHENE_LAYER_NM, MOS2_LAYER_NM, HBN_LAYER_NM,
                      MOSE2_LAYER_NM, WS2_LAYER_NM, WSE2_LAYER_NM)
    import estimator_v2 as E
    import multiregion_v2 as MR

    MATERIALS = {"graphene": (n_graphene, GRAPHENE_LAYER_NM), "hbn": (n_hbn, HBN_LAYER_NM),
                 "mos2": (n_mos2, MOS2_LAYER_NM), "mose2": (n_mose2, MOSE2_LAYER_NM),
                 "ws2": (n_ws2, WS2_LAYER_NM), "wse2": (n_wse2, WSE2_LAYER_NM)}
    if a.material.lower() not in MATERIALS:
        sys.exit(f"unknown material {a.material}")
    n_fn, layer_nm = MATERIALS[a.material.lower()]

    if a.na and a.na > 0.05:
        E.reflectance = lambda d, t, l, f: reflectance_NA(d, t, l, f, NA=a.na, n_nodes=6)
        mode = f"NA-averaged (NA={a.na})"
    else:
        mode = "normal incidence"
    E.TOX0 = float(a.oxide)
    E.SIG_TOX = float(a.oxide_sd)
    jax.clear_caches()

    # Capture the joint solution so the fitted oxide can be reported. The public
    # function returns only thicknesses; this wraps the batch polish rather than
    # modifying multiregion_v2.py.
    cap = {}
    _orig = MR._joint_polish_batch

    def _wrapped(*args, **kw):
        xs, Ls = _orig(*args, **kw)
        cap["xs"], cap["Ls"] = np.array(xs), np.array(Ls)
        return xs, Ls

    MR._joint_polish_batch = _wrapped

    import jax.numpy as jnp
    rows, t0 = [], time.time()
    for stem, (ca, ya, na_), (cb, yb, nb_), ybg, nbg in harvest_pairs(
            a.dataset, a.split, a.min_px, a.limit, a.mppi):
        try:
            r = MR.estimate_v2_multi(jnp.asarray(ya), jnp.asarray(yb), jnp.asarray(ybg),
                                     float(na_), float(nb_), float(nbg),
                                     n_fn=n_fn, k_basis=5)
        except Exception as e:
            print(f"  {stem} {ca}+{cb}: failed - {e}", flush=True)
            continue
        ox = float("nan")
        if "xs" in cap and len(cap["Ls"]):
            ox = float(cap["xs"][int(np.argmin(cap["Ls"]))][2])
        for cls, d_joint, d_single, npx in ((ca, r["d1"], r["singles"][0], na_),
                                            (cb, r["d2"], r["singles"][1], nb_)):
            d_true = cls * layer_nm
            rows.append(dict(image=stem, pair=f"{ca}+{cb}", label_layers=cls,
                             d_label_nm=round(d_true, 4),
                             d_joint_nm=round(float(d_joint), 4),
                             d_single_nm=round(float(d_single), 4),
                             res_joint_nm=round(float(d_joint) - d_true, 4),
                             res_single_nm=round(float(d_single) - d_true, 4),
                             oxide_joint_nm=round(ox, 2), n_px=npx,
                             posterior=round(float(r["L"]), 2), n_alts=len(r["alts"])))
        if len(rows) % 10 == 0:
            el = time.time() - t0
            print(f"  {len(rows)//2} pairs  {el/max(len(rows)//2,1):.1f}s/pair  "
                  f"elapsed {el/60:.1f}min", flush=True)

    if not rows:
        sys.exit("no pairs evaluated - try --list-only to see what is available")

    FAIL = lambda v: abs(v) >= a.fail_nm
    fj = [FAIL(r["res_joint_nm"]) for r in rows]
    fs = [FAIL(r["res_single_nm"]) for r in rows]
    b = sum(1 for x, y in zip(fs, fj) if x and not y)   # single failed, joint fixed
    c = sum(1 for x, y in zip(fs, fj) if y and not x)   # joint broke one
    n = b + c
    pval = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i)
                                  for i in range(0, min(b, c) + 1)) / 2 ** n)

    def med_mad(v):
        v = [x for x in v if not math.isnan(x)]
        if not v:
            return float("nan"), float("nan")
        m = st.median(v)
        return m, 1.4826 * st.median([abs(x - m) for x in v])

    ox_m, ox_d = med_mad([r["oxide_joint_nm"] for r in rows])
    L = []
    A = L.append
    A("MULTI-REGION (MODE-LATTICE) EVALUATION ON REAL MICROGRAPHS")
    A(f"dataset {a.dataset}  split {a.split}  material {a.material}")
    A(f"forward model: {mode}   oxide prior {a.oxide} +/- {a.oxide_sd} nm   "
      f"transfer function {_GAMMA}")
    A(f"pairs: {len(rows)//2}   region estimates: {len(rows)}")
    A("")
    A("PAIRED COMPARISON, identical flakes, single-region vs joint")
    A(f"  failures single-region  {sum(fs):>4}/{len(rows)}  ({100*sum(fs)/len(rows):.1f}%)")
    A(f"  failures joint          {sum(fj):>4}/{len(rows)}  ({100*sum(fj)/len(rows):.1f}%)")
    A(f"  fixed by joint          {b}")
    A(f"  broken by joint         {c}")
    A(f"  exact McNemar two-sided p = {pval:.4f}  (discordant n={n})")
    A("")
    A("CONTINUOUS THICKNESS vs LABEL x LAYER HEIGHT")
    for lab, key in (("single", "res_single_nm"), ("joint ", "res_joint_nm")):
        v = [abs(r[key]) for r in rows]
        A(f"  {lab}: median |residual| {st.median(v):.3f} nm   mean {sum(v)/len(v):.3f} nm")
    A("")
    A("JOINT-FIT OXIDE (no labels used)")
    A(f"  median {ox_m:.2f} nm   MAD {ox_d:.2f} nm   (prior {a.oxide})")
    A("")
    A("BY CLASS (exact layer agreement, nearest integer)")
    for k in sorted({r["label_layers"] for r in rows}):
        sub = [r for r in rows if r["label_layers"] == k]
        es = sum(1 for r in sub if round(r["d_single_nm"] / layer_nm) == k)
        ej = sum(1 for r in sub if round(r["d_joint_nm"] / layer_nm) == k)
        A(f"  {k}L (n={len(sub):3d}): single {100*es/len(sub):5.1f}%   "
          f"joint {100*ej/len(sub):5.1f}%")
    A("")
    A("The failure columns use labels and are reported as outcomes. The joint-fit")
    A("oxide uses none. The McNemar test is paired on identical flakes, which is")
    A("the standard this project applies to its own design decisions.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "per_region.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = "\n".join(L)
    open(os.path.join(a.out, "summary.txt"), "w").write(s + "\n")
    print("\n" + s)
    print(f"\nwrote {a.out}/per_region.csv, summary.txt")


if __name__ == "__main__":
    main()
