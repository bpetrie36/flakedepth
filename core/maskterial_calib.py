#!/usr/bin/env python3
"""
maskterial_calib.py - two-stage evaluation for white-balanced micrographs.

WHY THIS EXISTS
  SOME flake datasets are white-balanced on the bare substrate, so the wafer reads
  neutral gray; hBN_Thin is one (substrate [156,159,154], inter-image IQR
  [2.0,2.2,0.2]). Three free channel gains then fit ANY substrate color, the
  bare-substrate term carries no information, and the full joint estimate becomes
  under-determined: thickness, oxide and 5 illuminant coefficients float against
  6 observed numbers.

  OTHERS are not. GrapheneL's substrate reads [113,107,151] against a forward-model
  prediction of [0.110,0.098,0.177] at 90 nm oxide - both strongly blue. There the
  substrate term does carry information, the full joint estimate is usable, and this
  restricted mode is a robustness trade rather than a necessity. Check the substrate
  color against the model prediction before assuming which case you are in.

  Note on the gain prior: widening sigma_g does help somewhat on real images, where
  the overall exposure is physically arbitrary and fits near 4 against a prior centered
  at 1. On a paired 39-flake test, sigma_g 0.1 -> 1.5 tightened the label-free
  fitted-oxide MAD from 7.13 to 5.22 nm. It does not touch the joint ridge.

  The fix is to restore identifiability by fixing what the image cannot tell us:
    * illuminant is pinned (c = 0, reference SPD)
    * oxide is a single GLOBAL constant per dataset, calibrated once on the TRAIN split
    * per-image gains are then determined EXACTLY by the substrate (3 eqs, 3 unknowns)
    * thickness is the only free per-flake parameter: 3 observations, 1 unknown
  Evaluation happens on the TEST split, which the calibration never saw.

  This is the honest analogue of what learning baselines do (MaskTerial adapts with
  5-10 labeled images per material). Here the adaptation is one physical constant.

USAGE
  # stage 1: calibrate the oxide on the train split (uses labels)
  python maskterial_calib.py --dataset PATH --material graphene --calibrate --limit 80

  # stage 2: evaluate on the test split with that oxide (labels used only for scoring)
  python maskterial_calib.py --dataset PATH --material graphene --oxide VALUE --limit 200
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "core"), _os.path.join(_os.path.dirname(_HERE), "core")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, csv, glob, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from dtmm import (n_graphene, n_mos2, n_hbn, n_mose2, n_ws2, n_wse2, LAM_NM, _XYZ2RGB,
                  GRAPHENE_LAYER_NM, MOS2_LAYER_NM, HBN_LAYER_NM,
                  MOSE2_LAYER_NM, WS2_LAYER_NM, WSE2_LAYER_NM, reflectance_NA)
from scene import _CMF

VERSION = "2026-08-03-gamma"
MATERIALS = {"graphene": (n_graphene, GRAPHENE_LAYER_NM), "hbn": (n_hbn, HBN_LAYER_NM),
             "mos2": (n_mos2, MOS2_LAYER_NM), "mose2": (n_mose2, MOSE2_LAYER_NM),
             "ws2": (n_ws2, WS2_LAYER_NM), "wse2": (n_wse2, WSE2_LAYER_NM)}

def decode(x, mode):
    """Undo the camera transfer function. 'srgb' assumes standard sRGB encoding;
    'linear' assumes the file already holds linear radiance (common for scientific
    cameras writing raw-ish output); a number applies x**gamma."""
    if mode == "linear":
        return x
    if mode == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x ** float(mode)

def erode(m, it=3):
    try:
        from scipy import ndimage
        return ndimage.binary_erosion(m, iterations=it, border_value=0)
    except Exception:
        for _ in range(it):
            m = m & np.roll(m,1,0) & np.roll(m,-1,0) & np.roll(m,1,1) & np.roll(m,-1,1)
        return m

def comps(m, min_px):
    from scipy import ndimage
    lab, n = ndimage.label(m)
    return [lab == k for k in range(1, n+1) if (lab == k).sum() >= min_px]

def harvest(ds, split, min_px, limit, gamma='linear'):
    """Return list of (label_class, ratio_rgb) — the gain-invariant observable."""
    from PIL import Image
    out = []
    idir, mdir = _os.path.join(ds, f"{split}_images"), _os.path.join(ds, f"{split}_semantic_masks")
    if not _os.path.isdir(idir):
        _sys.exit(f"missing {idir}")
    for ip in sorted(glob.glob(_os.path.join(idir, "*"))):
        stem = _os.path.splitext(_os.path.basename(ip))[0]
        mp = glob.glob(_os.path.join(mdir, stem + ".*"))
        if not mp:
            continue
        img = decode(np.asarray(Image.open(ip).convert("RGB")).astype(np.float64) / 255.0, gamma)
        msk = np.asarray(Image.open(mp[0]))
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        bg = erode(msk == 0, 5)
        if bg.sum() < 5000:
            continue
        y_b = np.median(img[bg][::11], axis=0)
        if np.min(y_b) < 1e-4:
            continue
        for cls in sorted(set(np.unique(msk)) - {0}):
            for c in comps(msk == cls, min_px):
                core = erode(c, 3)
                if core.sum() < 80:
                    continue
                out.append((int(cls), np.median(img[core], axis=0) / y_b, stem))
                if limit and len(out) >= limit:
                    return out
    return out

def ratio_curve(tox, n_fn, na, d_grid):
    """Predicted gain-invariant flake/substrate RGB ratio vs thickness."""
    I = jnp.ones_like(LAM_NM)  # pinned flat illuminant; gains absorb the rest
    from scene import FAMILY
    I = jnp.asarray(FAMILY[3])
    nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    def col(d):
        R = jax.vmap(lambda l: reflectance_NA(d, tox, l, n_fn, NA=na, n_nodes=6))(LAM_NM)
        return _XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm)
    b = col(0.0)
    return np.array([np.array(col(float(d)) / b) for d in d_grid])

def invert(obs, curve, d_grid):
    """Nearest point on the predicted ratio curve, in log-color space."""
    e = np.linalg.norm(np.log(np.abs(curve) + 1e-9) - np.log(np.abs(obs) + 1e-9), axis=1)
    i = int(np.argmin(e))
    return float(d_grid[i]), float(e[i])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--material", required=True)
    p.add_argument("--na", type=float, default=0.45)
    p.add_argument("--min-px", dest="min_px", type=int, default=300)
    p.add_argument("--gamma", default="linear",
                   help="linear (default; MaskTerial imagery is linear) | srgb | a number. Decoding linear images as srgb rescales thickness by ~2x.")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--calibrate", action="store_true", help="stage 1: fit oxide on train split")
    p.add_argument("--oxide", type=float, help="stage 2: oxide from calibration")
    p.add_argument("--oxide-range", default="60,320,5")
    p.add_argument("--out", default="results_calib")
    a = p.parse_args()
    print(f"[maskterial_calib {VERSION}]", flush=True)
    n_fn, layer = MATERIALS[a.material.lower()]
    d_grid = np.concatenate([np.arange(0.2, 15, 0.05), np.arange(15, 160, 0.5)])

    split = "train" if a.calibrate else "test"
    t0 = time.time()
    print(f"harvesting {split} split...", flush=True)
    data = harvest(a.dataset, split, a.min_px, a.limit, a.gamma)
    print(f"  {len(data)} flakes in {time.time()-t0:.0f}s", flush=True)
    if not data:
        _sys.exit("no flakes found")

    if a.calibrate:
        lo, hi, st = [float(x) for x in a.oxide_range.split(",")]
        best = None
        print("calibrating oxide (fit = agreement between recovered thickness and label):", flush=True)
        for tox in np.arange(lo, hi + 1e-9, st):
            curve = ratio_curve(float(tox), n_fn, a.na, d_grid)
            errs, ds, ls = [], [], []
            for cls, obs, _ in data:
                d, e = invert(obs, curve, d_grid)
                ds.append(d); ls.append(cls); errs.append(e)
            ds, ls = np.array(ds), np.array(ls)
            # score: how well recovered thickness is proportional to label
            if len(set(ls)) > 1:
                A = np.vstack([ls, np.ones_like(ls)]).T
                sl, ic = np.linalg.lstsq(A, ds, rcond=None)[0]
                resid = np.std(ds - (sl * ls + ic))
                score = resid / max(abs(sl), 1e-6)
            else:
                score = np.std(ds)
            print(f"  tox={tox:6.1f} nm  color-fit={np.mean(errs):.4f}  "
                  f"layer-consistency={score:8.3f}", flush=True)
            # select on COLOUR FIT: how well the predicted ratio curve explains the
            # observed ratios. The layer-consistency column is reported for information
            # but is NOT the criterion: it degenerates to zero when the inversion
            # collapses all flakes onto one thickness.
            cf = float(np.mean(errs))
            if best is None or cf < best[0]:
                best = (cf, tox, score)
        print(f"\nBEST OXIDE: {best[1]:.1f} nm  (color-fit {best[0]:.4f}, layer-consistency {best[2]:.3f})")
        if best[0] > 0.15:
            print("WARNING: color-fit > 0.15 means the physics cannot reproduce the observed")
            print("         ratios at ANY oxide. The images carry processing beyond white")
            print("         balance and are not usable for absolute metrology.")
        print(f"Now run stage 2:\n  --oxide {best[1]:.1f} (omit --calibrate)")
        return

    if a.oxide is None:
        _sys.exit("stage 2 needs --oxide from the calibration run")
    curve = ratio_curve(a.oxide, n_fn, a.na, d_grid)
    rows = []
    for cls, obs, stem in data:
        d, e = invert(obs, curve, d_grid)
        rows.append(dict(image=stem, label=cls, d_hat_nm=round(d, 4),
                         layers_hat=round(d / layer, 3), color_resid=round(e, 5)))
    lab = np.array([r["label"] for r in rows]); hat = np.array([r["layers_hat"] for r in rows])
    rnd = np.round(hat).astype(int)
    _os.makedirs(a.out, exist_ok=True)
    with open(_os.path.join(a.out, "per_flake.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = [f"RESTRICTED-MODE EVALUATION (white-balanced images)",
         f"dataset {a.dataset}  split test  material {a.material}  NA {a.na}  gamma {a.gamma}",
         f"oxide {a.oxide} nm (calibrated on TRAIN split; test split unseen)",
         f"flakes: {len(rows)}", "",
         f"exact layer agreement:  {100*np.mean(rnd==lab):.1f}%",
         f"within 1 layer:         {100*np.mean(np.abs(rnd-lab)<=1):.1f}%"]
    for k in sorted(set(lab)):
        m = lab == k
        L.append(f"  {k}L (n={m.sum():4d}): exact {100*np.mean(rnd[m]==k):5.1f}%   "
                 f"median est {np.median(hat[m]):7.2f} layers")
    if len(set(lab)) > 1:
        A = np.vstack([lab, np.ones_like(lab)]).T
        sl, ic = np.linalg.lstsq(A, np.array([r["d_hat_nm"] for r in rows]), rcond=None)[0]
        r_ = np.corrcoef(lab, [r["d_hat_nm"] for r in rows])[0, 1]
        L += ["", f"thickness vs label: slope {sl:.4f} nm/layer (true layer height {layer:.3f}), "
                  f"intercept {ic:+.3f} nm, r = {r_:.4f}",
              "  slope near the layer height means the method tracks true thickness;",
              "  r near 1 means it orders flakes correctly even if scale is off."]
    L += ["", f"median color residual: {np.median([r['color_resid'] for r in rows]):.4f}",
          "  (large values mean the images carry processing beyond white balance)"]
    s = "\n".join(L)
    open(_os.path.join(a.out, "summary.txt"), "w").write(s + "\n")
    print("\n" + s)

if __name__ == "__main__":
    main()
