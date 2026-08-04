#!/usr/bin/env python3
"""
flakedepth.py - thickness of 2D-material flakes from a color micrograph.

ONE COMMAND. Point it at your own images and masks. It determines the acquisition
parameters it can determine, tells you the ones it cannot, and refuses data it
cannot interpret rather than returning a confident wrong number.

    python flakedepth.py --images IMG_DIR --masks MASK_DIR --material graphene \
                         --na 0.45 --oxide 90

INPUTS
  --images DIR    micrographs (png/tif/jpg). Any filenames.
  --masks  DIR    per-pixel masks, SAME filename stem as the image.
                  0 = bare substrate, 1..N = flake regions.
                  If mask values are layer counts, pass --labels-are-layers to score.
  --material      graphene | hbn | mos2 | mose2 | ws2 | wse2
  --na            objective numerical aperture (engraved on the barrel). REQUIRED:
                  it cannot be recovered from the image and a wrong value biases
                  thickness badly above NA 0.55.
  --oxide         wafer SiO2 thickness in nm from the wafer spec. If you don't know
                  it, use --scan-oxide (needs no labels: it minimizes color-fit
                  residual, not label agreement).

WHAT IT WORKS OUT FOR YOU
  * transfer function (sRGB vs linear) -- tested automatically, both scored
  * per-image channel gains (white balance / exposure) -- solved exactly from the
    bare substrate in each image
  * oxide thickness, if you ask for --scan-oxide

WHAT IT REFUSES
  If no oxide value can explain the observed flake/substrate color ratios, the run
  is flagged UNUSABLE. That happens when images carry processing beyond white balance
  (contrast enhancement, tone curves) or when the material is too low-contrast. This
  is a feature: the check needs no ground truth.
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
from scene import _CMF, FAMILY

VERSION = "1.2"
MATERIALS = {"graphene": (n_graphene, GRAPHENE_LAYER_NM), "hbn": (n_hbn, HBN_LAYER_NM),
             "mos2": (n_mos2, MOS2_LAYER_NM), "mose2": (n_mose2, MOSE2_LAYER_NM),
             "ws2": (n_ws2, WS2_LAYER_NM), "wse2": (n_wse2, WSE2_LAYER_NM)}
UNUSABLE = 0.15          # color-fit above this: images not interpretable
GOOD     = 0.08          # color-fit below this: clean

def decode(x, mode):
    if mode == "linear":
        return x
    if mode == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x ** float(mode)

def erode(m, it=3):
    from scipy import ndimage
    return ndimage.binary_erosion(m, iterations=it, border_value=0)

def harvest(img_dir, mask_dir, min_px, limit, gamma, erode_it=3, bg_erode=5):
    """-> list of (class, gain-invariant flake/substrate ratio, stem, n_px)"""
    from PIL import Image
    from scipy import ndimage
    out, exts = [], ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp")
    files = sorted({f for e in exts for f in glob.glob(_os.path.join(img_dir, e))})
    if not files:
        _sys.exit(f"no images found in {img_dir}")
    for ip in files:
        stem = _os.path.splitext(_os.path.basename(ip))[0]
        mp = [f for f in glob.glob(_os.path.join(mask_dir, stem + ".*"))]
        if not mp:
            continue
        img = decode(np.asarray(Image.open(ip).convert("RGB")).astype(np.float64) / 255.0, gamma)
        msk = np.asarray(Image.open(mp[0]))
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        bg = erode(msk == 0, bg_erode)
        if bg.sum() < 5000:
            continue
        y_b = np.median(img[bg][::11], axis=0)
        if np.min(y_b) < 1e-4:
            continue
        for cls in sorted(set(np.unique(msk)) - {0}):
            lab, n = ndimage.label(msk == cls)
            for k in range(1, n + 1):
                c = lab == k
                if c.sum() < min_px:
                    continue
                core = erode(c, erode_it)
                if core.sum() < 80:
                    continue
                out.append((int(cls), np.median(img[core], axis=0) / y_b, stem, int(core.sum())))
                if limit and len(out) >= limit:
                    return out
    return out

def ratio_curve(tox, n_fn, na, d_grid):
    I = jnp.asarray(FAMILY[3]); nrm = jnp.trapezoid(I * _CMF[1], LAM_NM)
    def col(d):
        R = jax.vmap(lambda l: reflectance_NA(d, tox, l, n_fn, NA=na, n_nodes=6))(LAM_NM)
        return _XYZ2RGB @ (jnp.trapezoid(I * R * _CMF, LAM_NM, axis=-1) / nrm)
    b = col(0.0)
    return np.array([np.array(col(float(d)) / b) for d in d_grid])

def invert(obs, curve, d_grid):
    e = np.linalg.norm(np.log(np.abs(curve) + 1e-9) - np.log(np.abs(obs) + 1e-9), axis=1)
    i = int(np.argmin(e))
    return float(d_grid[i]), float(e[i])

def fit_quality(data, tox, n_fn, na, d_grid):
    curve = ratio_curve(tox, n_fn, na, d_grid)
    r = [invert(o, curve, d_grid) for _, o, _, _ in data]
    return float(np.median([e for _, e in r])), [d for d, _ in r]

def main():
    p = argparse.ArgumentParser(description="Flake thickness from a color micrograph.")
    p.add_argument("--images", required=True); p.add_argument("--masks", required=True)
    p.add_argument("--material", required=True, choices=sorted(MATERIALS))
    p.add_argument("--na", type=float, required=True,
                   help="objective numerical aperture, engraved on the barrel next to the "
                        "magnification (e.g. '20x/0.45'). It is a property of the OBJECTIVE, "
                        "not the camera.")
    p.add_argument("--verify-na", action="store_true",
                   help="also fit NA from the images and warn if it disagrees with --na")
    p.add_argument("--oxide", type=float, help="SiO2 thickness (nm) from the wafer spec")
    p.add_argument("--scan-oxide", action="store_true", help="recover oxide from the images")
    p.add_argument("--oxide-range", default="60,320,10")
    p.add_argument("--gamma", default="auto", help="auto (default) | srgb | linear | number")
    p.add_argument("--labels-are-layers", action="store_true",
                   help="mask values are layer counts; score agreement")
    p.add_argument("--min-px", dest="min_px", type=int, default=300)
    p.add_argument("--erode", type=int, default=3,
                   help="pixels eroded from flake edges before sampling (default 3). "
                        "Increase if masks are loose; sensitivity should be checked.")
    p.add_argument("--bg-erode", dest="bg_erode", type=int, default=5,
                   help="pixels eroded from the substrate region (default 5)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="flakedepth_out")
    a = p.parse_args()
    if not a.oxide and not a.scan_oxide:
        p.error("give --oxide (from your wafer spec) or --scan-oxide")

    print(f"flakedepth {VERSION}  |  {a.material}, NA {a.na}", flush=True)
    n_fn, layer = MATERIALS[a.material]
    d_grid = np.concatenate([np.arange(0.2, 15, 0.05), np.arange(15, 160, 0.5)])
    gammas = ["srgb", "linear"] if a.gamma == "auto" else [a.gamma]
    lo, hi, st = [float(x) for x in a.oxide_range.split(",")]
    oxides = np.arange(lo, hi + 1e-9, st) if a.scan_oxide else [a.oxide]

    best = None
    for g in gammas:
        t0 = time.time()
        data = harvest(a.images, a.masks, a.min_px, a.limit, g, a.erode, a.bg_erode)
        if not data:
            _sys.exit("no flakes found: check --masks (0 = substrate, >0 = flake)")
        print(f"  [{g:6s}] {len(data)} flakes read in {time.time()-t0:.0f}s", flush=True)
        for tox in oxides:
            q, _ = fit_quality(data, float(tox), n_fn, a.na, d_grid)
            if len(oxides) > 1:
                print(f"     oxide {tox:6.1f} nm  color-fit {q:.4f}", flush=True)
            if best is None or q < best[0]:
                best = (q, g, float(tox), data)
    q, gamma, tox, data = best
    print(f"\nSELECTED: transfer function '{gamma}', oxide {tox:.1f} nm, color-fit {q:.4f}")
    if a.gamma == "auto":
        print("  (both sRGB and linear were tested; the lower-residual one was chosen)")

    if a.verify_na:
        NA_CANDIDATES = [0.25, 0.30, 0.40, 0.45, 0.55, 0.65, 0.75, 0.90]
        print("\nverifying NA against the images (objectives ship at discrete apertures):")
        rows_na = []
        for na_c in NA_CANDIDATES:
            qq = min(fit_quality(data, float(t), n_fn, na_c, d_grid)[0] for t in oxides) \
                 if len(oxides) > 1 else fit_quality(data, tox, n_fn, na_c, d_grid)[0]
            rows_na.append((qq, na_c))
            print(f"     NA {na_c:.2f}  color-fit {qq:.4f}" + ("   <- you specified" if abs(na_c-a.na)<1e-6 else ""))
        rows_na.sort()
        q_best, na_best = rows_na[0]
        q_user = next(qq for qq, nc in rows_na if abs(nc - a.na) < 1e-6) if any(
            abs(nc - a.na) < 1e-6 for _, nc in rows_na) else None
        print(f"  best-fitting NA: {na_best:.2f} (color-fit {q_best:.4f})")
        if q_user is not None and q_user > q_best * 1.25:
            print(f"  WARNING: your --na {a.na:.2f} fits worse ({q_user:.4f}) than NA {na_best:.2f} "
                  f"({q_best:.4f}).")
            print( "           A wrong NA silently rescales every thickness and is NOT caught by")
            print( "           the color-fit verdict. Check the number engraved on your objective.")
        elif q_user is not None:
            print(f"  consistent with your --na {a.na:.2f}.")
        print( "  Note: NA and oxide are partially degenerate; treat this as a cross-check,")
        print( "  not a replacement for reading the objective.")

    verdict = ("USABLE - clean" if q < GOOD else
               "USABLE - with caution" if q < UNUSABLE else "UNUSABLE")
    print(f"  VERDICT: {verdict}")
    if q >= UNUSABLE:
        print("  No oxide value explains the observed color ratios. Your images likely carry")
        print("  processing beyond white balance (contrast enhancement, tone curve), or the")
        print("  material is too low-contrast at these thicknesses. Thicknesses below are NOT")
        print("  trustworthy. Re-acquire with fixed white balance and no auto-contrast.")

    curve = ratio_curve(tox, n_fn, a.na, d_grid)
    rows = []
    for cls, obs, stem, npx in data:
        d, e = invert(obs, curve, d_grid)
        rows.append(dict(image=stem, mask_class=cls, n_px=npx, thickness_nm=round(d, 4),
                         layers=round(d / layer, 3), color_resid=round(e, 5),
                         flag=("ok" if e < UNUSABLE else "poor_fit")))
    _os.makedirs(a.out, exist_ok=True)
    with open(_os.path.join(a.out, "thickness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    L = [f"flakedepth {VERSION}", f"material {a.material}   NA {a.na}",
         f"transfer function: {gamma}   oxide: {tox:.1f} nm"
         + ("  (recovered from images)" if a.scan_oxide else "  (from wafer spec)"),
         f"color-fit residual: {q:.4f}   VERDICT: {verdict}", f"flakes: {len(rows)}", ""]
    d = np.array([r["thickness_nm"] for r in rows]); lay = d / layer
    L += [f"thickness: median {np.median(d):.3f} nm  range {d.min():.3f}-{d.max():.3f} nm",
          f"layers:    median {np.median(lay):.2f}   range {lay.min():.2f}-{lay.max():.2f}"]
    poor = sum(1 for r in rows if r["flag"] != "ok")
    L.append(f"flakes with poor individual fit: {poor}/{len(rows)}")
    if a.labels_are_layers:
        from scipy import stats
        cl = np.array([r["mask_class"] for r in rows]); rnd = np.round(lay)
        L += ["", "AGREEMENT WITH MASK LABELS (labels treated as layer counts)",
              f"  exact:     {100*np.mean(rnd==cl):.1f}%",
              f"  within 1:  {100*np.mean(np.abs(rnd-cl)<=1):.1f}%"]
        for k in sorted(set(cl)):
            m = cl == k
            L.append(f"    {k}L (n={m.sum():4d}): exact {100*np.mean(rnd[m]==k):5.1f}%  "
                     f"median est {np.median(lay[m]):6.2f}")
        if len(set(cl)) > 1:
            sl, ic, r_, _, se = stats.linregress(cl, d)
            L += ["", f"  recovered layer height: {sl:.4f} +/- {se:.4f} nm/layer "
                      f"(true {layer:.3f}, error {100*(sl-layer)/layer:+.1f}%)",
                  f"  intercept {ic:+.4f} nm   r = {r_:.4f}",
                  "  A recovered layer height matching the true spacing means absolute",
                  "  thickness is correct, not merely the ordering."]
    s = "\n".join(L)
    open(_os.path.join(a.out, "summary.txt"), "w").write(s + "\n")
    print("\n" + s)
    print(f"\nwrote {a.out}/thickness.csv and summary.txt")

if __name__ == "__main__":
    main()
