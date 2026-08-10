#!/usr/bin/env python3
"""
maskterial_eval.py - run the flake-thickness estimator on the public MaskTerial datasets.

This produces the real-data results for Section 5.10 of the paper. It needs no AFM and no
laboratory access: the MaskTerial datasets are public (Zenodo 15765514) and carry per-pixel
layer-class semantic masks, which supply both the flake regions and the bare substrate.

WHAT THE GROUND TRUTH IS, AND ISN'T
  MaskTerial labels are layer classes assigned by human annotation of optical images. They
  are the field's own standard for real-data validation (phi-Adapt and MaskTerial both report
  against labels of this kind), but they are NOT an independent physical measurement: agreement
  demonstrates that the method reproduces the community's labelling on the community's images,
  not that it recovers true thickness. State this in the paper.

USAGE
  python maskterial_eval.py --dataset /path/to/GrapheneH --material graphene \\
         --oxide 90 --na 0.45 --split test --out results_grapheneH/

  # if you do not know the wafer oxide, scan for it first (uses ~40 flakes, a few minutes):
  python maskterial_eval.py --dataset /path/to/GrapheneH --material graphene \\
         --oxide-scan --limit 40

OUTPUTS
  per_flake.csv   one row per annotated flake: label, estimate, residual, fitted nuisances
  summary.txt     the numbers that go into the paper
  confusion.txt   layer-label confusion matrix (nearest-integer assignment)
  diagnostics.txt conditions under which the numbers should not be trusted
"""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "core"), _os.path.join(_os.path.dirname(_HERE), "core")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, os, sys, csv, glob, json, time as _time
import numpy as np

import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import dtmm
from dtmm import (n_graphene, n_mos2, n_hbn, n_mose2, n_ws2, n_wse2,
                  GRAPHENE_LAYER_NM, MOS2_LAYER_NM, HBN_LAYER_NM,
                  MOSE2_LAYER_NM, WS2_LAYER_NM, WSE2_LAYER_NM, reflectance_NA)
import estimator_v2 as E

# layer heights (nm). WS2/WSe2/MoSe2 use in-plane ordinary-ray dispersion; see paper Sec. 3.1.
VERSION = "2026-08-03-oxidefix"

MATERIALS = {
    "graphene": (n_graphene, GRAPHENE_LAYER_NM),
    "hbn":      (n_hbn,      HBN_LAYER_NM),
    "mos2":     (n_mos2,     MOS2_LAYER_NM),
    "mose2":    (n_mose2,    MOSE2_LAYER_NM),
    "ws2":      (n_ws2,      WS2_LAYER_NM),
    "wse2":     (n_wse2,     WSE2_LAYER_NM),
}

def _robust(a):  # patched by patch_repo.py: robust
    a = np.asarray(a, dtype=float)
    m = float(np.median(a))
    return m, float(1.4826 * np.median(np.abs(a - m)))

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

# --- patched by patch_repo.py: transfer function is selectable and defaults to
# LINEAR, because the MaskTerial images are radiometrically linear. Override
# with the FLAKEDEPTH_GAMMA environment variable (linear | srgb | a number).
_FLAKEDEPTH_GAMMA = os.environ.get("FLAKEDEPTH_GAMMA", "linear")

def _decode(x):
    if _FLAKEDEPTH_GAMMA == "linear":
        return x
    if _FLAKEDEPTH_GAMMA == "srgb":
        return srgb_to_linear(x)
    return x ** float(_FLAKEDEPTH_GAMMA)
# --- end patch

def erode(mask, iters=2):
    try:
        from scipy import ndimage
        return ndimage.binary_erosion(mask, iterations=iters, border_value=0)
    except Exception:
        m = mask.copy()
        for _ in range(iters):
            m = (m & np.roll(m, 1, 0) & np.roll(m, -1, 0)
                   & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        return m

def components(mask, min_px):
    """Connected components. Uses scipy.ndimage when available (orders of magnitude
    faster on large flakes); falls back to a pure-Python flood fill otherwise."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        out = []
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() >= min_px:
                out.append(m)
        return out
    except Exception:
        pass
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not lab[yy, xx]):
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
    out = []
    for k in range(1, cur + 1):
        m = lab == k
        if m.sum() >= min_px:
            out.append(m)
    return out

def load_pair(img_path, mask_path):
    from PIL import Image
    img = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float64)
    img = _decode(img / 255.0)
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return img, mask.astype(np.int32)

def iter_flakes(ds, split, min_px, limit):
    img_dir = os.path.join(ds, f"{split}_images")
    msk_dir = os.path.join(ds, f"{split}_semantic_masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
        sys.exit(f"expected {img_dir} and {msk_dir}; check --dataset points at an unzipped "
                 f"MaskTerial dataset folder")
    n = 0
    for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        cands = glob.glob(os.path.join(msk_dir, stem + ".*"))
        if not cands:
            continue
        img, mask = load_pair(ip, cands[0])
        bg = erode(mask == 0, iters=4)
        if bg.sum() < 2000:
            continue
        y_b = np.median(img[bg][::7], axis=0)
        for cls in sorted(set(np.unique(mask)) - {0}):
            for comp in components(mask == cls, min_px):
                core = erode(comp, iters=3)
                if core.sum() < 60:
                    continue
                y_f = np.median(img[core], axis=0)
                yield stem, int(cls), y_f, int(core.sum()), y_b, int(bg.sum())
                n += 1
                if limit and n >= limit:
                    return

def run(args):
    mat = args.material.lower()
    if mat in MATERIALS:
        n_fn, layer_nm = MATERIALS[mat]
    else:
        sys.exit(f"unknown material '{mat}'")

    if args.na and args.na > 0.05:
        E.reflectance = lambda d, t, l, f: reflectance_NA(d, t, l, f, NA=args.na, n_nodes=6)
        mode = f"NA-averaged (NA={args.na})"
    else:
        mode = "normal incidence"
    # TOX0/SIG_TOX are captured by JAX at trace time; changing them without clearing
    # the cache silently reuses the previous compilation (verified failure mode).
    E.TOX0 = float(args.oxide)
    E.SIG_TOX = float(args.oxide_sd)
    jax.clear_caches()

    rows, warn = [], []
    _T0 = _time.time()
    for stem, cls, y_f, n_f, y_b, n_b in iter_flakes(args.dataset, args.split,
                                                     args.min_px, args.limit):
        try:
            r = E.estimate_v2(jnp.asarray(y_f), jnp.asarray(y_b), float(n_f), float(n_b),
                              n_fn=n_fn, k_basis=5)
        except Exception as e:
            warn.append(f"{stem} class {cls}: estimator failed - {e}")
            continue
        d_true = cls * layer_nm          # class index == layer count for these datasets
        rows.append(dict(image=stem, label_layers=cls, d_label_nm=round(d_true, 4),
                         d_hat_nm=round(r["d"], 4), layers_hat=round(r["d"] / layer_nm, 3),
                         sd_nm=round(r["sd"], 4), residual_nm=round(r["d"] - d_true, 4),
                         oxide_fit_nm=round(float(r["theta"][1]), 2),
                         gain_r=round(float(np.exp(r["theta"][2])), 3),
                         gain_g=round(float(np.exp(r["theta"][3])), 3),
                         gain_b=round(float(np.exp(r["theta"][4])), 3),
                         n_modes=len(r["modes"]), n_flake_px=n_f, posterior=round(r["L"], 2)))
        if len(rows) % 5 == 0:
            el = _time.time() - _T0
            rate = el / len(rows)
            eta = (args.limit - len(rows)) * rate if args.limit else None
            msg = f"  {len(rows)} flakes  {rate:.1f}s/flake  elapsed {el/60:.1f}min"
            if eta and eta > 0:
                msg += f"  eta {eta/60:.1f}min"
            print(msg, flush=True)
    if not rows:
        sys.exit("no flakes evaluated - check --dataset and --min_px")
    return rows, warn, mode, layer_nm

def summarize(rows, mode, layer_nm, args):
    lab = np.array([r["label_layers"] for r in rows])
    hat = np.array([r["layers_hat"] for r in rows])
    res = np.array([r["residual_nm"] for r in rows])
    ox = np.array([r["oxide_fit_nm"] for r in rows])
    rnd = np.round(hat).astype(int)
    L = []
    L.append("MASKTERIAL REAL-DATA EVALUATION")
    L.append(f"dataset {args.dataset}  split {args.split}  material {args.material}")
    L.append(f"forward model: {mode}   assumed oxide {args.oxide} +/- {args.oxide_sd} nm")
    L.append(f"flakes evaluated: {len(rows)}")
    L.append("")
    L.append("LAYER-COUNT AGREEMENT WITH ANNOTATION (the field's standard metric)")
    L.append(f"  exact:        {100*np.mean(rnd == lab):.1f}%")
    L.append(f"  within 1:     {100*np.mean(np.abs(rnd - lab) <= 1):.1f}%")
    for k in sorted(set(lab)):
        m = lab == k
        L.append(f"    {k}L (n={m.sum():4d}): exact {100*np.mean(rnd[m]==k):5.1f}%   "
                 f"median est. {np.median(hat[m]):.2f} layers")
    L.append("")
    L.append("CONTINUOUS THICKNESS vs LABEL x LAYER HEIGHT")
    L.append(f"  median |residual|  {np.median(np.abs(res)):.3f} nm")
    L.append(f"  mean |residual|    {np.abs(res).mean():.3f} nm")
    L.append(f"  median signed      {np.median(res):+.3f} nm")
    L.append("")
    L.append("SELF-CALIBRATION CONSISTENCY (falsifiable; needs no ground truth)")
    L.append(f"  fitted oxide across flakes: {_robust(ox)[0]:.2f} +/- {_robust(ox)[1]:.2f} nm "
             f"(assumed {args.oxide})")
    L.append(f"  -> flakes on one wafer must agree. Scatter >> wafer tolerance means the model")
    L.append(f"     is absorbing an unmodeled effect and the accuracy numbers above are void.")
    g = np.array([[r["gain_r"], r["gain_g"], r["gain_b"]] for r in rows])
    L.append(f"  fitted gains: R {_robust(g[:,0])[0]:.3f} +/- {_robust(g[:,0])[1]:.3f}  "
             f"G {_robust(g[:,1])[0]:.3f} +/- {_robust(g[:,1])[1]:.3f}  B {_robust(g[:,2])[0]:.3f} +/- {_robust(g[:,2])[1]:.3f}")
    amb = sum(1 for r in rows if r["n_modes"] > 1)
    L.append(f"  flagged ambiguous: {amb}/{len(rows)} ({100*amb/len(rows):.1f}%)")
    return "\n".join(L)

def confusion(rows):
    lab = np.array([r["label_layers"] for r in rows])
    rnd = np.round([r["layers_hat"] for r in rows]).astype(int)
    ks = sorted(set(lab) | set(rnd[(rnd > 0) & (rnd < 12)]))
    out = ["rows = annotated layer, cols = estimated layer", "      " + "".join(f"{k:>6}" for k in ks)]
    for a in sorted(set(lab)):
        out.append(f"{a:>5} " + "".join(f"{int(np.sum((lab==a)&(rnd==b))):>6}" for b in ks))
    return "\n".join(out)

def oxide_scan(args):
    """Try candidate wafer oxides; the correct one minimizes fitted-oxide scatter."""
    print("scanning candidate oxide thicknesses (consistency, not accuracy, is the test)\n")
    best = None
    for ox in [float(x) for x in args.oxide_scan_values.split(",")]:
        args.oxide, args.oxide_sd = ox, 40.0     # wide prior: let the data speak
        print(f"[{ox:g} nm] compiling + fitting {args.limit} flakes...", flush=True)
        rows, _, _, _ = run(args)
        f = np.array([r["oxide_fit_nm"] for r in rows])
        print(f"  assumed {ox:6.1f} nm -> fitted {_robust(f)[0]:7.2f} +/- {_robust(f)[1]:5.2f} nm  (n={len(rows)})")
        if best is None or f.std() < best[1]:
            best = (ox, f.std(), f.mean())
    print(f"\nmost self-consistent: assumed {best[0]} nm, fitted {best[2]:.2f} +/- {best[1]:.2f} nm")
    print("Re-run without --oxide-scan using that value and the wafer's true tolerance.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="unzipped MaskTerial dataset folder")
    p.add_argument("--material", required=True)
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--oxide", type=float, default=90.0)
    p.add_argument("--oxide-sd", type=float, default=10.0)
    p.add_argument("--na", type=float, default=0.45, help="20x objective is typically 0.40-0.45")
    p.add_argument("--min-px", dest="min_px", type=int, default=300)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="results")
    p.add_argument("--oxide-scan", action="store_true")
    p.add_argument("--oxide-scan-values", default="90,285,300")
    a = p.parse_args()
    print(f"[maskterial_eval {VERSION}]", flush=True)
    if a.oxide_scan:
        a.limit = a.limit or 40
        oxide_scan(a); return
    os.makedirs(a.out, exist_ok=True)
    rows, warn, mode, layer_nm = run(a)
    with open(os.path.join(a.out, "per_flake.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    s = summarize(rows, mode, layer_nm, a)
    open(os.path.join(a.out, "summary.txt"), "w").write(s + "\n")
    open(os.path.join(a.out, "confusion.txt"), "w").write(confusion(rows) + "\n")
    open(os.path.join(a.out, "diagnostics.txt"), "w").write("\n".join(warn) + "\n")
    print("\n" + s)
    print("\n" + confusion(rows))
    print(f"\nwrote {a.out}/per_flake.csv, summary.txt, confusion.txt, diagnostics.txt")

if __name__ == "__main__":
    main()
