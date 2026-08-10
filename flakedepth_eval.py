#!/usr/bin/env python3
"""
flakedepth_eval.py - real-micrograph evaluation for the flake-thickness method.

This is the script that produces Section 5.10 of the paper (the pending real-data
subsection). Everything it needs beyond the repo is listed in manifest.csv.

USAGE
  python flakedepth_eval.py --manifest manifest.csv --out results/

MANIFEST FORMAT (one row per flake; see make_manifest_template())
  flake_id, image_path, material, na, oxide_nm, oxide_sd_nm,
  flake_x, flake_y, bg_x, bg_y, patch_px,
  afm_height_nm, afm_sd_nm, raman_layers, gamma, notes

  flake_x/flake_y : pixel coordinates of a point INSIDE the flake, in the region
                    the AFM step height was measured on.
  bg_x/bg_y       : pixel coordinates of clean bare substrate, same field of view.
  patch_px        : half-width of the sampling square (16 is a good default).
  na              : objective numerical aperture, e.g. 0.90. REQUIRED.
  gamma           : 'linear' (default; MaskTerial imagery is linear), 'srgb',
                    or a numeric gamma value. Getting this wrong rescales
                    recovered thickness by ~2x with a clean per-image residual.
  afm_height_nm   : AFM step height. Leave blank if unknown; row still runs but
                    contributes only to the internal-consistency checks.

OUTPUTS
  results/per_flake.csv      one row per flake with estimate, uncertainty, residual
  results/summary.txt        the numbers that go in the paper
  results/agreement.png      optical vs AFM scatter with y=x and AFM error bars
  results/diagnostics.txt    warnings that indicate the data cannot support a claim
"""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _p in (_HERE, _os.path.join(_HERE, "core"), _os.path.join(_os.path.dirname(_HERE), "core")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, csv, os, sys, json
import numpy as np

# ------------------------------------------------------------------ imports from the repo
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import dtmm
from dtmm import (n_graphene, n_mos2, n_hbn, GRAPHENE_LAYER_NM, MOS2_LAYER_NM,
                  HBN_LAYER_NM, reflectance_NA)
import estimator_v2 as E

MATERIALS = {
    "graphene": (n_graphene, GRAPHENE_LAYER_NM),
    "mos2":     (n_mos2,     MOS2_LAYER_NM),
    "hbn":      (n_hbn,      HBN_LAYER_NM),
}

# ------------------------------------------------------------------ color handling
def to_linear(rgb01, gamma):
    """Undo the camera's transfer function. rgb01 in [0,1]."""
    if gamma == "linear":
        return rgb01
    if gamma == "srgb":
        return np.where(rgb01 <= 0.04045, rgb01 / 12.92, ((rgb01 + 0.055) / 1.055) ** 2.4)
    return rgb01 ** float(gamma)

def load_image(path):
    from PIL import Image
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"{path}: expected an RGB image, got shape {arr.shape}")
    arr = arr[:, :, :3].astype(np.float64)
    maxval = 65535.0 if arr.max() > 255 else 255.0
    return arr / maxval, ("16-bit" if maxval > 255 else "8-bit")

def sample_patch(img01, x, y, half, gamma):
    """Per-channel MEDIAN of a square patch, converted to linear RGB.
    Median (not mean) rejects dust, scratches and hot pixels."""
    h, w, _ = img01.shape
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    patch = img01[y0:y1, x0:x1, :].reshape(-1, 3)
    if patch.shape[0] < 25:
        raise ValueError("patch too small or coordinates outside image")
    lin = to_linear(patch, gamma)
    return np.median(lin, axis=0), lin.std(axis=0), patch.shape[0]

# ------------------------------------------------------------------ NA-aware physics
def install_na(na, n_nodes=6):
    """Patch the estimator's forward model to the objective actually used.
    MUST be called before the first jitted call. Paper Sec. 5.8: omitting this
    is benign at NA<=0.55 and catastrophic at NA 0.9."""
    if na is None or na <= 0.05:
        return "normal-incidence"
    E.reflectance = lambda d, t, l, n_fn: reflectance_NA(d, t, l, n_fn, NA=na, n_nodes=n_nodes)
    return f"NA-averaged (NA={na:.2f}, {n_nodes} pupil nodes)"

# ------------------------------------------------------------------ per-flake evaluation
def evaluate_row(row, warn):
    mat = row["material"].strip().lower()
    if mat not in MATERIALS:
        raise ValueError(f"unknown material '{mat}'")
    n_fn, layer_nm = MATERIALS[mat]

    img01, depth = load_image(row["image_path"])
    if depth == "8-bit":
        warn.append(f"{row['flake_id']}: 8-bit image (usable; see paper Sec. 5.9)")
    if row["image_path"].lower().endswith((".jpg", ".jpeg")):
        warn.append(f"{row['flake_id']}: JPEG source - compression artifacts bias region means")

    gamma = row.get("gamma", "linear").strip() or "linear"
    half = int(row.get("patch_px") or 16)
    y_f, sd_f, n_f = sample_patch(img01, int(row["flake_x"]), int(row["flake_y"]), half, gamma)
    y_b, sd_b, n_b = sample_patch(img01, int(row["bg_x"]), int(row["bg_y"]), half, gamma)

    if np.allclose(y_f, y_b, atol=2e-3):
        warn.append(f"{row['flake_id']}: flake and background colors nearly identical "
                    f"- check coordinates")
    if (sd_b / np.maximum(y_b, 1e-6)).max() > 0.15:
        warn.append(f"{row['flake_id']}: background patch is non-uniform "
                    f"(vignetting, dirt, or a flake edge in the patch)")

    # oxide prior from the wafer spec
    _new = (float(row["oxide_nm"]), float(row.get("oxide_sd_nm") or E.SIG_TOX))
    if (E.TOX0, E.SIG_TOX) != _new:
        E.TOX0, E.SIG_TOX = _new
        jax.clear_caches()

    r = E.estimate_v2(jnp.asarray(y_f), jnp.asarray(y_b), float(n_f), float(n_b),
                      n_fn=n_fn, k_basis=5)
    out = {
        "flake_id": row["flake_id"], "material": mat,
        "d_hat_nm": r["d"], "sd_nm": r["sd"], "layers_hat": r["d"] / layer_nm,
        "oxide_fit_nm": float(r["theta"][1]),
        "gain_r": float(np.exp(r["theta"][2])), "gain_g": float(np.exp(r["theta"][3])),
        "gain_b": float(np.exp(r["theta"][4])),
        "n_modes": len(r["modes"]),
        "alt_d_nm": r["alts"][0][0] if r["alts"] else "",
        "alt_gap": r["alts"][0][1] if r["alts"] else "",
        "posterior": r["L"],
        "y_f": list(np.round(y_f, 5)), "y_b": list(np.round(y_b, 5)),
    }

    # --- falsifiable internal checks: the fitted nuisances are physical claims ---
    dox = out["oxide_fit_nm"] - float(row["oxide_nm"])
    if abs(dox) > 4 * float(row.get("oxide_sd_nm") or 5.0):
        warn.append(f"{row['flake_id']}: fitted oxide {out['oxide_fit_nm']:.1f} nm is "
                    f"{dox:+.1f} nm from spec - suspect wrong NA, wrong gamma, or bad patch")
    if max(out["gain_r"], out["gain_g"], out["gain_b"]) > 1.6 or \
       min(out["gain_r"], out["gain_g"], out["gain_b"]) < 0.6:
        warn.append(f"{row['flake_id']}: extreme fitted gains - likely auto-white-balance "
                    f"or an unmodeled tone curve")
    if r["alts"] and r["alts"][0][1] < 10:
        warn.append(f"{row['flake_id']}: AMBIGUOUS - rival thickness at "
                    f"{r['alts'][0][0]:.1f} nm within {r['alts'][0][1]:.1f} posterior units")

    if row.get("afm_height_nm"):
        afm = float(row["afm_height_nm"])
        out["afm_nm"] = afm
        out["afm_sd_nm"] = float(row.get("afm_sd_nm") or 0.15)
        out["residual_nm"] = r["d"] - afm
        out["z_score"] = out["residual_nm"] / np.hypot(out["afm_sd_nm"], max(r["sd"], 1e-6))
    return out

# ------------------------------------------------------------------ reporting
def summarize(rows, path, physics_mode):
    have = [r for r in rows if "afm_nm" in r]
    L = []
    L.append("FLAKE-DEPTH REAL-DATA EVALUATION")
    L.append(f"forward model: {physics_mode}")
    L.append(f"flakes evaluated: {len(rows)}  with AFM ground truth: {len(have)}")
    L.append("")
    if have:
        res = np.array([r["residual_nm"] for r in have])
        afm = np.array([r["afm_nm"] for r in have])
        L.append("AGREEMENT WITH AFM")
        L.append(f"  median |residual|      {np.median(np.abs(res)):.3f} nm")
        L.append(f"  mean |residual|        {np.abs(res).mean():.3f} nm")
        L.append(f"  median signed residual {np.median(res):+.3f} nm   "
                 f"(a systematic offset here is the AFM adsorbed-water layer, not model error)")
        L.append(f"  RMS residual           {np.sqrt((res**2).mean()):.3f} nm")
        if len(have) > 2:
            r_ = np.corrcoef(afm, afm + res)[0, 1]
            sl, ic = np.polyfit(afm, afm + res, 1)
            L.append(f"  optical = {sl:.3f} x AFM + {ic:+.3f} nm   (r = {r_:.4f})")
            L.append(f"  slope is the meaningful test: 1.00 means the two measure the same "
                     f"quantity; the intercept absorbs the AFM offset.")
        thick = [r for r in have if r["afm_nm"] > 3.0]
        if thick:
            rt = np.array([r["residual_nm"] for r in thick])
            L.append(f"  restricted to AFM > 3 nm (where AFM is reliable, n={len(thick)}): "
                     f"median |residual| {np.median(np.abs(rt)):.3f} nm")
    L.append("")
    L.append("INTERNAL CONSISTENCY (no ground truth needed - these are falsifiable claims)")
    ox = np.array([r["oxide_fit_nm"] for r in rows])
    L.append(f"  fitted oxide: mean {ox.mean():.1f} nm, sd {ox.std():.1f} nm across flakes")
    L.append(f"    -> flakes on one wafer must agree. Scatter >> wafer tolerance means the")
    L.append(f"       model is absorbing an unmodeled effect (NA, gamma, or illumination).")
    amb = [r for r in rows if r["n_modes"] > 1]
    L.append(f"  flakes flagged ambiguous: {len(amb)}/{len(rows)}")
    open(path, "w").write("\n".join(L) + "\n")
    return "\n".join(L)

def make_plot(rows, path):
    have = [r for r in rows if "afm_nm" in r]
    if len(have) < 3:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    afm = np.array([r["afm_nm"] for r in have])
    opt = np.array([r["d_hat_nm"] for r in have])
    ea = np.array([r["afm_sd_nm"] for r in have])
    eo = np.array([r["sd_nm"] for r in have])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = [0, max(afm.max(), opt.max()) * 1.12]
    ax.plot(lim, lim, "k--", lw=1.2, label="y = x")
    ax.errorbar(afm, opt, xerr=ea, yerr=eo, fmt="o", ms=5, capsize=2,
                color="#2166ac", label="flakes")
    ax.set_xlabel("AFM step height (nm)"); ax.set_ylabel("optical estimate (nm)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(alpha=0.25)
    ax.legend(fontsize=9); ax.set_title("Optical thickness vs AFM ground truth")
    fig.tight_layout(); fig.savefig(path, dpi=160)

def make_manifest_template(path):
    cols = ["flake_id","image_path","material","na","oxide_nm","oxide_sd_nm",
            "flake_x","flake_y","bg_x","bg_y","patch_px","afm_height_nm","afm_sd_nm",
            "raman_layers","gamma","notes"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        w.writerow(["G01","images/G01.tif","graphene","0.90","285","5",
                    "742","518","120","96","16","1.05","0.15","1","srgb",
                    "monolayer confirmed by Raman 2D/G"])
    print(f"wrote template -> {path}")

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--out", default="results")
    ap.add_argument("--template", help="write a manifest template and exit")
    ap.add_argument("--na", type=float, help="override NA for all rows")
    a = ap.parse_args()
    if a.template:
        make_manifest_template(a.template); return
    if not a.manifest:
        ap.error("--manifest is required (or use --template)")
    os.makedirs(a.out, exist_ok=True)
    rows_in = list(csv.DictReader(open(a.manifest)))
    if not rows_in:
        sys.exit("manifest is empty")

    nas = {float(a.na if a.na else r["na"]) for r in rows_in}
    if len(nas) > 1:
        sys.exit(f"multiple NA values {sorted(nas)} in one manifest. The forward model is "
                 f"patched once per run; split the manifest by objective and run separately.")
    physics_mode = install_na(nas.pop())

    out, warn = [], []
    for r in rows_in:
        try:
            out.append(evaluate_row(r, warn))
            k = out[-1]
            print(f"{k['flake_id']:>8}  d = {k['d_hat_nm']:7.3f} +/- {k['sd_nm']:.3f} nm "
                  f"({k['layers_hat']:5.2f} L)" +
                  (f"   AFM {k['afm_nm']:.2f}  resid {k['residual_nm']:+.3f}" if "afm_nm" in k else ""))
        except Exception as e:
            warn.append(f"{r.get('flake_id','?')}: FAILED - {e}")
            print(f"{r.get('flake_id','?'):>8}  FAILED: {e}")

    if out:
        keys = sorted({k for r in out for k in r})
        with open(os.path.join(a.out, "per_flake.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
        print("\n" + summarize(out, os.path.join(a.out, "summary.txt"), physics_mode))
        make_plot(out, os.path.join(a.out, "agreement.png"))
    open(os.path.join(a.out, "diagnostics.txt"), "w").write("\n".join(warn) + "\n")
    if warn:
        print(f"\n{len(warn)} diagnostic(s) -> {a.out}/diagnostics.txt")
        for w_ in warn[:10]:
            print("  ! " + w_)

if __name__ == "__main__":
    main()
