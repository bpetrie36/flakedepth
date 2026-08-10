#!/usr/bin/env python3
"""
per_set.py - contrast and substrate color per ACQUISITION, not per dataset.

The dataset-level analysis is done: GrapheneL/M/H are overlapping samplings of
one pool of named acquisitions, shared images are pixel-identical, and the
GrapheneH anomaly lives entirely in the acquisitions L does not contain. This
regroups everything on the correct variable.

Each image is attributed to its acquisition using the union of every
{split}_set_name_to_uuid.json found across the datasets you pass. Stems are
DEDUPLICATED globally, so an acquisition appearing in two datasets is measured
once, not twice.

WHAT IT REPORTS, PER ACQUISITION
  substrate median RGB and IQR   different wafer or illumination shows here
  per-class flake contrast       the estimator's observable
  span (class4 - class1)         proxy for how fast contrast grows per label;
                                 a low span with normal absolute contrast means
                                 the labels do not track thickness the same way
  ratio to the reference set     one number per acquisition for ranking

USE IT TO
  decide which acquisitions belong in one oxide fit and which need their own,
  and which should be excluded from a graphene layer-height claim entirely.

USAGE
  python per_set.py --dataset L=D:\\MaskTerialData\\GrapheneL ^
                    M=D:\\MaskTerialData\\GrapheneM ^
                    H=D:\\MaskTerialData\\GrapheneH ^
                    --split test --out diag_full
"""
import argparse, glob, json, os, re, sys
import numpy as np
from PIL import Image

BG_ERODE, CORE_ERODE, BG_SUB = 5, 3, 11
MIN_BG_PX, MIN_CORE_PX, MIN_PX = 5000, 80, 300
UUIDISH = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def erode(m, it):
    from scipy import ndimage
    return ndimage.binary_erosion(m, iterations=it, border_value=0)


def build_index(paths):
    """stem -> acquisition name, from every uuid mapping in every dataset."""
    idx = {}
    for p in paths:
        for f in sorted(glob.glob(os.path.join(p, "meta_data", "*.json"))):
            try:
                raw = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            for k, v in raw.items():
                if not isinstance(v, list) or not v:
                    continue
                stems = [os.path.splitext(os.path.basename(str(x)))[0] for x in v]
                if sum(1 for s in stems if UUIDISH.match(s)) < max(1, len(stems) // 2):
                    continue
                for s in stems:
                    idx[s] = k
    return idx


def decode(x, mode):
    """Undo the camera transfer function, matching maskterial_calib.decode."""
    if mode == "linear":
        return x
    if mode == "srgb":
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x ** float(mode)


def harvest(ip, mp, gamma="linear"):
    raw = np.asarray(Image.open(ip).convert("RGB"))
    msk = np.asarray(Image.open(mp))
    if msk.ndim == 3:
        msk = msk[:, :, 0]
    bg = erode(msk == 0, BG_ERODE)
    if bg.sum() < MIN_BG_PX:
        return None, None
    bgpx = raw[bg][::BG_SUB].astype(np.float64)
    y_b = np.median(bgpx, axis=0)
    if np.min(y_b) < 1e-4:
        return None, None
    img = decode(raw.astype(np.float64) / 255.0, gamma)
    y_b = np.median(decode(bgpx / 255.0, gamma), axis=0)
    out = []
    from scipy import ndimage
    for cls in sorted(set(np.unique(msk)) - {0}):
        lab, n = ndimage.label(msk == cls)
        for k in range(1, n + 1):
            c = lab == k
            if c.sum() < MIN_PX:
                continue
            core = erode(c, CORE_ERODE)
            if core.sum() < MIN_CORE_PX:
                continue
            out.append((int(cls), np.median(img[core], axis=0) / y_b))
    return np.median(bgpx, axis=0), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref", default="", help="acquisition name to use as reference")
    ap.add_argument("--gamma", default="linear", help="linear | srgb | a number")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    specs = [s.split("=", 1) for s in a.dataset]
    idx = build_index([p for _, p in specs])
    print(f"index covers {len(idx)} images", flush=True)

    seen, sets = set(), {}
    for name, path in specs:
        for ip in sorted(glob.glob(os.path.join(path, f"{a.split}_images", "*"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            if stem in seen:
                continue
            acq = idx.get(stem)
            if acq is None:
                continue
            mps = glob.glob(os.path.join(path, f"{a.split}_semantic_masks", stem + ".*"))
            if not mps:
                continue
            seen.add(stem)
            y_b, fl = harvest(ip, mps[0], a.gamma)
            if y_b is None or not fl:
                continue
            d = sets.setdefault(acq, {"bg": [], "fl": [], "src": set()})
            d["bg"].append(y_b)
            d["fl"].extend(fl)
            d["src"].add(name)
        print(f"  scanned {name}", flush=True)

    if not sets:
        sys.exit("nothing attributed; check meta_data")

    classes = sorted({c for d in sets.values() for c, _ in d["fl"]})

    def contrast(d, cls):
        rs = [r for c, r in d["fl"] if c == cls]
        return (1 - np.median(rs, axis=0).mean()) if len(rs) >= 5 else None

    rows = []
    for acq, d in sets.items():
        cs = {c: contrast(d, c) for c in classes}
        bg = np.array(d["bg"])
        rows.append({
            "acq": acq, "n_img": len(d["bg"]), "n_fl": len(d["fl"]),
            "src": ",".join(sorted(d["src"])),
            "bg_med": np.median(bg, axis=0),
            "bg_iqr": np.subtract(*np.percentile(bg, [75, 25], axis=0)),
            "cs": cs,
            "span": (cs[classes[-1]] - cs[classes[0]])
                    if cs.get(classes[-1]) and cs.get(classes[0]) else None,
        })

    ref = a.ref
    if not ref:
        ok = [r for r in rows if all(r["cs"].get(c) for c in classes)]
        ref = max(ok, key=lambda r: r["n_fl"])["acq"] if ok else rows[0]["acq"]
    rr = next(r for r in rows if r["acq"] == ref)

    def rel(r):
        v = [r["cs"][c] / rr["cs"][c] for c in classes
             if r["cs"].get(c) and rr["cs"].get(c)]
        return float(np.mean(v)) if v else None

    for r in rows:
        r["rel"] = rel(r)
    rows.sort(key=lambda r: (r["rel"] is None, r["rel"]))

    L = []
    A = L.append
    A(f"===== PER-ACQUISITION ({a.split} split, decoded as {a.gamma}) =====")
    A(f"acquisitions measured {len(rows)}, unique images {len(seen)}")
    A(f"reference acquisition: {ref}")
    A("")
    w = max(len(r["acq"]) for r in rows) + 2
    A(f"  {'acquisition':<{w}}{'in':>7}{'imgs':>6}{'flk':>6}   substrate RGB      IQR R  "
      + "".join(f"{'c'+str(c):>8}" for c in classes) + f"{'span':>8}{'rel':>7}")
    for r in rows:
        cs = "".join(f"{r['cs'][c]:>8.4f}" if r["cs"].get(c) else f"{'-':>8}" for c in classes)
        A(f"  {r['acq']:<{w}}{r['src']:>7}{r['n_img']:>6}{r['n_fl']:>6}   "
          f"{r['bg_med'][0]:5.0f}{r['bg_med'][1]:5.0f}{r['bg_med'][2]:5.0f}  {r['bg_iqr'][0]:7.0f}  "
          + cs
          + (f"{r['span']:>8.4f}" if r["span"] else f"{'-':>8}")
          + (f"{r['rel']:>7.3f}" if r["rel"] else f"{'-':>7}"))
    A("")
    A("READING")
    A("  rel is mean per-class contrast relative to the reference acquisition.")
    A("  Acquisitions clustered near 1.00 share imaging conditions closely enough")
    A("  for one oxide fit. Outliers need their own fit, or exclusion.")
    A("  A substrate median far from the pack means a different wafer or lamp.")
    A("  A normal absolute contrast with a compressed span means the labels are")
    A("  not stepping in the same physical units, which no oxide refit will fix.")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        p = os.path.join(a.out, f"per_set_{a.split}_{a.gamma}.txt")
        open(p, "w", encoding="utf-8").write(txt + "\n")
        print(f"\nwritten to {p}")


if __name__ == "__main__":
    main()
