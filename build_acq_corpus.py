#!/usr/bin/env python3
"""
build_acq_corpus.py - re-partition the MaskTerial releases by ACQUISITION.

WHY
  maskterial_calib fits substrate oxide as ONE global constant per dataset.
  That is correct when a dataset is one acquisition and wrong when it mixes
  many, because the constant then lands wherever the mixture pulls it. The
  GrapheneL/M/H releases are overlapping samplings of ~24 named acquisitions,
  so no run against them has ever fitted one wafer.

  The fix needs no change to the estimator. Give it one acquisition per
  --dataset and it does the right thing.

WHAT IT BUILDS
  <out>/<acquisition>/train_images/          40% of that acquisition's images
  <out>/<acquisition>/train_semantic_masks/
  <out>/<acquisition>/test_images/           the other 60%
  <out>/<acquisition>/test_semantic_masks/
  <out>/manifest.csv                         stem, acquisition, sources, split
  <out>/build_report.txt

  Images are DEDUPLICATED globally: a stem appearing in several releases is
  written once. The split is deterministic (sorted stems, stride), so reruns
  reproduce it exactly and no image can leak from train to test.

  Where a stem exists in several releases the mask is taken from the FIRST
  --dataset listed, and the manifest records every release it appeared in.
  Labels agree 95-99% across releases, so this choice is small but it is
  yours to make, not silent: reorder --dataset to change it.

USAGE
  python build_acq_corpus.py --dataset L=D:\\MaskTerialData\\GrapheneL ^
        M=D:\\MaskTerialData\\GrapheneM H=D:\\MaskTerialData\\GrapheneH ^
        --out D:\\MaskTerialData\\by_acquisition --dry-run

  Drop --dry-run to actually write. Add --hardlink to avoid duplicating bytes
  (same volume only). Add --min-images N to skip acquisitions too small to fit.
"""
import argparse, csv, glob, json, os, re, shutil, sys

UUIDISH = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def build_index(paths):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, help="NAME=PATH pairs")
    ap.add_argument("--splits", nargs="+", default=["test", "train"],
                    help="source split folders to draw from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-frac", type=float, default=0.4)
    ap.add_argument("--min-images", type=int, default=12)
    ap.add_argument("--hardlink", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    specs = [s.split("=", 1) for s in a.dataset]
    idx = build_index([p for _, p in specs])

    # stem -> {acq, sources[], img path, mask path}
    found = {}
    for name, path in specs:
        for sp in a.splits:
            for ip in sorted(glob.glob(os.path.join(path, f"{sp}_images", "*"))):
                stem = os.path.splitext(os.path.basename(ip))[0]
                mps = glob.glob(os.path.join(path, f"{sp}_semantic_masks", stem + ".*"))
                if not mps:
                    continue
                rec = found.get(stem)
                if rec is None:
                    found[stem] = {"acq": idx.get(stem), "src": [name],
                                   "img": ip, "mask": mps[0]}
                elif name not in rec["src"]:
                    rec["src"].append(name)

    groups, orphans = {}, []
    for stem, rec in found.items():
        if rec["acq"]:
            groups.setdefault(rec["acq"], []).append(stem)
        else:
            orphans.append(stem)
    groups = {k: sorted(v) for k, v in groups.items()}

    R = []
    A = R.append
    A("===== ACQUISITION CORPUS =====")
    A(f"unique images across releases : {len(found)}")
    A(f"attributed to an acquisition  : {sum(len(v) for v in groups.values())}")
    A(f"UNATTRIBUTED (no mapping)     : {len(orphans)}")
    A(f"acquisitions                  : {len(groups)}")
    A(f"train fraction {a.train_frac}, min images {a.min_images}")
    A("")
    w = max((len(k) for k in groups), default=12) + 2
    A(f"  {'acquisition':<{w}}{'imgs':>6}{'train':>7}{'test':>6}   sources")
    built, skipped = [], []
    for acq in sorted(groups, key=lambda k: -len(groups[k])):
        stems = groups[acq]
        cut = max(1, min(9, round(a.train_frac * 10)))
        tr = [s for i, s in enumerate(stems) if (i % 10) < cut]
        te = [s for s in stems if s not in set(tr)]
        srcs = sorted({x for s in stems for x in found[s]["src"]})
        flag = "" if len(stems) >= a.min_images else "   SKIP (too few)"
        A(f"  {acq:<{w}}{len(stems):>6}{len(tr):>7}{len(te):>6}   {','.join(srcs)}{flag}")
        (built if len(stems) >= a.min_images else skipped).append((acq, tr, te))
    A("")
    A(f"would build {len(built)} acquisitions, skip {len(skipped)}")
    if orphans:
        A(f"{len(orphans)} unattributed images are NOT written. They belong to")
        A("acquisitions whose uuid mapping was never released, so they cannot be")
        A("assigned to a wafer and must not be pooled into one.")

    print("\n".join(R))
    if a.dry_run:
        print("\ndry run, nothing written")
        return

    os.makedirs(a.out, exist_ok=True)
    rows = []
    place = os.link if a.hardlink else shutil.copy2
    for acq, tr, te in built:
        safe = SAFE.sub("_", acq)
        for split, stems in (("train", tr), ("test", te)):
            for sub in (f"{split}_images", f"{split}_semantic_masks"):
                os.makedirs(os.path.join(a.out, safe, sub), exist_ok=True)
            for s in stems:
                rec = found[s]
                for src, sub in ((rec["img"], f"{split}_images"),
                                 (rec["mask"], f"{split}_semantic_masks")):
                    dst = os.path.join(a.out, safe, sub, os.path.basename(src))
                    if os.path.exists(dst):
                        continue
                    try:
                        place(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                rows.append({"stem": s, "acquisition": acq, "folder": safe,
                             "split": split, "sources": "|".join(rec["src"])})
        print(f"built {safe}: {len(tr)} train, {len(te)} test", flush=True)

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["stem", "acquisition", "folder", "split", "sources"])
        wtr.writeheader()
        wtr.writerows(rows)
    open(os.path.join(a.out, "build_report.txt"), "w", encoding="utf-8").write("\n".join(R) + "\n")
    print(f"\nmanifest.csv and build_report.txt written to {a.out}")


if __name__ == "__main__":
    main()
