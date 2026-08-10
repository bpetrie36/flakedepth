#!/usr/bin/env python3
"""
set_map.py - which acquisition does each image actually come from?

The MaskTerial releases carry two different metadata shapes:

  name -> uuid mapping    {"Set_Graphene-Calib-04-04-2022": ["<uuid>.png", ...]}
                          tells you WHICH images came from an acquisition
  set-name manifest       {"test sets": ["Set_Graphene-Calib-04-04-2022", ...]}
                          tells you only THAT the acquisition is included

GrapheneM has the first, GrapheneL has the second. Since the UUIDs are globally
unique, a mapping found in ANY dataset attributes those images in EVERY dataset.
So this builds one global stem -> set index from every mapping it can find, then
applies it to all three.

Why it matters: the estimator's restricted mode fits substrate oxide as a single
global constant per dataset. That is only sound if a dataset is one acquisition.
If a dataset mixes acquisitions taken months apart, the fitted constant lands
wherever the mixture pulls it, and the recovered layer height moves with the mix
rather than with the physics.

USAGE
  python set_map.py --dataset L=D:\\MaskTerialData\\GrapheneL ^
                    M=D:\\MaskTerialData\\GrapheneM ^
                    H=D:\\MaskTerialData\\GrapheneH --split test --out diag_full

Reads JSON and directory listings only. No images. Runs in seconds.
"""
import argparse
import glob
import json
import os
import re
import sys

UUIDISH = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def classify(path):
    """-> ('mapping', {set: {stems}}) | ('manifest', {key: [names]}) | (None, why)"""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as e:
        return None, f"unreadable ({e})"
    if not isinstance(raw, dict) or not raw:
        return None, "not a non-empty object"

    vals = [v for v in raw.values() if isinstance(v, list) and v]
    if not vals:
        return None, "no list values"
    sample = [str(x) for v in vals for x in v[:4]]
    stems = [os.path.splitext(os.path.basename(s))[0] for s in sample]
    uuidish = sum(1 for s in stems if UUIDISH.match(s))

    if uuidish >= max(1, len(stems) // 2):
        return "mapping", {k: {os.path.splitext(os.path.basename(str(x)))[0] for x in v}
                           for k, v in raw.items() if isinstance(v, list) and v}
    return "manifest", {k: [str(x) for x in v]
                        for k, v in raw.items() if isinstance(v, list)}


def scan_meta(ds):
    md = os.path.join(ds, "meta_data")
    out = []
    if not os.path.isdir(md):
        return out, "no meta_data/ directory"
    for p in sorted(glob.glob(os.path.join(md, "*.json"))):
        kind, payload = classify(p)
        out.append((os.path.basename(p), kind, payload))
    if not out:
        return out, "meta_data/ holds no .json: " + ", ".join(sorted(os.listdir(md))[:12])
    return out, ""


def image_stems(ds, split):
    return {os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(ds, f"{split}_images", "*"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, help="NAME=PATH pairs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    specs = []
    for spec in a.dataset:
        if "=" not in spec:
            sys.exit(f"expected NAME=PATH, got {spec}")
        specs.append(spec.split("=", 1))
    names = [n for n, _ in specs]

    L = []
    A = L.append
    A(f"===== ACQUISITION SETS ({a.split} split) =====")
    A("")

    global_index, declared = {}, {}
    A("-- metadata found --")
    for n, path in specs:
        files, why = scan_meta(path)
        if why:
            A(f"  {n}: {why}")
        for fname, kind, payload in files:
            if kind == "mapping":
                A(f"  {n}: {fname}  [name->uuid, {len(payload)} sets, "
                  f"{sum(len(v) for v in payload.values())} images]")
                for s, stems in payload.items():
                    for st in stems:
                        global_index[st] = s
                declared.setdefault(n, set()).update(payload)
            elif kind == "manifest":
                keys = [k for k in payload if a.split.lower() in k.lower()] or list(payload)
                got = {v for k in keys for v in payload[k]}
                A(f"  {n}: {fname}  [manifest, {len(got)} set names for this split]")
                declared.setdefault(n, set()).update(got)
            else:
                A(f"  {n}: {fname}  [unrecognised: {payload}]")
    A("")
    A(f"global stem->set index built from {len(global_index)} attributed images")
    A("")

    present = {n: image_stems(p, a.split) for n, p in specs}
    attributed = {n: {} for n in names}
    unattributed = {}
    for n in names:
        miss = 0
        for st in present[n]:
            s = global_index.get(st)
            if s is None:
                miss += 1
            else:
                attributed[n][s] = attributed[n].get(s, 0) + 1
        unattributed[n] = miss

    A("-- images on disk, attributed to an acquisition --")
    A("             " + "".join(f"{n:>12}" for n in names))
    A("  on disk    " + "".join(f"{len(present[n]):>12}" for n in names))
    A("  attributed " + "".join(f"{sum(attributed[n].values()):>12}" for n in names))
    A("  UNKNOWN    " + "".join(f"{unattributed[n]:>12}" for n in names))
    A("")

    all_sets = sorted(set(global_index.values()) | {s for v in declared.values() for s in v})
    w = max((len(s) for s in all_sets), default=10) + 2
    A("-- image count per acquisition ('decl' = declared but no uuid mapping) --")
    A(f"  {'acquisition':<{w}}" + "".join(f"{n:>10}" for n in names))
    for s in all_sets:
        cells = []
        for n in names:
            if s in attributed[n]:
                cells.append(f"{attributed[n][s]:>10}")
            elif s in declared.get(n, ()):
                cells.append(f"{'decl':>10}")
            else:
                cells.append(f"{'-':>10}")
        A(f"  {s:<{w}}" + "".join(cells))
    A("")

    A("-- acquisitions shared between datasets --")
    any_shared = False
    for s in all_sets:
        holders = [n for n in names if s in attributed[n] or s in declared.get(n, ())]
        if len(holders) > 1:
            any_shared = True
            A(f"  {s}: {', '.join(holders)}")
    if not any_shared:
        A("  none")
    A("")

    A("READING")
    A("  Every dataset with more than one acquisition violates the single-global-")
    A("  oxide assumption in restricted mode. The fitted constant is then a")
    A("  weighted compromise across wafers, and the recovered layer height moves")
    A("  with the mixture rather than with the optics. Refit per acquisition.")
    A("  An acquisition listed under two datasets is ONE piece of evidence, not")
    A("  two, and must not be counted twice as an independent replicate.")
    A("  A large UNKNOWN count means images no released mapping covers; those")
    A("  cannot be assigned to a wafer at all.")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"set_map_{a.split}.txt"), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
        print(f"\nwritten to {os.path.join(a.out, f'set_map_{a.split}.txt')}")


if __name__ == "__main__":
    main()
