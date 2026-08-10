#!/usr/bin/env python3
"""
collect_runs.py - assemble the per-acquisition table from a batch of runs.

Reads every <prefix>*/per_flake.csv under a directory and reports, per
acquisition, the numbers that belong in the paper:

  * label-free oxide recovery: median and MAD over ALL flakes, and over the
    subset the estimator itself flags as well determined (sd_nm below a
    threshold). Neither uses the annotations, so neither is circular.
  * layer agreement: exact and within-one.
  * ridge failures: flakes whose thickness is more than 1 nm from label x layer
    height. This DOES use labels and is reported as a failure rate, not as a
    selection criterion for anything above.
  * gains, decomposed into exposure (the green channel, physically arbitrary)
    and chromatic ratios R/G and B/G (the white balance, which should be near
    neutral if the camera applied none).

USAGE
  python collect_runs.py --dir D:\\MaskTerialData\\fd\\flakedepth --prefix fj_
  python collect_runs.py --dir . --prefix fj_ --csv per_acquisition.csv
"""
import argparse, csv, glob, math, os, statistics as st, sys

def med_mad(v):
    if not v: return float("nan"), float("nan")
    m = st.median(v)
    return m, 1.4826 * st.median([abs(x - m) for x in v])

def load(p):
    out = []
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out.append({k: (v if k == "image" else float(v)) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--prefix", default="fj_")
    ap.add_argument("--sd-max", type=float, default=0.05,
                    help="label-free quality filter on the estimator's own sd_nm")
    ap.add_argument("--fail-nm", type=float, default=1.0)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(a.dir, a.prefix + "*"))
                  if os.path.isfile(os.path.join(d, "per_flake.csv")))
    if not dirs:
        sys.exit(f"no {a.prefix}*/per_flake.csv under {a.dir}")

    rows, pooled = [], []
    for d in dirs:
        R = load(os.path.join(d, "per_flake.csv"))
        if not R: continue
        name = os.path.basename(d)[len(a.prefix):]
        ox_all = [r["oxide_fit_nm"] for r in R]
        sel = [r for r in R if r["sd_nm"] < a.sd_max]
        ox_sel = [r["oxide_fit_nm"] for r in sel]
        fails = [r for r in R if abs(r["residual_nm"]) >= a.fail_nm]
        exact = sum(1 for r in R if round(r["layers_hat"]) == round(r["label_layers"]))
        within = sum(1 for r in R if abs(round(r["layers_hat"]) - round(r["label_layers"])) <= 1)
        good = [r for r in R if abs(r["residual_nm"]) < a.fail_nm]
        expo = [r["gain_g"] for r in good] or [r["gain_g"] for r in R]
        rg = [r["gain_r"] / r["gain_g"] for r in good if r["gain_g"]] or [1.0]
        bg = [r["gain_b"] / r["gain_g"] for r in good if r["gain_g"]] or [1.0]
        m_all, d_all = med_mad(ox_all); m_sel, d_sel = med_mad(ox_sel)
        rows.append(dict(acq=name, n=len(R), n_sel=len(sel),
                         ox_all=m_all, mad_all=d_all, ox_sel=m_sel, mad_sel=d_sel,
                         exact=100*exact/len(R), within=100*within/len(R),
                         fail=100*len(fails)/len(R),
                         expo=st.median(expo), rg=st.median(rg), bg=st.median(bg),
                         resid=st.median([abs(r["residual_nm"]) for r in R])))
        pooled += R

    w = max(len(r["acq"]) for r in rows) + 1
    print(f"{'acquisition':<{w}}{'n':>4}{'oxide med':>11}{'MAD':>7}"
          f"{'oxide|sd':>10}{'MAD':>7}{'exact%':>8}{'w1%':>6}{'fail%':>7}"
          f"{'medres':>8}{'expo':>7}{'R/G':>6}{'B/G':>6}")
    for r in sorted(rows, key=lambda x: x["ox_sel"] if not math.isnan(x["ox_sel"]) else 1e9):
        print(f"{r['acq']:<{w}}{r['n']:>4}{r['ox_all']:>11.2f}{r['mad_all']:>7.2f}"
              f"{r['ox_sel']:>10.2f}{r['mad_sel']:>7.2f}{r['exact']:>8.1f}{r['within']:>6.1f}"
              f"{r['fail']:>7.1f}{r['resid']:>8.3f}{r['expo']:>7.2f}{r['rg']:>6.2f}{r['bg']:>6.2f}")

    ox = [r["ox_sel"] for r in rows if not math.isnan(r["ox_sel"])]
    m, d = med_mad(ox)
    f = [r["fail"] for r in rows]
    print(f"\nacquisitions: {len(rows)}   flakes: {len(pooled)}")
    print(f"per-acquisition oxide (label-free, sd-filtered): median {m:.2f} nm, "
          f"spread MAD {d:.2f} nm, range [{min(ox):.1f}, {max(ox):.1f}]")
    print(f"ridge failure rate across acquisitions: median {st.median(f):.1f}%, "
          f"range [{min(f):.1f}, {max(f):.1f}]")
    print("\nThe oxide columns use NO annotations. The failure column does, and is a")
    print("reported rate rather than a filter applied to anything else.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"\nwritten to {a.csv}")

if __name__ == "__main__":
    main()
