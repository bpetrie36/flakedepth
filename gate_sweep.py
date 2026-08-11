#!/usr/bin/env python3
"""
gate_sweep.py - is the fitted oxide a label-free detector for ridge failures?

BACKGROUND
  Per-flake thickness inversion puts 10-40% of real flakes into a spurious basin
  near 35-41 nm oxide. Section 4.1 shows these failures are UNIMODAL, so the
  ambiguity flag (n_modes > 1) does not catch them: on 299 flakes it fired 6 times
  and caught 2 of 91 failures, a recall of 2.2%.

  But the estimator also reports a fitted oxide per flake, and flakes on one wafer
  must agree. On a first look at three acquisitions, rejecting flakes whose fitted
  oxide sits more than 10 nm from the wafer value kept 202 flakes with ZERO
  failures and rejected 97 of which 94% genuinely failed. This sweeps that across
  every acquisition.

TWO REFERENCES, both label-free, reported side by side
  prior   the oxide passed to the run (--oxide). Needs a calibration sweep first.
  median  the median fitted oxide over that acquisition's own flakes. Needs
          nothing external at all, and is the per-flake form of the population
          check the tool already prints. Its weakness is stated below.

WHAT TO WATCH
  The median reference is only meaningful while most flakes are right. On an
  acquisition where the majority land on the ridge, the median IS the ridge and
  the gate inverts. The script reports each acquisition's failure rate alongside,
  so that case is visible rather than hidden.

  Failure means |estimate - label x layer height| >= --fail-nm. Labels are used
  ONLY to score the gate, never to build it.

USAGE
  python gate_sweep.py --dir . --prefix fj_
  python gate_sweep.py --dir . --prefix fj_ --priors priors.csv --csv gate.csv

  priors.csv, optional: two columns, acquisition name and the oxide passed to it.
"""
import argparse, csv, glob, math, os, statistics as st, sys


def med_mad(v):
    m = st.median(v)
    return m, 1.4826 * st.median([abs(x - m) for x in v])


def load(p):
    out = []
    for r in csv.DictReader(open(p, newline="")):
        try:
            out.append({k: (v if k == "image" else float(v)) for k, v in r.items()})
        except (TypeError, ValueError):
            continue
    return out


def stats(rows, dev_key, thr, fail_nm):
    keep = [r for r in rows if r[dev_key] <= thr]
    drop = [r for r in rows if r[dev_key] > thr]
    F = lambda r: abs(r["residual_nm"]) >= fail_nm
    nf = sum(1 for r in rows if F(r))
    kf = sum(1 for r in keep if F(r))
    df = sum(1 for r in drop if F(r))
    return dict(kept=len(keep), keptfail=kf, dropped=len(drop), dropfail=df, tot=len(rows),
                totfail=nf,
                keptrate=(100 * kf / len(keep)) if keep else float("nan"),
                prec=(100 * df / len(drop)) if drop else float("nan"),
                recall=(100 * df / nf) if nf else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--prefix", default="fj_")
    ap.add_argument("--fail-nm", dest="fail_nm", type=float, default=1.0)
    ap.add_argument("--thresholds", default="3,5,8,10,15,20,30")
    ap.add_argument("--report-at", dest="report_at", type=float, default=10.0)
    ap.add_argument("--priors", default="", help="optional csv: acquisition,oxide")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()

    priors = {}
    if a.priors and os.path.isfile(a.priors):
        for row in csv.reader(open(a.priors)):
            if len(row) >= 2:
                try:
                    priors[row[0].strip()] = float(row[1])
                except ValueError:
                    pass

    dirs = sorted(d for d in glob.glob(os.path.join(a.dir, a.prefix + "*"))
                  if os.path.isfile(os.path.join(d, "per_flake.csv")))
    if not dirs:
        sys.exit(f"no {a.prefix}*/per_flake.csv under {a.dir}")

    thrs = [float(x) for x in a.thresholds.split(",")]
    per_acq, pooled = [], []
    for d in dirs:
        R = load(os.path.join(d, "per_flake.csv"))
        if len(R) < 5:
            continue
        name = os.path.basename(d)[len(a.prefix):]
        ref_med, _ = med_mad([r["oxide_fit_nm"] for r in R])
        ref_pri = priors.get(name)
        for r in R:
            r["_dmed"] = abs(r["oxide_fit_nm"] - ref_med)
            r["_dpri"] = abs(r["oxide_fit_nm"] - ref_pri) if ref_pri else float("nan")
            r["_acq"] = name
        s = stats(R, "_dmed", a.report_at, a.fail_nm)
        per_acq.append(dict(acq=name, n=len(R), ref_med=ref_med, ref_pri=ref_pri,
                            failrate=100 * s["totfail"] / s["tot"], **s))
        pooled += R

    w = max(len(r["acq"]) for r in per_acq) + 1
    print(f"PER-ACQUISITION, gate = |fitted oxide - acquisition median| <= {a.report_at:g} nm")
    print(f"  {'acquisition':<{w}}{'n':>4}{'ref':>8}{'fail%':>7}{'kept':>6}"
          f"{'kept fail%':>12}{'rej':>5}{'rej fail%':>11}{'recall%':>9}")
    for r in sorted(per_acq, key=lambda x: -x["failrate"]):
        print(f"  {r['acq']:<{w}}{r['n']:>4}{r['ref_med']:>8.1f}{r['failrate']:>7.1f}"
              f"{r['kept']:>6}{r['keptrate']:>11.1f}%{r['dropped']:>5}{r['prec']:>10.1f}%"
              f"{r['recall']:>8.1f}%")

    print(f"\nPOOLED, n = {len(pooled)}   threshold sweep (median reference)")
    print(f"  {'gate':>7}{'kept':>7}{'kept fail%':>12}{'rejected':>10}{'precision':>11}{'recall':>9}")
    for t in thrs:
        s = stats(pooled, "_dmed", t, a.fail_nm)
        print(f"  {t:>5.0f}nm{s['kept']:>7}{s['keptrate']:>11.1f}%{s['dropped']:>10}"
              f"{s['prec']:>10.1f}%{s['recall']:>8.1f}%")

    if any(not math.isnan(r["_dpri"]) for r in pooled):
        sub = [r for r in pooled if not math.isnan(r["_dpri"])]
        print(f"\nPOOLED with the PRIOR as reference, n = {len(sub)}")
        print(f"  {'gate':>7}{'kept':>7}{'kept fail%':>12}{'rejected':>10}{'precision':>11}{'recall':>9}")
        for t in thrs:
            s = stats(sub, "_dpri", t, a.fail_nm)
            print(f"  {t:>5.0f}nm{s['kept']:>7}{s['keptrate']:>11.1f}%{s['dropped']:>10}"
                  f"{s['prec']:>10.1f}%{s['recall']:>8.1f}%")

    F = lambda r: abs(r["residual_nm"]) >= a.fail_nm
    tf = sum(1 for r in pooled if F(r))
    amb = [r for r in pooled if r["n_modes"] > 1]
    s = stats(pooled, "_dmed", a.report_at, a.fail_nm)
    print(f"\nAGAINST THE AMBIGUITY FLAG (n_modes > 1), same {len(pooled)} flakes")
    print(f"  total failures                {tf} ({100*tf/len(pooled):.1f}%)")
    print(f"  ambiguity flag fires on       {len(amb)}, catching {sum(1 for r in amb if F(r))}"
          f"  -> recall {100*sum(1 for r in amb if F(r))/tf if tf else 0:.1f}%")
    print(f"  oxide gate rejects            {s['dropped']}, catching {s['dropfail']}"
          f"  -> recall {s['recall']:.1f}%, precision {s['prec']:.1f}%")
    print(f"  flakes surviving the gate     {s['kept']}, of which {s['keptfail']} fail"
          f"  ({s['keptrate']:.1f}%)")

    print("\nCAUTION: the median reference assumes most flakes on an acquisition are")
    print("correct. Any acquisition above ~50% failure has a median that IS the ridge,")
    print("and its gate is meaningless. Read the fail% column before trusting a row.")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(per_acq[0]))
            wr.writeheader()
            wr.writerows(per_acq)
        print(f"\nwritten to {a.csv}")


if __name__ == "__main__":
    main()
