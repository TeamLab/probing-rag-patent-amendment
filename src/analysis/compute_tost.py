#!/usr/bin/env python3
"""
Median-based TOST equivalence test for the C5 prior-injection null (Appendix O).

For each model and each retrieval probe (F = random, G = structural), we pair the
per-case C5 score against the no-retrieval baseline, take the paired difference
(probe - baseline), and test whether the *median* paired difference is equivalent
to zero within the pre-registered +/-0.2 margin using two one-sided tests (TOST).

The test is median-based and computed by bootstrap (10,000 resamples, seed 42),
matching the resampling scheme already used for the paper's confidence intervals:

  lower one-sided:  H0 median <= -margin   vs   H1 median > -margin
  upper one-sided:  H0 median >= +margin   vs   H1 median < +margin
  TOST p = max(p_lower, p_upper)

Equivalence to zero within the margin is declared at alpha = 0.05 when BOTH
one-sided tests reject, equivalently when the 90% bootstrap CI (1 - 2*alpha) on
the median paired difference lies entirely inside [-margin, +margin].

This script recomputes a statistic FROM the released per-case C5 outputs. It does
not modify any model output, score, or other result.

Usage:
  python3 compute_tost.py \
      --c5 outputs/_analysis/c5_main.tsv \
      --margin 0.2 --alpha 0.05 --n-boot 10000 --seed 42
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


PROBES = ["F", "G"]           # retrieval conditions tested for prior injection
BASELINE = "baseline"


def load_percase_c5(path):
    """Return {(model, cond): {case: mean_c5_over_reps}}."""
    buckets = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            model = row["file"].split("__")[2]
            cond = row["cond"]
            case = row["case"]
            try:
                c5 = float(row["c5"])
            except (ValueError, TypeError):
                continue
            buckets[(model, cond)][case].append(c5)
    percase = {}
    for key, casemap in buckets.items():
        percase[key] = {c: float(np.mean(v)) for c, v in casemap.items()}
    return percase


def paired_diffs(percase, model, probe):
    """probe - baseline paired over the cases present in both conditions."""
    base = percase.get((model, BASELINE), {})
    prb = percase.get((model, probe), {})
    cases = sorted(set(base) & set(prb))
    return np.array([prb[c] - base[c] for c in cases]), cases


def bootstrap_median(diffs, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.median(diffs[idx], axis=1)


def tost_median(diffs, margin, alpha, n_boot, seed):
    obs_median = float(np.median(diffs))
    boot = bootstrap_median(diffs, n_boot, seed)
    p_lower = float(np.mean(boot <= -margin))   # H1: median > -margin
    p_upper = float(np.mean(boot >= margin))    # H1: median < +margin
    tost_p = max(p_lower, p_upper)
    lo90, hi90 = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    equivalent = (tost_p < alpha) and (lo90 >= -margin) and (hi90 <= margin)
    return {
        "n": len(diffs),
        "median": obs_median,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "tost_p": tost_p,
        "ci90": (float(lo90), float(hi90)),
        "ci95": (float(lo95), float(hi95)),
        "equivalent": bool(equivalent),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c5", required=True,
                    help="Path to c5_main.tsv (per-case, per-rep C5 scores).")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    percase = load_percase_c5(args.c5)
    models = sorted({m for (m, _c) in percase})

    print(f"Median-based bootstrap TOST on Delta-C5 (probe - baseline)")
    print(f"margin=+/-{args.margin}  alpha={args.alpha}  "
          f"n_boot={args.n_boot}  seed={args.seed}\n")
    hdr = (f"{'model':<12} {'probe':<6} {'n':>4} {'medianDC5':>10} "
           f"{'p_TOST':>8} {'90% CI':>20} {'equiv?':>7}")
    print(hdr)
    print("-" * len(hdr))

    all_equiv = True
    for model in models:
        for probe in PROBES:
            diffs, _cases = paired_diffs(percase, model, probe)
            if len(diffs) == 0:
                continue
            r = tost_median(diffs, args.margin, args.alpha,
                            args.n_boot, args.seed)
            all_equiv &= r["equivalent"]
            ci = f"[{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]"
            print(f"{model:<12} {probe:<6} {r['n']:>4} {r['median']:>+10.4f} "
                  f"{r['tost_p']:>8.4f} {ci:>20} "
                  f"{'YES' if r['equivalent'] else 'NO':>7}")

    print("\n" + ("All 8 cells: equivalence to zero within margin ESTABLISHED."
                  if all_equiv else
                  "WARNING: at least one cell did NOT reach equivalence."))


if __name__ == "__main__":
    main()
