"""Paired Wilcoxon signed-rank: baseline vs {F1,F5,G1,G5} (k-ablation),
per model, on C5 / C1(lexical) / C1(semantic) / C2 / C3 -- Holm-Bonferroni
within the new-test family (4 models x 4 headline metrics x 4 new
conditions = 64 tests; C3 reported but not part of the Holm family, same
convention as the main run treating C3 as non-headline). k=3 (F, G)
baseline comparisons are loaded from the EXISTING main-run tables (not
regenerated) and shown alongside for context, exactly as wilcoxon_denseG.py
showed baseline->F/G next to baseline->H.

STATUS: NOT YET RUNNABLE for the k=1/k=5 legs. Requires F1/F5/G1/G5 model
outputs, which have not been generated (no API credentials in the prep
environment -- see outputs/_analysis/k_ablation_run_config.json for cost
projection and the invocation once authorized). This script fails loudly
(exit 1, lists what's missing) rather than fabricate numbers, same
discipline as wilcoxon_denseG.py. It DOES run today for the k=3-only rows
(F, G vs baseline), since that data already exists -- see the
--k3-sanity-only flag.

Once F1/F5/G1/G5 have been generated and scored with the existing
deterministic pipeline, run with no arguments to get the full table:

  outputs/_analysis/c5_kablation_scores.tsv
  outputs/_analysis/c2_kablation_scores.tsv
  outputs/_analysis/c3_kablation_scores.tsv
  outputs/_analysis/c1_kablation_scores.tsv
  outputs/_analysis/c1_embedding_kablation_scores.tsv

each with the same file/case/cond/rep/<metric> column schema as the
main-run tables.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent  # experiment_run/
SUBMISSION = BASE.parent / "submission_repo"

MODELS = ["gpt5.4", "haiku4.5", "sonnet4", "gpt4o-mini"]
NEW_CONDS = ["F1", "F5", "G1", "G5"]
K_OF = {"F1": 1, "F5": 5, "G1": 1, "G5": 5, "F": 3, "G": 3}
BASE_PROBE_OF = {"F1": "F", "F5": "F", "G1": "G", "G5": "G", "F": "F", "G": "G"}

# (label, main-run existing-scores path, value col, kablation path, value col)
METRICS = [
    ("c5", SUBMISSION / "outputs/_analysis/c5_main.tsv", "c5",
           BASE / "outputs/_analysis/c5_kablation_scores.tsv", "c5"),
    ("c2", SUBMISSION / "outputs/_analysis/c2_main.tsv", "c2",
           BASE / "outputs/_analysis/c2_kablation_scores.tsv", "c2"),
    ("c3", SUBMISSION / "outputs/_analysis/c3_main.tsv", "c3",
           BASE / "outputs/_analysis/c3_kablation_scores.tsv", "c3"),
    ("c1_lexical", SUBMISSION / "outputs/_analysis/c1_main.tsv", "c1",
           BASE / "outputs/_analysis/c1_kablation_scores.tsv", "c1"),
    ("c1_semantic", BASE / "outputs/_analysis/c1_embedding_scores.tsv", "c1_embedding",
           BASE / "outputs/_analysis/c1_embedding_kablation_scores.tsv", "c1_embedding"),
]
HOLM_FAMILY_METRICS = {"c5", "c2", "c1_lexical", "c1_semantic"}  # C3 excluded, matches main-run non-headline treatment


def load(path, value_col):
    out = defaultdict(lambda: defaultdict(list))
    with path.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v = (r.get(value_col) or "").strip()
            if v in ("", "None"):
                continue
            model = r["file"].split("__")[2]
            case = r["case"]
            cond = r["cond"]
            try:
                val = float(v)
            except ValueError:
                continue
            out[(model, cond)][case].append(val)
    return out


def per_case_means(bucket):
    return {case: sum(vals) / len(vals) for case, vals in bucket.items()}


def bootstrap_ci_median(diffs, n_resamples=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    n = len(diffs)
    meds = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = diffs[rng.integers(0, n, n)]
        meds[i] = np.median(sample)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return lo, hi


def paired_test(base_means, probe_means):
    common = sorted(set(base_means) & set(probe_means))
    n = len(common)
    if n == 0:
        return None
    b = np.array([base_means[c] for c in common])
    p_ = np.array([probe_means[c] for c in common])
    diffs = p_ - b
    if np.all(diffs == 0):
        pval = 1.0
    else:
        _, pval = wilcoxon(p_, b, zero_method="wilcox", alternative="two-sided")
    lo, hi = bootstrap_ci_median(diffs)
    return dict(n=n, med_base=float(np.median(b)), med_probe=float(np.median(p_)),
                med_diff=float(np.median(diffs)), p=float(pval), ci=(float(lo), float(hi)))


def holm_correct(results, key_getter):
    """results: list of dicts each holding a test result under key_getter(r) -> result-dict-or-None."""
    testable = [(i, r) for i, r in enumerate(results) if key_getter(r) is not None]
    m = len(testable)
    if m == 0:
        return
    order = sorted(range(m), key=lambda k: key_getter(testable[k][1])["p"])
    running_max = 0.0
    holm_adj = [None] * m
    for rank, k in enumerate(order):
        p = key_getter(testable[k][1])["p"]
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        holm_adj[k] = running_max
    for k, (i, r) in enumerate(testable):
        key_getter(r)["p_holm"] = holm_adj[k]
    still = True
    for rank, k in enumerate(order):
        thresh = 0.05 / (m - rank)
        reject = still and key_getter(testable[k][1])["p"] <= thresh
        if not reject:
            still = False
        key_getter(testable[k][1])["holm_reject"] = reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k3-sanity-only", action="store_true",
                     help="run only the k=3 (F,G) vs baseline legs, which already have data -- "
                          "sanity-check the pipeline without needing new k=1/k=5 generations")
    args = ap.parse_args()

    if not args.k3_sanity_only:
        missing = [kp for _, _, _, kp, _ in METRICS if not kp.exists()]
        if missing:
            print("k-ablation score tables not found -- generation/scoring has not run yet:")
            for m in missing:
                print(f"  missing: {m}")
            print("\nSee outputs/_analysis/k_ablation_run_config.json for the invocation once "
                  "API credentials are authorized, or re-run with --k3-sanity-only to check the "
                  "k=3 (F,G) legs against existing main-run data only.")
            sys.exit(1)

    results = []  # each: dict(metric, model, cond, k, base_of, r=paired_test_result)
    for metric_label, main_path, main_col, kab_path, kab_col in METRICS:
        main_data = load(main_path, main_col)
        kab_data = load(kab_path, kab_col) if (not args.k3_sanity_only and kab_path.exists()) else {}
        for model in MODELS:
            base_means = per_case_means(main_data[(model, "baseline")])
            for cond in (["F", "G"] if args.k3_sanity_only else ["F", "G"] + NEW_CONDS):
                src = main_data if cond in ("F", "G") else kab_data
                probe_means = per_case_means(src[(model, cond)])
                r = paired_test(base_means, probe_means)
                results.append(dict(metric=metric_label, model=model, cond=cond,
                                     k=K_OF[cond], base_of=BASE_PROBE_OF[cond], r=r))

    # Holm-correct within the new-test family only (F1,F5,G1,G5), headline metrics only
    new_family = [r for r in results if r["cond"] in NEW_CONDS and r["metric"] in HOLM_FAMILY_METRICS]
    holm_correct(new_family, lambda r: r["r"])

    # ---- headline table: C5 (H1) and C2, all k, per model ----
    print("=" * 100)
    print("HEADLINE: C5 (H1 / template-dependence) and C2 (revision locality), baseline-vs-probe, by k")
    print("=" * 100)
    for metric in ["c5", "c2"]:
        print(f"\n--- {metric} ---")
        print(f"{'Model':<12}{'Probe':<8}{'k':<4}{'n':<5}{'p':<10}{'p_holm':<10}{'sig(Holm)':<11}{'med_diff':<12}{'95% CI'}")
        for model in MODELS:
            for cond in ["F1", "F", "F5", "G1", "G", "G5"]:
                row = next((r for r in results if r["metric"] == metric and r["model"] == model and r["cond"] == cond), None)
                if row is None or row["r"] is None:
                    print(f"{model:<12}{cond:<8}{K_OF[cond]:<4}{'--':<5}{'--':<10}{'--':<10}{'--':<11}{'--':<12}--")
                    continue
                r = row["r"]
                p_holm = r.get("p_holm")
                sig = ("YES" if r.get("holm_reject") else "no") if p_holm is not None else "n/a (k=3, not in Holm family)"
                ci_str = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
                p_holm_str = f"{p_holm:.4f}" if p_holm is not None else "--"
                print(f"{model:<12}{cond:<8}{K_OF[cond]:<4}{r['n']:<5}{r['p']:<10.4f}{p_holm_str:<10}{sig:<11}{r['med_diff']:<+12.4f}{ci_str}")

    # ---- monotonicity check: does |effect| trend with k? ----
    print("\n" + "=" * 100)
    print("MONOTONICITY: median-diff at k=1 -> k=3 -> k=5, per model/probe/metric")
    print("=" * 100)
    for metric in ["c5", "c2", "c3", "c1_lexical", "c1_semantic"]:
        for model in MODELS:
            for base_probe in ["F", "G"]:
                conds_in_order = [f"{base_probe}1", base_probe, f"{base_probe}5"]
                rows = [next((r for r in results if r["metric"] == metric and r["model"] == model and r["cond"] == c), None)
                        for c in conds_in_order]
                if any(row is None or row["r"] is None for row in rows):
                    continue
                diffs = [row["r"]["med_diff"] for row in rows]
                mono_up = diffs[0] <= diffs[1] <= diffs[2]
                mono_down = diffs[0] >= diffs[1] >= diffs[2]
                tag = "monotonic-up" if mono_up and diffs[0] != diffs[2] else \
                      "monotonic-down" if mono_down and diffs[0] != diffs[2] else \
                      "flat" if diffs[0] == diffs[1] == diffs[2] else "non-monotonic"
                print(f"{metric:<14}{model:<12}{base_probe:<4}k1={diffs[0]:+.4f}  k3={diffs[1]:+.4f}  k5={diffs[2]:+.4f}   [{tag}]")

    # ---- newly-significant-at-k1-or-k5-but-not-k3 flag ----
    print("\n" + "=" * 100)
    print("NEW SIGNIFICANCE CHECK: Holm-significant at k=1 or k=5 (new family) where k=3 was not "
          "significant in the main run (uncorrected p<0.05 there, since k=3 was never Holm-tested "
          "for these metrics per the paper's own stated limitation)")
    print("=" * 100)
    flagged = []
    for r in new_family:
        if r["r"] and r["r"].get("holm_reject"):
            k3_row = next((x for x in results if x["metric"] == r["metric"] and x["model"] == r["model"] and x["cond"] == r["base_of"]), None)
            k3_p = k3_row["r"]["p"] if (k3_row and k3_row["r"]) else None
            flagged.append((r["metric"], r["model"], r["cond"], r["r"]["p_holm"], k3_p))
    if flagged:
        for metric, model, cond, p_holm, k3_p in flagged:
            k3_p_str = f"{k3_p:.4f}" if k3_p is not None else "n/a"
            print(f"  {metric} / {model} / {cond} (k={K_OF[cond]}): Holm p={p_holm:.4f} SIGNIFICANT.  "
                  f"k=3 uncorrected p was {k3_p_str}.")
    else:
        print("  none -- no metric/model/probe becomes Holm-significant at k=1 or k=5 that wasn't already flagged at k=3.")

    print(f"\n(Holm family size: {len(new_family)} tests -- {len(NEW_CONDS)} new conditions x "
          f"{len(HOLM_FAMILY_METRICS)} headline metrics x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
