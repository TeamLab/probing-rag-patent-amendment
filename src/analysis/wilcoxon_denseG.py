"""Paired Wilcoxon signed-rank: baseline vs dense-G (H), per model, on
C5 / C1(lexical) / C1(semantic) / C2 — Holm-Bonferroni within this family
(4 models x 4 metrics = 16 tests). Also prints baseline-vs-F and
baseline-vs-G from the existing scored tables alongside, for a single
side-by-side table (dense-G's own Holm correction is computed only across
the 16 new dense-G tests, not re-mixed with the earlier F/G families).

STATUS: NOT YET RUNNABLE. Requires dense-G (condition H) model outputs,
which have not been generated (no API credentials in the prep environment;
see dense_G_run_config.json for cost projection and the invocation once
authorized). This script expects, once H has been generated and scored
with the existing deterministic pipeline:

  outputs/_analysis/c5_denseG_scores.tsv          (compute_c5.py over outputs/<model>_denseG/)
  outputs/_analysis/c2_denseG_scores.tsv          (compute_c2.py)
  outputs/_analysis/c1_denseG_scores.tsv          (compute_c1.py, lexical)
  outputs/_analysis/c1_embedding_denseG_scores.tsv (compute_c1_embedding.py, semantic)

each with the same file/case/cond/rep/<metric> column schema as the main-run
tables. Run with no arguments once those exist.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent  # experiment_run/
SUBMISSION = BASE.parent / "submission_repo"

MODELS = ["gpt5.4", "haiku4.5", "sonnet4", "gpt4o-mini"]

# (label, main-run existing-scores path, value col, denseG path, value col)
METRICS = [
    ("c5", SUBMISSION / "outputs/_analysis/c5_main.tsv", "c5",
           BASE / "outputs/_analysis/c5_denseG_scores.tsv", "c5"),
    ("c2", SUBMISSION / "outputs/_analysis/c2_main.tsv", "c2",
           BASE / "outputs/_analysis/c2_denseG_scores.tsv", "c2"),
    ("c1_lexical", SUBMISSION / "outputs/_analysis/c1_main.tsv", "c1",
           BASE / "outputs/_analysis/c1_denseG_scores.tsv", "c1"),
    ("c1_semantic", BASE / "outputs/_analysis/c1_embedding_scores.tsv", "c1_embedding",
           BASE / "outputs/_analysis/c1_embedding_denseG_scores.tsv", "c1_embedding"),
]


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


missing = [p for _, mainp, _, dgp, _ in METRICS for p in (dgp,) if not p.exists()]
if missing:
    print("dense-G score tables not found — generation/scoring has not run yet:")
    for m in missing:
        print(f"  missing: {m}")
    print("\nSee dense_G_run_config.json for the invocation once API credentials are authorized.")
    sys.exit(1)

results = []
for metric_label, main_path, main_col, dg_path, dg_col in METRICS:
    main_data = load(main_path, main_col)
    dg_data = load(dg_path, dg_col)
    for model in MODELS:
        base_means = per_case_means(main_data[(model, "baseline")])
        f_means = per_case_means(main_data[(model, "F")])
        g_means = per_case_means(main_data[(model, "G")])
        h_means = per_case_means(dg_data[(model, "H")])
        r_f = paired_test(base_means, f_means)
        r_g = paired_test(base_means, g_means)
        r_h = paired_test(base_means, h_means)
        results.append(dict(metric=metric_label, model=model, f=r_f, g=r_g, h=r_h))

# Holm-Bonferroni across the 16 new baseline-vs-H tests only
testable = [(i, r) for i, r in enumerate(results) if r["h"] is not None]
m = len(testable)
order = sorted(range(len(testable)), key=lambda k: testable[k][1]["h"]["p"])
holm_adj = [None] * len(testable)
running_max = 0.0
for rank, k in enumerate(order):
    p = testable[k][1]["h"]["p"]
    adj = min(1.0, (m - rank) * p)
    running_max = max(running_max, adj)
    holm_adj[k] = running_max
for k, (i, r) in enumerate(testable):
    r["h"]["p_holm"] = holm_adj[k]

still = True
reject = [False] * len(testable)
for rank, k in enumerate(order):
    thresh = 0.05 / (m - rank)
    if still and testable[k][1]["h"]["p"] <= thresh:
        reject[k] = True
    else:
        still = False
for k, (i, r) in enumerate(testable):
    r["h"]["holm_reject"] = reject[k]

print(f"{'Metric':<12}{'Model':<12}{'baseline->F p':<15}{'baseline->G p':<15}{'baseline->H p':<15}{'H Holm p':<12}{'H sig':<8}{'H med diff':<12}{'H 95% CI'}")
flagged_c5_increase = []
for r in results:
    h = r["h"]
    if h is None:
        continue
    f_p = f"{r['f']['p']:.4f}" if r["f"] else "--"
    g_p = f"{r['g']['p']:.4f}" if r["g"] else "--"
    holm_sig = "YES" if h.get("holm_reject") else "no"
    ci_str = f"[{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"
    print(f"{r['metric']:<12}{r['model']:<12}{f_p:<15}{g_p:<15}{h['p']:<15.4f}{h['p_holm']:<12.4f}{holm_sig:<8}{h['med_diff']:<+12.4f}{ci_str}")
    if r["metric"] == "c5" and h["med_diff"] > 0 and h.get("holm_reject"):
        flagged_c5_increase.append(r["model"])

print(f"\nModels with a Holm-significant C5 INCREASE under dense-G: {flagged_c5_increase or 'none'}")
