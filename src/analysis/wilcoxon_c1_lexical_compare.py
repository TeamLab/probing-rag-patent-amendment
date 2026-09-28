"""Paired Wilcoxon signed-rank: baseline vs F, baseline vs G, on lexical (trigram) C1.

Same method as wilcoxon_c1_embedding.py, run against the frozen trigram-Jaccard
c1_main.tsv instead of c1_embedding_scores.tsv, for direct side-by-side comparison
(see stats_note_c1_embedding_reanalysis.md, "Semantic vs lexical C1" section).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PATH = Path("/Users/josephamichikoleo/Documents/Claude/Projects/Probing Retrieval-Augmented Patent Claim Amendment/submission_repo/outputs/_analysis/c1_main.tsv")
VALUE_COL = "c1"
LABEL = "Lexical C1 (trigram Jaccard)"

MODELS = ["gpt5.4", "haiku4.5", "sonnet4", "gpt4o-mini"]
COMPARISONS = [("baseline", "F"), ("baseline", "G")]


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


data = load(PATH, VALUE_COL)
results = []
for model in MODELS:
    base_means = per_case_means(data[(model, "baseline")])
    for _, probe in COMPARISONS:
        probe_means = per_case_means(data[(model, probe)])
        common = sorted(set(base_means) & set(probe_means))
        n = len(common)
        if n == 0:
            results.append(dict(model=model, probe=probe, n=0, med_base=None,
                                 med_probe=None, med_diff=None, p=None, ci=None))
            continue
        b = np.array([base_means[c] for c in common])
        p_ = np.array([probe_means[c] for c in common])
        diffs = p_ - b
        med_base = float(np.median(b))
        med_probe = float(np.median(p_))
        med_diff = float(np.median(diffs))
        if np.all(diffs == 0):
            pval = 1.0
        else:
            stat, pval = wilcoxon(p_, b, zero_method="wilcox", alternative="two-sided")
        lo, hi = bootstrap_ci_median(diffs)
        results.append(dict(model=model, probe=probe, n=n, med_base=med_base,
                             med_probe=med_probe, med_diff=med_diff, p=float(pval),
                             ci=(float(lo), float(hi))))

# ---- Holm-Bonferroni across these 8 tests ----
ALPHA = 0.05
testable = [r for r in results if r["p"] is not None]
order = sorted(range(len(testable)), key=lambda i: testable[i]["p"])
m = len(testable)
holm_adj = [None] * m
running_max = 0.0
for rank, idx in enumerate(order):
    p = testable[idx]["p"]
    adj = min(1.0, (m - rank) * p)
    running_max = max(running_max, adj)
    holm_adj[idx] = running_max
for i, r in enumerate(testable):
    r["p_holm"] = holm_adj[i]

reject_holm = [False] * m
still_rejecting = True
for rank, idx in enumerate(order):
    threshold = ALPHA / (m - rank)
    p = testable[idx]["p"]
    if still_rejecting and p <= threshold:
        reject_holm[idx] = True
    else:
        still_rejecting = False
for i, r in enumerate(testable):
    r["holm_reject"] = reject_holm[i]

print(f"=== {LABEL} — baseline vs F, baseline vs G ===")
print(f"{'Model':<12}{'Probe':<7}{'n':<5}{'med base':<10}{'med probe':<10}{'med diff':<10}{'raw p':<10}{'Holm p (m=8)':<14}{'Holm sig':<9}{'95% CI (diff)'}")
for r in results:
    if r["n"] == 0:
        print(f"{r['model']:<12}{r['probe']:<7}{'0':<5}{'--':<10}{'--':<10}{'--':<10}{'--':<10}{'--':<14}{'--':<9}{'--'}")
        continue
    ci_str = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
    raw_sig = "*" if r["p"] < 0.05 else ""
    holm_sig = "YES" if r.get("holm_reject") else "no"
    print(f"{r['model']:<12}{r['probe']:<7}{r['n']:<5}{r['med_base']:<10.4f}{r['med_probe']:<10.4f}{r['med_diff']:<+10.4f}{r['p']:<9.4f}{raw_sig:<1}{r['p_holm']:<14.4f}{holm_sig:<9}{ci_str}")

sig_count = sum(1 for r in results if r["p"] is not None and r["p"] < 0.05)
holm_count = sum(1 for r in results if r.get("holm_reject"))
total_count = sum(1 for r in results if r["p"] is not None)
print(f"\nUncorrected significant at alpha=0.05: {sig_count} / {total_count}")
print(f"Holm-Bonferroni significant at family alpha=0.05 (m={m}, this family only): {holm_count} / {total_count}")
