#!/usr/bin/env python3
"""k-sweep summary stats — one command, stdlib only (no venv/deps).

Reads the raw run outputs and the pre-computed channel TSVs and prints:
  A. Coverage         (files per k, completeness vs 2400)
  B. Token & cost     (input/output tokens, $ cost, latency — per k and per model)
  C. Channel means    (C1/C2/C3/C5, F|G, per model per k)
  D. Robustness       (swing across k per channel/model — the reviewer numbers)

Usage:
    python3 scripts/k_sweep_stats.py
    python3 scripts/k_sweep_stats.py --ks 1 3 5 10 --root outputs/k_sweep
"""
from __future__ import annotations
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict

MODELS = ["sonnet4", "haiku4.5", "gpt5.4", "gpt4o-mini"]
CHANNELS = ["c1", "c2", "c3", "c5"]
EXPECTED_PER_K = 2400  # 100 cases x 2 probes x 4 models x 3 reps


def _model_cond_from_name(fn: str):
    parts = os.path.basename(fn).replace(".json", "").split("__")  # case__cond__model__rep
    return (parts[2], parts[1]) if len(parts) >= 4 else (None, None)


def load_tsv_means(tsv: str, col: str):
    """(model,cond) -> mean of channel col."""
    vals = defaultdict(list)
    if not os.path.exists(tsv):
        return {}
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            m, c = _model_cond_from_name(row.get("file", ""))
            v = row.get(col, "")
            if not m or v in ("", "None", None):
                continue
            try:
                vals[(m, c)].append(float(v))
            except ValueError:
                pass
    return {k: st.mean(v) for k, v in vals.items() if v}


def token_stats(kdir: str):
    """Aggregate tokens/cost/latency from result JSONs. -> (overall, per_model)."""
    overall = {"n": 0, "in": 0, "out": 0, "cost": 0.0, "lat": []}
    per = defaultdict(lambda: {"n": 0, "in": 0, "out": 0, "cost": 0.0})
    for fn in glob.glob(os.path.join(kdir, "*__rep*.json")):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        u = d.get("usage") or {}
        m = d.get("model_alias", "?")
        it, ot = u.get("input_tokens", 0) or 0, u.get("output_tokens", 0) or 0
        cost = d.get("cost_usd", 0.0) or 0.0
        overall["n"] += 1; overall["in"] += it; overall["out"] += ot; overall["cost"] += cost
        if d.get("latency_sec"): overall["lat"].append(d["latency_sec"])
        p = per[m]; p["n"] += 1; p["in"] += it; p["out"] += ot; p["cost"] += cost
    return overall, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/k_sweep")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    a = ap.parse_args()
    KS = a.ks

    # ---- A. coverage ----
    print("=" * 68); print("A. COVERAGE")
    total = 0
    for k in KS:
        n = len(glob.glob(f"{a.root}/k{k}/*__rep*.json")); total += n
        print(f"  k={k:<3} {n}/{EXPECTED_PER_K} result files"
              f"{'  ✓' if n == EXPECTED_PER_K else '  ← incomplete'}")
    print(f"  TOTAL {total}/{EXPECTED_PER_K*len(KS)}")

    # ---- B. token & cost ----
    print("=" * 68); print("B. TOKEN & COST")
    print(f"  {'k':<4}{'calls':>7}{'in_tok':>12}{'out_tok':>11}{'cost$':>9}{'lat_med':>9}")
    gt = {"n": 0, "in": 0, "out": 0, "cost": 0.0}
    per_model_tot = defaultdict(lambda: {"n": 0, "in": 0, "out": 0, "cost": 0.0})
    for k in KS:
        o, per = token_stats(f"{a.root}/k{k}")
        med = f"{st.median(o['lat']):.1f}s" if o["lat"] else "–"
        print(f"  {k:<4}{o['n']:>7}{o['in']:>12,}{o['out']:>11,}{o['cost']:>9.2f}{med:>9}")
        for kk in ("n", "in", "out", "cost"):
            gt[kk] += o[kk]
        for m, p in per.items():
            for kk in ("n", "in", "out", "cost"):
                per_model_tot[m][kk] += p[kk]
    print(f"  {'ALL':<4}{gt['n']:>7}{gt['in']:>12,}{gt['out']:>11,}{gt['cost']:>9.2f}")
    print("  by model (all k):")
    for m in MODELS:
        p = per_model_tot.get(m, {})
        if p.get("n"):
            print(f"    {m:<11}{p['n']:>6} calls  {p['in']:>11,} in  {p['out']:>10,} out  ${p['cost']:.2f}")

    # ---- C. channel means ----
    means = {}  # (channel,k) -> {(model,cond): mean}
    for k in KS:
        for ch in CHANNELS:
            means[(ch, k)] = load_tsv_means(f"{a.root}/_scores/k{k}/{ch}.tsv", ch)
    print("=" * 68); print("C. CHANNEL MEANS  (F | G)")
    for ch in CHANNELS:
        print(f"\n  --- {ch.upper()} ---")
        print(f"  {'model':<11}" + "".join(f"k={k:<10}" for k in KS))
        for m in MODELS:
            cells = []
            for k in KS:
                d = means[(ch, k)]
                F, G = d.get((m, "F")), d.get((m, "G"))
                cells.append(f"{F:.3f}|{G:.3f}" if F is not None and G is not None else "   –   ")
            print(f"  {m:<11}" + "".join(f"{c:<12}" for c in cells))

    # ---- D. robustness: swing across k ----
    print("=" * 68); print("D. ROBUSTNESS — max swing across k (max-min of mean)")
    print("   small swing + non-monotonic = 'k not load-bearing'")
    for ch in CHANNELS:
        print(f"\n  --- {ch.upper()} ---")
        for m in MODELS:
            for cond in ("F", "G"):
                series = [means[(ch, k)].get((m, cond)) for k in KS]
                series = [x for x in series if x is not None]
                if len(series) < 2:
                    continue
                swing = max(series) - min(series)
                mono = (all(series[i] <= series[i+1] for i in range(len(series)-1))
                        or all(series[i] >= series[i+1] for i in range(len(series)-1)))
                tag = "monotonic" if mono else "non-monotonic (noise)"
                print(f"    {m:<11} {cond}  swing={swing:.3f}  [{', '.join(f'{x:.3f}' for x in series)}]  {tag}")


if __name__ == "__main__":
    main()
