"""Summarize smoke outputs to files (avoid dumping model text to stdout).

Writes:
  outputs/smoke/_analysis/summary.tsv      — per-case length/cost/latency/stop
  outputs/smoke/_analysis/axes.tsv         — per-case axes
  outputs/smoke/_analysis/overall.json     — aggregate stats
  outputs/smoke/_analysis/by_case/<id>.txt — raw response_text, one per case

Run:  python3 scripts/analyze_smoke.py
Then: Read the files above instead of printing to the chat.
"""

import json
import glob
import os
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "outputs" / "smoke"
OUT = SMOKE_DIR / "_analysis"
(OUT / "by_case").mkdir(parents=True, exist_ok=True)


def load_cases():
    rows = []
    for p in sorted(glob.glob(str(SMOKE_DIR / "*.json"))):
        with open(p) as f:
            d = json.load(f)
        rows.append(d)
    return rows


def main():
    rows = load_cases()
    if not rows:
        print(f"No json found in {SMOKE_DIR}")
        return

    with open(OUT / "summary.tsv", "w") as f:
        f.write("case_id\tin_tok\tout_tok\tstop\tlat_sec\tcost_usd\tresp_chars\tuser_chars\n")
        for d in rows:
            u = d.get("usage", {}) or {}
            f.write(
                f"{d['case_id']}\t{u.get('input_tokens','')}\t{u.get('output_tokens','')}\t"
                f"{d.get('stop_reason','')}\t{d.get('latency_sec',''):.2f}\t"
                f"{d.get('cost_usd',0):.5f}\t{len(d.get('response_text',''))}\t{d.get('user_chars','')}\n"
            )

    with open(OUT / "axes.tsv", "w") as f:
        axes_keys = sorted({k for d in rows for k in (d.get("axes") or {}).keys()})
        f.write("case_id\t" + "\t".join(axes_keys) + "\n")
        for d in rows:
            a = d.get("axes") or {}
            f.write(d["case_id"] + "\t" + "\t".join(str(a.get(k, "")) for k in axes_keys) + "\n")

    def safe(xs):
        xs = [x for x in xs if x is not None]
        return {
            "n": len(xs),
            "min": min(xs) if xs else None,
            "max": max(xs) if xs else None,
            "mean": stats.mean(xs) if xs else None,
            "median": stats.median(xs) if xs else None,
        }

    overall = {
        "n_cases": len(rows),
        "model": rows[0].get("model"),
        "model_alias": rows[0].get("model_alias"),
        "temperature": rows[0].get("temperature"),
        "max_tokens": rows[0].get("max_tokens"),
        "response_chars": safe([len(d.get("response_text", "")) for d in rows]),
        "output_tokens": safe([(d.get("usage") or {}).get("output_tokens") for d in rows]),
        "input_tokens": safe([(d.get("usage") or {}).get("input_tokens") for d in rows]),
        "latency_sec": safe([d.get("latency_sec") for d in rows]),
        "cost_usd": safe([d.get("cost_usd") for d in rows]),
        "stop_reasons": {},
    }
    for d in rows:
        s = d.get("stop_reason", "unknown")
        overall["stop_reasons"][s] = overall["stop_reasons"].get(s, 0) + 1
    overall["total_cost_usd"] = sum(d.get("cost_usd", 0) for d in rows)

    with open(OUT / "overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    for d in rows:
        (OUT / "by_case" / f"{d['case_id']}.txt").write_text(d.get("response_text", ""))

    print(f"wrote: {OUT}/summary.tsv")
    print(f"wrote: {OUT}/axes.tsv")
    print(f"wrote: {OUT}/overall.json")
    print(f"wrote: {OUT}/by_case/*.txt  ({len(rows)} files)")


if __name__ == "__main__":
    main()
