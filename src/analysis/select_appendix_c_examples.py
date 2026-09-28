"""Select pass/fail worked examples for Appendix C, per channel.

Goal: for each of C1/C2/C3/C5, pick one illustrative "pass" case and
one illustrative "fail" case, stay within structural data we already
have (TSVs + α/β parsed JSON), do NOT read raw LLM response text in
full — only structural features and previously-extracted spans.

Heuristics:
- Prefer cases present in the 100-case cohort (so scores are from
  main experiment).
- Prefer within-case contrast: baseline vs one retrieval condition,
  same model, so the channel discriminates two conditions on the
  same input.
- When within-case contrast not available, fall back to cross-case
  extremes.

Reads:
  outputs/_analysis/c1_main.tsv
  outputs/_analysis/c2_main.tsv
  outputs/_analysis/c3_main.tsv
  outputs/_analysis/c5_main.tsv
  data/parsed/cohort_batch0_v1.json
  data/parsed/beta_parsed/*.json  (structural only)

Writes:
  docs/appendix_c_candidates.json
  docs/appendix_c_candidates.md  (human-readable candidate list)
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "_analysis"
OUT_JSON = ROOT / "docs" / "appendix_c_candidates.json"
OUT_MD = ROOT / "docs" / "appendix_c_candidates.md"

MODELS = ["sonnet4", "haiku4.5", "gpt5.4", "gpt4o-mini"]


def parse_model(fname: str) -> str:
    parts = fname.rsplit("__", 3)
    return parts[-2] if len(parts) >= 4 else ""


def load_tsv(name: str, value_col: str) -> list[dict]:
    rows = []
    with (SRC / name).open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get(value_col) in (None, ""):
                continue
            try:
                v = float(r[value_col])
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            model = parse_model(r["file"])
            if model not in MODELS:
                continue
            rec = dict(r)
            rec["model"] = model
            rec["value"] = v
            rows.append(rec)
    return rows


def avg_over_reps(rows: list[dict], case: str, cond: str, model: str) -> float | None:
    vs = [r["value"] for r in rows
          if r["case"] == case and r["cond"] == cond and r["model"] == model]
    return statistics.mean(vs) if vs else None


def pick_pass_fail_same_case(rows: list[dict], pass_hi: bool = True) -> dict:
    """Find a (case, model) where baseline and some perturbation give the
    largest spread of the channel value. Returns {"case", "model",
    "high_cond", "low_cond", "high_val", "low_val"} with high_val >= low_val.

    pass_hi=True means "higher value is the pass direction".
    """
    # index by (case, model) -> {cond: mean across reps}
    index: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    for r in rows:
        key = (r["case"], r["model"])
        index[key].setdefault(r["cond"], []).append(r["value"])
    # reduce to mean
    reduced: dict[tuple[str, str], dict[str, float]] = {}
    for key, conds in index.items():
        reduced[key] = {c: statistics.mean(vs) for c, vs in conds.items()}
    # find max spread where baseline and another condition both present
    best = None
    for (case, model), conds in reduced.items():
        if "baseline" not in conds:
            continue
        base = conds["baseline"]
        for c, v in conds.items():
            if c == "baseline":
                continue
            # We want: one value "good", one "bad". The spread picks the
            # most discriminative example.
            spread = abs(v - base)
            rec = (spread, case, model, c, v, base)
            if best is None or spread > best[0]:
                best = rec
    if best is None:
        return {}
    spread, case, model, c, v, base = best
    # order so high is pass direction
    high_cond, high_val, low_cond, low_val = (
        (c, v, "baseline", base) if v > base else ("baseline", base, c, v)
    )
    if not pass_hi:
        high_cond, low_cond = low_cond, high_cond
        high_val, low_val = low_val, high_val
    return {
        "case": case,
        "model": model,
        "high_cond": high_cond,
        "low_cond": low_cond,
        "high_val": high_val,
        "low_val": low_val,
        "spread": spread,
    }


def pick_cross_case_extremes(rows: list[dict], pass_hi: bool = True) -> dict:
    """Pick top-scoring and bottom-scoring (case, cond, model) triples."""
    # rep-mean per (case, cond, model)
    idx: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        idx[(r["case"], r["cond"], r["model"])].append(r["value"])
    entries = [(k, statistics.mean(v)) for k, v in idx.items()]
    entries.sort(key=lambda kv: kv[1], reverse=True)
    top = entries[0]
    bot = entries[-1]
    pass_entry, fail_entry = (top, bot) if pass_hi else (bot, top)
    return {
        "pass": {"case": pass_entry[0][0], "cond": pass_entry[0][1],
                 "model": pass_entry[0][2], "val": pass_entry[1]},
        "fail": {"case": fail_entry[0][0], "cond": fail_entry[0][1],
                 "model": fail_entry[0][2], "val": fail_entry[1]},
    }


def main():
    c1 = load_tsv("c1_main.tsv", "c1")
    c2 = load_tsv("c2_main.tsv", "c2")
    c3 = load_tsv("c3_main.tsv", "c3")
    c5 = load_tsv("c5_main.tsv", "c5")

    # Higher C1/C3 = pass; C2 close to 1 = pass (we'll rank by |C2-1|);
    # Lower C5 = pass.
    result = {}
    result["c1"] = {
        "same_case": pick_pass_fail_same_case(c1, pass_hi=True),
        "extremes": pick_cross_case_extremes(c1, pass_hi=True),
    }

    # For C2, we want closeness to 1. Transform value -> -|value - 1|,
    # high transformed = pass.
    c2_transformed = [
        {**r, "value": -abs(r["value"] - 1.0)} for r in c2
    ]
    result["c2"] = {
        "same_case": pick_pass_fail_same_case(c2_transformed, pass_hi=True),
        "extremes": pick_cross_case_extremes(c2_transformed, pass_hi=True),
    }

    result["c3"] = {
        "same_case": pick_pass_fail_same_case(c3, pass_hi=True),
        "extremes": pick_cross_case_extremes(c3, pass_hi=True),
    }
    result["c5"] = {
        "same_case": pick_pass_fail_same_case(c5, pass_hi=False),
        "extremes": pick_cross_case_extremes(c5, pass_hi=False),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    # human-readable digest
    lines = ["# Appendix C candidate selection\n"]
    for ch in ["c1", "c2", "c3", "c5"]:
        lines.append(f"## {ch.upper()}\n")
        sc = result[ch]["same_case"]
        ex = result[ch]["extremes"]
        if sc:
            lines.append(f"**Same-case contrast** (spread = {sc['spread']:.4f}):")
            lines.append(f"- case `{sc['case']}`, model `{sc['model']}`")
            lines.append(f"- high: `{sc['high_cond']}` value={sc['high_val']:.4f}")
            lines.append(f"- low:  `{sc['low_cond']}` value={sc['low_val']:.4f}\n")
        lines.append("**Cross-case extremes**:")
        lines.append(f"- pass: `{ex['pass']['case']}` cond `{ex['pass']['cond']}` model `{ex['pass']['model']}` val={ex['pass']['val']:.4f}")
        lines.append(f"- fail: `{ex['fail']['case']}` cond `{ex['fail']['cond']}` model `{ex['fail']['model']}` val={ex['fail']['val']:.4f}\n")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
