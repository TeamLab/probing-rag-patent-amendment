"""C5 — Template dependence metric.

Paper §5: "Frequency of the top-k canonical amendment phrases (mined
from the retrieval pool) in the generated text, normalized by claim
length. High value indicates reuse of boilerplate."

v0 operationalization (frozen 2026-04-18):
  - canonical phrases = top-50 4-grams by pool-frequency (from
    canonical_phrases_v1.json)
  - "generated text" = concatenated plain_body of all parsed claims in
    the response (markup stripped)
  - "claim length" = total char count of that concatenation
  - C5 = (sum of phrase occurrences) * 1000 / char_length
    — per-thousand-character rate

§5.6 differential refinement (v1, deferred): subtract TC-specific
baseline rate so ordinary TC phrasing registers near zero.

Usage:
    python3 scripts/compute_c5.py score \\
        --roots outputs/smoke outputs/integration_test \\
        --phrases data/parsed/canonical_phrases_v1.json \\
        --out outputs/_analysis/c5_scores.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from parse_response import parse as parse_response_text  # noqa: E402


WORD_RE = re.compile(r"[a-z][a-z\-']+")


def _tokens(text: str) -> list:
    return WORD_RE.findall((text or "").lower())


def _count_phrase(tokens: list, phrase_tokens: tuple) -> int:
    n = len(phrase_tokens)
    if n == 0 or len(tokens) < n:
        return 0
    hits = 0
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n]) == phrase_tokens:
            hits += 1
    return hits


def score_response(response_text: str, phrase_tokens_list: list) -> dict:
    parsed = parse_response_text(response_text or "")
    concat = "\n".join(p.plain_body for p in parsed if p.plain_body)
    if not concat:
        return {"c5": None, "char_len": 0, "n_claims": len(parsed),
                "total_hits": 0, "reason": "no_parsed_body"}
    toks = _tokens(concat)
    hits_total = 0
    per_phrase_counts = {}
    for pt in phrase_tokens_list:
        c = _count_phrase(toks, pt)
        if c:
            per_phrase_counts[" ".join(pt)] = c
        hits_total += c
    char_len = len(concat)
    c5 = (hits_total * 1000) / char_len if char_len else 0.0
    return {
        "c5": round(c5, 4),
        "char_len": char_len,
        "token_len": len(toks),
        "n_claims": len(parsed),
        "total_hits": hits_total,
        "top_phrases": sorted(per_phrase_counts.items(),
                               key=lambda x: -x[1])[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    aps = sub.add_parser("score")
    aps.add_argument("--roots", nargs="+", required=True, type=Path)
    aps.add_argument("--phrases", required=True, type=Path)
    aps.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ph = json.loads(args.phrases.read_text(encoding="utf-8"))
    phrase_tokens_list = [tuple(p["phrase"].split()) for p in ph["phrases"]]
    print(f"loaded {len(phrase_tokens_list)} canonical phrases")

    rows = []
    for root in args.roots:
        for f in sorted(root.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            s = score_response(d.get("response_text") or "", phrase_tokens_list)
            rows.append({
                "file": f.name,
                "case": d.get("case_id"),
                "cond": d.get("condition", "baseline"),
                "rep": d.get("rep", 0),
                **s,
            })

    cols = ["file", "case", "cond", "rep", "c5", "total_hits",
            "char_len", "token_len", "n_claims", "reason"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}")

    ok = [r for r in rows if r.get("c5") is not None]
    if ok:
        import statistics as st
        vals = sorted(r["c5"] for r in ok)
        print(f"C5 range: {vals[0]:.3f} .. {vals[-1]:.3f}  "
              f"median={st.median(vals):.3f}  mean={st.mean(vals):.3f}  n={len(ok)}")


if __name__ == "__main__":
    main()
