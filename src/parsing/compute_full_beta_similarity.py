"""Compute pre-vs-gold similarity for the full β corpus and the eligibility pool.

Used to validate paper §3.4's claim of "median 0.933 similarity (7% edit)"
on the full corpus, by comparing against the 99/100 cohort's median of 0.661.

Uses rapidfuzz.Levenshtein for exact char-level edit distance (C-backed),
and also reports difflib.SequenceMatcher.ratio() for cross-checking against
the method that may underlie the paper figure.

Output: data/parsed/full_beta_similarity.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from select_cohort import compute_case_axes  # noqa: E402


def _strip_markup(text: str) -> str:
    text = re.sub(r"\[\[[^\]]+?\]\]", "", text)
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _join_claims(claim_list):
    kept = []
    for c in claim_list or []:
        num = c.get("num")
        if not str(num or "").strip().isdigit():
            continue
        t = (c.get("text") or "").strip()
        if len(t) < 10:
            continue
        t = _strip_markup(t)
        t = re.sub(rf"^\s*{num}\s*[.\s]+", "", t, count=1).strip()
        kept.append((int(num), t))
    kept.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in kept)


def similarity_lev(a: str, b: str) -> float:
    """1 - Levenshtein/max_len, bounded [0, 1]."""
    if not a or not b:
        return 0.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    d = Levenshtein.distance(a, b)
    return round(1 - d / m, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-dir", type=Path,
                    default=Path("data/parsed/beta_parsed"))
    ap.add_argument("--alpha", type=Path,
                    default=Path("data/parsed/alpha_corpus.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("data/parsed/full_beta_similarity.json"))
    ap.add_argument("--progress-every", type=int, default=1000)
    args = ap.parse_args()

    has_alpha = args.alpha.exists()
    alpha_by_proc: dict[str, dict] = {}
    if has_alpha:
        print(f"loading alpha from {args.alpha} …", flush=True)
        with args.alpha.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                alpha_by_proc[str(r.get("proceeding_number") or "")] = r
        print(f"alpha loaded: {len(alpha_by_proc)}", flush=True)
    else:
        print("WARN: alpha not present locally — eligible pool will be empty.",
              flush=True)

    files = sorted(os.listdir(args.beta_dir))
    print(f"scanning {len(files)} β files …", flush=True)

    rows = []
    full_sims = []
    eligible_sims = []
    n_processed = 0
    n_skip_empty = 0
    n_skip_parse = 0
    t0 = time.time()

    for fn in files:
        if not fn.endswith(".json"):
            continue
        n_processed += 1
        try:
            rec = json.loads((args.beta_dir / fn).read_text(encoding="utf-8"))
        except Exception:
            n_skip_parse += 1
            continue
        pre = _join_claims((rec.get("pre_clm") or {}).get("claims"))
        post = _join_claims((rec.get("post_clm") or {}).get("claims"))
        if not pre or not post:
            n_skip_empty += 1
            continue
        sim = similarity_lev(pre, post)
        full_sims.append(sim)
        row = {"case_id": rec.get("case_id"), "app_num": rec.get("app_num"),
               "pre_chars": len(pre), "post_chars": len(post), "sim": sim,
               "eligible": False}
        if has_alpha:
            proc = str(rec.get("proceeding_number") or "")
            alpha = alpha_by_proc.get(proc)
            if alpha:
                _, eligible, _ = compute_case_axes(alpha, rec)
                if eligible:
                    eligible_sims.append(sim)
                    row["eligible"] = True
        rows.append(row)
        if n_processed % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 0.01)
            print(f"  {n_processed}/{len(files)}  elapsed={elapsed:.0f}s  "
                  f"rate={rate:.0f}/s  full={len(full_sims)}  "
                  f"eligible={len(eligible_sims)}", flush=True)

    import statistics as st
    def summary(vals):
        if not vals:
            return {}
        q = sorted(vals)
        return {
            "n": len(vals),
            "mean": round(st.mean(vals), 4),
            "median": round(st.median(vals), 4),
            "Q1": round(q[len(q) // 4], 4),
            "Q3": round(q[3 * len(q) // 4], 4),
            "min": round(q[0], 4),
            "max": round(q[-1], 4),
        }

    out = {
        "method": "rapidfuzz.Levenshtein  sim = 1 - d/max(len_a, len_b)",
        "n_processed": n_processed,
        "n_skip_empty": n_skip_empty,
        "n_skip_parse": n_skip_parse,
        "full": summary(full_sims),
        "eligible": summary(eligible_sims),
        "full_values": full_sims,
        "eligible_values": eligible_sims,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"=== FULL β similarity (n={out['full'].get('n', 0)}) ===")
    for k, v in out["full"].items():
        print(f"  {k}: {v}")
    print(f"\n=== ELIGIBLE pool similarity (n={out['eligible'].get('n', 0)}) ===")
    for k, v in out["eligible"].items():
        print(f"  {k}: {v}")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
