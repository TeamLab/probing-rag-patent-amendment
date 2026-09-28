"""Build the retrieval-pool index for Probes F (random) and G (structural).

Pool composition (paper §6.1 line 186 verbatim):
  pool = β corpus \\ cohort_batch0_v1

Per-case entry stores the *attacked claim* pre + post text and the three
features used by Probe G's deterministic matcher (statute section, statute
subsection, limitation-pattern indicators). Entries where these cannot be
extracted are skipped; skip reasons counted and reported.

Invariant asserted at build time: pool case_ids ∩ cohort case_ids = ∅.

Output: data/parsed/retrieval_pool_v1.json

Usage:
  python3 scripts/build_retrieval_pool.py \\
      --beta-parsed-dir data/parsed/beta_parsed \\
      --cohort data/parsed/cohort_batch0_v1.json \\
      --out data/parsed/retrieval_pool_v1.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# ---------- claim text utilities ------------------------------------------

def _strip_markup(text: str) -> str:
    text = re.sub(r"\[\[[^\]]+?\]\]", "", text or "")
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _claims_map(claim_list) -> dict:
    out: dict[int, str] = {}
    for c in claim_list or []:
        num = c.get("num")
        if not str(num or "").strip().isdigit():
            continue
        t = (c.get("text") or "").strip()
        if len(t) < 10:
            continue
        t = _strip_markup(t)
        t = re.sub(rf"^\s*{num}\s*[.\s]+", "", t, count=1).strip()
        out[int(num)] = t
    return out


def _first_rejected_num(rejection_instances: list) -> int | None:
    """First numeric claim mentioned in any rejection_instance.claims_raw."""
    for ri in rejection_instances or []:
        for raw in ri.get("claims_raw") or []:
            # accept "1", "1-5", "1, 3-5", "1 and 3" — take the first integer
            m = re.search(r"\b(\d+)\b", raw or "")
            if m:
                return int(m.group(1))
    return None


# ---------- feature extraction --------------------------------------------

# Patterns from draft_v0.3 §4.2 G matcher; frozen here with word boundaries.
RE_MEANS_PLUS_FN = re.compile(
    r"\bmeans for\s+\w+ing\b", re.IGNORECASE
)
RE_FUNCTIONAL = re.compile(
    r"\bconfigured to\b|\badapted to\b|\bfor\s+\w+ing\b",
    re.IGNORECASE,
)
RE_RANGE = re.compile(
    r"\bat least\b"
    r"|\bbetween\s+[^,.;]+?\s+and\b"
    r"|\b(?:greater|less|more|fewer)\s+than\b",
    re.IGNORECASE,
)


def _limitation_patterns(claim_text: str) -> list:
    hits = []
    if RE_MEANS_PLUS_FN.search(claim_text or ""):
        hits.append("means_plus_function")
    if RE_FUNCTIONAL.search(claim_text or ""):
        hits.append("functional")
    if RE_RANGE.search(claim_text or ""):
        hits.append("range")
    return hits


def _statute_subsection(ri: dict) -> str:
    subsec = ri.get("statute_subsection") or []
    return subsec[0] if subsec else ""


# ---------- main build loop -----------------------------------------------

def build_pool(beta_dir: Path, cohort_ids: set) -> dict:
    skip_counts: Counter = Counter()
    entries = []
    n_scanned = 0
    n_in_cohort = 0

    for fn in sorted(beta_dir.iterdir()):
        if fn.suffix != ".json":
            continue
        try:
            rec = json.loads(fn.read_text(encoding="utf-8"))
        except Exception:
            skip_counts["parse_error"] += 1
            continue
        case_id = rec.get("case_id")
        if case_id in cohort_ids:
            n_in_cohort += 1
            continue
        n_scanned += 1

        ctnf = rec.get("ctnf") or {}
        rinst = ctnf.get("rejection_instances") or []
        if not rinst:
            skip_counts["no_rejection"] += 1
            continue

        attacked_num = _first_rejected_num(rinst)
        if attacked_num is None:
            skip_counts["no_attacked_num"] += 1
            continue

        pre = _claims_map((rec.get("pre_clm") or {}).get("claims"))
        post = _claims_map((rec.get("post_clm") or {}).get("claims"))
        if attacked_num not in pre:
            skip_counts["attacked_not_in_pre"] += 1
            continue
        if attacked_num not in post:
            skip_counts["attacked_not_in_post"] += 1
            continue

        pre_text = pre[attacked_num]
        post_text = post[attacked_num]
        # Drop entries whose post_text is an implicit cancellation placeholder
        # (inserted by parser for <ImplicitClaim> ranges) — not useful as a
        # past-amendment exemplar.
        if post_text.startswith("(implicit,") or post_text.startswith("(range "):
            skip_counts["implicit_placeholder"] += 1
            continue

        ri0 = rinst[0]
        entry = {
            "case_id": case_id,
            "app_num": rec.get("app_num"),
            "statute": ri0.get("statute_section") or "",
            "subsection": _statute_subsection(ri0),
            "limitation_patterns": _limitation_patterns(pre_text),
            "attacked_claim_num": attacked_num,
            "pre_text": pre_text,
            "post_text": post_text,
        }
        entries.append(entry)

    # Deterministic ordering
    entries.sort(key=lambda e: (str(e.get("app_num") or ""), e["case_id"]))
    return {
        "n_scanned_after_cohort_exclusion": n_scanned,
        "n_in_cohort": n_in_cohort,
        "n_retrievable": len(entries),
        "skip_counts": dict(skip_counts),
        "entries": entries,
    }


# ---------- CLI ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cohort_ids = {c["case_id"] for c in cohort["cases"]}
    print(f"cohort: {len(cohort_ids)} case_ids to exclude")

    result = build_pool(args.beta_parsed_dir, cohort_ids)

    # Invariant
    pool_ids = {e["case_id"] for e in result["entries"]}
    overlap = pool_ids & cohort_ids
    assert not overlap, f"LEAKAGE: pool ∩ cohort = {sorted(overlap)[:5]}"
    print(f"leakage assertion passed (pool ∩ cohort = ∅)")

    out = {
        "pool_version": "v1",
        "cohort_excluded": args.cohort.name,
        "n_cohort_excluded": len(cohort_ids),
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== pool build summary ===")
    print(f"  scanned (after cohort exclusion): {result['n_scanned_after_cohort_exclusion']}")
    print(f"  in cohort (excluded):             {result['n_in_cohort']}")
    print(f"  retrievable:                      {result['n_retrievable']}")
    print(f"  skip counts:")
    for k, v in sorted(result["skip_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:30s} {v}")

    # Feature distribution
    stats = Counter()
    for e in result["entries"]:
        stats[f"statute={e['statute']}"] += 1
        for p in e["limitation_patterns"]:
            stats[f"pattern={p}"] += 1
    print(f"\n  feature distribution (top 12):")
    for k, v in stats.most_common(12):
        print(f"    {k:30s} {v}")

    size_mb = args.out.stat().st_size / 1e6
    print(f"\n  saved to {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
