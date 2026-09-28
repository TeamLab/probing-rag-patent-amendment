"""
Pattern analysis over parsed β corpus (structured 4-tuple JSONs).

Analyses:
  1. Amendment pattern clusters — classify each case by claim_diff composition
     (cancel-heavy / modify-heavy / new-heavy / mixed)
  2. CTNF rejection type distribution (101/102/103/112 + combinations)
  3. Prior art extraction sanity — patent-number pattern hit rate, name-only rate
  4. Cross-format stratification — DTD legacy vs NS ClaimsDocument amendment styles
  5. Decisive-limitation inference heuristic — for each case, try to match
     CTNF-cited claim numbers with diff'd claim numbers

Usage:
  python3 scripts/analyze_beta_patterns.py \
      --parsed-dir data/parsed/beta_parsed \
      --manifest   data/parsed/beta_100_manifest.jsonl \
      --out        data/parsed/beta_patterns.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RE_CLAIM_NUMBERS = re.compile(r"\b(\d+)(?:\s*[-\u2013]\s*(\d+))?\b")
RE_PATENT_NUMBER = re.compile(r"\(?\s*([\d,]{5,10})\s*\)?")


def extract_claim_numbers(raw_list):
    """Given strings like ['s 1-3 and 5 are', 'Claim 14-17'] return set of ints.
    Handles ranges ('1-3' -> {1,2,3}) and commas ('1, 5, 7')."""
    out = set()
    for raw in raw_list or []:
        # Normalize dashes
        s = str(raw).replace("\u2013", "-")
        for m in RE_CLAIM_NUMBERS.finditer(s):
            a = int(m.group(1))
            b = m.group(2)
            if b:
                try:
                    b = int(b)
                    if 0 < b - a < 200:  # sanity
                        out.update(range(a, b + 1))
                    else:
                        out.add(a)
                except ValueError:
                    out.add(a)
            else:
                out.add(a)
    return out


def classify_pattern(diff_summary):
    """Return one-word label for amendment style."""
    kept = diff_summary.get("kept", 0)
    mod = diff_summary.get("modified", 0)
    new = diff_summary.get("new", 0)
    canc = diff_summary.get("cancelled", 0)
    total = kept + mod + new + canc
    if total == 0:
        return "empty"

    # Dominant axis
    pct_canc = canc / total
    pct_new = new / total
    pct_mod = mod / total
    pct_kept = kept / total

    # Classification rules
    if pct_canc > 0.5 and pct_new > 0.3:
        return "cancel_and_replace"
    if pct_canc > 0.5:
        return "cancel_heavy"
    if pct_mod > 0.5 and pct_new < 0.15 and pct_canc < 0.15:
        return "modify_only"
    if pct_new > 0.3 and pct_mod > 0.3:
        return "modify_and_add"
    if pct_new > 0.5:
        return "new_heavy"
    if pct_kept > 0.5:
        return "mostly_kept"
    return "mixed"


def check_prior_art_pattern(refs):
    """Classify each prior art reference by structure."""
    with_pnum = 0
    name_only = 0
    suspicious = 0
    for ref in refs or []:
        if RE_PATENT_NUMBER.search(ref) and any(c.isdigit() for c in ref):
            with_pnum += 1
        elif re.match(r"^[A-Z][a-zA-Z\-\. ]{1,50}$", ref.strip()):
            name_only += 1
        else:
            suspicious += 1
    return with_pnum, name_only, suspicious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # Load manifest for outcome linkage
    outcomes = {}
    # Alpha corpus lookup (for outcome per proceeding)
    alpha_outcome = {}
    alpha_path = Path("data/parsed/alpha_corpus.jsonl")
    if alpha_path.exists():
        with alpha_path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                alpha_outcome[r.get("proceeding_number")] = r.get("subdecision")
                alpha_outcome[str(r.get("proceeding_number"))] = r.get("subdecision")

    patterns = Counter()
    patterns_by_outcome = defaultdict(Counter)     # outcome -> pattern -> count
    rejection_type_counts = Counter()              # single types
    rejection_combo_counts = Counter()             # sorted combos
    ctnf_fallback_cases = 0
    per_case_stats = []

    # Prior art aggregation
    pa_with_pnum = 0
    pa_name_only = 0
    pa_suspicious = 0
    pa_per_case = []

    # Format stratification
    format_pattern_xtab = defaultdict(Counter)     # (pre_fmt, post_fmt) -> pattern

    # Decisive-limitation heuristic
    decisive_hit = 0
    decisive_miss = 0
    decisive_na = 0  # no rejection or no diff changes

    for p in sorted(args.parsed_dir.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        case_id = rec.get("case_id")
        proc = str(rec.get("proceeding_number") or "")
        outcome = alpha_outcome.get(proc, "Unknown")

        diff = rec.get("claim_diff") or {}
        summary = diff.get("summary") or {}
        pattern = classify_pattern(summary)
        patterns[pattern] += 1
        patterns_by_outcome[outcome][pattern] += 1

        ctnf = rec.get("ctnf") or {}
        if ctnf.get("regex_fallback_used"):
            ctnf_fallback_cases += 1

        # Rejection types: collect statute sections
        rej_instances = ctnf.get("rejection_instances") or []
        types_this_case = set()
        for ri in rej_instances:
            st = ri.get("statute_section") or ""
            if st:
                types_this_case.add(st)
            # Alternative field from regex fallback
            # (already in statute_section above)
        for t in types_this_case:
            rejection_type_counts[t] += 1
        if types_this_case:
            rejection_combo_counts["+".join(sorted(types_this_case))] += 1

        # Prior-art structure
        refs = ctnf.get("prior_art_refs") or []
        pn, no, sus = check_prior_art_pattern(refs)
        pa_with_pnum += pn
        pa_name_only += no
        pa_suspicious += sus
        pa_per_case.append(len(refs))

        # Cross-format
        pre_fmt = (rec.get("pre_clm") or {}).get("format")
        post_fmt = (rec.get("post_clm") or {}).get("format")
        fmt_key = f"{pre_fmt}→{post_fmt}"
        format_pattern_xtab[fmt_key][pattern] += 1

        # Decisive-limitation heuristic
        # Claims mentioned in rejection_instances[*].claims_raw → ints
        rejected_nums = set()
        for ri in rej_instances:
            rejected_nums |= extract_claim_numbers(ri.get("claims_raw") or [])
        # Claims changed (modified+cancelled+new) in diff
        changed_nums = set()
        for c in diff.get("per_claim") or []:
            if c.get("status") in ("modified", "cancelled"):
                try:
                    changed_nums.add(int(c.get("num")))
                except (ValueError, TypeError):
                    pass
        if not rejected_nums or not changed_nums:
            decisive_na += 1
        elif rejected_nums & changed_nums:
            decisive_hit += 1
        else:
            decisive_miss += 1

        per_case_stats.append({
            "case_id": case_id,
            "proceeding_number": proc,
            "outcome": outcome,
            "pattern": pattern,
            "diff_summary": summary,
            "rejection_types": sorted(types_this_case),
            "n_prior_art": len(refs),
            "pre_fmt": pre_fmt,
            "post_fmt": post_fmt,
            "rejected_claim_nums": sorted(rejected_nums),
            "changed_claim_nums": sorted(changed_nums),
            "decisive_match": bool(rejected_nums & changed_nums)
                             if (rejected_nums and changed_nums) else None,
        })

    def pct(counter, tot):
        return {k: {"n": v, "pct": round(100 * v / max(tot, 1), 1)}
                for k, v in counter.most_common()}

    n = len(per_case_stats)
    total_pa = pa_with_pnum + pa_name_only + pa_suspicious

    summary = {
        "n_cases": n,
        "ctnf_regex_fallback_used": ctnf_fallback_cases,
        "amendment_patterns": pct(patterns, n),
        "amendment_patterns_by_outcome": {
            k: dict(v.most_common()) for k, v in patterns_by_outcome.items()
        },
        "rejection_type_single_counts": pct(rejection_type_counts, n),
        "rejection_type_combos_top15": dict(rejection_combo_counts.most_common(15)),
        "prior_art_structure": {
            "total_refs": total_pa,
            "with_patent_number_pct": round(100 * pa_with_pnum / max(total_pa, 1), 1),
            "name_only_pct": round(100 * pa_name_only / max(total_pa, 1), 1),
            "suspicious_pct": round(100 * pa_suspicious / max(total_pa, 1), 1),
            "refs_per_case": {
                "min": min(pa_per_case) if pa_per_case else 0,
                "median": sorted(pa_per_case)[n // 2] if pa_per_case else 0,
                "max": max(pa_per_case) if pa_per_case else 0,
                "mean": round(sum(pa_per_case) / max(n, 1), 1),
            },
        },
        "format_x_pattern": {k: dict(v) for k, v in format_pattern_xtab.items()},
        "decisive_limitation_heuristic": {
            "hit": decisive_hit,
            "miss": decisive_miss,
            "na_missing_side": decisive_na,
            "hit_rate_of_scored": round(
                100 * decisive_hit / max(decisive_hit + decisive_miss, 1), 1),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    # Also dump per-case for downstream use
    per_case_path = args.out.with_name(args.out.stem + "_per_case.jsonl")
    with per_case_path.open("w", encoding="utf-8") as fh:
        for row in per_case_stats:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPer-case rows → {per_case_path}")


if __name__ == "__main__":
    main()
