"""
Stratified analysis over α corpus (13,749 PTAB cases).

Produces:
  - outcome × issue_type crosstab
  - coverage flags crosstab (published claims × prior-art JSON × claim quote)
  - tech_center distribution
  - decision_date timeline
  - "eligible for β selection" subset stats (full_4tuple_with_quote_proxy)

Usage:
  python3 scripts/analyze_alpha_strata.py \
      --alpha data/parsed/alpha_corpus.jsonl \
      --out   data/parsed/alpha_strata.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    outcome_counts = Counter()
    issue_counts = Counter()
    issue_per_case = []
    tech_center_counts = Counter()
    art_unit_counts = Counter()
    decision_year_counts = Counter()

    flag_counts = Counter()
    flag_and_outcome = defaultdict(Counter)          # flag -> outcome -> count
    outcome_by_issue = defaultdict(Counter)           # outcome -> issue -> count
    eligible_by_year = defaultdict(Counter)           # year -> outcome -> count (eligible only)
    eligible_by_issue = defaultdict(Counter)          # issue -> outcome -> count (eligible only)

    quote_count_dist = Counter()
    pub_claim_count_dist = Counter()

    total = 0
    eligible = 0

    with args.alpha.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            total += 1
            flags = r.get("flags") or {}
            sub = r.get("subdecision") or "Unknown"
            outcome_counts[sub] += 1

            issues = r.get("issue_type") or []
            if isinstance(issues, list):
                if not issues:
                    issue_counts["(none)"] += 1
                    issue_per_case.append(0)
                else:
                    issue_per_case.append(len(issues))
                    # Sort + join for "102+103" style label
                    key = "+".join(sorted(issues))
                    issue_counts[key] += 1
                    for i in issues:
                        outcome_by_issue[sub][i] += 1

            tc = r.get("tech_center") or "?"
            tech_center_counts[str(tc)] += 1
            au = r.get("art_unit") or "?"
            art_unit_counts[str(au)] += 1

            dd = r.get("decision_date") or ""
            year = dd[-4:] if len(dd) >= 4 else "?"
            decision_year_counts[year] += 1

            for k, v in flags.items():
                if v:
                    flag_counts[k] += 1
                    flag_and_outcome[k][sub] += 1

            # Quote distribution bucket
            n_quotes = len(r.get("appeal_time_claim_quotes") or [])
            bucket = (
                "0" if n_quotes == 0 else
                "1-2" if n_quotes <= 2 else
                "3-5" if n_quotes <= 5 else
                "6-10" if n_quotes <= 10 else
                "11-20" if n_quotes <= 20 else ">20"
            )
            quote_count_dist[bucket] += 1

            # Published claim count bucket
            pub = r.get("published") or {}
            pc = pub.get("claims") or []
            n_pc = len(pc) if isinstance(pc, list) else 0
            pc_bucket = (
                "0" if n_pc == 0 else
                "1-10" if n_pc <= 10 else
                "11-20" if n_pc <= 20 else
                "21-40" if n_pc <= 40 else ">40"
            )
            pub_claim_count_dist[pc_bucket] += 1

            if flags.get("full_4tuple_with_quote_proxy"):
                eligible += 1
                eligible_by_year[year][sub] += 1
                if isinstance(issues, list):
                    for i in issues:
                        eligible_by_issue[i][sub] += 1

    # Summarize
    def pct(counter, total):
        return {k: {"n": v, "pct": round(100 * v / max(total, 1), 2)}
                for k, v in counter.most_common()}

    summary = {
        "total_cases": total,
        "eligible_cases": eligible,
        "eligible_ratio": round(100 * eligible / max(total, 1), 2),
        "outcome_overall": pct(outcome_counts, total),
        "issue_type_overall": dict(issue_counts.most_common(20)),
        "issues_per_case": {
            "median": sorted(issue_per_case)[len(issue_per_case) // 2] if issue_per_case else 0,
            "mean": round(sum(issue_per_case) / max(len(issue_per_case), 1), 2),
            "max": max(issue_per_case) if issue_per_case else 0,
        },
        "tech_center_top10": dict(tech_center_counts.most_common(10)),
        "art_unit_top10": dict(art_unit_counts.most_common(10)),
        "decision_year_counts": dict(sorted(decision_year_counts.items())),
        "flag_counts": pct(flag_counts, total),
        "outcome_by_issue": {k: dict(v) for k, v in outcome_by_issue.items()},
        "quote_count_distribution": dict(quote_count_dist),
        "pub_claim_count_distribution": dict(pub_claim_count_dist),
        "eligible_outcome_by_year": {k: dict(v) for k, v in eligible_by_year.items()},
        "eligible_outcome_by_issue": {k: dict(v) for k, v in eligible_by_issue.items()},
    }

    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("eligible_outcome_by_year", "decision_year_counts")},
                     indent=2, ensure_ascii=False))
    print(f"\nFull report → {args.out}")


if __name__ == "__main__":
    main()
