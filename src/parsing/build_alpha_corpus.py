"""
Build α (partial ground-truth) corpus by joining:
  - pilot_bench/benchmark/benchmark_dataset.json  (13,749 cases)
  - ptab_patents_hf/extracted/{proceeding_number}_*_{suffix}/
        ApplicantPatent/<pub>.json  (published claims)
        PriorArtPatent/<pub>.json   (prior art, subset)

For each case we emit one JSONL row with:
  case_id, proceeding_number, patent_number, tech_center, art_unit,
  decision_date, subdecision, issue_type,
  published_claims (list)         <- pre-amendment approximation
  examiner_findings (str)          <- rejection rationale
  appellant_arguments (str)        <- applicant's amendment rationale
  ptab_opinion (str)               <- full PTAB decision
  appeal_time_claim_quotes (list)  <- regex-extracted quotes
  prior_art_available (bool)
  prior_art_sample_title (str or None)
  flags (dict of coverage / quality booleans)

Usage:
  cd /data2/hsm2026/vPClaimGeneration
  python3 scripts/build_alpha_corpus.py \
      --benchmark /data2/hsm2026/vPatentTest/data/pilot_bench/benchmark/benchmark_dataset.json \
      --extracted /data2/hsm2026/vPatentTest/data/ptab_patents_hf/extracted \
      --out data/parsed/alpha_corpus.jsonl \
      --stats data/parsed/alpha_corpus_stats.json \
      --limit 50      # 0 = all

Output: one JSON object per line. Stats file = summary over all rows.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Regex patterns for appeal-time claim language quotes inside ptab_opinion.
# Each returns a list of (match_text, match_len) pairs.
QUOTE_PATTERNS = [
    # Long double-quoted strings (likely claim recitation)
    ("long_dq", re.compile(r'"([^"]{60,1200})"')),
    # "claim N recites/states/provides/reads ..." phrase
    ("claim_recites", re.compile(
        r'[Cc]laim\s+\d+[a-z]?\s*(?:recites?|states?|provides?|reads?|requires?)[^.]{20,500}\.'
    )),
    # "as claimed" excerpted language
    ("as_claimed", re.compile(r'(?:as\s+claimed|the\s+claimed)[^.]{10,300}\.')),
]


def extract_quotes(text: str):
    out = []
    for label, pat in QUOTE_PATTERNS:
        for m in pat.finditer(text or ""):
            s = m.group(0)
            out.append({"pattern": label, "span": [m.start(), m.end()], "text": s[:800]})
    return out


def find_case_dir(extracted_root: Path, proceeding_number: str):
    """Given a proceeding_number like '2012011198', locate its case dir under
    extracted_root. PTAB uses suffixes like _DECISION, _Mail_Decision, etc.
    """
    # Common suffixes observed in sample scan
    for suffix in ("_DECISION", "_Mail_Decision"):
        p = extracted_root / f"{proceeding_number}{suffix}"
        if p.is_dir():
            return p
    # Fallback: glob
    hits = list(extracted_root.glob(f"{proceeding_number}_*"))
    return hits[0] if hits else None


def read_applicant_patent(case_dir: Path):
    """Return list of claim dicts from ApplicantPatent/<pub>.json, preferring
    current-version file (not 'original/' which is known to be corrupt).
    """
    ap_dir = case_dir / "ApplicantPatent"
    if not ap_dir.is_dir():
        return None, None, "no_applicant_patent_dir"
    candidates = [p for p in ap_dir.glob("*.json") if p.is_file()]
    if not candidates:
        return None, None, "no_applicant_json"
    # Take largest file as primary (structured version)
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    primary = candidates[0]
    try:
        data = json.loads(primary.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return None, None, f"applicant_parse_error:{type(e).__name__}"
    claims = data.get("claims")
    title = data.get("title") or data.get("invention_title")
    pub = primary.stem
    if not isinstance(claims, list):
        return None, title, f"claims_not_list:{type(claims).__name__}"
    # Normalize claim text access (schema varies)
    normalized = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        ct = c.get("claim_text")
        if isinstance(ct, dict):
            text = ct.get("text", "")
        elif isinstance(ct, str):
            text = ct
        else:
            text = c.get("text", "") or ""
        normalized.append({
            "claim_num": c.get("claim_num") or c.get("claim_number"),
            "depends_on": c.get("depends_on"),
            "text": (text or "")[:4000],
        })
    return {"publication": pub, "title": title, "claims": normalized}, title, "ok"


def prior_art_info(case_dir: Path):
    pa_dir = case_dir / "PriorArtPatent"
    if not pa_dir.is_dir():
        return False, None
    jsons = [p for p in pa_dir.glob("*.json") if p.is_file()]
    if not jsons:
        return False, None
    jsons.sort(key=lambda p: p.stat().st_size, reverse=True)
    try:
        data = json.loads(jsons[0].read_text(encoding="utf-8", errors="replace"))
        title = data.get("title") or data.get("invention_title")
    except Exception:
        title = None
    return True, title


def process_case(case: dict, extracted_root: Path):
    proc = str(case.get("proceeding_number", ""))
    case_dir = find_case_dir(extracted_root, proc) if proc else None

    row = {
        "case_id": case.get("case_id"),
        "proceeding_number": proc,
        "patent_number": case.get("patent_number"),
        "tech_center": case.get("tech_center"),
        "art_unit": case.get("art_unit"),
        "decision_date": case.get("decision_date"),
        "subdecision": (case.get("ground_truth") or {}).get("subdecision"),
        "issue_type": (case.get("ground_truth") or {}).get("issue_type"),
        "examiner_findings": case.get("examiner_findings") or "",
        "appellant_arguments": case.get("appellant_arguments") or "",
        "ptab_opinion": case.get("ptab_opinion") or "",
    }

    # Attach published claims from ApplicantPatent
    if case_dir is None:
        row["published"] = None
        row["published_status"] = "case_dir_not_found"
    else:
        pub_data, title, status = read_applicant_patent(case_dir)
        row["published"] = pub_data
        row["published_status"] = status
        row["title"] = title

    # Prior art
    if case_dir is not None:
        has_pa, pa_title = prior_art_info(case_dir)
        row["prior_art_available"] = has_pa
        row["prior_art_sample_title"] = pa_title
    else:
        row["prior_art_available"] = False
        row["prior_art_sample_title"] = None

    # Extract appeal-time claim quotes from ptab_opinion
    row["appeal_time_claim_quotes"] = extract_quotes(row["ptab_opinion"])

    # Coverage flags
    row["flags"] = {
        "has_examiner_findings": len(row["examiner_findings"]) > 100,
        "has_appellant_args": len(row["appellant_arguments"]) > 100,
        "has_ptab_opinion": len(row["ptab_opinion"]) > 100,
        "has_published_claims": bool(row["published"] and row["published"]["claims"]),
        "has_prior_art": row["prior_art_available"],
        "has_any_claim_quote": len(row["appeal_time_claim_quotes"]) > 0,
        "has_long_dq_quote": any(q["pattern"] == "long_dq" for q in row["appeal_time_claim_quotes"]),
    }
    row["flags"]["full_4tuple_minus_postclaim"] = all([
        row["flags"]["has_examiner_findings"],
        row["flags"]["has_published_claims"],
        row["flags"]["has_prior_art"],
    ])
    row["flags"]["full_4tuple_with_quote_proxy"] = (
        row["flags"]["full_4tuple_minus_postclaim"] and row["flags"]["has_any_claim_quote"]
    )

    return row


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}

    flag_counts = Counter()
    for r in rows:
        for k, v in r["flags"].items():
            if v:
                flag_counts[k] += 1

    status_counts = Counter(r.get("published_status") for r in rows)

    # Claim quote statistics
    quote_per_case = [len(r["appeal_time_claim_quotes"]) for r in rows]
    quote_per_case.sort()
    med_quotes = quote_per_case[n // 2]

    # Published claim stats
    pub_claim_counts = [
        len(r["published"]["claims"]) if r.get("published") else 0 for r in rows
    ]
    pub_claim_counts_sorted = sorted(pub_claim_counts)
    med_pub_claims = pub_claim_counts_sorted[n // 2]

    # Field size stats
    ef_sizes = sorted(len(r["examiner_findings"]) for r in rows)
    op_sizes = sorted(len(r["ptab_opinion"]) for r in rows)

    return {
        "n": n,
        "flag_pct": {k: round(100 * v / n, 1) for k, v in flag_counts.items()},
        "published_status_counts": dict(status_counts.most_common()),
        "quote_per_case_median": med_quotes,
        "quote_per_case_max": max(quote_per_case) if quote_per_case else 0,
        "published_claims_median": med_pub_claims,
        "examiner_findings_chars_median": ef_sizes[n // 2],
        "ptab_opinion_chars_median": op_sizes[n // 2],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--extracted", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    bench = json.loads(args.benchmark.read_text(encoding="utf-8"))
    cases = bench["cases"]
    if args.limit and args.limit < len(cases):
        cases = cases[: args.limit]

    print(f"Processing {len(cases)} cases", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(cases):
            row = process_case(case, args.extracted)
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(cases)}", flush=True)

    stats = summarize(rows)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(stats, indent=2))
    print(f"\nRows → {args.out}")
    print(f"Stats → {args.stats}")


if __name__ == "__main__":
    main()
