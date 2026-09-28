"""
Cross-match applicant XML (post-amendment claims) ↔ ptab_patents_hf/extracted
to determine whether rejection rationale text is recoverable per-case.

Strategy:
1. applicant XML filename = application number (stem).
2. ptab_patents_hf/extracted/<appeal_id>_<decision_type>/ contains PTAB
   decisions. Each decision text references application number somewhere.
3. grep application numbers inside extracted decision bodies; build map
   app_num → [appeal dirs].
4. For matched cases, inspect decision doc for a "rejection rationale"
   paragraph (examiner findings restated by PTAB).

This script is exploratory — output a feasibility report, not production pairs.

Usage:
    python scripts/cross_match_ptab.py \
        --applicant /data2/hsm2026/vPatentTest/data/patent_spec_xml/applicant \
        --ptab     /data2/hsm2026/vPatentTest/data/ptab_patents_hf/extracted \
        --out      data/parsed/cross_match_report.json \
        --sample   100
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


APP_NUM_RE = re.compile(r"\b(\d{2})[\s/,]?(\d{3})[\s/,]?(\d{3})\b")


def normalize_app(n: str) -> str:
    digits = re.sub(r"\D", "", n)
    return digits[-8:] if len(digits) >= 8 else digits


def collect_applicant_app_nums(applicant_dir: Path) -> set:
    return {p.stem for p in applicant_dir.glob("*.xml")}


def scan_ptab_case(case_dir: Path) -> dict:
    """Return {'appeal_id': ..., 'app_nums_mentioned': [...], 'text_size': ...}
    by reading any .txt / .xml / .json files within a case dir.
    """
    mentions: set = set()
    size = 0
    for f in case_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        size += len(text)
        for m in APP_NUM_RE.findall(text):
            joined = "".join(m)
            mentions.add(normalize_app(joined))
    return {
        "appeal_id": case_dir.name,
        "app_nums_mentioned": sorted(mentions),
        "text_size": size,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--applicant", required=True, type=Path)
    ap.add_argument("--ptab", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=0,
                    help="If > 0, sample this many PTAB case dirs for quick scan.")
    args = ap.parse_args()

    app_pool = collect_applicant_app_nums(args.applicant)
    print(f"applicant XML app_nums: {len(app_pool)}")

    ptab_cases = sorted([p for p in args.ptab.iterdir() if p.is_dir()])
    print(f"ptab cases total: {len(ptab_cases)}")
    if args.sample and args.sample < len(ptab_cases):
        ptab_cases = ptab_cases[: args.sample]

    per_case = []
    matches: dict = defaultdict(list)  # app_num -> [appeal_id]
    for i, c in enumerate(ptab_cases):
        row = scan_ptab_case(c)
        per_case.append(row)
        for app_num in row["app_nums_mentioned"]:
            if app_num in app_pool:
                matches[app_num].append(row["appeal_id"])
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(ptab_cases)}")

    summary = {
        "n_applicant_xml": len(app_pool),
        "n_ptab_cases_scanned": len(ptab_cases),
        "n_ptab_cases_total": len(ptab_cases),
        "n_unique_apps_matched": len(matches),
        "match_rate_vs_applicant": round(len(matches) / max(len(app_pool), 1), 4),
        "samples_with_multi_appeals": {
            k: v for k, v in matches.items() if len(v) > 1
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "matches": matches, "per_case": per_case[:50]}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote match map → {args.out}")


if __name__ == "__main__":
    main()
