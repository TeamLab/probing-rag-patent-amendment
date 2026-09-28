"""
Diagnostic scan of the applicant-XML corpus to understand
claim-amendment markup before writing the full parser.

Goals:
1. Count files with claim amendment section (sanity vs grep baseline 595).
2. Tag census across namespace (pat:*) — especially markup tags for
   additions/deletions. We need to know if underlined/strikethrough
   markup exists as explicit tags, or if only [[deleted]] brackets are used.
3. OCR quality proxy: <pat:OCRConfidenceData> density per file.
4. ClaimStatusCategory distribution (amended / original / cancelled / new).
5. [[...]] bracket occurrence pattern.

Usage:
    cd /data2/hsm2026/vPClaimGeneration
    python scripts/inspect_applicant_xml.py \
        --src /data2/hsm2026/vPatentTest/data/patent_spec_xml/applicant \
        --out data/parsed/inspect_report.json \
        --limit 0       # 0 = all; use e.g. 50 for quick smoke
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CLAIM_AMENDMENT_MARKERS = re.compile(
    r"(Amendments to the Claims|Listing of Claims|In the Claims)",
    re.IGNORECASE,
)
STATUS_RE = re.compile(r"<pat:ClaimStatusCategory>([^<]+)</pat:ClaimStatusCategory>")
TAG_RE = re.compile(r"<(pat:[A-Za-z][A-Za-z0-9]*)[ >/]")
BRACKET_DELETE_RE = re.compile(r"\[\[[^\[\]]{1,200}\]\]")
OCR_CONF_RE = re.compile(r"<pat:OCRConfidenceData[^>]*>([^<]+)</pat:OCRConfidenceData>")


def scan_file(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"file": path.name, "error": str(e)}

    has_claim_amend = bool(CLAIM_AMENDMENT_MARKERS.search(text))
    statuses = Counter(STATUS_RE.findall(text))
    tags = Counter(TAG_RE.findall(text))

    ocr_hits = OCR_CONF_RE.findall(text)
    ocr_density = len(ocr_hits) / max(len(text), 1)

    bracket_deletes = BRACKET_DELETE_RE.findall(text)

    return {
        "file": path.name,
        "app_num": path.stem,
        "chars": len(text),
        "has_claim_amendment": has_claim_amend,
        "status_counts": dict(statuses),
        "tag_sample": dict(tags.most_common(15)),
        "ocr_conf_count": len(ocr_hits),
        "ocr_density": round(ocr_density, 6),
        "bracket_delete_count": len(bracket_deletes),
        "bracket_delete_samples": bracket_deletes[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = scan all")
    args = ap.parse_args()

    files = sorted(args.src.glob("*.xml"))
    if args.limit:
        files = files[: args.limit]

    print(f"Scanning {len(files)} files from {args.src}")

    rows = []
    global_tags: Counter = Counter()
    global_statuses: Counter = Counter()
    n_with_claim_amend = 0
    n_with_brackets = 0

    for i, p in enumerate(files):
        row = scan_file(p)
        rows.append(row)
        if row.get("has_claim_amendment"):
            n_with_claim_amend += 1
        if row.get("bracket_delete_count", 0) > 0:
            n_with_brackets += 1
        for t, c in row.get("tag_sample", {}).items():
            global_tags[t] += c
        for s, c in row.get("status_counts", {}).items():
            global_statuses[s] += c

        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(files)}")

    summary = {
        "src": str(args.src),
        "n_files": len(files),
        "n_with_claim_amendment": n_with_claim_amend,
        "n_with_bracket_deletes": n_with_brackets,
        "global_tag_top30": dict(global_tags.most_common(30)),
        "global_status_counts": dict(global_statuses),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {"summary": summary, "per_file": rows}
    args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-file rows + summary → {args.out}")


if __name__ == "__main__":
    main()
