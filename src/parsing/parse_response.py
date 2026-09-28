"""Parse model-generated amendment responses into structured claim records.

Format (confirmed empirically on 14 sample responses, 2026-04-18):
  - Each claim starts with "N. (Status) ..." on its own line
  - Status ∈ {Original, Currently amended, Previously presented,
              Cancelled, New, Withdrawn} (case-insensitive)
  - Additions marked with `__text__` (double underscore)
  - Deletions marked with `[[text]]` (double bracket)

This parser is lenient about preamble/postamble noise but strict about
the header line: anything before the first `N. (Status)` is discarded,
anything that doesn't match the header pattern inside the claim body is
kept as body text.

API:
    parse(text) -> list[ClaimRecord]

    ClaimRecord = {
      "num": int,
      "status": str,           # normalized: see STATUS_MAP
      "status_raw": str,
      "raw_body": str,         # body as returned by model (with markup)
      "plain_body": str,       # markup stripped
      "additions": list[str],  # text spans model added
      "deletions": list[str],  # text spans model deleted
    }

CLI:
    python3 scripts/parse_response.py <response_file.json> \\
        [--out parsed_response.json] [--tsv]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


# ---- status normalization -----------------------------------------------

STATUS_MAP = {
    "original": "original",
    "currently amended": "currently_amended",
    "currently-amended": "currently_amended",
    "previously presented": "previously_presented",
    "previously-presented": "previously_presented",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "new": "new",
    "withdrawn": "withdrawn",
    "amended": "currently_amended",        # loose fallback
}


def _normalize_status(s: str) -> str:
    key = (s or "").strip().lower()
    return STATUS_MAP.get(key, "unknown")


# ---- core parser --------------------------------------------------------

# Claim header variants observed across models (2026-04-19):
#   A. "1. (Currently amended)"      — canonical USPTO (Claude, GPT-5.4)
#   B. "1 (Currently amended)"       — no period between num and paren (gpt-4o-mini ~20%)
#   C. "Claim 1 (Currently amended)" — word prefix  (gpt-4o-mini ~25%)
#   D. "(Currently amended) Claim 1" — reversed order (gpt-4o-mini ~10%)
#   E. "1. Currently amended:"       — no parens (rare)
# All variants captured into (num, status_raw). Tried in order; first match wins.

_HDR_CANONICAL = re.compile(
    r"^\s*(\d+)\s*\.\s*\(\s*([^)\n]+?)\s*\)\s*:?\s*",
    re.MULTILINE,
)
_HDR_NO_DOT = re.compile(
    r"^\s*(\d+)\s+\(\s*([^)\n]+?)\s*\)\s*:?\s*",
    re.MULTILINE,
)
_HDR_WORD_PREFIX = re.compile(
    r"^\s*Claim\s+(\d+)\s*\(\s*([^)\n]+?)\s*\)\s*:?\s*",
    re.MULTILINE | re.IGNORECASE,
)
_HDR_REVERSED = re.compile(
    r"^\s*\(\s*([^)\n]+?)\s*\)\s+Claim\s+(\d+)\s*:?\s*",
    re.MULTILINE | re.IGNORECASE,
)

CLAIM_HEADER_PATTERNS = [
    ("canonical",    _HDR_CANONICAL,    1, 2),
    ("no_dot",       _HDR_NO_DOT,       1, 2),
    ("word_prefix",  _HDR_WORD_PREFIX,  1, 2),
    ("reversed",     _HDR_REVERSED,     2, 1),  # num=grp2, status=grp1
]


def _find_all_headers(text):
    """Return list of (match_start, match_end, num, status_raw, variant).
    Uses all patterns, deduplicates by (start, num) keeping earliest pattern."""
    found = {}
    for variant, pat, gn, gs in CLAIM_HEADER_PATTERNS:
        for m in pat.finditer(text):
            key = (m.start(), int(m.group(gn)))
            if key not in found:
                found[key] = (m.start(), m.end(),
                              int(m.group(gn)), m.group(gs).strip(), variant)
    # sort by position
    return sorted(found.values(), key=lambda x: x[0])


ADDITION_RE = re.compile(r"__([^_]+?)__")
DELETION_RE = re.compile(r"\[\[([^\]]+?)\]\]")


@dataclass
class ClaimRecord:
    num: int
    status: str
    status_raw: str
    raw_body: str
    plain_body: str
    additions: list[str]
    deletions: list[str]


def parse(text: str) -> list[ClaimRecord]:
    """Parse response text into a list of claim records.

    Greedy header matching on line boundaries; body of claim N spans from
    the char after its header to the start of claim N+1's header.
    """
    if not text:
        return []

    headers = _find_all_headers(text)
    if not headers:
        return []

    records: list[ClaimRecord] = []
    for i, (h_start, h_end, num, status_raw, _variant) in enumerate(headers):
        body_start = h_end
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        raw_body = text[body_start:body_end].strip()

        status = _normalize_status(status_raw)

        additions = [x.strip() for x in ADDITION_RE.findall(raw_body)]
        deletions = [x.strip() for x in DELETION_RE.findall(raw_body)]

        # plain_body: strip markup AND remove deleted spans from plain text
        plain = ADDITION_RE.sub(r"\1", raw_body)   # keep added text
        plain = DELETION_RE.sub("", plain)         # remove deleted text
        plain = re.sub(r"\s+", " ", plain).strip()

        records.append(ClaimRecord(
            num=num, status=status, status_raw=status_raw,
            raw_body=raw_body, plain_body=plain,
            additions=additions, deletions=deletions,
        ))
    return records


# ---- structural summary over a list of parses ---------------------------

def summarize(records: list[ClaimRecord]) -> dict:
    """Numeric indicators only — no body text — safe for logs/agents."""
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    total_add = sum(len(r.additions) for r in records)
    total_del = sum(len(r.deletions) for r in records)
    add_chars = sum(len(a) for r in records for a in r.additions)
    del_chars = sum(len(d) for r in records for d in r.deletions)
    return {
        "n_claims_parsed": len(records),
        "status_counts": by_status,
        "has_amended_or_new": any(
            r.status in ("currently_amended", "new") for r in records
        ),
        "total_additions": total_add,
        "total_deletions": total_del,
        "addition_chars": add_chars,
        "deletion_chars": del_chars,
        "claim_nums": [r.num for r in records],
        "plain_body_chars_mean": (
            round(sum(len(r.plain_body) for r in records) / len(records))
            if records else 0
        ),
    }


# ---- CLI (single-file convenience) --------------------------------------

def _cli_one(resp_path: Path, out_path: Path | None) -> None:
    d = json.loads(resp_path.read_text(encoding="utf-8"))
    text = d.get("response_text", "") or ""
    records = parse(text)
    summary = summarize(records)
    result = {
        "case_id": d.get("case_id"),
        "condition": d.get("condition", "baseline"),
        "rep": d.get("rep", 0),
        "summary": summary,
        "claims": [asdict(r) for r in records],
    }
    if out_path:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    # Console: numeric summary only
    print(json.dumps({k: v for k, v in result.items() if k != "claims"},
                     ensure_ascii=False, indent=2))


def _cli_batch(roots: Iterable[Path], out_tsv: Path) -> None:
    cols = ["file", "case", "cond", "rep", "n_claims_parsed",
            "amended_or_new", "total_additions", "total_deletions",
            "addition_chars", "deletion_chars", "plain_body_chars_mean",
            "status_original", "status_currently_amended",
            "status_previously_presented", "status_cancelled",
            "status_new", "status_withdrawn", "status_unknown"]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_parse_ok = 0
    with out_tsv.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for root in roots:
            for f in sorted(root.glob("*.json")):
                if f.name.startswith("_"):
                    continue
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                text = d.get("response_text", "") or ""
                recs = parse(text)
                s = summarize(recs)
                row = {
                    "file": f.name,
                    "case": d.get("case_id"),
                    "cond": d.get("condition", "baseline"),
                    "rep": d.get("rep", 0),
                    "n_claims_parsed": s["n_claims_parsed"],
                    "amended_or_new": int(s["has_amended_or_new"]),
                    "total_additions": s["total_additions"],
                    "total_deletions": s["total_deletions"],
                    "addition_chars": s["addition_chars"],
                    "deletion_chars": s["deletion_chars"],
                    "plain_body_chars_mean": s["plain_body_chars_mean"],
                }
                for st in ("original", "currently_amended",
                           "previously_presented", "cancelled",
                           "new", "withdrawn", "unknown"):
                    row[f"status_{st}"] = s["status_counts"].get(st, 0)
                fh.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
                n += 1
                if s["n_claims_parsed"] > 0:
                    n_parse_ok += 1
    print(f"parsed {n} files, {n_parse_ok} had ≥1 claim ({100*n_parse_ok/max(n,1):.0f}%)")
    print(f"wrote: {out_tsv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", type=Path,
                    help="directories of response JSONs for batch parse")
    ap.add_argument("--tsv", type=Path,
                    help="output TSV digest for batch mode")
    ap.add_argument("--one", type=Path,
                    help="single response file (prints summary)")
    ap.add_argument("--out", type=Path,
                    help="output parsed JSON for single-file mode")
    args = ap.parse_args()

    if args.one:
        _cli_one(args.one, args.out)
        return
    if args.roots and args.tsv:
        _cli_batch(args.roots, args.tsv)
        return
    ap.error("use --one <file> [--out ...] OR --roots <dirs> --tsv <out>")


if __name__ == "__main__":
    main()
