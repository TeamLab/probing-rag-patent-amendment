"""Structural counts over model responses — no raw text to stdout/file.

Aggregates pattern hits across all response files so we can design the
parser without leaking raw patent text to the agent's context.

Writes: outputs/_analysis/response_patterns.tsv (counts only)
"""
from __future__ import annotations
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path


PATTERNS = {
    # Claim-header candidates
    "hdr_claim_N_paren_status": r"^\s*Claim\s+\d+\s*\([^)]+\)\s*[:.]?",
    "hdr_N_dot":                 r"^\s*\d+\s*\.\s",
    "hdr_N_paren_status":        r"^\s*\d+\s*\([^)]+\)\s*[:.]?",
    "hdr_claim_N_only":          r"^\s*Claim\s+\d+\s*[:.]?\s",

    # Status tokens (case-insensitive count)
    "status_currently_amended": r"\(currently amended\)",
    "status_original":          r"\(original\)",
    "status_cancelled":         r"\(cancel?led\)",
    "status_new":               r"\(new\)",
    "status_previously":        r"\(previously presented\)",
    "status_withdrawn":         r"\(withdrawn\)",

    # Markup
    "markup_bracket_delete":    r"\[\[[^\]]+\]\]",
    "markup_underline":         r"__[^_]+__",
    "markup_strikethrough_alt": r"~~[^~]+~~",
    "markup_single_underscore": r"(?<!_)_[A-Za-z][^_]*_(?!_)",

    # Preamble / commentary indicators (system prompt forbids these)
    "preamble_here_are":        r"^\s*(here are|below are|the following)",
    "preamble_md_heading":      r"^#+\s",
    "preamble_backtick_fence":  r"^```",
    "commentary_explanation":   r"(explanation|reasoning|note:|comment:)",

    # Claim body phrases (for sanity: should occur many times)
    "body_comprising":          r"\bcomprising\b",
    "body_wherein":             r"\bwherein\b",
    "body_said":                r"\bsaid\b",
    "body_configured_to":       r"\bconfigured to\b",
}


def inspect(text: str) -> dict:
    low = text.lower()
    out: dict[str, int] = {}
    for name, pat in PATTERNS.items():
        flags = re.IGNORECASE | re.MULTILINE if not name.startswith("markup") else re.MULTILINE
        out[name] = len(re.findall(pat, text, flags=flags))
    # first/last non-blank lines — pattern-class only, no content
    lines = [l for l in text.split("\n") if l.strip()]
    out["n_nonblank_lines"] = len(lines)
    out["total_chars"] = len(text)
    # classify first/last line shape
    out["first_line_is_claim_hdr"] = 1 if (lines and re.match(PATTERNS["hdr_claim_N_paren_status"], lines[0], re.I)) else 0
    out["last_line_is_claim_hdr"] = 1 if (lines and re.match(PATTERNS["hdr_claim_N_paren_status"], lines[-1], re.I)) else 0
    out["first_line_has_body"] = 1 if (lines and re.search(r"\b(comprising|wherein|said|configured)\b", lines[0], re.I)) else 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, type=Path,
                    help="directories containing *.json response files")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    files: list[Path] = []
    for r in args.roots:
        files.extend(sorted(r.glob("*.json")))
    files = [f for f in files if not f.name.startswith("_")]

    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rt = d.get("response_text")
        if rt is None:
            continue
        row = {"file": f.name, "case": d.get("case_id"),
               "cond": d.get("condition", "baseline"),
               "model": d.get("model_alias"),
               **inspect(rt)}
        rows.append(row)

    if not rows:
        print("no response files found")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["file", "case", "cond", "model"] + list(PATTERNS) + \
           ["n_nonblank_lines", "total_chars",
            "first_line_is_claim_hdr", "last_line_is_claim_hdr",
            "first_line_has_body"]
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    # console aggregates
    print(f"N files: {len(rows)}")
    print("\n=== pattern hit counts (sum across all files) ===")
    agg = Counter()
    files_with = Counter()
    for r in rows:
        for name in PATTERNS:
            agg[name] += r[name]
            if r[name] > 0:
                files_with[name] += 1
    w = max(len(k) for k in PATTERNS)
    for name in PATTERNS:
        print(f"  {name:<{w}}  total={agg[name]:5d}  files_with≥1={files_with[name]:3d}/{len(rows)}")

    print("\n=== first/last line classification ===")
    fl_claim = sum(r["first_line_is_claim_hdr"] for r in rows)
    ll_claim = sum(r["last_line_is_claim_hdr"] for r in rows)
    fl_body = sum(r["first_line_has_body"] for r in rows)
    print(f"  first line is claim header       : {fl_claim}/{len(rows)}")
    print(f"  last line is claim header        : {ll_claim}/{len(rows)}")
    print(f"  first line has claim body phrases: {fl_body}/{len(rows)}")

    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
