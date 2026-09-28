"""Structural / repetition metrics for smoke responses — NO raw text in output.

Writes numeric + categorical indicators only so the agent can Read the digest
without exposing model-generated text (triggers Usage Policy false-positives).

Run:  python3 scripts/sample_smoke_structure.py
Read: outputs/smoke/_analysis/structural_digest.txt
"""
from __future__ import annotations
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "outputs" / "smoke"
ANALYSIS = SMOKE / "_analysis"
OUT = ANALYSIS / "structural_digest.txt"

# Whitelisted domain keywords — we only report COUNTS, never surrounding text.
KEYWORDS = [
    "claim", "claims", "prior art", "obvious", "obviousness",
    "affirm", "affirmed", "reverse", "reversed", "reject", "rejection",
    "anticipat", "motivation", "combine", "teach", "disclos",
    "examiner", "applicant", "patent", "specification", "embodiment",
]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())


def line_shape(text: str) -> dict:
    lines = text.split("\n")
    blank = sum(1 for l in lines if not l.strip())
    numbered = sum(1 for l in lines if re.match(r"^\s*\d+[.)\]]\s", l))
    bullet = sum(1 for l in lines if re.match(r"^\s*[-*•]\s", l))
    header_like = sum(
        1 for l in lines
        if l.strip() and (l.strip().endswith(":") or (l.strip().isupper() and len(l.strip()) > 3))
    )
    non_blank = [l for l in lines if l.strip()]
    lens = [len(l) for l in non_blank] or [0]
    return {
        "lines": len(lines),
        "non_blank": len(non_blank),
        "blank": blank,
        "numbered": numbered,
        "bullet": bullet,
        "header_like": header_like,
        "line_len_mean": round(st.mean(lens), 1),
        "line_len_max": max(lens),
        "line_len_p95": sorted(lens)[int(0.95 * (len(lens) - 1))] if lens else 0,
    }


def paragraph_shape(text: str) -> dict:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return {"paragraphs": 0, "para_chars_mean": 0, "para_chars_max": 0}
    lens = [len(p) for p in paras]
    return {
        "paragraphs": len(paras),
        "para_chars_mean": round(st.mean(lens), 1),
        "para_chars_max": max(lens),
    }


def repetition_metrics(text: str) -> dict:
    """Detect pathological repetition WITHOUT exposing the repeated text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    line_dup_ratio = 1.0 - (len(set(lines)) / len(lines)) if lines else 0.0
    # top-k line repeat counts (counts only, never the lines themselves)
    line_counts = Counter(lines)
    top_line_repeats = sorted(line_counts.values(), reverse=True)[:5]

    words = tokenize_words(text)
    n = len(words)
    uniq_w = len(set(words))
    word_div = uniq_w / n if n else 0.0

    # 5-gram diversity — low value signals looping/copypasta
    if n >= 5:
        grams = [tuple(words[i:i + 5]) for i in range(n - 4)]
        g_counts = Counter(grams)
        g_div = len(g_counts) / len(grams)
        top_gram_repeats = sorted(g_counts.values(), reverse=True)[:5]
    else:
        g_div = 1.0
        top_gram_repeats = []

    return {
        "word_count": n,
        "unique_word_ratio": round(word_div, 3),
        "line_dup_ratio": round(line_dup_ratio, 3),
        "top5_line_repeat_counts": top_line_repeats,
        "fivegram_diversity": round(g_div, 3),
        "top5_fivegram_repeat_counts": top_gram_repeats,
    }


def keyword_hits(text: str) -> dict:
    low = text.lower()
    return {k: low.count(k) for k in KEYWORDS}


def digest_row(d: dict) -> dict:
    rt = d.get("response_text", "") or ""
    return {
        "case_id": d.get("case_id"),
        "axes": d.get("axes"),
        "chars": len(rt),
        "out_tok": (d.get("usage") or {}).get("output_tokens"),
        "stop": d.get("stop_reason"),
        "shape": line_shape(rt),
        "para": paragraph_shape(rt),
        "rep": repetition_metrics(rt),
        "kw": keyword_hits(rt),
    }


def fmt_block(label: str, row: dict) -> list[str]:
    out = []
    out.append("=" * 72)
    out.append(
        f"[{label}] {row['case_id']}  chars={row['chars']}  out_tok={row['out_tok']}  stop={row['stop']}"
    )
    out.append(f"axes: {row['axes']}")
    out.append("shape:     " + "  ".join(f"{k}={v}" for k, v in row["shape"].items()))
    out.append("paragraph: " + "  ".join(f"{k}={v}" for k, v in row["para"].items()))
    out.append("repetition:")
    for k, v in row["rep"].items():
        out.append(f"  {k}: {v}")
    out.append("keyword_counts:")
    kw_nonzero = {k: v for k, v in row["kw"].items() if v}
    for k, v in sorted(kw_nonzero.items(), key=lambda x: -x[1]):
        out.append(f"  {k}: {v}")
    out.append("")
    return out


def main() -> None:
    files = sorted(SMOKE.glob("*.json"))
    digests = [digest_row(json.loads(f.read_text())) for f in files]
    digests.sort(key=lambda r: r["chars"])

    picks = [
        ("SHORTEST", digests[0]),
        ("MEDIAN", digests[len(digests) // 2]),
        ("LONGEST", digests[-1]),
    ]

    lines_out: list[str] = []
    lines_out.append(f"# structural digest — {len(digests)} smoke cases  (NO raw text)")
    lines_out.append(
        f"# char range: {digests[0]['chars']} .. {digests[-1]['chars']}"
    )
    lines_out.append("")

    # per-case compact table across ALL cases (numeric only)
    lines_out.append("## per-case compact (all cases)")
    header = [
        "case_id", "chars", "out_tok", "lines", "non_blank", "numbered", "bullet",
        "paragraphs", "para_max", "uniq_word_ratio", "line_dup_ratio",
        "fivegram_div", "top5gram_max",
    ]
    lines_out.append("\t".join(header))
    for r in digests:
        top5g = r["rep"]["top5_fivegram_repeat_counts"]
        lines_out.append("\t".join([
            str(r["case_id"]),
            str(r["chars"]),
            str(r["out_tok"]),
            str(r["shape"]["lines"]),
            str(r["shape"]["non_blank"]),
            str(r["shape"]["numbered"]),
            str(r["shape"]["bullet"]),
            str(r["para"]["paragraphs"]),
            str(r["para"]["para_chars_max"]),
            str(r["rep"]["unique_word_ratio"]),
            str(r["rep"]["line_dup_ratio"]),
            str(r["rep"]["fivegram_diversity"]),
            str(top5g[0] if top5g else 0),
        ]))
    lines_out.append("")

    # detail blocks for 3 picks
    lines_out.append("## detail (SHORTEST / MEDIAN / LONGEST)")
    lines_out.append("")
    for label, r in picks:
        lines_out.extend(fmt_block(label, r))

    OUT.write_text("\n".join(lines_out))
    print(f"wrote: {OUT}")
    print(f"picks: {[(lbl, r['case_id'], r['chars']) for lbl, r in picks]}")


if __name__ == "__main__":
    main()
