"""Mine top-k canonical amendment phrases from the retrieval pool for C5.

Paper §5: "top-k canonical amendment phrases (mined from the retrieval
pool)". We use 4-grams over whitespace-tokenized pool post_text,
lowercased, punctuation-stripped. Minimum pool frequency filter keeps
only phrases with meaningful corpus support.

Output: data/parsed/canonical_phrases_v1.json
  {
    "version": "v1",
    "source_pool": "retrieval_pool_v1.json",
    "ngram": 4,
    "min_pool_freq": 20,
    "top_k": 50,
    "phrases": [{"phrase": "...", "pool_count": N, "pool_docs": M}, ...]
  }

Usage:
  python3 scripts/mine_canonical_phrases.py \\
      --pool data/parsed/retrieval_pool_v1.json \\
      --out data/parsed/canonical_phrases_v1.json \\
      --ngram 4 --top-k 50 --min-pool-freq 20
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[a-z][a-z\-']+")

# Tokens that appear only via SVG/XML/OCR markup leakage, never in real
# claim language. Phrases containing any of these are rejected as
# extraction artifacts.
ARTIFACT_TOKENS = frozenset({
    "svg", "clm", "chemistryblack", "chemistrywhite",
    "xmlns", "xsi", "xlink", "href",
    "inkscape", "sodipodi", "rdf", "cdata",
})


def _tokens(text: str) -> list:
    return WORD_RE.findall((text or "").lower())


def _is_artifact_phrase(phrase: str) -> bool:
    return any(t in ARTIFACT_TOKENS for t in phrase.split())


def _ngrams(tokens: list, n: int):
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i:i + n])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-pool-freq", type=int, default=20)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    entries = pool["entries"]
    print(f"loaded pool: {len(entries)} entries")

    total_ngram_counts: Counter = Counter()
    doc_freq: Counter = Counter()  # number of entries containing the phrase

    for e in entries:
        toks = _tokens(e["post_text"])
        seen_in_entry: set = set()
        for g in _ngrams(toks, args.ngram):
            total_ngram_counts[g] += 1
            seen_in_entry.add(g)
        for g in seen_in_entry:
            doc_freq[g] += 1

    print(f"unique {args.ngram}-grams: {len(total_ngram_counts)}")

    # Filter: min pool freq + reject extraction artifacts
    filtered = [
        {"phrase": g, "pool_count": n, "pool_docs": doc_freq[g]}
        for g, n in total_ngram_counts.items()
        if n >= args.min_pool_freq and not _is_artifact_phrase(g)
    ]
    # Sort by pool_count desc
    filtered.sort(key=lambda x: (-x["pool_count"], x["phrase"]))
    top = filtered[: args.top_k]
    print(f"after min_pool_freq≥{args.min_pool_freq}: {len(filtered)} phrases "
          f"→ top {len(top)}")

    out = {
        "version": "v1",
        "source_pool": args.pool.name,
        "ngram": args.ngram,
        "min_pool_freq": args.min_pool_freq,
        "top_k": args.top_k,
        "n_entries_mined": len(entries),
        "phrases": top,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\nsaved to {args.out}")
    print(f"\ntop 20 canonical phrases:")
    for i, p in enumerate(top[:20], 1):
        print(f"  {i:2d}. \"{p['phrase']}\"  count={p['pool_count']}  "
              f"docs={p['pool_docs']}")


if __name__ == "__main__":
    main()
