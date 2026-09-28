#!/usr/bin/env python3
"""Prompt token stats per probe x k (GPT cl100k tokenizer) + truncation rate.

Backs the 'more exemplars => longer prompts' / context-budget argument for k=3.
Requires tiktoken:  pip install tiktoken

Usage:
    python3 scripts/prompt_token_stats.py
    python3 scripts/prompt_token_stats.py --ks 1 3 5 10
"""
import argparse, glob, json, statistics as st
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tok = lambda s: len(enc.encode(s))
TRUNC = "[... truncated ...]"


def stats(files):
    toks, trunc = [], 0
    for f in files:
        d = json.load(open(f))
        toks.append(tok(d.get("system", "") + d.get("user", "")))
        if TRUNC in d.get("user", ""):
            trunc += 1
    toks.sort()
    return (min(toks), int(st.median(toks)), int(st.mean(toks)), max(toks),
            round(100 * trunc / len(toks), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    a = ap.parse_args()

    print(f"{'probe':5} {'k':>3} {'n':>4} {'min':>6} {'median':>7} {'mean':>6} {'max':>7} {'trunc%':>7}")
    for probe in ("F", "G"):
        for k in a.ks:
            fs = glob.glob(f"data/parsed/prompts_k{k}/*__{probe}.json")
            if not fs:
                print(f"{probe:5} {k:>3}    –  (no files at data/parsed/prompts_k{k})")
                continue
            mn, md, mean, mx, tr = stats(fs)
            print(f"{probe:5} {k:>3} {len(fs):>4} {mn:>6} {md:>7} {mean:>6} {mx:>7} {tr:>7}")


if __name__ == "__main__":
    main()
