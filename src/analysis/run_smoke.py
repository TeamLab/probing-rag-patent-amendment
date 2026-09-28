"""
Smoke test: run baseline amendment generation on N cohort cases with a single
model (Sonnet 4 by default) via the Anthropic Messages API.

Purpose: validate the end-to-end pipeline before launching the full Stage-1
experiment. We check:
  - Prompt renders and fits context
  - API call succeeds and returns structured response
  - Response format (amended claim listing) is parseable
  - Token usage and per-case latency are within expectation

Inputs:
  - data/parsed/cohort_batch0.json
  - data/parsed/prompts_smoke/<case_id>__baseline.json   (rendered prompts)
  - env: ANTHROPIC_API_KEY_1 (from .env)

Output (per case):
  - outputs/smoke/<case_id>__baseline__<model_slug>.json
      { prompt_info, model, temperature, max_tokens, response_text,
        usage, latency_sec, timestamp }

Usage:
  export ANTHROPIC_API_KEY_1=...
  python3 scripts/run_smoke.py \\
      --cohort data/parsed/cohort_batch0.json \\
      --prompts-dir data/parsed/prompts_smoke \\
      --out-dir outputs/smoke \\
      --n 10 --model sonnet4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://api.anthropic.com/v1/messages"

MODELS = {
    "sonnet4": "claude-sonnet-4-20250514",
    "haiku4.5": "claude-haiku-4-5-20251001",
    "opus4.6": "claude-opus-4-6",
}

PRICE_PER_M = {
    # Rough cost estimates for smoke-level budget sanity (USD / 1M tokens)
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
    "claude-opus-4-6": (15.0, 75.0),
}


def call_claude(system: str, user: str, *, model: str, api_key: str,
                temperature: float = 0.3, max_tokens: int = 4096):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    t0 = time.time()
    resp = requests.post(API_URL, headers=headers, json=body, timeout=180)
    latency = round(time.time() - t0, 2)
    resp.raise_for_status()
    return resp.json(), latency


def estimate_cost(usage, model):
    inp_p, out_p = PRICE_PER_M.get(model, (0, 0))
    return round(
        usage.get("input_tokens", 0) / 1e6 * inp_p
        + usage.get("output_tokens", 0) / 1e6 * out_p,
        5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--prompts-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="sonnet4", choices=list(MODELS.keys()))
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--key-env", default="ANTHROPIC_API_KEY_1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get(args.key_env, "").strip()
    if not api_key and not args.dry_run:
        print(f"ERROR: env var {args.key_env} not set.", file=sys.stderr)
        sys.exit(2)

    model_id = MODELS[args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"][: args.n]
    print(f"Smoke: {len(cases)} cases × model={args.model} ({model_id})")

    total_cost = 0.0
    total_input = 0
    total_output = 0
    failures = []

    for i, c in enumerate(cases):
        case_id = c["case_id"]
        prompt_path = args.prompts_dir / f"{case_id}__baseline.json"
        if not prompt_path.exists():
            failures.append((case_id, "prompt_missing"))
            continue
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))

        if args.dry_run:
            print(f"  [{i+1}/{len(cases)}] DRY {case_id}  "
                  f"user_chars={prompt['user_chars']}")
            continue

        try:
            data, latency = call_claude(
                prompt["system"], prompt["user"],
                model=model_id, api_key=api_key,
                temperature=args.temperature, max_tokens=args.max_tokens,
            )
        except requests.HTTPError as e:
            failures.append((case_id, f"http_{e.response.status_code}"))
            print(f"  [{i+1}/{len(cases)}] FAIL {case_id}  {e.response.status_code}  "
                  f"{e.response.text[:200]}", file=sys.stderr)
            continue

        response_text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                response_text += block.get("text", "")
        usage = data.get("usage", {}) or {}
        cost = estimate_cost(usage, model_id)
        total_cost += cost
        total_input += usage.get("input_tokens", 0)
        total_output += usage.get("output_tokens", 0)

        out = {
            "case_id": case_id,
            "proceeding_number": prompt["proceeding_number"],
            "app_num": prompt["app_num"],
            "axes": prompt["axes"],
            "model": model_id,
            "model_alias": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "user_chars": prompt["user_chars"],
            "response_text": response_text,
            "usage": usage,
            "stop_reason": data.get("stop_reason"),
            "latency_sec": latency,
            "cost_usd": cost,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out_path = args.out_dir / f"{case_id}__baseline__{args.model}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        resp_chars = len(response_text)
        print(f"  [{i+1}/{len(cases)}] OK  {case_id}  "
              f"in={usage.get('input_tokens','?')}  out={usage.get('output_tokens','?')}  "
              f"${cost}  {latency}s  resp_chars={resp_chars}")

    print()
    print(f"Total tokens: in={total_input}  out={total_output}  cost=${round(total_cost, 4)}")
    if failures:
        print(f"Failures: {len(failures)}")
        for cid, reason in failures:
            print(f"  {cid}: {reason}")


if __name__ == "__main__":
    main()
