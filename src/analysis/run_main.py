"""Main Stage-1 experiment runner with resume + retry + atomic write.

Executes per-(case, condition, rep) model calls, skipping completed ones.
Idempotent: re-running the same command resumes where it left off.

File layout:
  <prompts_dir>/<case_id>__<cond>.json
  <out_dir>/<case_id>__<cond>__<model>__rep<r>.json
  <out_dir>/_manifest.jsonl      (append-only, one line per call attempt)
  <out_dir>/_failures.jsonl      (append-only, hard-fail only)

Fairness/design decisions (see docs/experiment_plan_check.md):
  - condition order shuffled per (case, rep) via deterministic seed → removes
    temporal model-drift confound across conditions
  - per-call system/user prompt sha256 stored → post-hoc prompt-drift audit
  - per-condition status aggregated in final summary → surfaces asymmetric
    hard-failure concentration (e.g., probe C context-length)

Usage:
  python3 scripts/run_main.py \\
      --cohort data/parsed/cohort_batch0.json \\
      --prompts-dir data/parsed/prompts_main \\
      --out-dir outputs/main \\
      --model sonnet4 \\
      --conditions baseline,A,B,C,D,E,F,G \\
      --reps 3 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Alias → (vendor, canonical model id). Vendor drives API format.
MODELS = {
    # Anthropic
    "sonnet4":  ("anthropic", "claude-sonnet-4-20250514"),
    "haiku4.5": ("anthropic", "claude-haiku-4-5-20251001"),
    "opus4.6":  ("anthropic", "claude-opus-4-6"),
    # OpenAI — model IDs per 2026-04 catalog; override via --model-id if needed.
    "gpt5.4":     ("openai",    "gpt-5.4"),
    "gpt5-mini":  ("openai",    "gpt-5-mini"),
    # gpt-4o-mini: pre-registered fallback for smaller-OpenAI slot when
    # gpt-5-mini exhibits format-following failure (smoke 2026-04-18 confirmed).
    "gpt4o-mini": ("openai",    "gpt-4o-mini"),
}

# OpenRouter model slugs (used only with --openrouter). OpenRouter is
# OpenAI-API-compatible, so ALL models route through the OpenAI request/response
# format regardless of underlying vendor.
# ⚠️ CONFIRM THESE before a real run — OpenRouter's catalog (fetched 2026-07)
# does NOT list the pinned 2025-era Sonnet-4 / Haiku-4.5 or gpt-4o-mini. The
# entries below fall back to rolling "-latest" aliases or nearest models, which
# are NOT the exact models the paper pinned. Override per-run with --model-id.
OPENROUTER_SLUGS = {
    "sonnet4":    "anthropic/claude-sonnet-latest",  # ⚠ rolling alias, not pinned 4
    "haiku4.5":   "anthropic/claude-haiku-latest",   # ⚠ rolling alias, not pinned 4.5
    "gpt5.4":     "openai/gpt-5.4",                   # exact match available
    "gpt4o-mini": "openai/gpt-mini-latest",           # ⚠ gpt-4o-mini not on OpenRouter
    "opus4.6":    "anthropic/claude-opus-latest",
    "gpt5-mini":  "openai/gpt-mini-latest",
}

# (input_$/M, output_$/M). Missing entries → cost=0 (logged but not scored).
PRICE_PER_M = {
    "claude-sonnet-4-20250514":  (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8,  4.0),
    "claude-opus-4-6":           (15.0, 75.0),
    # OpenAI estimates (user should verify current pricing)
    "gpt-5.4":     (1.25, 10.0),
    "gpt-5-mini":  (0.25,  2.0),
    "gpt-4o-mini": (0.15,  0.6),
    # OpenRouter slugs (approx; OR passes through provider price + small margin)
    "anthropic/claude-sonnet-latest": (3.0, 15.0),
    "anthropic/claude-haiku-latest":  (0.8,  4.0),
    "openai/gpt-5.4":                 (1.25, 10.0),
    "openai/gpt-mini-latest":         (0.25,  2.0),
}

# Default env var per vendor — CLI --key-env overrides.
DEFAULT_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY_1",
    "openai":    "OPENAI_API_KEY_1",
}

HARD_FAIL_STATUSES = {400, 401, 403, 404}


# ---------- small utils ---------------------------------------------------

def iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def jsonl_append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffle_conditions(conds: list[str], case_id: str, rep: int, seed: int) -> list[str]:
    """Deterministic shuffle per (case, rep) to remove condition-order drift confound."""
    key = f"{case_id}__rep{rep}__seed{seed}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = random.Random(h)
    out = list(conds)
    rng.shuffle(out)
    return out


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        d.get("response_text") is not None
        and d.get("usage") is not None
        and d.get("stop_reason") is not None
    )


def estimate_cost(usage: dict, model: str) -> float:
    inp_p, out_p = PRICE_PER_M.get(model, (0.0, 0.0))
    return round(
        usage.get("input_tokens", 0) / 1e6 * inp_p
        + usage.get("output_tokens", 0) / 1e6 * out_p,
        5,
    )


# ---------- API calls ----------------------------------------------------

def call_anthropic_raw(system: str, user: str, *, model: str, api_key: str,
                       temperature: float, max_tokens: int, timeout: float):
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
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body,
                         timeout=timeout)
    latency = round(time.time() - t0, 2)
    resp.raise_for_status()
    return resp.json(), latency


# OpenAI models that support only default temperature (reasoning + mini
# families). Paper §6.3 also commits GPT-5 mini to API-default temperature.
OPENAI_DEFAULT_TEMP_ONLY = {
    "gpt-5-mini", "gpt-5-nano",
    "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini",
}


def call_openai_raw(system: str, user: str, *, model: str, api_key: str,
                    temperature: float, max_tokens: int, timeout: float,
                    api_url: str = OPENAI_API_URL):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # GPT-5 series uses max_completion_tokens; older models accept max_tokens.
        "max_completion_tokens": max_tokens,
    }
    if model not in OPENAI_DEFAULT_TEMP_ONLY:
        body["temperature"] = temperature

    def _post():
        t = time.time()
        r = requests.post(api_url, headers=headers, json=body,
                          timeout=timeout)
        return r, round(time.time() - t, 2)

    resp, latency = _post()

    # Fallback 1: older model rejects max_completion_tokens → retry with max_tokens
    if resp.status_code == 400 and "max_completion_tokens" in (resp.text or ""):
        body.pop("max_completion_tokens", None)
        body["max_tokens"] = max_tokens
        resp, latency = _post()

    # Fallback 2: unknown model rejects temperature → drop and retry
    if resp.status_code == 400 and "temperature" in (resp.text or ""):
        body.pop("temperature", None)
        resp, latency = _post()

    resp.raise_for_status()
    return resp.json(), latency


def _extract_text(data: dict, vendor: str) -> str:
    if vendor == "anthropic":
        out = ""
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                out += block.get("text", "")
        return out
    if vendor == "openai":
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""
    return ""


def _extract_usage(data: dict, vendor: str) -> dict:
    u = data.get("usage") or {}
    if vendor == "anthropic":
        return u
    if vendor == "openai":
        return {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "total_tokens": u.get("total_tokens", 0),
        }
    return {}


def _extract_stop(data: dict, vendor: str):
    if vendor == "anthropic":
        return data.get("stop_reason")
    if vendor == "openai":
        choices = data.get("choices") or []
        return choices[0].get("finish_reason") if choices else None
    return None


def call_with_retry(system: str, user: str, *, vendor: str, model: str,
                    api_key: str, temperature: float, max_tokens: int,
                    max_retries: int = 4, timeout: float = 180.0,
                    api_url: str | None = None):
    """Retry 429/5xx/timeout/conn only. 4xx (except 429) = hard fail."""
    if vendor == "anthropic":
        raw_fn = call_anthropic_raw
        extra = {}
    elif vendor == "openai":
        raw_fn = call_openai_raw
        extra = {"api_url": api_url or OPENAI_API_URL}
    else:
        return None, 0.0, 0, ("bad_vendor", vendor)

    attempts = 0
    last_err = None
    while True:
        attempts += 1
        try:
            data, latency = raw_fn(
                system, user,
                model=model, api_key=api_key,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout, **extra,
            )
            return data, latency, attempts, None
        except requests.HTTPError as e:
            status = e.response.status_code
            body = (e.response.text or "")[:300]
            last_err = (f"http_{status}", body)
            if status in HARD_FAIL_STATUSES:
                return None, 0.0, attempts, last_err
            if status == 429:
                ra = e.response.headers.get("retry-after", "")
                try:
                    wait = float(ra) if ra else 2 ** (attempts - 1)
                except ValueError:
                    wait = 2 ** (attempts - 1)
                wait = min(max(wait, 1), 60)
            elif 500 <= status < 600:
                wait = min(2 ** (attempts - 1), 60)
            else:
                # Unknown 4xx (e.g., 408) — one retry path
                wait = min(2 ** (attempts - 1), 30)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = (type(e).__name__, str(e)[:300])
            wait = min(2 ** (attempts - 1), 60)
        except Exception as e:
            # Unknown — do not retry
            return None, 0.0, attempts, (type(e).__name__, str(e)[:300])

        if attempts > max_retries:
            return None, 0.0, attempts, last_err
        print(f"    retry {attempts}/{max_retries} after {wait:.0f}s  ({last_err[0]})",
              file=sys.stderr)
        time.sleep(wait)


# ---------- prompt loading ------------------------------------------------

def load_prompt(prompts_dir: Path, case_id: str, cond: str) -> dict | None:
    p = prompts_dir / f"{case_id}__{cond}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------- main ----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--prompts-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model", default="sonnet4", choices=list(MODELS.keys()),
                    help="alias from MODELS dict. Each alias pins a vendor + canonical id.")
    ap.add_argument("--model-id", default=None,
                    help="override the canonical model id (keeps vendor from alias)")
    ap.add_argument("--conditions", default="baseline,A,B,C,D,E,F,G",
                    help="comma-separated condition names; prompt files must exist")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42,
                    help="global seed for per-(case,rep) condition shuffle")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--key-env", default=None,
                    help="override env var name (defaults from vendor: ANTHROPIC_API_KEY_1 / OPENAI_API_KEY_1; OPENROUTER_API_KEY with --openrouter)")
    ap.add_argument("--openrouter", action="store_true",
                    help="route all calls through OpenRouter (single OPENROUTER_API_KEY, OpenAI-compatible format). Model slug from OPENROUTER_SLUGS unless --model-id given.")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run completed calls (overrides skip)")
    ap.add_argument("--skip-failed", action="store_true",
                    help="skip tuples already in _failures.jsonl (don't retry hard fails)")
    ap.add_argument("--case-limit", type=int, default=0,
                    help="0 = all cases; otherwise first N cases (for partial runs)")
    ap.add_argument("--case-slice", default=None,
                    help='"start:end" Python-slice over the cohort (for within-model parallelism). '
                         'Applied after --case-limit if both set.')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions:
        print("ERROR: --conditions empty", file=sys.stderr)
        sys.exit(2)

    if args.openrouter:
        # OpenRouter is OpenAI-API-compatible for every model → force "openai"
        # request/response format and point at the OpenRouter endpoint.
        vendor = "openai"
        api_url = OPENROUTER_API_URL
        model_id = args.model_id or OPENROUTER_SLUGS.get(args.model)
        if not model_id:
            print(f"ERROR: no OpenRouter slug for alias '{args.model}'. "
                  f"Pass --model-id with an OpenRouter slug.", file=sys.stderr)
            sys.exit(2)
        key_env = args.key_env or "OPENROUTER_API_KEY"
    else:
        vendor, default_model_id = MODELS[args.model]
        model_id = args.model_id or default_model_id
        api_url = None
        key_env = args.key_env or DEFAULT_KEY_ENV[vendor]
    api_key = os.environ.get(key_env, "").strip()
    if not api_key and not args.dry_run:
        print(f"ERROR: env var {key_env} not set (vendor={vendor}).",
              file=sys.stderr)
        sys.exit(2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "_manifest.jsonl"
    failures_path = args.out_dir / "_failures.jsonl"

    # Known-hard-fail set for --skip-failed
    skip_set: set[tuple[str, str, int]] = set()
    if args.skip_failed and failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                skip_set.add((r["case"], r["cond"], r["rep"]))
            except Exception:
                continue

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"]
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    if args.case_slice:
        start_s, end_s = args.case_slice.split(":")
        start_i = int(start_s) if start_s else 0
        end_i = int(end_s) if end_s else len(cases)
        cases = cases[start_i:end_i]

    total_planned = len(cases) * len(conditions) * args.reps
    print(f"[plan] cases={len(cases)}  cond={len(conditions)}  reps={args.reps}  "
          f"→ total={total_planned} calls  vendor={vendor}  "
          f"model={args.model} ({model_id})  key_env={key_env}")
    print(f"[plan] conditions = {conditions}")
    print(f"[plan] out_dir = {args.out_dir}  skip_failed={args.skip_failed}  "
          f"overwrite={args.overwrite}")

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "skip": 0, "fail": 0})
    total_cost = 0.0
    total_in = 0
    total_out = 0
    n_done = 0
    t_start = time.time()

    for ci, c in enumerate(cases):
        case_id = c["case_id"]
        for rep in range(args.reps):
            cond_order = shuffle_conditions(conditions, case_id, rep, args.seed)
            for cond in cond_order:
                n_done += 1
                out_path = args.out_dir / f"{case_id}__{cond}__{args.model}__rep{rep}.json"

                # skip: already complete
                if not args.overwrite and is_complete(out_path):
                    stats[cond]["skip"] += 1
                    jsonl_append(manifest_path, {
                        "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                        "model": args.model, "status": "skip", "attempts": 0,
                    })
                    continue

                # skip: known hard-fail
                if (case_id, cond, rep) in skip_set:
                    stats[cond]["skip"] += 1
                    jsonl_append(manifest_path, {
                        "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                        "model": args.model, "status": "skip_failed",
                    })
                    continue

                prompt = load_prompt(args.prompts_dir, case_id, cond)
                if prompt is None:
                    stats[cond]["fail"] += 1
                    err = ("prompt_missing",
                           f"{case_id}__{cond}.json not in {args.prompts_dir}")
                    jsonl_append(failures_path, {
                        "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                        "model": args.model, "status": "fail", "attempts": 0,
                        "err": err[0], "err_detail": err[1],
                    })
                    jsonl_append(manifest_path, {
                        "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                        "model": args.model, "status": "fail", "err": err[0],
                    })
                    print(f"  [{n_done}/{total_planned}] MISS {case_id}/{cond}/rep{rep}  prompt missing",
                          file=sys.stderr)
                    continue

                sys_sha = sha16(prompt["system"])
                usr_sha = sha16(prompt["user"])

                if args.dry_run:
                    stats[cond]["skip"] += 1
                    print(f"  [{n_done}/{total_planned}] DRY {case_id}/{cond}/rep{rep}  "
                          f"user_chars={prompt.get('user_chars')}")
                    continue

                data, latency, attempts, err = call_with_retry(
                    prompt["system"], prompt["user"],
                    vendor=vendor, model=model_id, api_key=api_key,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    max_retries=args.max_retries, timeout=args.timeout,
                    api_url=api_url,
                )

                if err is not None:
                    stats[cond]["fail"] += 1
                    jsonl_append(failures_path, {
                        "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                        "model": args.model, "status": "fail", "attempts": attempts,
                        "err": err[0], "err_detail": err[1],
                        "sys_sha": sys_sha, "usr_sha": usr_sha,
                    })
                    jsonl_append(manifest_path, {
                        "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                        "model": args.model, "status": "fail", "attempts": attempts,
                        "err": err[0],
                    })
                    print(f"  [{n_done}/{total_planned}] FAIL {case_id}/{cond}/rep{rep}  "
                          f"attempts={attempts}  err={err[0]}", file=sys.stderr)
                    continue

                response_text = _extract_text(data, vendor)
                usage = _extract_usage(data, vendor)
                stop_reason = _extract_stop(data, vendor)
                cost = estimate_cost(usage, model_id)
                total_cost += cost
                total_in += usage.get("input_tokens", 0)
                total_out += usage.get("output_tokens", 0)

                record = {
                    "case_id": case_id,
                    "condition": cond,
                    "rep": rep,
                    "proceeding_number": prompt.get("proceeding_number"),
                    "app_num": prompt.get("app_num"),
                    "axes": prompt.get("axes"),
                    "probe_meta": prompt.get("probe_meta"),
                    "vendor": vendor,
                    "model": model_id,
                    "model_alias": args.model,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "user_chars": prompt.get("user_chars"),
                    "sys_sha": sys_sha,
                    "usr_sha": usr_sha,
                    "response_text": response_text,
                    "usage": usage,
                    "stop_reason": stop_reason,
                    "latency_sec": latency,
                    "cost_usd": cost,
                    "attempts": attempts,
                    "timestamp": iso_utc(),
                }
                atomic_write_text(out_path, json.dumps(record, ensure_ascii=False, indent=2))

                stats[cond]["ok"] += 1
                jsonl_append(manifest_path, {
                    "ts": iso_utc(), "case": case_id, "cond": cond, "rep": rep,
                    "vendor": vendor, "model": args.model,
                    "status": "ok", "attempts": attempts,
                    "cost": cost, "lat": latency,
                    "in_tok": usage.get("input_tokens"),
                    "out_tok": usage.get("output_tokens"),
                    "stop": stop_reason,
                    "sys_sha": sys_sha, "usr_sha": usr_sha,
                })
                print(f"  [{n_done}/{total_planned}] OK   {case_id}/{cond}/rep{rep}  "
                      f"in={usage.get('input_tokens','?')}  out={usage.get('output_tokens','?')}  "
                      f"${cost}  {latency}s  att={attempts}")

    # ---- final summary ----
    elapsed = time.time() - t_start
    print()
    print(f"[done] elapsed={elapsed/60:.1f} min  total_in={total_in}  total_out={total_out}  "
          f"total_cost=${round(total_cost, 4)}")
    print("[per-condition]")
    print(f"  {'cond':<10} {'ok':>6} {'skip':>6} {'fail':>6}")
    grand = {"ok": 0, "skip": 0, "fail": 0}
    for cond in conditions:
        s = stats[cond]
        for k in grand:
            grand[k] += s[k]
        print(f"  {cond:<10} {s['ok']:>6} {s['skip']:>6} {s['fail']:>6}")
    print(f"  {'TOTAL':<10} {grand['ok']:>6} {grand['skip']:>6} {grand['fail']:>6}")
    if grand["fail"] > 0:
        print(f"  → see {failures_path} for details; re-run same command to retry, "
              f"or --skip-failed to ignore.")


if __name__ == "__main__":
    main()
