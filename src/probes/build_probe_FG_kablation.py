"""k-ablation prompt builder for Probes F (random retrieval) and G
(structural-match retrieval).

Renders new conditions F1, F5, G1, G5 (k=1 and k=5) on top of the SAME
frozen baseline prompts used by the k=3 main run (data/parsed/prompts_main/
<case_id>__baseline.json, byte-identical reuse — not regenerated, not
reconstructed from beta_parsed/alpha). k=3 (conditions F, G) already exists
from the main run and is not re-rendered here.

Everything in this script is local compute only: no API calls, no spend.

Design notes / caveats (documented, not hidden defaults):
  - Probe G is deterministic top-k by score, so G1's pick is exactly G3's
    (frozen main-run G) top-1, and G5's first 3 picks are exactly G3's
    3 picks (G's ranking doesn't change with k) -- true nesting.
  - Probe F uses random.Random(seed).sample(pool, k=k) with the SAME
    per-case seed tag ("F") as the frozen k=3 run, for reproducibility.
    Python's random.sample() does NOT guarantee that the k=1 result is a
    prefix of the k=3 result or that k=5 is a superset -- sample() takes a
    different internal code path depending on k relative to len(pool), so
    F1/F3/F5 picks for a given case are independently-drawn (same-seeded,
    but not nested) subsets of the same 4,221-entry pool. This is called
    out explicitly so it isn't mistaken for a nested design.

Usage:
  python3 scripts/build_probe_FG_kablation.py \\
      --cohort data/parsed/cohort_batch0.json \\
      --beta-parsed-dir data/parsed/beta_parsed \\
      --prompts-main-dir data/parsed/prompts_main \\
      --out-dir data/parsed/prompts_kablation \\
      --k 1 5 --conditions F G --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Reuse Probe G's feature extraction / scoring and the shared block
# formatter verbatim from the frozen builder, so the k-ablation is
# byte-for-byte consistent with the k=3 main run wherever k is held fixed.
from build_probe_prompts import (  # noqa: E402
    _format_retrieved_block,
    _limitation_patterns,  # noqa: F401  (imported for parity / no-op check)
    _test_case_features,
    _g_score,
)

PROBE_RETRIEVAL_POOL_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "parsed" / "retrieval_pool_v1.json"
)

_POOL_CACHE: list | None = None


def _load_pool() -> list:
    global _POOL_CACHE
    if _POOL_CACHE is None:
        if not PROBE_RETRIEVAL_POOL_PATH.exists():
            raise FileNotFoundError(f"retrieval pool not found at {PROBE_RETRIEVAL_POOL_PATH}")
        _POOL_CACHE = json.loads(PROBE_RETRIEVAL_POOL_PATH.read_text(encoding="utf-8"))["entries"]
    return _POOL_CACHE


def _seed_from_case(case_id: str, tag: str) -> int:
    h = hashlib.sha256(f"{case_id}::{tag}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def apply_probe_F_k(base_prompt: dict, k: int) -> dict:
    pool = _load_pool()
    case_id = base_prompt["case_id"]
    seed = _seed_from_case(case_id, "F")  # same seed tag as frozen k=3 F
    rng = random.Random(seed)
    picks = rng.sample(pool, k=k)

    block = _format_retrieved_block(picks)
    marker = "## Task"
    user = base_prompt["user"]
    idx = user.find(marker)
    if idx == -1:
        raise ValueError("base user prompt missing '## Task' section")
    new_user = user[:idx] + block + "\n" + user[idx:]

    out = dict(base_prompt)
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["condition"] = f"F{k}"
    out["probe_meta"] = {
        "mode": "random_retrieval",
        "k": k,
        "k_ablation_of": "F",
        "seed_rule": "sha256(case_id + '::F')[:8]  (same tag as frozen k=3 F; NOT nested vs k=3 -- see script docstring)",
        "seed_hex": f"{seed:08x}",
        "retrieved_case_ids": [p["case_id"] for p in picks],
        "retrieved_app_nums": [p["app_num"] for p in picks],
        "retrieved_chars_total": sum(len(p["pre_text"]) + len(p["post_text"]) for p in picks),
        "pool_version": "v1",
    }
    return out


def apply_probe_G_k(base_prompt: dict, beta_rec: dict, k: int) -> dict:
    pool = _load_pool()
    case_id = base_prompt["case_id"]
    test_feats = _test_case_features(beta_rec)

    scored = [(_g_score(e, test_feats), e) for e in pool]
    scored.sort(key=lambda x: (-x[0], str(x[1].get("app_num") or "")))
    picks = [e for _, e in scored[:k]]
    pick_scores = [round(s, 4) for s, _ in scored[:k]]

    block = _format_retrieved_block(picks)
    marker = "## Task"
    user = base_prompt["user"]
    idx = user.find(marker)
    if idx == -1:
        raise ValueError("base user prompt missing '## Task' section")
    new_user = user[:idx] + block + "\n" + user[idx:]

    out = dict(base_prompt)
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["condition"] = f"G{k}"
    out["probe_meta"] = {
        "mode": "structural_retrieval",
        "k": k,
        "k_ablation_of": "G",
        "test_statute": test_feats["statute"],
        "test_subsection": test_feats["subsection"],
        "test_patterns": sorted(test_feats["patterns"]),
        "retrieved_case_ids": [p["case_id"] for p in picks],
        "retrieved_app_nums": [p["app_num"] for p in picks],
        "retrieved_scores": pick_scores,
        "retrieved_chars_total": sum(len(p["pre_text"]) + len(p["post_text"]) for p in picks),
        "pool_version": "v1",
        "nested_vs_k3": "yes -- deterministic top-k ranking, G{k}'s picks are a prefix/superset of frozen k3 G's picks",
    }
    return out


BUILDERS = {"F": apply_probe_F_k, "G": apply_probe_G_k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--prompts-main-dir", required=True, type=Path,
                     help="dir containing frozen <case_id>__baseline.json (byte-identical reuse)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--k", nargs="+", type=int, required=True, choices=[1, 5],
                     help="k values to render (1 and/or 5; k=3 already exists from the main run)")
    ap.add_argument("--conditions", nargs="+", default=["F", "G"], choices=["F", "G"])
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_rendered = 0
    n_missing_baseline = 0
    n_missing_beta = 0
    for c in cases:
        case_id = c["case_id"]
        base_path = args.prompts_main_dir / f"{case_id}__baseline.json"
        if not base_path.exists():
            n_missing_baseline += 1
            continue
        base_prompt = json.loads(base_path.read_text(encoding="utf-8"))

        beta_path = args.beta_parsed_dir / f"{c['app_num']}.json"
        beta_rec = None
        if beta_path.exists():
            beta_rec = json.loads(beta_path.read_text(encoding="utf-8"))
        elif "G" in args.conditions:
            n_missing_beta += 1
            continue

        for cond in args.conditions:
            for k in args.k:
                if cond == "F":
                    probe = apply_probe_F_k(base_prompt, k)
                else:
                    probe = apply_probe_G_k(base_prompt, beta_rec, k)
                out_path = args.out_dir / f"{case_id}__{probe['condition']}.json"
                out_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
                n_rendered += 1

    print(f"Rendered {n_rendered} prompts "
          f"({n_missing_baseline} missing baseline, {n_missing_beta} missing beta) -> {args.out_dir}")


if __name__ == "__main__":
    main()
