"""Build Probe C decoy / mechanism-true index.

For each cohort case (v1), select ONE prior-art reference to inject
from the retrieval pool (non-cohort β). Subcondition assigned per
case via hash-mod-2:
  - C_decoy    : wording-similar, different statute (surface trap)
  - C_mechtrue : wording-dissimilar, same statute (grounded signal)

Wording similarity = token Jaccard over the union of each case's
`ctnf.prior_art_refs` strings. Statute = `rejection_instances[0].statute_section`.

Output: data/parsed/probe_c_index_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _tokens(text: str) -> frozenset:
    return frozenset(WORD_RE.findall((text or "").lower()))


def _pa_token_set(prior_art_refs: list) -> frozenset:
    """Union of tokens across all prior_art_ref strings for a case."""
    out = set()
    for ref in prior_art_refs or []:
        out.update(WORD_RE.findall((ref or "").lower()))
    return frozenset(out)


def _first_pa_ref(prior_art_refs: list) -> str:
    """Pick first non-empty reference for injection."""
    for r in prior_art_refs or []:
        r = (r or "").strip()
        if r:
            return r
    return ""


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _assign_subcondition(case_id: str) -> str:
    h = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 2
    return "C_decoy" if h == 0 else "C_mechtrue"


def _case_facts(beta_rec: dict) -> dict:
    ctnf = beta_rec.get("ctnf") or {}
    rinst = ctnf.get("rejection_instances") or []
    statute = rinst[0].get("statute_section") if rinst else None
    statute = statute or ""
    refs = ctnf.get("prior_art_refs") or []
    return {
        "statute": statute,
        "prior_art_refs": refs,
        "pa_tokens": _pa_token_set(refs),
        "first_pa_ref": _first_pa_ref(refs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cohort_cases = cohort["cases"]
    cohort_app_nums = {c["app_num"]: c["case_id"] for c in cohort_cases}

    # Build pool facts (all non-cohort β with statute + prior_art_refs)
    print("scanning β for pool facts ...")
    pool_facts = []
    n_scanned = 0
    n_skipped = 0
    for fn in sorted(args.beta_parsed_dir.iterdir()):
        if fn.suffix != ".json":
            continue
        try:
            rec = json.loads(fn.read_text(encoding="utf-8"))
        except Exception:
            continue
        case_id = rec.get("case_id")
        if case_id in cohort_app_nums.values():
            continue
        facts = _case_facts(rec)
        if not facts["statute"] or not facts["first_pa_ref"]:
            n_skipped += 1
            continue
        pool_facts.append({
            "case_id": case_id,
            "app_num": rec.get("app_num"),
            "statute": facts["statute"],
            "first_pa_ref": facts["first_pa_ref"],
            "pa_tokens": facts["pa_tokens"],
        })
        n_scanned += 1
    print(f"pool facts: {n_scanned} usable (skipped {n_skipped})")

    # Per-cohort-case selection
    out_cases: dict[str, dict] = {}
    skip_counts: Counter = Counter()
    subcond_counts: Counter = Counter()

    for c in cohort_cases:
        case_id = c["case_id"]
        bp = args.beta_parsed_dir / f"{c['app_num']}.json"
        if not bp.exists():
            skip_counts["no_beta"] += 1
            continue
        beta = json.loads(bp.read_text(encoding="utf-8"))
        facts = _case_facts(beta)
        if not facts["statute"]:
            out_cases[case_id] = {"mode": "skipped", "reason": "no_test_statute"}
            skip_counts["no_test_statute"] += 1
            continue
        if not facts["pa_tokens"]:
            out_cases[case_id] = {"mode": "skipped", "reason": "no_test_pa"}
            skip_counts["no_test_pa"] += 1
            continue

        sub = _assign_subcondition(case_id)
        test_tokens = facts["pa_tokens"]
        test_statute = facts["statute"]

        # filter candidates
        if sub == "C_decoy":
            cands = [p for p in pool_facts if p["statute"] != test_statute]
            # maximize similarity; tie → lowest app_num
            scored = [(_jaccard(test_tokens, p["pa_tokens"]), p) for p in cands]
            scored.sort(key=lambda x: (-x[0], str(x[1].get("app_num") or "")))
        else:  # C_mechtrue
            cands = [p for p in pool_facts if p["statute"] == test_statute]
            # minimize similarity; tie → lowest app_num
            scored = [(_jaccard(test_tokens, p["pa_tokens"]), p) for p in cands]
            scored.sort(key=lambda x: (x[0], str(x[1].get("app_num") or "")))

        if not scored:
            out_cases[case_id] = {"mode": "skipped",
                                  "reason": f"no_candidate_for_{sub}",
                                  "subcondition": sub,
                                  "test_statute": test_statute}
            skip_counts[f"no_candidate_for_{sub}"] += 1
            continue

        sim, best = scored[0]
        out_cases[case_id] = {
            "mode": sub,
            "subcondition": sub,
            "test_statute": test_statute,
            "n_test_pa_refs": len(facts["prior_art_refs"]),
            "n_test_pa_tokens": len(test_tokens),
            "injected_ref": best["first_pa_ref"],
            "injected_source_case": best["case_id"],
            "injected_source_app": best["app_num"],
            "injected_source_statute": best["statute"],
            "wording_similarity": round(sim, 4),
            "n_candidates": len(scored),
        }
        subcond_counts[sub] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"version": "v1",
                    "cohort_file": args.cohort.name,
                    "pool_size": n_scanned,
                    "n_cohort": len(cohort_cases),
                    "subcondition_counts": dict(subcond_counts),
                    "skip_counts": dict(skip_counts),
                    "cases": out_cases},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\n=== summary ===")
    print(f"  subcondition counts: {dict(subcond_counts)}")
    print(f"  skip counts: {dict(skip_counts)}")
    print(f"  saved to {args.out}")
    # quick stats on similarity scores per subcondition
    decoy_sims = [v["wording_similarity"] for v in out_cases.values()
                  if v.get("mode") == "C_decoy"]
    mech_sims = [v["wording_similarity"] for v in out_cases.values()
                 if v.get("mode") == "C_mechtrue"]
    if decoy_sims:
        decoy_sims.sort()
        print(f"  C_decoy sim (should be HIGH):  "
              f"n={len(decoy_sims)} min={decoy_sims[0]:.3f} "
              f"median={decoy_sims[len(decoy_sims)//2]:.3f} "
              f"max={decoy_sims[-1]:.3f}")
    if mech_sims:
        mech_sims.sort()
        print(f"  C_mechtrue sim (should be LOW): "
              f"n={len(mech_sims)} min={mech_sims[0]:.3f} "
              f"median={mech_sims[len(mech_sims)//2]:.3f} "
              f"max={mech_sims[-1]:.3f}")


if __name__ == "__main__":
    main()
