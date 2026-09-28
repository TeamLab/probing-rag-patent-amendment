"""
Select a 100-case stratified test cohort for Phase 1 probe experiments.

Six axes, with marginal targets matching the full β corpus's distribution
(§6.1 of draft_v0). Greedy deficit-minimization algorithm selects cases
iteratively, each step choosing the case that reduces the largest unmet
marginal need across all six axes.

Inputs:
  - data/parsed/alpha_corpus.jsonl          (outcome / statute / tech / year)
  - data/parsed/beta_parsed/*.json          (amendment pattern / XML format)

Output:
  - data/parsed/cohort_batch0.json          (100 case_ids + per-case axis values + marginal report)

Usage:
  python3 scripts/select_cohort.py \\
      --alpha   data/parsed/alpha_corpus.jsonl \\
      --beta-parsed-dir data/parsed/beta_parsed \\
      --out     data/parsed/cohort_batch0.json \\
      --n 100 --seed 42
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


# ---- Marginal targets (sum to n) -----------------------------------------
# See draft_v0 §6.1. Targets sum to 100 for n=100.
TARGETS = {
    "outcome": {
        "Affirmed": 40,
        "Reversed": 40,
        "AIP": 20,
    },
    "statute": {
        "103": 35,         # 103 alone
        "102+103": 20,
        "103+112": 15,
        "101-rel": 15,     # any combo including 101
        "112": 10,         # 112 alone
        "other": 5,
    },
    "tech": {
        "3600": 20,
        "3700": 17,
        "1700": 14,
        "2400": 12,
        "2100": 12,
        "other": 25,
    },
    "pattern": {
        "modify_only": 45,
        "cancel_heavy": 20,
        "mixed": 15,
        "new_or_add": 10,  # new_heavy + modify_and_add
        "other": 10,       # mostly_kept + cancel_and_replace + empty
    },
    "format": {
        "dtd_legacy": 25,
        "ns_claims": 75,
    },
    "year": {
        "2010-14": 25,
        "2015-19": 45,
        "2020-25": 30,
    },
}


# ---- Axis extraction per case --------------------------------------------

def _outcome_bin(sub):
    if not sub:
        return "other"
    if sub == "Affirmed":
        return "Affirmed"
    if sub == "Reversed":
        return "Reversed"
    if sub.startswith("Affirmed-in-Part"):
        return "AIP"
    return "other"


def _statute_bin(issues):
    if not isinstance(issues, list):
        return "other"
    s = set(str(x) for x in issues)
    if s == {"103"}:
        return "103"
    if s == {"102", "103"}:
        return "102+103"
    if s == {"103", "112"}:
        return "103+112"
    if "101" in s:
        return "101-rel"
    if s == {"112"}:
        return "112"
    return "other"


def _tech_bin(tc):
    tc = str(tc or "")
    return tc if tc in {"3600", "3700", "1700", "2400", "2100"} else "other"


def _year_bin(dd):
    dd = str(dd or "")
    # "MM-DD-YYYY" format in PILOT-Bench
    y = dd[-4:] if len(dd) >= 4 else ""
    if not y.isdigit():
        return "unknown"
    yi = int(y)
    if yi <= 2014:
        return "2010-14"
    if yi <= 2019:
        return "2015-19"
    return "2020-25"


def _pattern_bin(diff_summary):
    """Map 8 diff classes → 5 bins matching TARGETS."""
    if not diff_summary:
        return "other"
    kept = diff_summary.get("kept", 0)
    mod = diff_summary.get("modified", 0)
    new = diff_summary.get("new", 0)
    canc = diff_summary.get("cancelled", 0)
    total = kept + mod + new + canc
    if total == 0:
        return "other"
    pct_canc = canc / total
    pct_new = new / total
    pct_mod = mod / total
    if pct_canc > 0.5:
        return "cancel_heavy"
    if pct_mod > 0.5 and pct_new < 0.15 and pct_canc < 0.15:
        return "modify_only"
    if pct_new > 0.3:
        return "new_or_add"
    if 0.15 <= pct_mod <= 0.5 and 0.15 <= pct_canc <= 0.5:
        return "mixed"
    if pct_mod >= 0.3 and pct_new >= 0.15:
        return "new_or_add"
    return "other"


def _format_bin(post_fmt):
    if post_fmt == "dtd_legacy":
        return "dtd_legacy"
    if post_fmt == "ns_claims":
        return "ns_claims"
    return "other"


def compute_case_axes(alpha_row: dict, beta_rec: dict):
    """Return (axes, eligible, reasons).

    Pre-registered eligibility criteria (data-design, not post-hoc):
    probe-based amendment study requires both an identifiable rejection
    and at least one structured prior-art reference (Probe C decoy
    construction requires the latter). Cases failing these are excluded
    from the cohort pool but remain in the released corpus.
    """
    axes = {
        "outcome": _outcome_bin(alpha_row.get("subdecision")),
        "statute": _statute_bin(alpha_row.get("issue_type")),
        "tech": _tech_bin(alpha_row.get("tech_center")),
        "year": _year_bin(alpha_row.get("decision_date")),
        "pattern": _pattern_bin((beta_rec.get("claim_diff") or {}).get("summary")),
        "format": _format_bin((beta_rec.get("post_clm") or {}).get("format")),
    }
    eligible = True
    reasons = []
    if axes["outcome"] == "other":
        eligible = False
        reasons.append("rare_outcome")
    if axes["year"] == "unknown":
        eligible = False
        reasons.append("unknown_year")
    ctnf = beta_rec.get("ctnf") or {}
    n_rej = len(ctnf.get("rejection_instances") or [])
    n_pa = len(ctnf.get("prior_art_refs") or [])
    if n_rej < 1:
        eligible = False
        reasons.append("no_structured_rejection")
    if n_pa < 1:
        eligible = False
        reasons.append("no_structured_prior_art")
    pre_claims = (beta_rec.get("pre_clm") or {}).get("claims") or []
    if not pre_claims:
        eligible = False
        reasons.append("no_pre_claim")
    return axes, eligible, reasons


# ---- Greedy cohort selection ---------------------------------------------

def greedy_select(pool, targets, n=100, seed=42):
    """Iteratively pick cases that minimize unmet marginal need.

    Score(case) = sum over axes of max(target[axis][case.axis_bin] - current[axis][bin], 0).
    Higher score = this case plugs a larger hole.
    """
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)

    selected = []
    taken = set()
    current = {ax: Counter() for ax in targets}

    def deficit_for(bin_name, axis):
        return max(targets[axis].get(bin_name, 0) - current[axis][bin_name], 0)

    def case_score(case):
        return sum(deficit_for(case["axes"][ax], ax) for ax in targets)

    for _ in range(n):
        best = None
        best_score = -1
        for c in pool:
            if c["case_id"] in taken:
                continue
            s = case_score(c)
            if s > best_score:
                best_score = s
                best = c
        if best is None:
            break
        # If best_score == 0 we can no longer improve targets — pick anything still available
        selected.append(best)
        taken.add(best["case_id"])
        for ax in targets:
            current[ax][best["axes"][ax]] += 1

    return selected, current


# ---- Main ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Build pool: inner join on app_num (beta) / proceeding_number (alpha)
    alpha_by_proc = {}
    with args.alpha.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            proc = str(r.get("proceeding_number") or "")
            alpha_by_proc[proc] = r

    pool = []
    n_beta = 0
    n_linked = 0
    n_eligible = 0
    exclusion_counts = Counter()
    for p in sorted(args.beta_parsed_dir.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_beta += 1
        proc = str(rec.get("proceeding_number") or "")
        alpha_row = alpha_by_proc.get(proc)
        if not alpha_row:
            exclusion_counts["no_alpha_link"] += 1
            continue
        n_linked += 1
        axes, eligible, reasons = compute_case_axes(alpha_row, rec)
        if not eligible:
            for r in reasons:
                exclusion_counts[r] += 1
            continue
        n_eligible += 1
        pool.append({
            "case_id": rec.get("case_id"),
            "proceeding_number": proc,
            "app_num": rec.get("app_num"),
            "axes": axes,
        })

    print(f"β parsed: {n_beta}  α-linked: {n_linked}  eligible: {n_eligible}")
    print("Exclusions:")
    for reason, n in exclusion_counts.most_common():
        print(f"  {reason}: {n}")

    if len(pool) < args.n:
        print(f"WARNING: pool {len(pool)} < n {args.n}; selecting all.")

    selected, current = greedy_select(pool, TARGETS, n=args.n, seed=args.seed)

    # Marginal deficit report
    deficit = {ax: {} for ax in TARGETS}
    for ax, tgt in TARGETS.items():
        for bin_name, target_v in tgt.items():
            achieved = current[ax][bin_name]
            deficit[ax][bin_name] = {"target": target_v, "achieved": achieved,
                                     "delta": achieved - target_v}

    out = {
        "batch_id": "batch_0",
        "seed": args.seed,
        "n_requested": args.n,
        "n_selected": len(selected),
        "pool_size": len(pool),
        "exclusion_counts": dict(exclusion_counts),
        "eligibility_criteria": [
            "outcome in {Affirmed, Reversed, AIP}",
            "decision year parseable",
            "rejection_instances >= 1 (structured or regex-fallback)",
            "prior_art_refs >= 1 (structured)",
            "pre-amendment claim list non-empty",
        ],
        "targets": TARGETS,
        "marginal_achieved": {ax: dict(c) for ax, c in current.items()},
        "marginal_deficit": deficit,
        "cases": [{"case_id": c["case_id"], "proceeding_number": c["proceeding_number"],
                   "app_num": c["app_num"], "axes": c["axes"]}
                  for c in selected],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSelected {len(selected)} cases → {args.out}")
    print("\n=== Marginal match (target → achieved) ===")
    for ax, tgt in TARGETS.items():
        print(f"[{ax}]")
        for bin_name, target_v in tgt.items():
            ach = current[ax][bin_name]
            delta = ach - target_v
            mark = "OK" if delta == 0 else (f"+{delta}" if delta > 0 else str(delta))
            print(f"  {bin_name:15s}  target={target_v:3d}  achieved={ach:3d}  [{mark}]")


if __name__ == "__main__":
    main()
