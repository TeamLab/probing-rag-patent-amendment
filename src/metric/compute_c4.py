"""C4 — Robustness (direction-match across probes).

Paper §5: "For any pair of probe conditions A vs A' ... whether the
model's amendment shifts in the predicted direction. Scored per probe
and averaged within a model."

Aggregate metric — not per-response. Takes the scored TSVs from
compute_c1/c2/c3/c5, joins them by (case, cond, rep), computes per-case
baseline-vs-probe Δ per channel, then checks against the pre-registered
direction table. Output: per-probe-channel hit rates plus a grand C4.

For C sub-conditions (C_decoy vs C_mechtrue), the response JSONs are
required so that probe_meta.subcondition can map each C row to its
subcondition. TSVs alone don't distinguish.

Usage:
    python3 scripts/compute_c4.py \\
        --c1 outputs/_analysis/c1_scores.tsv \\
        --c2 outputs/_analysis/c2_scores.tsv \\
        --c3 outputs/_analysis/c3_scores.tsv \\
        --c5 outputs/_analysis/c5_scores.tsv \\
        --response-roots outputs/smoke outputs/integration_test \\
        --out outputs/_analysis/c4_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


# ==========================================================================
# Pre-registered direction table (frozen 2026-04-18)
# ==========================================================================

PREDICTED_DIRECTIONS = {
    # (probe, channel) -> "UP" | "DOWN" | "FLAT"
    ("D", "c5"): "UP",
    ("F", "c5"): "UP",
    ("G", "c5"): "UP",
    ("B", "c1"): "FLAT",
    ("B", "c3"): "FLAT",
    ("C_decoy", "c1"): "DOWN",
    ("C_mechtrue", "c1"): "UP",
}

FLAT_THRESHOLD = 0.05


def _direction_hit(delta: float, predicted: str) -> int:
    if predicted == "UP":
        return 1 if delta > 0 else 0
    if predicted == "DOWN":
        return 1 if delta < 0 else 0
    if predicted == "FLAT":
        return 1 if abs(delta) < FLAT_THRESHOLD else 0
    return 0


# ==========================================================================
# Loading
# ==========================================================================

def _load_tsv(path: Path, col: str) -> dict:
    """Return {(case, cond, rep): float(col)} for rows with non-null col."""
    out = {}
    with path.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v = (r.get(col) or "").strip()
            if v in ("", "None"):
                continue
            try:
                out[(r["case"], r["cond"], int(r["rep"] or 0))] = float(v)
            except ValueError:
                continue
    return out


def _load_subconditions(roots: list) -> dict:
    """Map (case, rep) → subcondition for C responses, from probe_meta."""
    out: dict[tuple, str] = {}
    for root in roots:
        for f in sorted(Path(root).glob("*__C__*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            sub = (d.get("probe_meta") or {}).get("subcondition")
            if sub:
                out[(d.get("case_id"), int(d.get("rep") or 0))] = sub
    return out


# ==========================================================================
# Core aggregation
# ==========================================================================

def _case_cond_means(by_key: dict) -> dict:
    """Aggregate across reps → {(case, cond): mean}."""
    bucket = defaultdict(list)
    for (case, cond, rep), v in by_key.items():
        bucket[(case, cond)].append(v)
    return {k: mean(v) for k, v in bucket.items() if v}


def _split_C_by_sub(by_key: dict, subs: dict) -> dict:
    """Rewrite C rows into C_decoy / C_mechtrue using subcondition map.
    Rows without a subcondition are dropped (skipped C cases).
    Returns same shape dict but cond ∈ {C_decoy, C_mechtrue} instead of "C".
    """
    out = {}
    for (case, cond, rep), v in by_key.items():
        if cond != "C":
            out[(case, cond, rep)] = v
            continue
        sub = subs.get((case, rep))
        if not sub:
            continue  # skipped C → no signal
        out[(case, sub, rep)] = v
    return out


def compute_c4(channel_scores: dict, subs: dict) -> dict:
    """channel_scores: {'c1': dict, 'c2': dict, 'c3': dict, 'c5': dict}
       dict entries: {(case, cond, rep): float}
    Returns per-probe-channel hit-rate summary.
    """
    # Rewrite C rows via subcondition
    rewritten = {ch: _split_C_by_sub(s, subs) for ch, s in channel_scores.items()}

    # Means per (case, cond) per channel
    means = {ch: _case_cond_means(s) for ch, s in rewritten.items()}

    # Per (probe, channel) predicted: collect Δ across cases
    results = {}
    per_probe_hits = defaultdict(list)
    for (probe, channel), predicted in PREDICTED_DIRECTIONS.items():
        ch_means = means[channel]
        # Find cases where both baseline and probe are present
        cases_b = {c for (c, cond) in ch_means if cond == "baseline"}
        cases_p = {c for (c, cond) in ch_means if cond == probe}
        joint = cases_b & cases_p
        if not joint:
            results[(probe, channel)] = {
                "predicted": predicted,
                "n_cases": 0,
                "hit_rate": None,
                "reason": "no_paired_cases",
            }
            continue
        hits = []
        deltas = []
        for case in sorted(joint):
            base = ch_means[(case, "baseline")]
            probe_val = ch_means[(case, probe)]
            delta = probe_val - base
            deltas.append(delta)
            hits.append(_direction_hit(delta, predicted))
        hit_rate = sum(hits) / len(hits)
        results[(probe, channel)] = {
            "predicted": predicted,
            "n_cases": len(joint),
            "hit_rate": round(hit_rate, 3),
            "delta_mean": round(mean(deltas), 4),
            "delta_min": round(min(deltas), 4),
            "delta_max": round(max(deltas), 4),
        }
        per_probe_hits[probe].extend(hits)

    # Per-probe C4 (avg across channels)
    per_probe = {}
    for probe, hits in per_probe_hits.items():
        per_probe[probe] = {
            "c4": round(sum(hits) / len(hits), 3) if hits else None,
            "n_hits": sum(hits),
            "n_total": len(hits),
        }

    all_hits = [h for hs in per_probe_hits.values() for h in hs]
    grand = round(sum(all_hits) / len(all_hits), 3) if all_hits else None

    return {
        "flat_threshold": FLAT_THRESHOLD,
        "predicted_directions": {f"{p}:{c}": d
                                  for (p, c), d in PREDICTED_DIRECTIONS.items()},
        "per_probe_channel": {f"{p}:{c}": v
                               for (p, c), v in results.items()},
        "per_probe": per_probe,
        "c4_grand": grand,
    }


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", required=True, type=Path)
    ap.add_argument("--c2", required=True, type=Path)
    ap.add_argument("--c3", required=True, type=Path)
    ap.add_argument("--c5", required=True, type=Path)
    ap.add_argument("--response-roots", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    channel_scores = {
        "c1": _load_tsv(args.c1, "c1"),
        "c2": _load_tsv(args.c2, "c2"),
        "c3": _load_tsv(args.c3, "c3"),
        "c5": _load_tsv(args.c5, "c5"),
    }
    for ch, s in channel_scores.items():
        print(f"loaded {ch}: {len(s)} rows")

    subs = _load_subconditions(args.response_roots)
    print(f"C subconditions: {len(subs)} response files")

    result = compute_c4(channel_scores, subs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"\n=== C4 summary ===")
    print(f"  grand C4: {result['c4_grand']}")
    print(f"\n  per probe-channel:")
    for key, v in result["per_probe_channel"].items():
        n = v.get("n_cases", 0)
        rate = v.get("hit_rate")
        rate_s = f"{rate:.3f}" if rate is not None else "N/A"
        pred = v.get("predicted")
        print(f"    {key:20s} pred={pred:5s} n={n:3d} hit_rate={rate_s}")
    print(f"\n  per probe:")
    for probe, v in result["per_probe"].items():
        print(f"    {probe:12s} c4={v['c4']}  ({v['n_hits']}/{v['n_total']})")
    print(f"\n  saved to {args.out}")


if __name__ == "__main__":
    main()
