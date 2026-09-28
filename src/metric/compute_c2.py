"""C2 — Revision locality metric.

Definition (draft_v0.3 §5):
    C2 = editdist(gen, pre) / editdist(gold_post, pre)

Where:
  - `pre`       = pre-amendment claim text (β.pre_clm.claims, plain text)
  - `gold_post` = gold post-amendment text  (β.post_clm.claims, plain text)
  - `gen`       = model-generated amendment text (parse_response.plain_body)

All three are formed by joining plain claim-body text across all
numbered claims (sorted by claim number) with single-newline separators
— character-level Levenshtein distance via rapidfuzz.

C2 interpretation:
  ≈ 1.0  model edited at the same scale as the gold amendment
  << 1   under-editing (model barely touched claims)
  >> 1   over-editing  (model rewrote more than needed)

Two-stage pipeline:
  Stage 1 — Per-case baseline cache:
      python3 scripts/compute_c2.py baseline \\
          --beta-parsed-dir data/parsed/beta_parsed \\
          --cohort data/parsed/cohort_batch0.json \\
          --out data/parsed/c2_baseline.json

  Stage 2 — Per-response C2:
      python3 scripts/compute_c2.py score \\
          --roots outputs/smoke outputs/integration_test \\
          --beta-parsed-dir data/parsed/beta_parsed \\
          --baseline data/parsed/c2_baseline.json \\
          --out outputs/_analysis/c2_scores.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# rapidfuzz has Levenshtein distance in C — required for reasonable speed
from rapidfuzz.distance import Levenshtein

# parse_response is local
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from parse_response import parse as parse_response_text  # noqa: E402


# -------- text extraction helpers --------------------------------------

def _is_numeric_claim_num(n) -> bool:
    return str(n or "").strip().isdigit()


def _strip_markup(text: str) -> str:
    """Remove any leftover amendment-format markup from β claim text."""
    # β stores pre/post with some markup; strip to plain for fair comparison.
    text = re.sub(r"\[\[[^\]]+?\]\]", "", text)
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)           # stray XML tags if any
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _join_beta_claims(claim_list) -> str:
    """Join plain text of numeric claims, sorted by claim number, with '\\n'."""
    kept = []
    for c in claim_list or []:
        num = c.get("num")
        if not _is_numeric_claim_num(num):
            continue
        t = (c.get("text") or "").strip()
        if len(t) < 10:
            continue
        t = _strip_markup(t)
        # β often prefixes the claim text with its own number — normalize out
        t = re.sub(rf"^\s*{num}\s*[.\s]+", "", t, count=1).strip()
        kept.append((int(num), t))
    kept.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in kept)


def _join_parsed_gen(parsed_claims) -> str:
    """Join plain_body of generated claims, sorted by number, with '\\n'."""
    kept = [(r.num, r.plain_body) for r in parsed_claims
            if r.plain_body and r.num is not None]
    kept.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in kept if t)


def _editdist(a: str, b: str) -> int:
    return Levenshtein.distance(a, b)


# -------- Stage 1: per-case baseline cache -----------------------------

def build_baseline(beta_dir: Path, cohort_path: Path) -> dict:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    out = {}
    n_ok = 0
    n_missing = 0
    for c in cohort["cases"]:
        case_id = c["case_id"]
        app = c["app_num"]
        bp = beta_dir / f"{app}.json"
        if not bp.exists():
            n_missing += 1
            continue
        beta = json.loads(bp.read_text(encoding="utf-8"))
        pre_text = _join_beta_claims((beta.get("pre_clm") or {}).get("claims"))
        post_text = _join_beta_claims((beta.get("post_clm") or {}).get("claims"))
        if not pre_text or not post_text:
            continue
        d = _editdist(pre_text, post_text)
        denom = d if d > 0 else 1  # avoid div-by-zero
        out[case_id] = {
            "app_num": app,
            "pre_chars": len(pre_text),
            "post_chars": len(post_text),
            "editdist_pre_gold": d,
            "denom": denom,
            "similarity_ref": round(1 - d / max(len(pre_text), len(post_text)), 4),
        }
        n_ok += 1
    return {"n_ok": n_ok, "n_missing": n_missing, "cases": out}


# -------- Stage 2: per-response C2 --------------------------------------

def score_response(resp_json: dict, beta: dict, denom: int) -> dict:
    gen_claims = parse_response_text(resp_json.get("response_text", "") or "")
    gen_text = _join_parsed_gen(gen_claims)
    pre_text = _join_beta_claims((beta.get("pre_clm") or {}).get("claims"))
    if not pre_text:
        return {"n_claims_parsed": len(gen_claims),
                "editdist_pre_gen": None, "c2": None, "reason": "no_pre"}
    if not gen_text:
        return {"n_claims_parsed": 0, "editdist_pre_gen": None,
                "c2": None, "reason": "no_gen"}
    d_pre_gen = _editdist(pre_text, gen_text)
    return {
        "n_claims_parsed": len(gen_claims),
        "pre_chars": len(pre_text),
        "gen_chars": len(gen_text),
        "editdist_pre_gen": d_pre_gen,
        "denom": denom,
        "c2": round(d_pre_gen / denom, 4),
    }


# -------- CLI ----------------------------------------------------------

def cmd_baseline(args) -> None:
    result = build_baseline(args.beta_parsed_dir, args.cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"computed baseline for {result['n_ok']} cases "
          f"({result['n_missing']} missing β)")
    # quick distribution
    sims = sorted(v["similarity_ref"] for v in result["cases"].values())
    if sims:
        print(f"similarity_ref range: {sims[0]:.3f} .. {sims[-1]:.3f}  "
              f"median={sims[len(sims)//2]:.3f}")


def cmd_score(args) -> None:
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    base_cases = baseline["cases"]

    rows = []
    for root in args.roots:
        for f in sorted(root.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            case_id = d.get("case_id")
            cond = d.get("condition", "baseline")
            rep = d.get("rep", 0)
            if case_id not in base_cases:
                rows.append({"file": f.name, "case": case_id, "cond": cond,
                             "rep": rep, "c2": None, "reason": "no_baseline"})
                continue
            app = base_cases[case_id]["app_num"]
            bp = args.beta_parsed_dir / f"{app}.json"
            if not bp.exists():
                rows.append({"file": f.name, "case": case_id, "cond": cond,
                             "rep": rep, "c2": None, "reason": "no_beta"})
                continue
            beta = json.loads(bp.read_text(encoding="utf-8"))
            denom = base_cases[case_id]["denom"]
            s = score_response(d, beta, denom)
            rows.append({"file": f.name, "case": case_id, "cond": cond,
                         "rep": rep, **s})

    cols = ["file", "case", "cond", "rep", "n_claims_parsed",
            "pre_chars", "gen_chars", "editdist_pre_gen", "denom", "c2", "reason"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}")

    # summary
    ok = [r for r in rows if r.get("c2") is not None]
    if ok:
        c2s = sorted(r["c2"] for r in ok)
        print(f"C2 range: {c2s[0]:.3f} .. {c2s[-1]:.3f}  "
              f"median={c2s[len(c2s)//2]:.3f}  n={len(ok)}")
        fail = len(rows) - len(ok)
        if fail:
            print(f"failed: {fail} (see tsv)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    apb = sub.add_parser("baseline")
    apb.add_argument("--beta-parsed-dir", required=True, type=Path)
    apb.add_argument("--cohort", required=True, type=Path)
    apb.add_argument("--out", required=True, type=Path)
    apb.set_defaults(fn=cmd_baseline)

    aps = sub.add_parser("score")
    aps.add_argument("--roots", nargs="+", required=True, type=Path)
    aps.add_argument("--beta-parsed-dir", required=True, type=Path)
    aps.add_argument("--baseline", required=True, type=Path)
    aps.add_argument("--out", required=True, type=Path)
    aps.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
