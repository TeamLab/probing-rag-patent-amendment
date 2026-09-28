"""C1 — Grounding alignment metric.

Paper §5: "Overlap between the limitation(s) modified in the generated
amendment and the limitation(s) named as rejected in the CTNF
DataField#1 and its associated reasoning paragraph. Computed at
limitation-token level after claim-number alignment."

v0 operationalization (frozen 2026-04-18):
  - Token unit: 3-grams (sliding trigrams over lowercased word tokens)
  - "Modified tokens" of claim n: symmetric_difference(
        3grams(pre[n]), 3grams(gen[n]))
  - "Rejected tokens" of claim n: 3grams(pre[n]) ∩ 3grams(ctnf.full_text),
    restricted to claims actually listed in rejection_instances.claims_raw
  - Per-claim C1_n = Jaccard(modified_n, rejected_n)
  - Case-level C1 = mean over claims present in BOTH pre and gen
    AND with |rejected_n| > 0

Two-stage pipeline (parallels compute_c2.py):
  Stage 1 — Per-case baseline cache (rejected sets, can be precomputed once):
      python3 scripts/compute_c1.py baseline \\
          --beta-parsed-dir data/parsed/beta_parsed \\
          --cohort data/parsed/cohort_batch0_v1.json \\
          --out data/parsed/c1_baseline_v1.json

  Stage 2 — Per-response C1:
      python3 scripts/compute_c1.py score \\
          --roots outputs/smoke outputs/integration_test \\
          --beta-parsed-dir data/parsed/beta_parsed \\
          --baseline data/parsed/c1_baseline_v1.json \\
          --out outputs/_analysis/c1_scores.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from parse_response import parse as parse_response_text  # noqa: E402


# ---------- tokenization ---------------------------------------------------

WORD_RE = re.compile(r"[a-z][a-z\-']+")


def _trigrams(text: str) -> frozenset:
    """Lowercased word trigrams as a set."""
    if not text:
        return frozenset()
    words = WORD_RE.findall(text.lower())
    if len(words) < 3:
        return frozenset()
    return frozenset(" ".join(words[i:i + 3]) for i in range(len(words) - 2))


def _strip_markup(text: str) -> str:
    text = re.sub(r"\[\[[^\]]+?\]\]", "", text or "")
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- claim maps -----------------------------------------------------

def _claims_map_from_beta(claim_list) -> dict:
    out: dict[int, str] = {}
    for c in claim_list or []:
        num = c.get("num")
        if not str(num or "").strip().isdigit():
            continue
        t = (c.get("text") or "").strip()
        if len(t) < 10:
            continue
        t = _strip_markup(t)
        t = re.sub(rf"^\s*{num}\s*[.\s]+", "", t, count=1).strip()
        out[int(num)] = t
    return out


def _claims_map_from_response(parsed_claims) -> dict:
    return {r.num: r.plain_body for r in parsed_claims
            if r.num is not None and r.plain_body}


def _rejected_claim_nums(ctnf: dict) -> set:
    """Return set of claim nums listed in rejection_instances.claims_raw."""
    nums: set[int] = set()
    for ri in (ctnf or {}).get("rejection_instances") or []:
        for raw in ri.get("claims_raw") or []:
            # forms seen: "1", "1-5", "1, 3-5", "1 and 3"
            for m in re.finditer(r"(\d+)\s*-\s*(\d+)", raw):
                a, b = int(m.group(1)), int(m.group(2))
                if 0 < b - a < 200:
                    nums.update(range(a, b + 1))
            for m in re.finditer(r"\b(\d+)\b", re.sub(r"\d+\s*-\s*\d+", "", raw)):
                nums.add(int(m.group(1)))
    return nums


# ---------- C1 core --------------------------------------------------------

def compute_case_rejected(beta: dict) -> dict:
    """Per-case precomputable: rejected trigrams per claim.

    Returns {claim_num: {'rejected': frozenset, 'pre_grams': frozenset}}
    restricted to claims listed in rejection_instances.
    """
    pre = _claims_map_from_beta((beta.get("pre_clm") or {}).get("claims"))
    ctnf = beta.get("ctnf") or {}
    ctnf_grams = _trigrams(ctnf.get("full_text") or "")
    rej_nums = _rejected_claim_nums(ctnf)
    out = {}
    for num, text in pre.items():
        if num not in rej_nums:
            continue
        pre_grams = _trigrams(text)
        rejected = pre_grams & ctnf_grams
        if not rejected:
            continue
        out[num] = {
            "pre_chars": len(text),
            "pre_grams_n": len(pre_grams),
            "rejected_grams_n": len(rejected),
            # store the sets as sorted lists for JSON-ability
            "pre_grams": sorted(pre_grams),
            "rejected": sorted(rejected),
        }
    return out


def score_response(beta: dict, response_text: str,
                   rejected_cache: dict) -> dict:
    """Compute C1 for one model response against its case's rejected cache."""
    gen_parsed = parse_response_text(response_text or "")
    gen_map = _claims_map_from_response(gen_parsed)

    pre_map = _claims_map_from_beta((beta.get("pre_clm") or {}).get("claims"))

    per_claim = []
    skipped_no_gen = 0
    skipped_no_rejected = 0
    for num_key, info in rejected_cache.items():
        # JSON roundtrip turns int keys into strings — cast back.
        try:
            num = int(num_key)
        except (ValueError, TypeError):
            continue
        if num not in pre_map:
            continue
        if num not in gen_map:
            skipped_no_gen += 1
            continue
        pre_grams = frozenset(info["pre_grams"])
        rejected = frozenset(info["rejected"])
        if not rejected:
            skipped_no_rejected += 1
            continue
        gen_grams = _trigrams(gen_map[num])
        modified = pre_grams.symmetric_difference(gen_grams)
        inter_n = len(modified & rejected)
        mod_n = len(modified)
        rej_n = len(rejected)
        # Primary: Jaccard. Auxiliary: recall / precision / F1.
        union_n = mod_n + rej_n - inter_n
        jac = inter_n / union_n if union_n else 0.0
        recall = inter_n / rej_n if rej_n else 0.0
        prec = inter_n / mod_n if mod_n else 0.0
        f1 = 2 * inter_n / (mod_n + rej_n) if (mod_n + rej_n) else 0.0
        per_claim.append({
            "num": num,
            "c1": round(jac, 4),
            "c1_recall": round(recall, 4),
            "c1_prec": round(prec, 4),
            "c1_f1": round(f1, 4),
            "modified_n": mod_n,
            "rejected_n": rej_n,
            "overlap_n": inter_n,
        })

    if not per_claim:
        return {"c1": None, "c1_recall": None, "c1_prec": None, "c1_f1": None,
                "n_claims": 0, "reason": "no_aligned_claims",
                "skipped_no_gen": skipped_no_gen,
                "skipped_no_rejected": skipped_no_rejected}
    n = len(per_claim)
    return {
        "c1": round(sum(x["c1"] for x in per_claim) / n, 4),
        "c1_recall": round(sum(x["c1_recall"] for x in per_claim) / n, 4),
        "c1_prec": round(sum(x["c1_prec"] for x in per_claim) / n, 4),
        "c1_f1": round(sum(x["c1_f1"] for x in per_claim) / n, 4),
        "n_claims": n,
        "skipped_no_gen": skipped_no_gen,
        "skipped_no_rejected": skipped_no_rejected,
        "per_claim": per_claim,
    }


# ---------- CLI ------------------------------------------------------------

def cmd_baseline(args) -> None:
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    out = {"cases": {}, "n_ok": 0, "n_no_rejected": 0}
    for c in cohort["cases"]:
        cid = c["case_id"]
        bp = args.beta_parsed_dir / f"{c['app_num']}.json"
        if not bp.exists():
            continue
        beta = json.loads(bp.read_text(encoding="utf-8"))
        rej = compute_case_rejected(beta)
        if not rej:
            out["n_no_rejected"] += 1
            continue
        out["cases"][cid] = {"app_num": c["app_num"], "rejected": rej}
        out["n_ok"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"baseline computed: {out['n_ok']} cases with rejected grams  "
          f"({out['n_no_rejected']} cases had no rejected-gram overlap)")
    # quick stats on rejected-set size
    sizes = [sum(ci["rejected_grams_n"] for ci in case["rejected"].values())
             for case in out["cases"].values()]
    if sizes:
        sizes.sort()
        print(f"  per-case total rejected 3grams: "
              f"min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")


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
                             "rep": rep, "c1": None, "reason": "no_baseline"})
                continue
            app = base_cases[case_id]["app_num"]
            bp = args.beta_parsed_dir / f"{app}.json"
            if not bp.exists():
                rows.append({"file": f.name, "case": case_id, "cond": cond,
                             "rep": rep, "c1": None, "reason": "no_beta"})
                continue
            beta = json.loads(bp.read_text(encoding="utf-8"))
            rej_cache = base_cases[case_id]["rejected"]
            s = score_response(beta, d.get("response_text") or "", rej_cache)
            rows.append({"file": f.name, "case": case_id, "cond": cond,
                         "rep": rep, **s})

    cols = ["file", "case", "cond", "rep",
            "c1", "c1_recall", "c1_prec", "c1_f1",
            "n_claims", "skipped_no_gen", "skipped_no_rejected", "reason"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}")
    ok = [r for r in rows if r.get("c1") is not None]
    if ok:
        vals = sorted(r["c1"] for r in ok)
        import statistics as st
        print(f"C1 range: {vals[0]:.3f} .. {vals[-1]:.3f}  "
              f"median={st.median(vals):.3f}  mean={st.mean(vals):.3f}  n={len(ok)}")


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
