"""C3 — Scope preservation metric.

Paper §5: "Semantic overlap between the generated amendment and the
invention core of the original claim, computed as the Jaccard overlap
of noun phrases after limitation-removal normalization. Reduction
indicates over-narrowing."

v0 operationalization (frozen 2026-04-18):
  - "Noun phrases after limitation-removal normalization" is
    approximated by **content-word unigrams** — tokens surviving a
    stopword filter that strips general-English stopwords AND
    patent-claim-language tokens (comprising, wherein, said,
    configured, plurality, etc.). This is a no-dependency
    approximation; v1 upgrade path is spaCy noun-phrase chunking.
  - Per claim n in pre ∩ gen:
      c3_n = Jaccard(content_tokens(pre_n), content_tokens(gen_n))
  - Case-level C3 = mean over qualifying claims
  - Primary = Jaccard; auxiliary = recall / precision / F1

Paper §5.6 secondary baseline (BERTScore) is deferred to post-Stage 1.

Usage:
    python3 scripts/compute_c3.py score \\
        --roots outputs/smoke outputs/integration_test \\
        --beta-parsed-dir data/parsed/beta_parsed \\
        --cohort data/parsed/cohort_batch0_v1.json \\
        --out outputs/_analysis/c3_scores.tsv
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


# ---------- stopword filter ------------------------------------------------
#
# Two layers:
#   (1) General English function words (determiners, prepositions, common
#       conjunctions, auxiliaries, basic quantifiers).
#   (2) Patent-claim-language tokens (transition terms, relational verbs,
#       claim-drafting boilerplate) that carry zero "invention core"
#       content when removed.

GENERAL_STOP = frozenset("""
    a an the of in on at by for from to with without into onto upon
    is are was were be been being has have had having do does did done
    not no nor or and but if then than as so such also only just very
    may might can could would should must will shall about which who whom
    whose where when why how what that these those this those there
    its it their they them he she him her his hers us we our ours you your
    all any each both some more less many much several few most least
    same other same different like unlike including
""".split())

PATENT_STOP = frozenset("""
    comprising comprises comprise consisting consists include includes
    including included having has have had contain contains containing
    wherein whereby whereas whereupon whereof
    said therein therefor thereof therebetween thereto therewith thereon
    above said first second third fourth fifth sixth seventh eighth ninth
    tenth eleventh twelfth
    plurality plural one two three four five six seven eight nine ten
    according accordingly further furthermore moreover additionally
    substantially approximately generally relatively essentially
    configured adapted arranged operable capable coupled connected
    operatively disposed positioned situated located mounted formed
    provided presented
    claim claims recited defined mentioned described disclosed illustrated
    element elements component components means system method apparatus
    device devices portion portions section sections member members
    preferred preferably optional optionally respective respectively
    original amended previously presented cancelled canceled new withdrawn
    currently preamble transition limitation limitations
""".split())

STOPWORDS = GENERAL_STOP | PATENT_STOP

WORD_RE = re.compile(r"[a-z][a-z\-']+")
MIN_LEN = 4


def content_tokens(text: str) -> set:
    """Lowercase content tokens: length≥4, not in stopword list."""
    return {t for t in WORD_RE.findall((text or "").lower())
            if len(t) >= MIN_LEN and t not in STOPWORDS}


# ---------- claim maps -----------------------------------------------------

def _strip_markup(text: str) -> str:
    text = re.sub(r"\[\[[^\]]+?\]\]", "", text or "")
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _claims_map_beta(claim_list) -> dict:
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


# ---------- C3 scoring ----------------------------------------------------

def score_response(beta: dict, response_text: str) -> dict:
    pre_map = _claims_map_beta((beta.get("pre_clm") or {}).get("claims"))
    if not pre_map:
        return {"c3": None, "n_claims": 0, "reason": "no_pre"}
    gen_parsed = parse_response_text(response_text or "")
    gen_map = _claims_map_from_response(gen_parsed)
    if not gen_map:
        return {"c3": None, "n_claims": 0, "reason": "no_gen"}

    per_claim = []
    for num, pre_text in pre_map.items():
        if num not in gen_map:
            continue
        P = content_tokens(pre_text)
        G = content_tokens(gen_map[num])
        if not P and not G:
            continue
        inter = len(P & G)
        union = len(P | G)
        jac = inter / union if union else 0.0
        recall = inter / len(P) if P else 0.0
        prec = inter / len(G) if G else 0.0
        f1 = 2 * inter / (len(P) + len(G)) if (len(P) + len(G)) else 0.0
        per_claim.append({
            "num": num,
            "c3": round(jac, 4),
            "c3_recall": round(recall, 4),
            "c3_prec": round(prec, 4),
            "c3_f1": round(f1, 4),
            "pre_n": len(P), "gen_n": len(G), "overlap_n": inter,
        })
    if not per_claim:
        return {"c3": None, "c3_recall": None, "c3_prec": None, "c3_f1": None,
                "n_claims": 0, "reason": "no_aligned_claims"}
    n = len(per_claim)
    return {
        "c3": round(sum(x["c3"] for x in per_claim) / n, 4),
        "c3_recall": round(sum(x["c3_recall"] for x in per_claim) / n, 4),
        "c3_prec": round(sum(x["c3_prec"] for x in per_claim) / n, 4),
        "c3_f1": round(sum(x["c3_f1"] for x in per_claim) / n, 4),
        "n_claims": n,
        "per_claim": per_claim,
    }


# ---------- CLI ----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    aps = sub.add_parser("score")
    aps.add_argument("--roots", nargs="+", required=True, type=Path)
    aps.add_argument("--beta-parsed-dir", required=True, type=Path)
    aps.add_argument("--cohort", required=True, type=Path)
    aps.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    app_by_case = {c["case_id"]: c["app_num"] for c in cohort["cases"]}

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
            if case_id not in app_by_case:
                rows.append({"file": f.name, "case": case_id,
                             "cond": d.get("condition", "baseline"),
                             "rep": d.get("rep", 0),
                             "c3": None, "reason": "not_in_cohort"})
                continue
            bp = args.beta_parsed_dir / f"{app_by_case[case_id]}.json"
            if not bp.exists():
                rows.append({"file": f.name, "case": case_id,
                             "cond": d.get("condition", "baseline"),
                             "rep": d.get("rep", 0),
                             "c3": None, "reason": "no_beta"})
                continue
            beta = json.loads(bp.read_text(encoding="utf-8"))
            s = score_response(beta, d.get("response_text") or "")
            rows.append({"file": f.name, "case": case_id,
                         "cond": d.get("condition", "baseline"),
                         "rep": d.get("rep", 0), **s})

    cols = ["file", "case", "cond", "rep",
            "c3", "c3_recall", "c3_prec", "c3_f1",
            "n_claims", "reason"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}")
    ok = [r for r in rows if r.get("c3") is not None]
    if ok:
        import statistics as st
        vals = sorted(r["c3"] for r in ok)
        print(f"C3 range: {vals[0]:.3f} .. {vals[-1]:.3f}  "
              f"median={st.median(vals):.3f}  mean={st.mean(vals):.3f}  n={len(ok)}")


if __name__ == "__main__":
    main()
