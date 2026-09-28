"""
Inspect a selected cohort for quality and sanity before experiment launch.

Checks:
  A. Marginal target vs achieved (cross-check select_cohort output)
  B. Cell minimum — each target bin has ≥ min_cell_size members
  C. 2-way crosstabs on the interesting pairs (outcome × statute, outcome ×
     pattern, tech × statute, format × year)
  D. Duplicate detection (case_id, proceeding_number, app_num)
  E. Per-case health metrics: n published claims, rejection_instances,
     prior_art_refs, diff sizes
  F. Edge cases: zero rejection_instances, zero prior_art, empty published
  G. Expected correlation sanity checks
  H. Sample 3 cases: case_id, title, pre-claim head, rejection head

Usage:
  python3 scripts/inspect_cohort.py \\
      --cohort data/parsed/cohort_batch0.json \\
      --beta-parsed-dir data/parsed/beta_parsed \\
      --alpha data/parsed/alpha_corpus.jsonl \\
      --out data/parsed/cohort_batch0_inspection.md
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_alpha_by_proc(alpha_path):
    out = {}
    with alpha_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[str(r.get("proceeding_number") or "")] = r
    return out


def crosstab(cases, ax_a, ax_b):
    ct = defaultdict(Counter)
    for c in cases:
        ct[c["axes"][ax_a]][c["axes"][ax_b]] += 1
    return ct


def render_crosstab(ct, title, a_name, b_name, out):
    b_keys = sorted({k for row in ct.values() for k in row})
    a_keys = sorted(ct.keys())
    out.append(f"### {title} ({a_name} × {b_name})\n")
    out.append("| " + a_name + " \\ " + b_name + " | " + " | ".join(b_keys) + " | total |")
    out.append("|" + "---|" * (len(b_keys) + 2))
    for a in a_keys:
        row_total = sum(ct[a].values())
        cells = [str(ct[a].get(b, 0)) for b in b_keys]
        out.append("| " + a + " | " + " | ".join(cells) + f" | **{row_total}** |")
    col_totals = {b: sum(ct[a].get(b, 0) for a in a_keys) for b in b_keys}
    out.append("| **total** | " + " | ".join(f"**{col_totals[b]}**" for b in b_keys) + f" | **{sum(col_totals.values())}** |")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-cell-size", type=int, default=5)
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"]
    targets = cohort["targets"]
    marginal_achieved = cohort["marginal_achieved"]

    alpha_by_proc = load_alpha_by_proc(args.alpha)

    out = []
    out.append(f"# Cohort Inspection — {cohort.get('batch_id', 'batch_0')}")
    out.append("")
    out.append(f"- Seed: {cohort.get('seed')}")
    out.append(f"- Selected: {len(cases)} / requested {cohort.get('n_requested')}")
    out.append(f"- Pool size: {cohort.get('pool_size')}")
    out.append("")

    # A. Marginal target vs achieved
    out.append("## A. Marginal target vs achieved")
    out.append("")
    for ax, tgt in targets.items():
        out.append(f"### {ax}")
        out.append("| bin | target | achieved | delta |")
        out.append("|---|---|---|---|")
        for bin_name, v in tgt.items():
            ach = marginal_achieved.get(ax, {}).get(bin_name, 0)
            delta = ach - v
            mark = "OK" if delta == 0 else f"{'+' if delta > 0 else ''}{delta}"
            out.append(f"| {bin_name} | {v} | {ach} | {mark} |")
        out.append("")

    # B. Cell minimum
    out.append("## B. Cell minimum check (min = {})".format(args.min_cell_size))
    out.append("")
    issues = []
    for ax, tgt in targets.items():
        for bin_name, target_v in tgt.items():
            ach = marginal_achieved.get(ax, {}).get(bin_name, 0)
            if target_v >= args.min_cell_size and ach < args.min_cell_size:
                issues.append(f"- {ax}[{bin_name}]: achieved {ach} < min {args.min_cell_size} (target {target_v})")
    out.append("✓ All cells pass minimum." if not issues else "⚠ Cells below minimum:")
    out.extend(issues)
    out.append("")

    # C. 2-way crosstabs
    out.append("## C. 2-way crosstabs")
    out.append("")
    for a, b in [("outcome", "statute"),
                 ("outcome", "pattern"),
                 ("tech", "statute"),
                 ("format", "year")]:
        ct = crosstab(cases, a, b)
        render_crosstab(ct, f"{a} × {b}", a, b, out)

    # D. Duplicates
    out.append("## D. Duplicates")
    out.append("")
    case_ids = [c["case_id"] for c in cases]
    procs = [c["proceeding_number"] for c in cases]
    apps = [c["app_num"] for c in cases]
    dup = {
        "case_id": [k for k, v in Counter(case_ids).items() if v > 1],
        "proceeding_number": [k for k, v in Counter(procs).items() if v > 1],
        "app_num": [k for k, v in Counter(apps).items() if v > 1],
    }
    for k, v in dup.items():
        out.append(f"- {k}: {len(v)} duplicates" + (f" — {v[:5]}" if v else " OK"))
    out.append("")

    # E. Per-case health
    out.append("## E. Per-case health metrics (from beta_parsed)")
    out.append("")
    n_pub_claims = []
    n_rej = []
    n_pa = []
    ctnf_fallback = 0
    empty_pub = 0
    zero_rej = 0
    zero_pa = 0
    short_pub_claim = 0
    for c in cases:
        bp = args.beta_parsed_dir / f"{c['app_num']}.json"
        if not bp.exists():
            continue
        rec = json.loads(bp.read_text(encoding="utf-8"))
        pre_claims = (rec.get("pre_clm") or {}).get("claims") or []
        n_pub_claims.append(len(pre_claims))
        if not pre_claims:
            empty_pub += 1
        ctnf = rec.get("ctnf") or {}
        rj = ctnf.get("rejection_instances") or []
        n_rej.append(len(rj))
        if not rj:
            zero_rej += 1
        if ctnf.get("regex_fallback_used"):
            ctnf_fallback += 1
        pa = ctnf.get("prior_art_refs") or []
        n_pa.append(len(pa))
        if not pa:
            zero_pa += 1
        # Short independent claim check
        for cl in pre_claims:
            if cl.get("num") == "1" and len(cl.get("text", "")) < 50:
                short_pub_claim += 1
                break

    def stat(xs):
        if not xs:
            return "(empty)"
        xs = sorted(xs)
        n = len(xs)
        return f"min={xs[0]} median={xs[n//2]} mean={round(sum(xs)/n, 1)} max={xs[-1]}"

    out.append(f"- Pre-claim count per case: {stat(n_pub_claims)}")
    out.append(f"- Rejection instances per case: {stat(n_rej)}")
    out.append(f"- Prior-art refs per case: {stat(n_pa)}")
    out.append(f"- CTNF regex fallback used: {ctnf_fallback} / {len(cases)}")
    out.append("")

    # F. Edge cases
    out.append("## F. Edge-case counts")
    out.append("")
    out.append(f"- Empty pre-claim: {empty_pub}")
    out.append(f"- Zero rejection_instances: {zero_rej}")
    out.append(f"- Zero prior_art_refs: {zero_pa}")
    out.append(f"- Claim 1 too short (<50 chars): {short_pub_claim}")
    out.append("")

    # G. Sanity correlations
    out.append("## G. Expected correlation sanity")
    out.append("")
    # DTD legacy should concentrate in older years
    fmt_year_ct = crosstab(cases, "format", "year")
    dtd_legacy_years = fmt_year_ct.get("dtd_legacy", Counter())
    ns_years = fmt_year_ct.get("ns_claims", Counter())
    out.append(f"- DTD legacy year split: {dict(dtd_legacy_years)}  "
               f"(expected: concentrated in 2010-14 and 2015-19)")
    out.append(f"- NS claims year split:  {dict(ns_years)}  "
               f"(expected: concentrated in 2015-19 and 2020-25)")

    # §101 should concentrate in software TCs (2100, 2400, 3600)
    tc_st_ct = crosstab(cases, "tech", "statute")
    stat_101 = {tc: cnt.get("101-rel", 0) for tc, cnt in tc_st_ct.items()}
    out.append(f"- §101-related by tech center: {stat_101}  "
               f"(expected: 3600/2100/2400 higher; 1700 low)")

    # cancel_heavy should correlate with Reversed (β 269 observation)
    out_pat_ct = crosstab(cases, "outcome", "pattern")
    ch_rev = out_pat_ct.get("Reversed", Counter()).get("cancel_heavy", 0)
    ch_aff = out_pat_ct.get("Affirmed", Counter()).get("cancel_heavy", 0)
    out.append(f"- cancel_heavy: Reversed={ch_rev} vs Affirmed={ch_aff}  "
               f"(expected: Reversed proportionally higher; β-269 base: 18%→26%)")
    out.append("")

    # H. Sample cases
    out.append("## H. Sample 3 cases (first 3 in cohort)")
    out.append("")
    for c in cases[:3]:
        bp = args.beta_parsed_dir / f"{c['app_num']}.json"
        if not bp.exists():
            continue
        rec = json.loads(bp.read_text(encoding="utf-8"))
        alpha_row = alpha_by_proc.get(c["proceeding_number"], {})
        title = (rec.get("pre_clm") or {}).get("claims")
        title_text = alpha_row.get("title", "(no title)")
        pre_head = ""
        pre = (rec.get("pre_clm") or {}).get("claims") or []
        if pre:
            for cl in pre:
                if str(cl.get("num")) == "1":
                    pre_head = (cl.get("text") or "")[:300]
                    break
        out.append(f"### {c['case_id']} (app_num={c['app_num']})")
        out.append(f"- Axes: {c['axes']}")
        out.append(f"- Title: {title_text[:120]}")
        out.append(f"- Subdecision: {alpha_row.get('subdecision')}")
        out.append(f"- Pre-claim 1 head (300 chars):")
        out.append(f"  > {pre_head}")
        rejs = (rec.get("ctnf") or {}).get("rejection_instances") or []
        if rejs:
            r = rejs[0]
            out.append(f"- First rejection: §{r.get('statute_section')} — type={r.get('rejection_type')} — refs={r.get('prior_art_refs')[:2]}")
        out.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"Inspection report → {args.out}")
    print()
    print("\n".join(out[:80]))  # head


if __name__ == "__main__":
    main()
