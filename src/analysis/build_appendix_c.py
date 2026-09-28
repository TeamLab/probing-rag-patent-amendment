"""Build Appendix C — channel-by-channel worked examples.

For each of C1/C2/C3/C5, show one pass / fail pair from the main data
with structural numbers only (no raw LLM text dumps beyond 1-line
snippets already extracted by the parser).

Reads:
  outputs/_analysis/c{1,2,3,5}_main.tsv
  data/parsed/cohort_batch0_v1.json
  data/parsed/beta_parsed/<app>.json  (structural only)

Writes:
  docs/appendix_c_worked_examples.md
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "_analysis"
BETA = ROOT / "data" / "parsed" / "beta_parsed"
COHORT = ROOT / "data" / "parsed" / "cohort_batch0_v1.json"
OUT = ROOT / "docs" / "appendix_c_worked_examples.md"

# pre-selected (case, model, high_cond, low_cond) per channel
PICKS = {
    "c1": {
        "case": "PILOT-2019005776", "model": "gpt4o-mini",
        "pass_cond": "A", "fail_cond": "baseline",
        "pass_label": "A (one-limitation perturbation)",
        "fail_label": "baseline (no perturbation)",
        "pass_header": "higher C1 (more aligned)",
        "fail_header": "lower C1 (off-target)",
    },
    "c2": {
        "case": "PILOT-2020002555", "model": "gpt4o-mini",
        "pass_cond": "baseline", "fail_cond": "F",
        "pass_label": "baseline",
        "fail_label": "F (random retrieval)",
        "pass_header": "under-edit",
        "fail_header": "catastrophic over-edit",
    },
    "c3": {
        "case": "PILOT-2020003998", "model": "haiku4.5",
        "pass_cond": "baseline", "fail_cond": "D",
        "pass_label": "baseline",
        "fail_label": "D (boilerplate injection)",
        "pass_header": "scope preserved",
        "fail_header": "scope collapsed",
    },
    "c5": {
        "case": "PILOT-2013007708", "model": "gpt4o-mini",
        "pass_cond": "baseline", "fail_cond": "G",
        "pass_label": "baseline",
        "fail_label": "G (structural retrieval)",
        "pass_header": "low template reuse",
        "fail_header": "template inflation",
    },
}

MODEL_LABELS = {
    "sonnet4": "Claude Sonnet 4",
    "haiku4.5": "Claude Haiku 4.5",
    "gpt5.4": "GPT-5.4",
    "gpt4o-mini": "gpt-4o-mini",
}


def load_cohort_meta() -> dict:
    d = json.load(COHORT.open())
    cases = d.get("cases") if isinstance(d, dict) else d
    return {c["case_id"]: c for c in cases}


def load_beta(app_num: str) -> dict | None:
    p = BETA / f"{app_num}.json"
    if not p.exists():
        return None
    return json.load(p.open())


def load_tsv_filtered(name: str, case: str, model: str) -> list[dict]:
    rows = []
    with (SRC / name).open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            parts = r["file"].rsplit("__", 3)
            m = parts[-2] if len(parts) >= 4 else ""
            if r["case"] == case and m == model:
                rows.append(r)
    return rows


def rep_mean(rows: list[dict], cond: str, numeric_cols: list[str]) -> dict:
    matching = [r for r in rows if r["cond"] == cond]
    out = {"n_reps": len(matching)}
    for col in numeric_cols:
        vs = []
        for r in matching:
            v = r.get(col, "")
            if v in (None, ""):
                continue
            try:
                vs.append(float(v))
            except ValueError:
                continue
        out[col] = statistics.mean(vs) if vs else None
    return out


def first_claim_preview(beta: dict, nchar: int = 180) -> str:
    claims = beta.get("pre_clm", {}).get("claims", [])
    c1 = next((c for c in claims if str(c.get("num")) == "1"), claims[0] if claims else None)
    if not c1:
        return "(claim 1 not found)"
    text = (c1.get("text") or "").strip()
    return text[:nchar] + ("…" if len(text) > nchar else "")


def rejection_summary(beta: dict) -> dict:
    ctnf = beta.get("ctnf") or {}
    insts = ctnf.get("rejection_instances") or []
    if not insts:
        return {"statute": None, "claims_raw": None, "snippet": None}
    i0 = insts[0]
    rt = i0.get("rejection_type") or []
    snippet = rt[0][:200] if rt and isinstance(rt[0], str) else None
    return {
        "statute": i0.get("statute_section"),
        "claims_raw": (i0.get("claims_raw") or [None])[0],
        "snippet": snippet,
    }


def build():
    cohort = load_cohort_meta()
    lines = ["# Appendix C — Worked examples per channel", ""]
    lines.append("This appendix maps each channel's numeric output to a")
    lines.append("concrete high / low example drawn from the main experiment.")
    lines.append("Each pair shows: the case, the input context, the two")
    lines.append("conditions being contrasted, the channel's value under")
    lines.append("each, and the structural reason the score moved. Raw")
    lines.append("model-output text is not reproduced beyond the counts and")
    lines.append("limits already captured by the parser. Scores here are")
    lines.append("illustrative: they show what the channel *responds to*,")
    lines.append("not where the population median sits.")
    lines.append("")

    channel_info = {
        "c1": {
            "title": "C1 — Grounding alignment",
            "blurb": ("C1 measures the overlap between limitations modified "
                      "in the generated amendment and limitations named as "
                      "rejected in the CTNF. **Higher = more aligned.**"),
            "tsv": "c1_main.tsv",
            "cols": ["c1", "c1_recall", "c1_prec", "c1_f1", "n_claims",
                     "skipped_no_gen", "skipped_no_rejected"],
            "value_col": "c1",
        },
        "c2": {
            "title": "C2 — Revision locality",
            "blurb": ("C2 = editdist(gen, pre) / editdist(gold_post, pre). "
                      "**1.0 = model edited at gold scale; <<1 = under-edit; "
                      ">>1 = over-rewrite.**"),
            "tsv": "c2_main.tsv",
            "cols": ["c2", "pre_chars", "gen_chars", "editdist_pre_gen",
                     "denom", "n_claims_parsed"],
            "value_col": "c2",
        },
        "c3": {
            "title": "C3 — Scope preservation",
            "blurb": ("C3 = Jaccard overlap of noun phrases between generated "
                      "amendment and the invention core of the original "
                      "claim. **Higher = scope better preserved.** Drop "
                      "indicates over-narrowing."),
            "tsv": "c3_main.tsv",
            "cols": ["c3", "c3_recall", "c3_prec", "c3_f1", "n_claims"],
            "value_col": "c3",
        },
        "c5": {
            "title": "C5 — Template dependence",
            "blurb": ("C5 = canonical-phrase hit rate per 1000 characters, "
                      "where canonical phrases are mined from the retrieval "
                      "pool. **Lower = less boilerplate; higher = more "
                      "template reuse.**"),
            "tsv": "c5_main.tsv",
            "cols": ["c5", "total_hits", "char_len", "token_len", "n_claims"],
            "value_col": "c5",
        },
    }

    for ch, pick in PICKS.items():
        info = channel_info[ch]
        meta = cohort.get(pick["case"])
        if meta is None:
            lines.append(f"## {info['title']}")
            lines.append(f"(case `{pick['case']}` not in cohort metadata — skipped)")
            continue
        app_num = meta.get("app_num")
        beta = load_beta(app_num) if app_num else None

        lines.append(f"## {info['title']}")
        lines.append("")
        lines.append(info["blurb"])
        lines.append("")
        lines.append(f"**Case.** `{pick['case']}` (app {app_num}). "
                     f"Axes: {meta['axes']['outcome']} · §{meta['axes']['statute']} · "
                     f"TC {meta['axes']['tech']} · {meta['axes']['year']} · "
                     f"{meta['axes']['pattern']}.")
        lines.append("")
        if beta:
            lines.append(f"*Claim 1 (pre-amendment, preview):* "
                         f"`{first_claim_preview(beta)}`")
            rej = rejection_summary(beta)
            if rej.get("snippet"):
                lines.append(f"*Rejection:* §{rej['statute']} against "
                             f"claim {rej['claims_raw']} — "
                             f"`{rej['snippet']}…`")
            lines.append("")

        rows = load_tsv_filtered(info["tsv"], pick["case"], pick["model"])
        pass_stats = rep_mean(rows, pick["pass_cond"], info["cols"])
        fail_stats = rep_mean(rows, pick["fail_cond"], info["cols"])

        lines.append(f"**Model.** {MODEL_LABELS[pick['model']]}.")
        lines.append("")
        lines.append(f"| feature | {pick['pass_label']} — {pick['pass_header']} | {pick['fail_label']} — {pick['fail_header']} |")
        lines.append("|---|---|---|")
        for col in info["cols"]:
            p = pass_stats.get(col)
            f = fail_stats.get(col)
            p_str = f"{p:.4f}" if isinstance(p, float) else "—"
            f_str = f"{f:.4f}" if isinstance(f, float) else "—"
            label = col
            if col in (info["value_col"],):
                label = f"**{col}** (channel output)"
            lines.append(f"| {label} | {p_str} | {f_str} |")
        lines.append(f"| reps averaged | {pass_stats.get('n_reps')} | {fail_stats.get('n_reps')} |")
        lines.append("")
        lines.append(_explanation(ch, pick, pass_stats, fail_stats))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("**How to read across channels.** A model that scores **high** on C1 and C3, near 1 on C2, and **low** on C5 has produced an amendment that (i) touches the limitations the examiner actually attacked, (ii) edits at a scale comparable to the registered amendment, (iii) preserves the invention's scope, and (iv) avoids canonical-phrase boilerplate. When retrieval moves any single one of these in the wrong direction, the channel registers it; the commercial-RAG scorecard in §8.2 counts how many of those cells move in the claimed direction.")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


def _explanation(ch: str, pick: dict, p: dict, f: dict) -> str:
    if ch == "c1":
        return (
            f"**Read.** Under `{pick['fail_cond']}` the model's modified "
            f"limitations overlapped the rejected limitations at rate "
            f"{f.get('c1', 0):.3f} — near zero, meaning the model's edits "
            f"landed largely off-target. Under `{pick['pass_cond']}` the "
            f"overlap rose to {p.get('c1', 0):.3f}, an order-of-magnitude "
            f"shift on the same case and model. The C1 channel catches the "
            f"difference between an amendment that addresses the examiner's "
            f"specific attack and one that edits elsewhere in the claim. "
            f"(C1 in absolute terms is a sparse rate across all cases — the "
            f"channel's value lies in the *within-case contrast* it "
            f"surfaces, not in population averages.)"
        )
    if ch == "c2":
        p_c2 = p.get("c2", 0)
        f_c2 = f.get("c2", 0)
        return (
            f"**Read.** The gold amendment on this case has edit distance "
            f"{p.get('denom', 0):.0f} characters. Under `{pick['pass_cond']}` "
            f"the model's C2 = {p_c2:.2f} — substantially *under-editing* "
            f"the claim at about {p_c2*100:.0f}% of gold scale. Under "
            f"`{pick['fail_cond']}` C2 jumps to {f_c2:.2f} — more than "
            f"{f_c2:.0f}× the gold scale, a rewrite of the whole claim. "
            f"Both conditions miss the gold scale, but the channel "
            f"registers the two misses as categorically different: one "
            f"under-edit, one catastrophic over-edit. C2 therefore captures "
            f"the *direction and magnitude* of scale mismatch, not a "
            f"binary pass / fail."
        )
    if ch == "c3":
        return (
            f"**Read.** Under `{pick['pass_cond']}` the Jaccard noun-phrase "
            f"overlap with the invention core was {p.get('c3', 0):.3f} "
            f"(precision {p.get('c3_prec', 0):.2f}, recall "
            f"{p.get('c3_recall', 0):.2f}) — essentially complete "
            f"preservation. Under `{pick['fail_cond']}` C3 collapsed to "
            f"{f.get('c3', 0):.3f}, meaning the amendment introduced so "
            f"much new noun-phrase content (and/or dropped so much of the "
            f"original's) that the invention's scope no longer overlaps. "
            f"This is the over-narrowing / redirection failure mode C3 is "
            f"designed to catch."
        )
    if ch == "c5":
        return (
            f"**Read.** Under `{pick['pass_cond']}` canonical-phrase hits "
            f"totaled {p.get('total_hits', 0):.1f} across "
            f"{p.get('char_len', 0):.0f} characters (C5 = {p.get('c5', 0):.2f} "
            f"per 1000 chars). Under `{pick['fail_cond']}` hits rose to "
            f"{f.get('total_hits', 0):.1f} across {f.get('char_len', 0):.0f} "
            f"characters (C5 = {f.get('c5', 0):.2f}), more than 3× the "
            f"baseline rate. C5 therefore registers when retrieval pushes "
            f"the model into a boilerplate-heavy regime — the "
            f"template-injection failure mode the paper examines."
        )
    return ""


if __name__ == "__main__":
    build()
