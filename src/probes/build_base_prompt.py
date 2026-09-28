"""
Build the baseline (no-probe) prompt for a given case from the β corpus.

Reads:
  - contexts/system.txt            — system role + formatting instructions
  - contexts/user_base.txt         — user template with {placeholders}
  - data/parsed/beta_parsed/<app_num>.json
  - data/parsed/alpha_corpus.jsonl (for title, tech_center)

Output:
  {"system": str, "user": str, "case_id": str, "app_num": str}

Usage:
  python3 scripts/build_base_prompt.py \\
      --cohort data/parsed/cohort_batch0.json \\
      --beta-parsed-dir data/parsed/beta_parsed \\
      --alpha data/parsed/alpha_corpus.jsonl \\
      --case-index 0 \\
      --out-dir data/parsed/prompts_smoke/
  # or --all to render all cohort cases
"""

import argparse
import json
import re
from pathlib import Path


CTNF_MAX_CHARS = 8000   # truncate rejection body if extremely long


def load_templates(contexts_dir: Path):
    sys_text = (contexts_dir / "system.txt").read_text(encoding="utf-8")
    user_tpl = (contexts_dir / "user_base.txt").read_text(encoding="utf-8")
    return sys_text, user_tpl


def format_pre_claims(claims):
    lines = []
    for c in claims:
        num_raw = c.get("num") or ""
        # Only accept numeric claim numbers (drop boundary-data / header junk).
        if not str(num_raw).isdigit():
            continue
        status = (c.get("status") or "").strip().lower()
        if status in ("", "unlabeled"):
            status = "original"
        text = (c.get("text") or "").strip()
        if not text or len(text) < 30:
            continue
        # Drop leading "N." when the claim text begins with its own number —
        # redundant with the prefix we are adding.
        text = re.sub(rf"^{num_raw}\s*[.\s]+", "", text).strip()
        lines.append(f"Claim {num_raw} ({status}): {text}")
    return "\n\n".join(lines) if lines else "(no claim text available)"


def _format_statute(stat):
    if not stat:
        return "§?"
    return f"§{stat}"


def format_rejection(ctnf):
    """Build a compact yet faithful rejection description.

    Strategy: show structured rejection_instances (one per grounds) followed by
    the full examiner body text (truncated to CTNF_MAX_CHARS).
    """
    parts = []
    for i, ri in enumerate(ctnf.get("rejection_instances") or [], start=1):
        claims = ", ".join(ri.get("claims_raw") or []) or "(unspecified)"
        statute = _format_statute(ri.get("statute_section"))
        subsec = ri.get("statute_subsection") or []
        subsec_str = f"({subsec[0]})" if subsec else ""
        rtype = ri.get("rejection_type") or []
        rtype_str = ", ".join(rtype) if rtype else ""
        refs = ri.get("prior_art_refs") or []
        ref_str = "; ".join(refs[:3]) if refs else "(none listed)"
        line = (f"Rejection {i}: Claim(s) {claims} rejected under 35 U.S.C. "
                f"{statute}{subsec_str}"
                + (f" as {rtype_str}" if rtype_str else "")
                + (f" over {ref_str}." if refs else "."))
        parts.append(line)

    if not parts:
        parts.append("(No structured rejection instances parsed; see full text below.)")

    full_text = (ctnf.get("full_text") or "").strip()
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    # Trim at footer markers so examiner signature / PAIR boilerplate /
    # filing-address paragraphs don't contaminate the prompt.
    FOOTER_MARKERS = (
        r"Any inquiry concerning (?:this|the) communication",
        r"The prior art made of record and not relied upon",
        r"\bConclusion\b",
        r"Information regarding the status of an application",
        r"Commissioner of Patents and Trademarks",
    )
    earliest = len(full_text)
    for pat in FOOTER_MARKERS:
        m = re.search(pat, full_text)
        if m and m.start() > 400:  # avoid chopping rejection body too early
            earliest = min(earliest, m.start())
    if earliest < len(full_text):
        full_text = full_text[:earliest].rstrip()

    if len(full_text) > CTNF_MAX_CHARS:
        full_text = full_text[:CTNF_MAX_CHARS] + "\n[... truncated ...]"

    parts.append("---\nExaminer body (verbatim):\n" + full_text)
    return "\n\n".join(parts)


def format_prior_art(ctnf):
    refs = ctnf.get("prior_art_refs") or []
    if not refs:
        return "(no structured references; see rejection body)"
    seen = []
    seen_set = set()
    for r in refs:
        r = (r or "").strip()
        if r and r not in seen_set:
            seen.append(r)
            seen_set.add(r)
    return "\n".join(f"- {r}" for r in seen[:15])


def build_prompt(case_axes: dict, beta_rec: dict, alpha_row: dict,
                 sys_text: str, user_tpl: str):
    title = alpha_row.get("title") or "(no title)"
    app_num = beta_rec.get("app_num") or ""
    tech_center = alpha_row.get("tech_center") or "(unknown)"

    pre_claims_text = format_pre_claims(
        (beta_rec.get("pre_clm") or {}).get("claims") or []
    )
    ctnf = beta_rec.get("ctnf") or {}
    rejection_text = format_rejection(ctnf)
    prior_art_list = format_prior_art(ctnf)

    user = user_tpl.format(
        title=title[:160],
        app_num=app_num,
        tech_center=tech_center,
        pre_claims=pre_claims_text,
        rejection_text=rejection_text,
        prior_art_list=prior_art_list,
    )

    return {
        "system": sys_text,
        "user": user,
        "case_id": beta_rec.get("case_id"),
        "proceeding_number": beta_rec.get("proceeding_number"),
        "app_num": app_num,
        "axes": case_axes,
        "n_pre_claims": len((beta_rec.get("pre_clm") or {}).get("claims") or []),
        "n_rejection_instances": len(ctnf.get("rejection_instances") or []),
        "n_prior_art_refs": len(ctnf.get("prior_art_refs") or []),
        "user_chars": len(user),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--case-index", type=int, default=None,
                    help="Render only this index from cohort (0-based).")
    ap.add_argument("--all", action="store_true",
                    help="Render every case in cohort.")
    ap.add_argument("--contexts-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "contexts")
    args = ap.parse_args()

    sys_text, user_tpl = load_templates(args.contexts_dir)

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"]

    alpha_by_proc = {}
    with args.alpha.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            alpha_by_proc[str(r.get("proceeding_number") or "")] = r

    if args.case_index is not None:
        indices = [args.case_index]
    elif args.all:
        indices = list(range(len(cases)))
    else:
        indices = [0]   # default: first case

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        if idx >= len(cases):
            continue
        c = cases[idx]
        bp = args.beta_parsed_dir / f"{c['app_num']}.json"
        if not bp.exists():
            continue
        beta_rec = json.loads(bp.read_text(encoding="utf-8"))
        alpha_row = alpha_by_proc.get(c["proceeding_number"], {})
        prompt = build_prompt(c["axes"], beta_rec, alpha_row, sys_text, user_tpl)
        out_path = args.out_dir / f"{c['case_id']}__baseline.json"
        out_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        # brief console summary
        print(f"[{idx}] {c['case_id']} app={c['app_num']}  "
              f"pre={prompt['n_pre_claims']}  rej={prompt['n_rejection_instances']}  "
              f"pa={prompt['n_prior_art_refs']}  user_chars={prompt['user_chars']}")

    print(f"\nRendered prompts → {args.out_dir}")


if __name__ == "__main__":
    main()
