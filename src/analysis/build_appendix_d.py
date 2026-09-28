"""Build Appendix Table D.1: full probe × channel delta table (all models).

Reads: outputs/_analysis/main_aggregate.tsv
Writes: docs/appendix_d_table.md (markdown table)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "_analysis" / "main_aggregate.tsv"
OUT = ROOT / "docs" / "appendix_d_table.md"

MODELS = [
    ("sonnet4", "Claude Sonnet 4"),
    ("haiku4.5", "Claude Haiku 4.5"),
    ("gpt5.4", "GPT-5.4"),
    ("gpt4o-mini", "gpt-4o-mini"),
]
METRICS = ["c1", "c2", "c3", "c5"]
CONDITIONS = ["baseline", "A", "B", "C", "D", "E", "F", "G"]


def load() -> dict:
    out = {}
    with SRC.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["metric"], row["model"], row["condition"])] = {
                "median": float(row["median"]),
                "n": int(row["n_cases"]),
            }
    return out


def fmt_delta(v: float, metric: str) -> str:
    if metric == "c2":
        return f"{v:+.3f}"
    if metric == "c5":
        return f"{v:+.2f}"
    return f"{v:+.4f}"


def fmt_base(v: float, metric: str) -> str:
    if metric == "c2":
        return f"{v:.3f}"
    if metric == "c5":
        return f"{v:.2f}"
    return f"{v:.4f}"


def main():
    data = load()
    lines = []
    lines.append("# Appendix Table D.1 — Probe × channel medians (full)")
    lines.append("")
    lines.append("Values are per-case medians across 100 cohort cases × 3 replicates.")
    lines.append("baseline: raw median (no intervention). A–G: Δ from baseline.")
    lines.append("C1 = grounding alignment (rate, overlap limitation-tokens / case).")
    lines.append("C2 = revision locality (ratio editdist(gen,pre)/editdist(gold,pre)).")
    lines.append("C3 = scope preservation (Jaccard NP overlap).")
    lines.append("C5 = template dependence (canonical-phrase rate per 1000 chars).")
    lines.append("")

    for key, label in MODELS:
        lines.append(f"## {label} (`{key}`)")
        lines.append("")
        header = "| condition | " + " | ".join(m.upper() for m in METRICS) + " | n |"
        sep = "|---|" + "|".join(["---"] * (len(METRICS) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        # baseline row
        row_cells = ["baseline"]
        min_n = 9999
        for m in METRICS:
            rec = data.get((m, key, "baseline"))
            if rec is None:
                row_cells.append("—")
            else:
                row_cells.append(fmt_base(rec["median"], m))
                min_n = min(min_n, rec["n"])
        row_cells.append(str(min_n))
        lines.append("| " + " | ".join(row_cells) + " |")
        # probe rows as deltas
        for cond in CONDITIONS[1:]:
            row_cells = [cond]
            min_n = 9999
            for m in METRICS:
                base = data.get((m, key, "baseline"))
                rec = data.get((m, key, cond))
                if rec is None or base is None:
                    row_cells.append("—")
                else:
                    d = rec["median"] - base["median"]
                    row_cells.append(fmt_delta(d, m))
                    min_n = min(min_n, rec["n"])
            row_cells.append(str(min_n))
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    # H1/H2/H3 verdict footer
    lines.append("## Pre-registered verdicts (reference copy)")
    lines.append("")
    lines.append("| model | ΔF (C5) | ΔG (C5) | H1 (ΔC5>0.2) | H2 (\\|ΔF−ΔG\\|<0.1, same sign) | H3 (both ≥ 0) |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in MODELS:
        base = data.get(("c5", key, "baseline"))
        f_val = data.get(("c5", key, "F"))
        g_val = data.get(("c5", key, "G"))
        df = f_val["median"] - base["median"]
        dg = g_val["median"] - base["median"]
        h1 = "supported" if max(abs(df), abs(dg)) > 0.2 and (df > 0.2 or dg > 0.2) else (
            "boundary" if max(df, dg) >= 0.2 else "not supported"
        )
        same_sign = (df > 0 and dg > 0) or (df < 0 and dg < 0)
        h2 = "SUPPORTED" if (abs(df - dg) < 0.1 and same_sign) else "not supported"
        h3 = "SUPPORTED" if (df >= 0 and dg >= 0) else "not supported"
        lines.append(f"| {label} | {df:+.2f} | {dg:+.2f} | {h1} | {h2} | {h3} |")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
