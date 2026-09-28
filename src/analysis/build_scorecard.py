"""Build the §7 commercial-RAG scorecard table.

For each (model, channel) we report:
  - direction vs the advertised commercial-RAG claim (✓ supports / ✗ opposite / · null)
  - signed Δ magnitude (F-G average)
  - pass Y/N given a documented threshold

Thresholds (documented in the markdown output):
  C1 grounding  : support requires ΔC1 ≥ +0.005   (~30% of baseline median)
  C2 locality   : support requires distance-to-gold to drop by ≥ 20% of its
                  baseline value
  C3 scope      : support requires ΔC3 ≥ +0.02
  C5 template   : support requires ΔC5 ≤ −0.20     (mirror of H1 threshold)

Reads:   outputs/_analysis/main_aggregate.tsv
Writes:  docs/scorecard_commercial_rag.md
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "_analysis" / "main_aggregate.tsv"
OUT = ROOT / "docs" / "scorecard_commercial_rag.md"

MODELS = [
    ("sonnet4", "Claude Sonnet 4"),
    ("haiku4.5", "Claude Haiku 4.5"),
    ("gpt5.4", "GPT-5.4"),
    ("gpt4o-mini", "gpt-4o-mini"),
]

# thresholds
C1_THR = 0.005
C2_DIST_REDUCTION_FRAC = 0.20
C3_THR = 0.02
C5_THR = -0.20  # negative = reduction


def load():
    out = {}
    with SRC.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["metric"], r["model"], r["condition"])] = float(r["median"])
    return out


def direction(delta: float, support_sign: int, thr: float) -> tuple[str, str]:
    """Return (Y/N/neutral, arrow) relative to the support direction.

    support_sign = +1 if positive delta supports the claim, −1 otherwise.
    """
    support_delta = delta * support_sign
    if support_delta >= thr:
        return "Y", "✓"
    if support_delta <= -thr:
        return "N", "✗"
    return "·", "·"


def row_entry(score: str, arrow: str, detail: str) -> str:
    return f"**{score}** {arrow} ({detail})"


def build():
    m = load()
    lines = []
    lines.append("# Commercial-RAG scorecard (§7)")
    lines.append("")
    lines.append("Each cell reports whether retrieval (F/G average Δ) moved the model")
    lines.append("in the direction the advertised commercial-RAG property would require,")
    lines.append("with magnitude exceeding a documented threshold.")
    lines.append("")
    lines.append("Legend:")
    lines.append("- **Y** ✓ — retrieval moved the score meaningfully toward the claim")
    lines.append("- **N** ✗ — retrieval moved the score meaningfully *against* the claim")
    lines.append("- **·** — retrieval null or sub-threshold")
    lines.append("")
    lines.append("Thresholds:")
    lines.append(f"- **C1 grounding**: Y requires ΔC1 ≥ +{C1_THR:.3f} (about 30% of baseline median).")
    lines.append(f"  N if ΔC1 ≤ −{C1_THR:.3f}.")
    lines.append("- **C2 edit size closer to gold**: Y requires distance-to-1 to drop by")
    lines.append(f"  ≥ {int(C2_DIST_REDUCTION_FRAC*100)}% of its baseline value. N if it grows by")
    lines.append(f"  ≥ {int(C2_DIST_REDUCTION_FRAC*100)}% of baseline.")
    lines.append(f"- **C3 scope preservation**: Y requires ΔC3 ≥ +{C3_THR:.2f}. N if ΔC3 ≤ −{C3_THR:.2f}.")
    lines.append(f"- **C5 template reduction**: Y requires ΔC5 ≤ {C5_THR:.2f} (mirror of H1).")
    lines.append(f"  N if ΔC5 ≥ {-C5_THR:.2f}.")
    lines.append("")
    lines.append("## Scorecard — per-model × per-claim")
    lines.append("")
    lines.append("| Model | Retrieval grounds output (ΔC1 up) | Retrieval matches gold edit size (C2→1) | Retrieval preserves scope (ΔC3 up) | Retrieval reduces boilerplate (ΔC5 down) |")
    lines.append("|---|---|---|---|---|")

    count_y = 0
    count_n = 0
    count_null = 0

    for key, label in MODELS:
        # C1
        dC1 = ((m[("c1", key, "F")] - m[("c1", key, "baseline")]) +
               (m[("c1", key, "G")] - m[("c1", key, "baseline")])) / 2
        c1_score, c1_arrow = direction(dC1, +1, C1_THR)

        # C2 distance to gold
        b2 = m[("c2", key, "baseline")]
        avg2 = (m[("c2", key, "F")] + m[("c2", key, "G")]) / 2
        dist_b = abs(b2 - 1.0)
        dist_a = abs(avg2 - 1.0)
        reduction_frac = (dist_b - dist_a) / dist_b if dist_b > 0 else 0.0
        if reduction_frac >= C2_DIST_REDUCTION_FRAC:
            c2_score, c2_arrow = "Y", "✓"
        elif reduction_frac <= -C2_DIST_REDUCTION_FRAC:
            c2_score, c2_arrow = "N", "✗"
        else:
            c2_score, c2_arrow = "·", "·"

        # C3
        dC3 = ((m[("c3", key, "F")] - m[("c3", key, "baseline")]) +
               (m[("c3", key, "G")] - m[("c3", key, "baseline")])) / 2
        c3_score, c3_arrow = direction(dC3, +1, C3_THR)

        # C5 reduction — support direction negative, so support_sign = -1
        dC5 = ((m[("c5", key, "F")] - m[("c5", key, "baseline")]) +
               (m[("c5", key, "G")] - m[("c5", key, "baseline")])) / 2
        c5_score, c5_arrow = direction(dC5, -1, abs(C5_THR))

        # tally
        for s in (c1_score, c2_score, c3_score, c5_score):
            if s == "Y":
                count_y += 1
            elif s == "N":
                count_n += 1
            else:
                count_null += 1

        # format
        c1_cell = row_entry(c1_score, c1_arrow, f"ΔC1 = {dC1:+.4f}")
        c2_cell = row_entry(c2_score, c2_arrow,
                             f"dist {dist_b:.3f}→{dist_a:.3f}, {reduction_frac*100:+.0f}%")
        c3_cell = row_entry(c3_score, c3_arrow, f"ΔC3 = {dC3:+.4f}")
        c5_cell = row_entry(c5_score, c5_arrow, f"ΔC5 = {dC5:+.2f}")

        lines.append(f"| {label} | {c1_cell} | {c2_cell} | {c3_cell} | {c5_cell} |")

    lines.append("")
    lines.append(f"**Totals across 16 cells**: Y = {count_y}, N = {count_n}, · (null) = {count_null}.")
    lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append("Of the 16 claim × model combinations in this table, only a small minority")
    lines.append("meet the threshold for \"retrieval meaningfully supports the advertised")
    lines.append("property\" (**Y** cells). Several cells are **N** — retrieval moved the")
    lines.append("score meaningfully *against* the claim — which the aggregate medians and")
    lines.append("one-sided scoring used in earlier charts washed out to zero. See Appendix")
    lines.append("for per-metric worked examples mapping each number to a concrete failure")
    lines.append("mode.")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
