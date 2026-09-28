"""Compute channel correlation matrix (C1/C2/C3/C5) for §5.6 / §7.6.

For each case × condition × model, we first average across reps to a single
(case, cond, model) score per channel, then compute Pearson and Spearman
correlations between the four channels on the pooled (case, cond, model)
population. Per-model matrices are also reported.

Reads:
  outputs/_analysis/c{1,2,3,5}_main.tsv
Writes:
  docs/channel_correlations.md
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "outputs" / "_analysis"
OUT = ROOT / "docs" / "channel_correlations.md"

MODELS = ["sonnet4", "haiku4.5", "gpt5.4", "gpt4o-mini"]
MODEL_LABELS = {
    "sonnet4": "Claude Sonnet 4",
    "haiku4.5": "Claude Haiku 4.5",
    "gpt5.4": "GPT-5.4",
    "gpt4o-mini": "gpt-4o-mini",
}
METRICS = ["c1", "c2", "c3", "c5"]


def parse_model_from_file(fname: str) -> str:
    # Pattern: PILOT-XXXXXXXXXX__{cond}__{model}__rep{r}.json
    parts = fname.rsplit("__", 3)
    if len(parts) < 4:
        return ""
    return parts[-2]


def load_metric(path: Path, col: str) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            try:
                v = float(r[col])
            except (ValueError, KeyError):
                continue
            if not math.isfinite(v):
                continue
            model = parse_model_from_file(r["file"])
            if model not in MODELS:
                continue
            rows.append({
                "case": r["case"],
                "cond": r["cond"],
                "model": model,
                "rep": int(r["rep"]),
                "val": v,
            })
    return rows


def average_reps(rows: list[dict]) -> dict:
    """Return {(case, cond, model): mean across reps}."""
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["case"], r["cond"], r["model"])].append(r["val"])
    return {k: mean(v) for k, v in bucket.items()}


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    return pearson(rank(xs), rank(ys))


def build_matrix(per_key: dict, subset_model: str | None) -> tuple[list[list[float]], list[list[float]], int]:
    """Return (pearson_mat, spearman_mat, n_samples)."""
    keys = set()
    for m, d in per_key.items():
        if subset_model is None:
            keys.update(d.keys())
        else:
            keys.update(k for k in d if k[2] == subset_model)
    aligned = []
    for k in keys:
        vec = []
        ok = True
        for m in METRICS:
            if k not in per_key[m]:
                ok = False
                break
            vec.append(per_key[m][k])
        if ok:
            aligned.append(vec)
    n = len(aligned)
    pear = [[0.0] * len(METRICS) for _ in METRICS]
    sp = [[0.0] * len(METRICS) for _ in METRICS]
    if n < 3:
        for i in range(len(METRICS)):
            for j in range(len(METRICS)):
                pear[i][j] = float("nan")
                sp[i][j] = float("nan")
        return pear, sp, n
    cols = list(zip(*aligned))
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            pear[i][j] = pearson(list(cols[i]), list(cols[j]))
            sp[i][j] = spearman(list(cols[i]), list(cols[j]))
    return pear, sp, n


def fmt_row(row: list[float]) -> str:
    cells = []
    for v in row:
        if math.isnan(v):
            cells.append("—")
        else:
            flag = "**" if abs(v) > 0.7 else ""
            cells.append(f"{flag}{v:+.2f}{flag}")
    return " | ".join(cells)


def main():
    per_key = {}
    per_key["c1"] = average_reps(load_metric(SRC_DIR / "c1_main.tsv", "c1"))
    per_key["c2"] = average_reps(load_metric(SRC_DIR / "c2_main.tsv", "c2"))
    per_key["c3"] = average_reps(load_metric(SRC_DIR / "c3_main.tsv", "c3"))
    per_key["c5"] = average_reps(load_metric(SRC_DIR / "c5_main.tsv", "c5"))

    lines = []
    lines.append("# §5.6 / §7.6 — Channel correlation matrix (main data)")
    lines.append("")
    lines.append("Units: per-(case, condition, model) rep-averaged channel values.")
    lines.append("Cells: Pearson on upper-triangle labels, Spearman on lower-triangle labels.")
    lines.append("|ρ| > 0.7 flagged in **bold**.")
    lines.append("")

    # pooled
    pear, sp, n = build_matrix(per_key, None)
    lines.append(f"## Pooled across all models (n={n})")
    lines.append("")
    lines.append("**Pearson** (symmetric)")
    lines.append("")
    header = "|  | " + " | ".join(m.upper() for m in METRICS) + " |"
    sep = "|---|" + "|".join(["---"] * len(METRICS)) + "|"
    lines.append(header)
    lines.append(sep)
    for i, m in enumerate(METRICS):
        lines.append(f"| **{m.upper()}** | {fmt_row(pear[i])} |")
    lines.append("")
    lines.append("**Spearman** (symmetric)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for i, m in enumerate(METRICS):
        lines.append(f"| **{m.upper()}** | {fmt_row(sp[i])} |")
    lines.append("")

    # per-model
    for model in MODELS:
        pear, sp, n = build_matrix(per_key, model)
        lines.append(f"## {MODEL_LABELS[model]} (n={n})")
        lines.append("")
        lines.append("**Pearson**")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for i, m in enumerate(METRICS):
            lines.append(f"| **{m.upper()}** | {fmt_row(pear[i])} |")
        lines.append("")
        lines.append("**Spearman**")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for i, m in enumerate(METRICS):
            lines.append(f"| **{m.upper()}** | {fmt_row(sp[i])} |")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
