"""Build Figure 2: 4-model × (probe × channel) delta heatmap panels.

Reads: outputs/_analysis/main_aggregate.tsv
Writes: paper/figures/figure2_heatmap.png, paper/figures/figure2_heatmap.pdf

Each of 4 panels (one per model) shows rows=probes A/B/C/D/E/F/G, cols=channels
C1/C2/C3/C5, cells = (probe_median - baseline_median). C4 is excluded (stored
separately in c4_summary.json with a different structure).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "_analysis" / "main_aggregate.tsv"
OUT_DIR = ROOT / "paper" / "figures"

MODELS = ["sonnet4", "haiku4.5", "gpt5.4", "gpt4o-mini"]
MODEL_LABELS = {
    "sonnet4": "Claude Sonnet 4",
    "haiku4.5": "Claude Haiku 4.5",
    "gpt5.4": "GPT-5.4",
    "gpt4o-mini": "gpt-4o-mini",
}
METRICS = ["c1", "c2", "c3", "c5"]
METRIC_LABELS = {
    "c1": "C1\n(grounding)",
    "c2": "C2\n(locality)",
    "c3": "C3\n(scope)",
    "c5": "C5\n(template)",
}
PROBES = ["A", "B", "C", "D", "E", "F", "G"]
PROBE_LABELS = {
    "A": "A (limitation)",
    "B": "B (rationale)",
    "C": "C (decoy)",
    "D": "D (boilerplate)",
    "E": "E (hint)",
    "F": "F (rand retr)",
    "G": "G (struct retr)",
}


def load_medians() -> dict:
    """Return {(metric, model, condition): median}."""
    out = {}
    with SRC.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["metric"], row["model"], row["condition"])
            out[key] = float(row["median"])
    return out


def build_delta_matrix(medians: dict, model: str) -> np.ndarray:
    """Rows = probes, cols = metrics. Cells = probe_median - baseline_median."""
    mat = np.zeros((len(PROBES), len(METRICS)))
    for i, probe in enumerate(PROBES):
        for j, metric in enumerate(METRICS):
            base = medians.get((metric, model, "baseline"))
            val = medians.get((metric, model, probe))
            mat[i, j] = (val - base) if (base is not None and val is not None) else np.nan
    return mat


def _rescale_c5(mat: np.ndarray) -> np.ndarray:
    """C5 is per-1000-char (values ~0.1–0.3 scale); others are rate-of-overlap
    on a smaller scale (~0.01–0.05). Rescale C5 column for consistent colormap.
    We report both raw C5 and scaled — here we divide C5 by 10 so a visible
    Δ=+0.20 on C5 matches Δ=+0.02 visually on C1/C3. Annotations show raw values.
    """
    scaled = mat.copy()
    scaled[:, 3] = scaled[:, 3] / 10.0
    return scaled


def plot():
    medians = load_medians()
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    all_raw = np.stack([build_delta_matrix(medians, m) for m in MODELS])
    # Separate scales: annotate raw values, color by scaled
    vmax = 0.3  # covers c1/c2/c3 deltas; c5 rescaled to same visual range
    vmin = -vmax

    for ax, model in zip(axes, MODELS):
        raw = build_delta_matrix(medians, model)
        scaled = _rescale_c5(raw)
        im = ax.imshow(scaled, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

        for i in range(len(PROBES)):
            for j in range(len(METRICS)):
                val = raw[i, j]
                if np.isnan(val):
                    continue
                # Show raw delta; smaller font for C5 which has wider range
                fmt = "{:+.2f}" if METRICS[j] == "c5" else "{:+.3f}"
                txt_color = "white" if abs(scaled[i, j]) > 0.18 else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=8, color=txt_color)

        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
        ax.set_yticks(range(len(PROBES)))
        ax.set_yticklabels([PROBE_LABELS[p] for p in PROBES], fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xlabel("channel", fontsize=9)

    axes[0].set_ylabel("probe", fontsize=9)
    fig.suptitle("Probe × channel Δ from baseline (median across 100 cases × 3 reps)",
                 fontsize=12, y=1.02)

    # Shared colorbar (scaled); note about C5 rescale
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.015,
                         pad=0.02, shrink=0.75)
    cbar.set_label("Δ (C1/C2/C3 raw; C5 rescaled ÷10 for color)", fontsize=8)

    OUT_DIR.mkdir(exist_ok=True)
    out_png = OUT_DIR / "figure2_heatmap.png"
    out_pdf = OUT_DIR / "figure2_heatmap.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    plot()
