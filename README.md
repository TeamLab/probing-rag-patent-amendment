# Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment

Artifacts for the AACL paper *Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment* — Josepha Michiko Leo\*, Hyun-seok Min\*, Yehoon Jang, Irvan Zidny, Jin-Woo Chung, Sungchul Choi (\*equal contribution).

Retrieval-augmented generation is widely used in professional writing, but whether retrieval grounds revision or merely injects templates is rarely tested where "correct" has a definable meaning. Patent claim amendment supplies that signal: the examiner names the attacked limitation and cites prior art, giving per-case ground truth. Across 9,600 pre-registered calls on four frontier LLMs, no tested model shows classical prior-injection behavior; retrieval effects are small and direction-inconsistent between random and structural retrieval.

## Status

Released — the parsed corpus, all 9,600 model outputs, the full pipeline (`src/`), and the analysis scripts all live in this repository and regenerate the tables in the paper. Analysis and scoring run offline from the released raw outputs; only regenerating the model calls needs API access. 

## Three released artifacts

1. **Corpus** — 7,385 USPTO prosecution cases with XML-aligned pre/post claims, rejection, and cited prior art.
2. **Probe battery** — seven probes comparing random and structural-match retrieval as two policies under a fixed prompt scaffold.
3. **Five-channel metric** — deterministic, requiring no LLM evaluation (C1–C3 and C5 in the main paper, C4 supplementary).

## Repository layout

| Path | Contents |
|---|---|
| `data/corpus/` | Parsed JSONL corpus of 7,385 four-tuples |
| `data/index/` | Application-number index for zero-cost reconstruction from USPTO ODP |
| `data/cohort/` | `cohort_batch0.json`|
| `src/parsing/` | ODP retrieval, XML parsing, per-claim diff (kept / modified / new / cancelled) |
| `src/cohort/select_cohort.py` | Six-axis iterative marginal matching, batch-0 seed 42 |
| `src/probes/` | The seven probe templates and the shared prompt scaffold |
| `src/metric/` | Five-channel scoring (C1–C5) |
| `src/analysis/` | Scripts regenerating every table from raw model outputs |
| `results/raw/` | Raw model responses (9,600 calls) |
| `results/strata/` | Per-stratum TSVs: statute section, technology center, XML format, amendment pattern |
| `docs/` | Corpus, probe, metric, and experiment documentation |

## Documentation

- [docs/corpus.md](docs/corpus.md) — source, alignment procedure, summary statistics, release form
- [docs/probes.md](docs/probes.md) — probes A–G and what each one manipulates
- [docs/metric.md](docs/metric.md) — the five channels and their validity argument
- [docs/experiments.md](docs/experiments.md) — cohort, retrieval pool, model matrix, pre-registered hypotheses

## Reproducing

```bash
pip install -r requirements.txt
```

Scoring and analysis run offline from the released raw outputs and need no API access. Regenerating the model calls requires API access to the four evaluated models (Claude Sonnet 4, Claude Haiku 4.5, GPT-5.4, GPT-4o-mini); inference used temperature 0.3, three replicates per (case, probe, model) with separate sampling seeds, and retrieval depth k=3.

The corpus can also be rebuilt from scratch: the application-number index plus `src/parsing/` reconstructs every four-tuple from the USPTO Open Data Portal at no cost.

## Data statement

The corpus derives from USPTO Open Data Portal records of US patent prosecution, starting from the PILOT-Bench PTAB-appeal subset. USPTO ODP data carries no copyright restriction. These are administrative records about patent applications; applicant, attorney, and examiner names appear as they do in the public record. No annotation was crowdsourced, and the attorney validation study reported in the paper was conducted with a licensed practitioner.

## Licenses

- Code: MIT — [`LICENSE`](LICENSE)
- Corpus, index, cohort, and result files: CC BY 4.0 — [`LICENSE-data`](LICENSE-data)

## Citation

See [`CITATION.cff`](CITATION.cff). The BibTeX entry will carry the ACL Anthology identifier once the paper appears.
