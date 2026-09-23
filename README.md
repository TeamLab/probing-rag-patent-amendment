# Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment

Artifacts for the AACL paper *Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment* (Leo, Min, Jang, Zidny, Chung, Choi).

The paper asks whether retrieval grounds patent claim amendment or merely injects templates, using USPTO prosecution records where the examiner names the attacked limitation and cites prior art, so that "correct" has a definable meaning. Across 9,600 pre-registered calls on four frontier LLMs, no tested model shows classical prior-injection behavior.

## Status

Skeleton. The corpus, code, and result files are not uploaded yet — see [Release checklist](#release-checklist).

## What will be released

| Artifact | Path | Description |
|---|---|---|
| Corpus | `data/corpus/` | 7,385 USPTO prosecution four-tuples (pre/post claims, rejection, cited prior art) as JSONL |
| Application-number index | `data/index/` | Application numbers permitting zero-cost reconstruction from USPTO ODP |
| Cohort | `data/cohort/` | `cohort_batch0.json`, `cohort_seeds.json` for batch replication |
| Parsing pipeline | `src/parsing/` | XML parsing and four-tuple alignment |
| Cohort selection | `src/cohort/select_cohort.py` | Deterministic six-axis marginal matching, seed 42 |
| Probe battery | `src/probes/` | Seven probe prompt templates under a fixed prompt scaffold |
| Metric | `src/metric/` | Deterministic five-channel metric (C1–C5), no LLM evaluation |
| Analysis | `src/analysis/` | Scripts regenerating every table from raw model outputs |
| Stratified results | `results/strata/` | Per-stratum TSVs (statute section, technology center, XML format, amendment pattern) |

## Reproducing

```bash
pip install -r requirements.txt
```

Reproducing the model calls requires API access to the four evaluated LLMs (Claude Sonnet 4, Claude Haiku 4.5, GPT-5.4, GPT-4o-mini). Per-model cost estimates and inference settings are in the paper's appendices. Scoring and analysis run offline from the released raw outputs and need no API access.

## Data statement

The corpus is derived from USPTO Open Data Portal records of US patent prosecution. USPTO ODP data carries no copyright restriction. The records are administrative documents about patent applications, not personal data collections; applicant and attorney names appear as they do in the public record. No annotation was crowdsourced.

## Licenses

- Code: MIT (`LICENSE`)
- Corpus, index, and result files: CC BY 4.0 (`LICENSE-data`)

## Citation

See `CITATION.cff`. The BibTeX entry will be updated with the ACL Anthology identifier once the paper appears.

## Release checklist

- [ ] Upload corpus JSONL and the application-number index
- [ ] Upload parsing, cohort, probe, metric, and analysis code
- [ ] Upload raw model outputs and per-stratum TSVs
- [ ] Pin dependency versions in `requirements.txt`
- [ ] Verify a clean checkout regenerates every table in the paper
- [ ] Switch the repository to public and confirm the URL in the paper resolves
- [ ] Update `CITATION.cff` with the Anthology identifier
