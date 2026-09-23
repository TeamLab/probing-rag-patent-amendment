# Release checklist

What has to land before this repository goes public and the URL in the paper resolves.

## Data

- [ ] `data/corpus/` — parsed JSONL of 7,385 four-tuples
- [ ] `data/index/` — application-number index for USPTO ODP reconstruction
- [ ] `data/cohort/` — `cohort_batch0.json`, `cohort_seeds.json`
- [ ] Spot-check that no field carries anything beyond the public ODP record

## Code

- [ ] `src/parsing/` — ODP retrieval, XML parsing, per-claim diff
- [ ] `src/cohort/select_cohort.py` — six-axis matching, seed 42
- [ ] `src/probes/` — seven probe templates and the shared scaffold
- [ ] `src/metric/` — C1–C5 scoring, including semantic-C1 variant
- [ ] `src/analysis/` — table regeneration scripts
- [ ] `requirements.txt` — pinned versions
- [ ] A clean checkout regenerates every table in the paper

## Results

- [ ] `results/raw/` — raw model responses for the 9,600 calls
- [ ] `results/strata/` — per-stratum TSVs (statute section, technology center, XML format, amendment pattern)
- [ ] Per-pair C4 tables
- [ ] Dense-G and k-sweep outputs

## Release

- [ ] API keys, tokens, and internal paths removed from code and notebooks
- [ ] README status section updated (drop "skeleton")
- [ ] Switch the repository to public
- [ ] Confirm the paper's URL resolves: `https://github.com/TeamLab/probing-rag-patent-amendment`
- [ ] `CITATION.cff` updated with the ACL Anthology identifier
- [ ] Tag a release matching the camera-ready
