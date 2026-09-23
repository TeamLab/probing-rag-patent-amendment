# Corpus

7,385 prosecution cases in which every amendment is aligned at the XML level with its triggering rejection, the cited prior art, and the pre- and post-amendment claim text.

## Source and alignment

Starting point: the PILOT-Bench PTAB-appeal subset (13,749 cases). Each case is enriched through the USPTO Open Data Portal (ODP):

1. Resolve the application number.
2. Retrieve the file-wrapper document list.
3. Identify the first non-final rejection (CTNF).
4. Take the immediately preceding Claims filing as the pre-amendment state.
5. Take the first Claims filing after the rejection as the post-amendment state.

For each pre/post claim pair the pipeline computes a per-claim difference status — kept, modified, new, cancelled — using claim-number alignment. Modified claims additionally carry a SequenceMatcher similarity ratio and the list of added and removed spans. This difference is the ground truth the generated amendments are scored against.

## Summary statistics

| Quantity | Value |
|---|---|
| Parsed cases | 7,385 |
| Cases admitting a case-level C2 denominator | 5,755 |
| Per-claim actions that are modifications | 64.5% (median similarity 0.933, ~7% character-level edit) |
| Cancellations and new claims | 32.5% |
| Scorable cases where rejected and post-amendment-modified claim numbers overlap | 94.5% |

First-rejection statute breakdown: §102 (2,337), §112 (1,245), §101 (950), §103 (738), remainder mixed.

## Derived splits

- **Cohort** — 100 cases, stratified on six axes (see [experiments.md](experiments.md)).
- **Retrieval pool** — 4,221 cases, indexed by (statute section, statute subsection, limitation pattern), used by Probes F and G. Disjoint from the cohort by construction: the pool is the corpus with the cohort cases removed.
- **Replication reserve** — remaining cases, held for the sampling-bias replication described in the paper's appendix.

## Release form

Two release paths, both provided:

- the parsed JSONL corpus, directly usable; and
- the application-number index plus parsing code, which reconstructs the corpus from USPTO ODP at no cost.

## Position against related corpora

This corpus is the only one among Patent-CR, PANORAMA, PEDANTIC, and PILOT-Bench that aligns all four elements — rejection context, cited prior art, pre/post claim pair, and amendment diff — at the XML level. Patent-CR provides the pre/post pair but omits the rejection and prior-art channels; PANORAMA, PEDANTIC, and PILOT-Bench target judgment, classification, or upstream retrieval rather than amendment generation.
