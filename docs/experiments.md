# Experiments

The entire protocol was fixed before any model call was made.

## Cohort

100 test cases, drawn by joint stratified sampling across six axes so that the cohort matches the marginal distributions of the 4,270-case eligibility pool while guaranteeing a minimum cell size of 5:

1. Subdecision outcome (Affirmed / Reversed / Affirmed-in-Part)
2. Rejection statute
3. Technology center
4. Amendment pattern (modify_only / cancel_heavy / mixed / new_or_add / other)
5. XML format era (dtd_legacy / ns_claims)
6. Decision year bin

Selection is iterative marginal matching rather than cross-product enumeration: each pool case is scored by the rarity of its six-axis vector relative to the targets, cases that move the cohort toward all six target marginals are greedily accepted, and the seed breaks ties. Batch-0 seed: **42**. Code: `src/cohort/select_cohort.py`.

Eligibility filter for the 4,270-case pool: parse health (pre- and post-claims parseable, at least one CTNF rejection instance, statute section identifiable on the first rejection); axis coverage (outcome in {Affirmed, Reversed, Affirmed-in-Part}); no obvious data defect (pre-claim 1 at least 50 characters, at least one modified or cancelled per-claim action).

## Retrieval pool

4,221 cases, indexed by (statute section, statute subsection, limitation pattern). Disjoint from the cohort by construction.

## Model matrix

A 2×2 factorial across vendor family and model tier:

- **Stage 1** — Claude Sonnet 4, GPT-5.4 (full probe battery)
- **Stage 2** — Claude Haiku 4.5, GPT-4o-mini (conditional on Stage-1 effect-size criteria)

Each case is scored under 8 conditions × 3 replicates = 24 evaluations per case per model, giving within-model paired comparisons with >80% power at d = 0.3, α = 0.05.

Inference settings: temperature 0.3 for all models; three independent replicates per (case, probe, model) with separate sampling seeds; retrieval k = 3.

Totals: 9,600 model calls, 97.4% parse success overall (93.0% on GPT-4o-mini to 100% on GPT-5.4). Per-response scores aggregate as per-case mean across replicates, then median across cases within (model, condition).

## Pre-registered hypotheses

- **H1 (retrieval as prior-injection anchor)** — Probe F raises C5 over baseline by at least 0.2 and does not improve C1 by more than 0.1.
- **H2 (similarity as second-order)** — Probes F and G yield statistically indistinguishable shifts on C1, C2, and C5 (|Δ_F − Δ_G| ≤ 0.1 per channel, or same sign with no significant ranking).
- **H3 (template-recycling inflation)** — retrieval does not push template dependence below baseline on either probe.

## Statistics

Per-model aggregation to a per-case mean, retrieval condition paired against the no-retrieval baseline case by case, paired Wilcoxon signed-rank test. p-values are Holm–Bonferroni corrected within the C5 test family; bootstrap 95% CIs on the median paired difference use 10,000 resamples (seed 42). Equivalence against the pre-registered ±0.2 margin is assessed by two one-sided tests on the median ΔC5.
