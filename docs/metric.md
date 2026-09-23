# Five-channel metric

Every channel is a function of text, parser output, and the XML gold post-amendment claim. No LLM acts as judge: the four models cluster into distinct behavioral patterns under retrieval perturbation, so picking one as judge would install that model's signature as the evaluation axis.

| Channel | What it measures | Definition |
|---|---|---|
| **C1** grounding alignment | Is the model editing the limitation the examiner attacked? | Overlap between limitations modified in the generated amendment and limitations named as rejected, at limitation-token level after claim-number alignment |
| **C2** revision locality | Did the model edit at the right scale? | `editdist(gen, pre) / editdist(gold_post, pre)`; 1 matches the gold edit scale, ≪1 is under-editing, ≫1 is over-rewriting |
| **C3** scope preservation | Did the amendment over-narrow? | Jaccard overlap of noun phrases between the generated amendment and the original claim's invention core, after limitation-removal normalization |
| **C4** robustness | Did the amendment shift in the predicted direction? | Defined on pairs of probe conditions, scored per probe and averaged within a model; supplementary only |
| **C5** template dependence | Did retrieval inflate canonical phrasing? | Frequency of canonical amendment phrases mined from the retrieval pool, per 1,000 characters of generated claim text |

## Reading C5

C5's absolute level is **not** a measure of prior injection — canonical phrasing is normal in competent drafting, so a high C5 may reflect legitimate convention. Prior injection is diagnosed only from **ΔC5**, the retrieval-induced change over the no-retrieval baseline for the same case and model, so convention present at baseline is differenced out. C5 remains a proxy that cannot by itself separate appropriate phrasing from recycling, and a lower C5 does not automatically mean better grounding.

## Why C4 is supplementary

C4 is defined on probe pairs rather than on individual amendments, so it is reported separately to keep the main tables dimensionally consistent with the per-amendment channels. Grand C4 on the cohort across the seven pre-registered (probe, channel) pairs is 0.56 (n=2,276 case-model contrasts).

## Validity

The design follows Campbell and Fiske's multitrait-multimethod logic: each channel ties to a specific XML-record feature rather than an LLM judgment. Three legs:

1. **Construct checks** — within-case checks that each channel moves in the direction a patent practitioner would predict.
2. **Near-independence** — pooled across 3,062 (case, condition, model) triples, all off-diagonal correlations |ρ| ≤ 0.29, below the pre-registered |ρ| > 0.7 flag threshold.
3. **Attorney study** — a 30–50-case double-rated study committed post-review; a 10-case pre-release sanity sample had all four channels moving in the direction the reading attorney predicted.

Legs 1 and 2 carry the metric-validity argument in the paper; the formal κ leg is deferred to the post-review study.

## Robustness variants shipped with the paper

- **Semantic C1** — the same limitations rescored with a paraphrase-sensitive embedding scorer (`all-MiniLM-L6-v2`) instead of trigram overlap. Lexical and semantic C1 are only loosely concordant (Spearman ρ = 0.37), so this is a genuine second measurement; the retrieval null is unchanged under it.
- **Dense-G** — Probe G's deterministic matcher swapped for an embedding-cosine top-3 retriever, holding pool, k, exemplar fields, injection format, and base prompt byte-identical.
- **Retrieval depth** — k ∈ {1, 3, 5, 10} with everything else fixed.
