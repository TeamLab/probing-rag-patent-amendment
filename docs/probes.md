# Probe battery

Seven probes, all sharing one input template (pre-amendment claim, rejection rationale, cited prior art) and differing only in what is manipulated. Each probe's expected effect on the five channels is fixed in advance.

## Observational probes (A–E)

These characterize each model's baseline amendment signature under input perturbation.

| Probe | Manipulation | Predicted signal |
|---|---|---|
| **A — Claim Truncation** | Modify one decisive limitation in the rejected independent claim by deletion, addition, paraphrase, or antonym swap | A grounded model follows the perturbation in C1; a model ignoring the rejection signal shows C1 invariance |
| **B — Rationale Shuffling** | Replace one sentence of the examiner findings with a semantically inconsistent fragment, or permute sentence order | A rationale-grounded model holds C1 stable across the coherence break; a surface tracker degrades |
| **C — Decoy Citation** | Substitute either a wording-similar reference that does not teach the mechanism, or a wording-dissimilar one that does | Separates lexical from mechanistic engagement with the cited art |
| **D — Boilerplate Injection** | Prepend canonical patent boilerplate to the drafting context | A grounded model leaves C5 and C3 largely unchanged; a prior-injection-prone model inflates C5 |
| **E — Drafting Hint** | Add one of four hints: "make minimal amendment", "preserve scope", "focus only on novelty", "avoid unnecessary narrowing" | Stable instruction-following produces shifts on C2 and C3 in the predicted directions |

## Retrieval probes (F, G)

These intervene on the retrieval channel as a random-versus-structural policy contrast under a fixed insertion template.

- **F — Random Retrieval.** Retrieve k past amendments from the pool (excluding the test case and its ancestors), selected uniformly at random, and prepend them. Random retrieval realizes prior injection in its strongest form: any effect cannot be attributed to similarity-driven grounding, because selection is similarity-agnostic by construction.
- **G — Structural-Match Retrieval.** Retrieve k past amendments matched on (i) statute section, (ii) statute subsection, and (iii) a coarse limitation-pattern feature derived from the attacked claim. The deterministic matcher isolates the structural-similarity signal, so any shift attributable to the match itself surfaces in the F-versus-G delta.

Both use the same k (default 3) and the same insertion template. G uses a deterministic feature-based matcher rather than a dense retriever, to keep the contrast interpretable as a structural-similarity test.

## Reading the contrast

F, G, and the no-retrieval baseline all sit inside the same prompt scaffold, so F-versus-G identifies the effect of switching retrieval policy under a fixed scaffold — not the unconditional effect of adding retrieval. The empty prior-injector cell reported in the paper is a finding about the deterministic structural-similarity policy that Probe G realizes, not about retrieval-augmented generation in general. Dense-retriever configurations are a different point in the same policy-contrast design and can be evaluated with this framework without modification; the paper reports one such dense-G run.
