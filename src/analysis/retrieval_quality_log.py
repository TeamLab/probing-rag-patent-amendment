"""Retrieval-quality log: random-F vs structural-G vs dense-G (H), compared
on a common footing.

For each condition, for each cohort case, look up its actual retrieved
app_nums (from the rendered prompt files' probe_meta — F/G already have
these; H was just rendered by build_probe_H_denseG.py) and compute, against
the SAME query used for H:

  - mean query-exemplar cosine similarity, using the dense-G embedding space
    (all-MiniLM-L6-v2) applied post-hoc even to F/G's picks — so cosine sim
    is comparable across all three conditions, not just native to H.
  - structural tag overlap: the same 0-3 score Probe G's own selection rule
    uses (statute match + subsection match + Jaccard(limitation_patterns)),
    applied post-hoc to F and H's picks too — so "how structurally relevant"
    is comparable across all three conditions, not just native to G.

Inputs (read-only): prompts_main/<case>__{F,G}.json, prompts_denseG/<case>__H.json
and its _query_log.json, outputs/_analysis/dense_g_index/{pool_embeddings.npy,pool_meta.json},
data/parsed/retrieval_pool.json.

Output: outputs/_analysis/retrieval_quality_log.tsv (per case x condition x pick)
and a printed summary table.
"""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent  # experiment_run/
POOL_PATH = BASE / "data" / "parsed" / "retrieval_pool.json"
INDEX_DIR = BASE / "outputs" / "_analysis" / "dense_g_index"
PROMPTS_MAIN = BASE / "data" / "parsed" / "prompts_main"
PROMPTS_DENSE_G = BASE / "data" / "parsed" / "prompts_denseG"
OUT_TSV = BASE / "outputs" / "_analysis" / "retrieval_quality_log.tsv"

CONDITIONS = ["F", "G", "H"]


def _g_score(entry_statute, entry_subsection, entry_patterns, test_statute, test_subsection, test_patterns):
    s = 0.0
    if test_statute and entry_statute == test_statute:
        s += 1.0
    if test_subsection and entry_subsection == test_subsection:
        s += 1.0
    tp, ep = set(test_patterns), set(entry_patterns or [])
    if tp or ep:
        inter = len(tp & ep)
        union = len(tp | ep)
        s += inter / union if union else 0.0
    return s


pool = json.loads(POOL_PATH.read_text())
pool_by_case = {e["case_id"]: e for e in pool["entries"]}

pool_embeds = np.load(INDEX_DIR / "pool_embeddings.npy")
pool_meta = json.loads((INDEX_DIR / "pool_meta.json").read_text())["entries_order"]
case_to_row = {m["case_id"]: i for i, m in enumerate(pool_meta)}

query_log = json.loads((PROMPTS_DENSE_G / "_query_log.json").read_text())

rows = []
for case_id, qinfo in query_log.items():
    qvec = np.array(qinfo["query_embedding"], dtype=np.float32)
    test_statute = qinfo["test_statute"]
    test_subsection = qinfo["test_subsection"]
    test_patterns = qinfo["test_patterns"]

    for cond in CONDITIONS:
        if cond == "H":
            fp = PROMPTS_DENSE_G / f"{case_id}__H.json"
        else:
            fp = PROMPTS_MAIN / f"{case_id}__{cond}.json"
        if not fp.exists():
            continue
        meta = json.loads(fp.read_text())["probe_meta"]
        retrieved_case_ids = meta["retrieved_case_ids"]
        for rank, rcid in enumerate(retrieved_case_ids):
            row_i = case_to_row.get(rcid)
            cos = float(np.dot(qvec, pool_embeds[row_i])) if row_i is not None else None
            entry = pool_by_case.get(rcid)
            if entry is not None:
                gscore = _g_score(entry["statute"], entry["subsection"], entry["limitation_patterns"],
                                   test_statute, test_subsection, test_patterns)
            else:
                gscore = None
            rows.append(dict(case=case_id, cond=cond, rank=rank, retrieved_case=rcid,
                              cosine=cos, structural_score=gscore))

cols = ["case", "cond", "rank", "retrieved_case", "cosine", "structural_score"]
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_TSV.open("w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
print(f"wrote {len(rows)} rows -> {OUT_TSV}")

print(f"\n{'Condition':<12}{'n picks':<10}{'mean cosine':<14}{'mean structural (0-3)':<24}{'statute-match rate':<20}")
for cond in CONDITIONS:
    crows = [r for r in rows if r["cond"] == cond]
    cos_vals = [r["cosine"] for r in crows if r["cosine"] is not None]
    struct_vals = [r["structural_score"] for r in crows if r["structural_score"] is not None]
    # statute-match rate needs the raw match, recompute quickly
    n_statute_match = 0
    n_scored = 0
    for r in crows:
        entry = pool_by_case.get(r["retrieved_case"])
        qinfo = query_log[r["case"]]
        if entry is None:
            continue
        n_scored += 1
        if qinfo["test_statute"] and entry["statute"] == qinfo["test_statute"]:
            n_statute_match += 1
    label = {"F": "random-F", "G": "structural-G", "H": "dense-G"}[cond]
    mean_cos = sum(cos_vals) / len(cos_vals) if cos_vals else float("nan")
    mean_struct = sum(struct_vals) / len(struct_vals) if struct_vals else float("nan")
    statute_rate = n_statute_match / n_scored if n_scored else float("nan")
    print(f"{label:<12}{len(crows):<10}{mean_cos:<14.4f}{mean_struct:<24.4f}{statute_rate:<20.4f}")
