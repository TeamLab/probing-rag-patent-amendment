"""Probe H ("dense-G") — dense-retriever realization of Probe G.

Controlled swap of the retrieval MECHANISM only, holding everything else in
Probe G's design fixed: same pool (data/parsed/retrieval_pool.json, v1, 4,221
entries, pool ∩ cohort = ∅ by construction), same k=3, same exemplar fields
injected (app_num/pre_text/post_text via the identical `_format_retrieved_block`
from build_probe_prompts.py), same injection point (before "## Task"), same
base prompt (reused byte-identical from data/parsed/prompts_main/<case>__baseline.json,
so the base prompt has zero drift from what F/G/baseline used).

What changes vs Probe G: selection is by embedding-cosine similarity instead
of the deterministic (statute, subsection, limitation-pattern-Jaccard) score.
Note this is a *deliberate departure* from the frozen pre-registration
(probe_decisions.md:658-660 explicitly rejected embedding-based retrieval for
G "so the comparison against F is interpretable") — dense-G is a new,
additional condition for testing whether retrieval-mechanism choice itself
matters, not a replacement for G.

Query/candidate representation: embed (rejection_text + pre_text) per
candidate, using the SAME embedding model as the C1 semantic reanalysis
(sentence-transformers/all-MiniLM-L6-v2), for methodological consistency
across this session's embedding-based work.

"rejection_text" extraction: the retrieval pool schema (build_retrieval_pool.py)
stores only statute/subsection/limitation_patterns (extracted features), not
the underlying rejection *text* — CTNF full_text is 15-45K chars of largely
boilerplate office-action text (see metric_decisions.md's own C1 limitation:
"CTNF full_text includes non-rejection boilerplate"), unsuitable to embed
wholesale with a 256-token-window model. This script extracts the paragraph(s)
of ctnf.full_text that (a) mention the attacked claim number and (b) contain a
rejection-verb keyword, as a pragmatic proxy for "the rejection rationale
paragraph". Applied identically to pool candidates and to queries.

Dedup policy (a gap in the existing F/G builders, per this session's code
review — neither dedupes on app_num across its 3 picks): candidates are
walked in ranked order and skipped if their app_num was already selected for
this case, or if case_id somehow equals the query case_id (defensive; the
pool/cohort disjointness invariant already guarantees this can't happen).

Two-stage pipeline:
  Stage 1 — build pool embedding index (expensive, run once):
      python3 scripts/build_probe_H_denseG.py index \\
          --pool data/parsed/retrieval_pool.json \\
          --beta-parsed-dir <submission_repo>/data/parsed/beta_parsed \\
          --out-dir outputs/_analysis/dense_g_index

  Stage 2 — render Probe H prompts for the cohort:
      python3 scripts/build_probe_H_denseG.py render \\
          --cohort data/parsed/cohort_batch0.json \\
          --beta-parsed-dir <submission_repo>/data/parsed/beta_parsed \\
          --prompts-main-dir data/parsed/prompts_main \\
          --index-dir outputs/_analysis/dense_g_index \\
          --out-dir data/parsed/prompts_denseG
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from build_probe_prompts import (  # noqa: E402
    _format_retrieved_block,
    _test_case_features,
    _claims_map_beta,
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
H_K = 3

# ---------- rejection-paragraph extraction (shared: pool + query) ---------

REJECTION_VERB_RE = re.compile(
    r"\brejected\b|\bunpatentable\b|\banticipated\b|\bobvious(ness)?\b",
    re.IGNORECASE,
)


def _claim_mention_re(num: int) -> re.Pattern:
    return re.compile(rf"\bclaims?\b[^.\n]{{0,40}}\b{num}\b", re.IGNORECASE)


def extract_rejection_paragraph(full_text: str, attacked_num, max_chars: int = 1500) -> str:
    """Pragmatic proxy for 'the rejection rationale paragraph' for one claim.

    Paragraphs of ctnf.full_text that mention the attacked claim number AND
    a rejection-verb keyword are concatenated (capped at max_chars). Falls
    back to the first max_chars of full_text if nothing matches (rare; still
    far better than embedding a 40K-char blob whose first 256 tokens are
    boilerplate ("DETAILED ACTION...") for every case alike).
    """
    text = full_text or ""
    if not text:
        return ""
    paras = re.split(r"\n\s*\n", text)
    mention_re = _claim_mention_re(attacked_num) if attacked_num is not None else None
    hits = []
    for p in paras:
        if not REJECTION_VERB_RE.search(p):
            continue
        if mention_re is not None and not mention_re.search(p):
            continue
        hits.append(p.strip())
    if not hits:
        # relax: any rejection-verb paragraph (claim-number match failed)
        hits = [p.strip() for p in paras if REJECTION_VERB_RE.search(p)]
    if not hits:
        return text[:max_chars]
    joined = "\n".join(hits)
    return joined[:max_chars]


def _query_text(rejection_text: str, pre_text: str) -> str:
    return f"{rejection_text}\n\n{pre_text}".strip()


# ---------- Stage 1: pool embedding index ----------------------------------

def cmd_index(args) -> None:
    from sentence_transformers import SentenceTransformer

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    entries = pool["entries"]
    print(f"pool: {len(entries)} entries")

    beta_cache: dict = {}

    def get_beta(app_num):
        if app_num not in beta_cache:
            bp = args.beta_parsed_dir / f"{app_num}.json"
            beta_cache[app_num] = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None
        return beta_cache[app_num]

    texts = []
    meta = []
    for i, e in enumerate(entries):
        beta = get_beta(e["app_num"])
        full_text = ((beta or {}).get("ctnf") or {}).get("full_text") or ""
        rej = extract_rejection_paragraph(full_text, e.get("attacked_claim_num"))
        texts.append(_query_text(rej, e["pre_text"]))
        meta.append({"case_id": e["case_id"], "app_num": e["app_num"]})
        if (i + 1) % 500 == 0:
            print(f"  built text {i + 1}/{len(entries)}")

    print(f"Loading {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} pool candidates ...")
    embeds = model.encode(texts, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "pool_embeddings.npy", embeds.astype(np.float32))
    (args.out_dir / "pool_meta.json").write_text(
        json.dumps({"pool_version": pool.get("pool_version", "v1"),
                    "embedding_model": EMBEDDING_MODEL_NAME,
                    "n_entries": len(entries), "entries_order": meta},
                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {embeds.shape} -> {args.out_dir}/pool_embeddings.npy")


# ---------- Stage 2: render Probe H prompts ---------------------------------

def cmd_render(args) -> None:
    from sentence_transformers import SentenceTransformer

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    entries = pool["entries"]
    pool_embeds = np.load(args.index_dir / "pool_embeddings.npy")
    pool_meta = json.loads((args.index_dir / "pool_meta.json").read_text())["entries_order"]
    assert len(pool_meta) == len(entries) == pool_embeds.shape[0]

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    beta_cache: dict = {}

    def get_beta(app_num):
        if app_num not in beta_cache:
            bp = args.beta_parsed_dir / f"{app_num}.json"
            beta_cache[app_num] = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else None
        return beta_cache[app_num]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    query_log = {}  # case_id -> {"query_embedding": [...], "test_feats": {...}}
    n_rendered = 0
    n_missing = 0

    for c in cases:
        case_id = c["case_id"]
        app_num = c["app_num"]
        beta_path = args.beta_parsed_dir / f"{app_num}.json"
        base_path = args.prompts_main_dir / f"{case_id}__baseline.json"
        if not beta_path.exists() or not base_path.exists():
            n_missing += 1
            continue
        beta_rec = json.loads(beta_path.read_text(encoding="utf-8"))
        base_prompt = json.loads(base_path.read_text(encoding="utf-8"))

        test_feats = _test_case_features(beta_rec)
        attacked = test_feats.get("attacked_num")
        pre = _claims_map_beta((beta_rec.get("pre_clm") or {}).get("claims"))
        pre_text = pre.get(attacked, "") if attacked is not None else ""
        full_text = ((beta_rec.get("ctnf") or {}).get("full_text")) or ""
        rej_text = extract_rejection_paragraph(full_text, attacked)
        qtext = _query_text(rej_text, pre_text)
        qvec = model.encode([qtext], normalize_embeddings=True)[0]

        sims = pool_embeds @ qvec  # cosine (both normalized)
        ranked_idx = np.argsort(-sims)

        picks = []
        pick_scores = []
        seen_app_nums = set()
        for idx in ranked_idx:
            m = pool_meta[idx]
            if m["case_id"] == case_id:
                continue  # defensive; pool/cohort are disjoint by construction
            if m["app_num"] in seen_app_nums:
                continue  # dedup on app_num across the 3 picks
            picks.append(entries[idx])
            pick_scores.append(round(float(sims[idx]), 4))
            seen_app_nums.add(m["app_num"])
            if len(picks) == H_K:
                break

        block = _format_retrieved_block(picks)
        marker = "## Task"
        user = base_prompt["user"]
        mi = user.find(marker)
        if mi == -1:
            raise ValueError(f"{case_id}: base prompt missing '## Task' marker")
        new_user = user[:mi] + block + "\n" + user[mi:]

        out = dict(base_prompt)
        out["user"] = new_user
        out["user_chars"] = len(new_user)
        out["condition"] = "H"
        out["probe_meta"] = {
            "mode": "dense_retrieval",
            "label": "dense-G (controlled retrieval-mechanism swap of Probe G)",
            "k": H_K,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "query_text_fields": ["rejection_paragraph_extract", "pre_text(attacked_claim)"],
            "test_statute": test_feats["statute"],
            "test_subsection": test_feats["subsection"],
            "test_patterns": sorted(test_feats["patterns"]),
            "retrieved_case_ids": [p["case_id"] for p in picks],
            "retrieved_app_nums": [p["app_num"] for p in picks],
            "retrieved_scores": pick_scores,  # cosine similarity, not G's 0-3 structural score
            "retrieved_chars_total": sum(len(p["pre_text"]) + len(p["post_text"]) for p in picks),
            "pool_version": pool.get("pool_version", "v1"),
            "dedup_policy": "exclude_self_case_id; exclude_duplicate_app_num_within_pick",
        }
        out_path = args.out_dir / f"{case_id}__H.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        n_rendered += 1

        query_log[case_id] = {
            "query_embedding": qvec.astype(np.float32).tolist(),
            "test_statute": test_feats["statute"],
            "test_subsection": test_feats["subsection"],
            "test_patterns": sorted(test_feats["patterns"]),
        }

    (args.out_dir / "_query_log.json").write_text(
        json.dumps(query_log, ensure_ascii=False), encoding="utf-8")
    print(f"\nRendered {n_rendered} Probe H prompts ({n_missing} missing β/baseline) -> {args.out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    api = sub.add_parser("index")
    api.add_argument("--pool", required=True, type=Path)
    api.add_argument("--beta-parsed-dir", required=True, type=Path)
    api.add_argument("--out-dir", required=True, type=Path)
    api.set_defaults(fn=cmd_index)

    apr = sub.add_parser("render")
    apr.add_argument("--cohort", required=True, type=Path)
    apr.add_argument("--beta-parsed-dir", required=True, type=Path)
    apr.add_argument("--prompts-main-dir", required=True, type=Path)
    apr.add_argument("--index-dir", required=True, type=Path)
    apr.add_argument("--out-dir", required=True, type=Path)
    apr.add_argument("--pool", required=True, type=Path)
    apr.set_defaults(fn=cmd_render)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
