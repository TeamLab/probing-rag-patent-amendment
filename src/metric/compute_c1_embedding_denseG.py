"""C1 embedding (semantic) scoring for the dense-G (condition H) responses.

Identical method to compute_c1_embedding.py (same frozen modified_n/rejected_n
definitions, same all-MiniLM-L6-v2 cosine swap) — only the input source
changes: the 4 outputs/<model>_denseG/ directories from the real dense-G
generation run, instead of the recovered main-run responses.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path("/Users/josephamichikoleo/Documents/Claude/Projects/Probing Retrieval-Augmented Patent Claim Amendment")
SCRIPTS = PROJECT / "experiment_run" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from parse_response import parse as parse_response_text  # noqa: E402

RESPONSE_DIRS = [
    PROJECT / "experiment_run" / "outputs" / "sonnet4_denseG",
    PROJECT / "experiment_run" / "outputs" / "haiku4.5_denseG",
    PROJECT / "experiment_run" / "outputs" / "gpt5.4_denseG",
    PROJECT / "experiment_run" / "outputs" / "gpt4o-mini_denseG",
]
BASELINE_CACHE = PROJECT / "experiment_run" / "data" / "parsed" / "c1_baseline.json"
BETA_PARSED_DIR = PROJECT / "submission_repo" / "data" / "parsed" / "beta_parsed"
OUT_TSV = PROJECT / "experiment_run" / "outputs" / "_analysis" / "c1_embedding_denseG_scores.tsv"

WORD_RE = re.compile(r"[a-z][a-z\-']+")


def _trigrams(text: str) -> frozenset:
    if not text:
        return frozenset()
    words = WORD_RE.findall(text.lower())
    if len(words) < 3:
        return frozenset()
    return frozenset(" ".join(words[i:i + 3]) for i in range(len(words) - 2))


def _strip_markup(text: str) -> str:
    text = re.sub(r"\[\[[^\]]+?\]\]", "", text or "")
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _claims_map_from_beta(claim_list) -> dict:
    out = {}
    for c in claim_list or []:
        num = c.get("num")
        if not str(num or "").strip().isdigit():
            continue
        t = (c.get("text") or "").strip()
        if len(t) < 10:
            continue
        t = _strip_markup(t)
        t = re.sub(rf"^\s*{num}\s*[.\s]+", "", t, count=1).strip()
        out[int(num)] = t
    return out


def _claims_map_from_response(parsed_claims) -> dict:
    return {r.num: r.plain_body for r in parsed_claims
            if r.num is not None and r.plain_body}


def _bag(trigram_set) -> str:
    return " ".join(sorted(trigram_set))


print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading rejected cache (frozen baseline)...")
baseline = json.loads(BASELINE_CACHE.read_text())
base_cases = baseline["cases"]

print("Embedding rejected-span bags (cached per case/claim)...")
rejected_bag_texts = []
rejected_bag_keys = []
for case_id, info in base_cases.items():
    for num_key, cinfo in info["rejected"].items():
        rejected = frozenset(cinfo["rejected"])
        rejected_bag_keys.append((case_id, num_key))
        rejected_bag_texts.append(_bag(rejected))

rejected_embeds = model.encode(rejected_bag_texts, batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True)
rejected_embed_map = {k: v for k, v in zip(rejected_bag_keys, rejected_embeds)}

beta_cache = {}


def get_beta(app_num):
    if app_num not in beta_cache:
        bp = BETA_PARSED_DIR / f"{app_num}.json"
        beta_cache[app_num] = json.loads(bp.read_text()) if bp.exists() else None
    return beta_cache[app_num]


print("Scanning response files & building modified-span bags...")
response_files = []
for d in RESPONSE_DIRS:
    response_files.extend(sorted(d.glob("*.json")))
print(f"  {len(response_files)} response files across {len(RESPONSE_DIRS)} model dirs")

pending_rows = []
modified_bag_texts = []
modified_bag_index = []

for i, f in enumerate(response_files):
    d = json.loads(f.read_text())
    case_id = d.get("case_id")
    cond = d.get("condition", "H")
    rep = d.get("rep", 0)
    if case_id not in base_cases:
        pending_rows.append(dict(file=f.name, case=case_id, cond=cond, rep=rep,
                                  reason="no_baseline", per_claim=None))
        continue
    app_num = base_cases[case_id]["app_num"]
    beta = get_beta(app_num)
    if beta is None:
        pending_rows.append(dict(file=f.name, case=case_id, cond=cond, rep=rep,
                                  reason="no_beta", per_claim=None))
        continue
    gen_parsed = parse_response_text(d.get("response_text") or "")
    gen_map = _claims_map_from_response(gen_parsed)
    pre_map = _claims_map_from_beta((beta.get("pre_clm") or {}).get("claims"))
    rej_cache = base_cases[case_id]["rejected"]

    per_claim = []
    for num_key, cinfo in rej_cache.items():
        try:
            num = int(num_key)
        except (ValueError, TypeError):
            continue
        if num not in pre_map or num not in gen_map:
            continue
        rejected = frozenset(cinfo["rejected"])
        if not rejected:
            continue
        pre_grams = frozenset(cinfo["pre_grams"])
        gen_grams = _trigrams(gen_map[num])
        modified = pre_grams.symmetric_difference(gen_grams)
        row_idx = len(per_claim)
        per_claim.append(dict(num=num, num_key=num_key, modified_n=len(modified),
                               rejected_n=len(rejected)))
        modified_bag_texts.append(_bag(modified))
        modified_bag_index.append((len(pending_rows), row_idx, case_id, num_key))

    pending_rows.append(dict(file=f.name, case=case_id, cond=cond, rep=rep,
                              reason="" if per_claim else "no_aligned_claims",
                              per_claim=per_claim))
    if (i + 1) % 500 == 0:
        print(f"  parsed {i + 1}/{len(response_files)}")

print(f"Embedding {len(modified_bag_texts)} modified-span bags...")
modified_embeds = model.encode(modified_bag_texts, batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True)

print("Computing cosine similarities & aggregating case-level c1_embedding...")
for (row_i, claim_i, case_id, num_key), vec in zip(modified_bag_index, modified_embeds):
    rej_vec = rejected_embed_map.get((case_id, num_key))
    cos = float(np.dot(vec, rej_vec)) if rej_vec is not None else None
    pending_rows[row_i]["per_claim"][claim_i]["c1_embedding"] = cos

rows_out = []
for r in pending_rows:
    if not r["per_claim"]:
        rows_out.append(dict(file=r["file"], case=r["case"], cond=r["cond"], rep=r["rep"],
                              c1_embedding="", n_claims=0, reason=r["reason"]))
        continue
    vals = [c["c1_embedding"] for c in r["per_claim"] if c.get("c1_embedding") is not None]
    if not vals:
        rows_out.append(dict(file=r["file"], case=r["case"], cond=r["cond"], rep=r["rep"],
                              c1_embedding="", n_claims=0, reason="no_valid_claims"))
        continue
    rows_out.append(dict(file=r["file"], case=r["case"], cond=r["cond"], rep=r["rep"],
                          c1_embedding=round(sum(vals) / len(vals), 4), n_claims=len(vals),
                          reason=""))

cols = ["file", "case", "cond", "rep", "c1_embedding", "n_claims", "reason"]
OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_TSV.open("w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows_out:
        fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

ok = [r for r in rows_out if r["c1_embedding"] != ""]
print(f"\nwrote {len(rows_out)} rows -> {OUT_TSV}")
print(f"  scored: {len(ok)}  unscored: {len(rows_out) - len(ok)}")
if ok:
    vals = sorted(r["c1_embedding"] for r in ok)
    import statistics as st
    print(f"  c1_embedding range: {vals[0]:.3f} .. {vals[-1]:.3f}  "
          f"median={st.median(vals):.3f}  mean={st.mean(vals):.3f}")
