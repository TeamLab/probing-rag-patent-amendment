"""C1 embedding reanalysis — drop-in similarity-function swap on the frozen C1 definition.

Motivation: metric_decisions.md documents C1's frozen primary op as trigram-set
Jaccard(modified_n, rejected_n), with a known limitation (not specific to C1's own
section, but the sister limitation documented for C2/C3): exact-token overlap is
paraphrase-insensitive. A model that rewords examiner-cited language with synonyms
registers zero token overlap despite being semantically grounded. This script keeps
the FROZEN modified_n / rejected_n set definitions (same code path, same trigram
extraction, same claim alignment, same rejected-cache) and swaps ONLY the final
similarity function:

    trigram version:   c1_n = Jaccard(modified_n, rejected_n)                 [exact set overlap]
    embedding version: c1_embedding_n = cosine(embed(bag(modified_n)), embed(bag(rejected_n)))

where bag(S) = " ".join(sorted(S)) turns a trigram set into a pseudo-sentence fed to
a sentence embedding model (all-MiniLM-L6-v2, local, no LLM/API calls — consistent
with "no LLM in eval loop" §5.6 in spirit, though note this project's pre-registration
explicitly reserved embedding metrics as *secondary/validity-check only*, never primary;
this script produces a secondary/validity-check artifact, not a replacement primary).

Case-level c1_embedding = mean over the SAME claims (n_claims) that qualified in the
original trigram scoring, for an apples-to-apples concordance comparison.

Inputs (read-only):
  - raw responses: outputs_main/*.json (extracted from Desktop Cleaned.zip, outputs/main/)
  - rejected cache: c1_baseline.json (frozen, from experiment_run/data/parsed)
  - beta_parsed dir: submission_repo/data/parsed/beta_parsed (pre claim text)
  - parse_response.py: experiment_run/scripts (claim parser)

Output: c1_embedding_scores.tsv
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "src" / "parsing"
sys.path.insert(0, str(SCRIPTS))
from parse_response import parse as parse_response_text  # noqa: E402

RESPONSES_DIR = REPO_ROOT / "results" / "raw"
BASELINE_CACHE = REPO_ROOT / "data" / "parsed" / "c1_baseline.json"
BETA_PARSED_DIR = REPO_ROOT / "data" / "parsed" / "beta_parsed"
OUT_TSV = REPO_ROOT / "results" / "analysis" / "c1_embedding_scores.tsv"

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

# Pre-embed rejected bags once per (case, claim) — these don't depend on model/cond/rep.
print("Embedding rejected-span bags (cached per case/claim)...")
rejected_bag_texts = []
rejected_bag_keys = []  # (case_id, claim_num)
for case_id, info in base_cases.items():
    for num_key, cinfo in info["rejected"].items():
        rejected = frozenset(cinfo["rejected"])
        rejected_bag_keys.append((case_id, num_key))
        rejected_bag_texts.append(_bag(rejected))

rejected_embeds = model.encode(rejected_bag_texts, batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True)
rejected_embed_map = {k: v for k, v in zip(rejected_bag_keys, rejected_embeds)}

# Load beta files once per app_num (cached).
beta_cache = {}


def get_beta(app_num):
    if app_num not in beta_cache:
        bp = BETA_PARSED_DIR / f"{app_num}.json"
        beta_cache[app_num] = json.loads(bp.read_text()) if bp.exists() else None
    return beta_cache[app_num]


print("Scanning response files & building modified-span bags...")
response_files = sorted(RESPONSES_DIR.glob("*.json"))
_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
if _limit:
    response_files = response_files[:_limit]
print(f"  {len(response_files)} response files")

pending_rows = []  # dicts with everything except c1_embedding, plus modified_bag text
modified_bag_texts = []
modified_bag_index = []  # index into pending_rows, correlates 1:1 with modified_bag_texts entries appended

for i, f in enumerate(response_files):
    d = json.loads(f.read_text())
    case_id = d.get("case_id")
    cond = d.get("condition", "baseline")
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
    if (i + 1) % 1000 == 0:
        print(f"  parsed {i + 1}/{len(response_files)}")

print(f"Embedding {len(modified_bag_texts)} modified-span bags "
      f"(this is the expensive step)...")
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
