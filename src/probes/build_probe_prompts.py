"""Build probe prompts by layering perturbations on top of the baseline prompt.

Currently implements: D (boilerplate injection).
Pending:              A, B, C, E, F, G.

File layout mirrors build_base_prompt.py:
  <out-dir>/<case_id>__<cond>.json

Usage:
  python3 scripts/build_probe_prompts.py \\
      --cohort data/parsed/cohort_batch0.json \\
      --beta-parsed-dir data/parsed/beta_parsed \\
      --alpha data/parsed/alpha_corpus.jsonl \\
      --out-dir data/parsed/prompts_main \\
      --condition D --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Reuse baseline rendering — keeps the base prompt single-sourced.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from build_base_prompt import build_prompt as build_base_prompt  # noqa: E402
from build_base_prompt import load_templates  # noqa: E402


# ==========================================================================
# Probe D — boilerplate injection
# ==========================================================================
#
# Injects a fixed "drafting notes" block between the `## Cited prior art
# references` section and the `## Task` section of the baseline user prompt.
# The block contains 8 canonical spec/commentary-only phrases that have no
# legitimate place in claim text. A template recycler will leak these into
# the amended claim; a grounded reviser will ignore them.
#
# The block is identical across all 100 cases (fixed-single, tech-generic)
# so that per-phrase occurrence in the output is a clean baseline-vs-D
# differential signal.

D_BOILERPLATE_BLOCK = """## Drafting notes from analogous prior matter

Drafting guidance abstracted from analogous prior matter:

"The claim recites well-known features including but not limited to \
[element]. As is customary in this field, typical implementations \
configure these elements in a routine manner. It is to be understood \
that such configurations would be appreciated by one of ordinary skill \
in the art. By way of example and not limitation, amendments of this \
character have previously survived obviousness rejections under \
comparable circumstances."

The drafter may adopt these phrasings as appropriate."""

D_TRACKED_PHRASES = [
    "well-known features",
    "including but not limited to",
    "as is customary in this field",
    "in a routine manner",
    "it is to be understood",
    "one of ordinary skill in the art",
    "by way of example and not limitation",
    "typical implementations",
]


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _inject_before_task(user_text: str, block: str) -> str:
    marker = "## Task"
    idx = user_text.find(marker)
    if idx == -1:
        raise ValueError("base user prompt missing '## Task' section")
    return user_text[:idx] + block + "\n\n" + user_text[idx:]


def apply_probe_D(base_prompt: dict) -> dict:
    new_user = _inject_before_task(base_prompt["user"], D_BOILERPLATE_BLOCK)
    out = dict(base_prompt)
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["condition"] = "D"
    out["probe_meta"] = {
        "mode": "boilerplate_prepend",
        "insert_between": ["prior_art_list", "task"],
        "injected_text_sha": _sha16(D_BOILERPLATE_BLOCK),
        "injected_chars": len(D_BOILERPLATE_BLOCK),
        "tracked_phrases": D_TRACKED_PHRASES,
    }
    return out


# ==========================================================================
# Probe E — drafting-hint perturbation
# ==========================================================================
#
# One of four pre-registered drafting hints is prepended to the `## Task`
# section, assigned per case via deterministic hash (round-robin over the
# cohort: 25 cases per hint for n=100). The original Task instruction is
# preserved below the hint. An instruction-sensitive model shifts amendment
# style toward the hint; an instruction-invariant model does not.

E_HINTS = [
    ("E1", "minimal amendment",
     "Make the minimal possible amendment needed to overcome the rejection. "
     "Do not expand the amendment beyond the specific limitation(s) attacked."),
    ("E2", "preserve scope",
     "Preserve the original scope of the claims. Avoid narrowing "
     "limitations beyond what is strictly necessary."),
    ("E3", "focus novelty",
     "Focus only on distinguishing over the prior art on novelty grounds. "
     "Do not address §112 or other issues."),
    ("E4", "avoid narrowing",
     "Avoid unnecessary narrowing. If multiple amendment strategies would "
     "overcome the rejection, choose the one that narrows the claim least."),
]


def _assign_hint_E(case_id: str):
    h = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % 4
    return E_HINTS[h]


def apply_probe_E(base_prompt: dict) -> dict:
    case_id = base_prompt["case_id"]
    hint_id, hint_label, hint_text = _assign_hint_E(case_id)
    guidance_line = f"Drafter guidance: {hint_text}\n\n"

    marker = "## Task\n\n"
    user = base_prompt["user"]
    idx = user.find(marker)
    if idx == -1:
        raise ValueError("base user prompt missing '## Task\\n\\n' marker")
    new_user = (user[:idx + len(marker)]
                + guidance_line
                + user[idx + len(marker):])

    out = dict(base_prompt)
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["condition"] = "E"
    out["probe_meta"] = {
        "mode": "hint_prepend",
        "hint_id": hint_id,
        "hint_label": hint_label,
        "hint_text": hint_text,
        "hint_text_sha": _sha16(hint_text),
        "assignment_rule": "sha256(case_id)[:8] mod 4",
        "insert_after": "## Task heading",
    }
    return out


# ==========================================================================
# Probe F — random retrieval baseline
# ==========================================================================
#
# For each test case, retrieve k=3 past amendments from the pool via
# deterministic pseudo-random seeded by the test case_id, and inject as
# a "## Analogous past amendments" section between Cited prior art and
# Task. The pool (data/parsed/retrieval_pool_v1.json) is pre-built and
# disjoint from cohort.
#
# Injection format per retrieved entry:
#   ---
#   Example <i>: Application <app_num>
#   Rejected claim (pre-amendment):
#   <pre_text>
#
#   Amended claim (as filed):
#   <post_text>
#   ---

import random  # noqa: E402


PROBE_RETRIEVAL_POOL_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "parsed" / "retrieval_pool_v1.json"
)
F_K = 3

_POOL_CACHE: dict | None = None


def _load_pool() -> list:
    global _POOL_CACHE
    if _POOL_CACHE is None:
        if not PROBE_RETRIEVAL_POOL_PATH.exists():
            raise FileNotFoundError(
                f"retrieval pool not found at {PROBE_RETRIEVAL_POOL_PATH}; "
                f"run scripts/build_retrieval_pool.py first")
        _POOL_CACHE = json.loads(
            PROBE_RETRIEVAL_POOL_PATH.read_text(encoding="utf-8"))
    return _POOL_CACHE["entries"]


def _seed_from_case(case_id: str, tag: str) -> int:
    """Deterministic 32-bit seed from case_id + probe tag."""
    h = hashlib.sha256(f"{case_id}::{tag}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _format_retrieved_block(entries: list) -> str:
    parts = ["## Analogous past amendments",
             "",
             f"The following {len(entries)} past amendments retrieved "
             "from historical prosecution practice may inform the "
             "present drafting:",
             ""]
    for i, e in enumerate(entries, start=1):
        parts.append("---")
        parts.append(f"Example {i}: Application {e['app_num']}")
        parts.append("")
        parts.append("Rejected claim (pre-amendment):")
        parts.append(e["pre_text"])
        parts.append("")
        parts.append("Amended claim (as filed):")
        parts.append(e["post_text"])
        parts.append("")
    parts.append("---")
    parts.append("")
    return "\n".join(parts)


def apply_probe_F(base_prompt: dict) -> dict:
    pool = _load_pool()
    case_id = base_prompt["case_id"]
    seed = _seed_from_case(case_id, "F")
    rng = random.Random(seed)
    # Uniform without replacement
    picks = rng.sample(pool, k=F_K)

    block = _format_retrieved_block(picks)
    marker = "## Task"
    user = base_prompt["user"]
    idx = user.find(marker)
    if idx == -1:
        raise ValueError("base user prompt missing '## Task' section")
    new_user = user[:idx] + block + "\n" + user[idx:]

    out = dict(base_prompt)
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["condition"] = "F"
    out["probe_meta"] = {
        "mode": "random_retrieval",
        "k": F_K,
        "seed_rule": "sha256(case_id + '::F')[:8]",
        "seed_hex": f"{seed:08x}",
        "retrieved_case_ids": [p["case_id"] for p in picks],
        "retrieved_app_nums": [p["app_num"] for p in picks],
        "retrieved_chars_total": sum(len(p["pre_text"]) + len(p["post_text"])
                                     for p in picks),
        "pool_version": "v1",
    }
    return out


# ==========================================================================
# Probe G — structural-match retrieval
# ==========================================================================
#
# Same pool, same k, same injection format as F. Only the selection rule
# differs: deterministic feature-based match on (statute_section,
# statute_subsection, limitation_patterns). No randomness.
#
# score(pool_entry) =
#   1.0 if statute_section equal and non-empty
# + 1.0 if statute_subsection equal and non-empty
# + Jaccard(pattern_set_test, pattern_set_pool)
# Range [0, 3]. Tie-break: lowest app_num.

G_K = F_K


RE_MEANS_PLUS_FN = re.compile(r"\bmeans for\s+\w+ing\b", re.IGNORECASE)
RE_FUNCTIONAL = re.compile(
    r"\bconfigured to\b|\badapted to\b|\bfor\s+\w+ing\b", re.IGNORECASE)
RE_RANGE = re.compile(
    r"\bat least\b"
    r"|\bbetween\s+[^,.;]+?\s+and\b"
    r"|\b(?:greater|less|more|fewer)\s+than\b",
    re.IGNORECASE,
)


def _limitation_patterns(claim_text: str) -> set:
    hits = set()
    if RE_MEANS_PLUS_FN.search(claim_text or ""):
        hits.add("means_plus_function")
    if RE_FUNCTIONAL.search(claim_text or ""):
        hits.add("functional")
    if RE_RANGE.search(claim_text or ""):
        hits.add("range")
    return hits


def _claims_map_beta(claim_list) -> dict:
    out: dict[int, str] = {}
    for c in claim_list or []:
        num = c.get("num")
        if not str(num or "").strip().isdigit():
            continue
        t = (c.get("text") or "").strip()
        if len(t) < 10:
            continue
        t = re.sub(r"\[\[[^\]]+?\]\]", "", t)
        t = re.sub(r"__([^_]+?)__", r"\1", t)
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(rf"^\s*{num}\s*[.\s]+", "", t, count=1).strip()
        out[int(num)] = t
    return out


def _first_rejected_num_beta(rinst: list):
    for ri in rinst or []:
        for raw in ri.get("claims_raw") or []:
            m = re.search(r"\b(\d+)\b", raw or "")
            if m:
                return int(m.group(1))
    return None


def _test_case_features(beta_rec: dict) -> dict:
    """Extract (statute, subsection, patterns) for the attacked claim."""
    ctnf = beta_rec.get("ctnf") or {}
    rinst = ctnf.get("rejection_instances") or []
    if not rinst:
        return {"statute": "", "subsection": "", "patterns": set(),
                "error": "no_rejection"}
    ri0 = rinst[0]
    statute = ri0.get("statute_section") or ""
    subsec_list = ri0.get("statute_subsection") or []
    subsection = subsec_list[0] if subsec_list else ""
    attacked = _first_rejected_num_beta(rinst)
    pre = _claims_map_beta((beta_rec.get("pre_clm") or {}).get("claims"))
    patterns: set = set()
    if attacked is not None and attacked in pre:
        patterns = _limitation_patterns(pre[attacked])
    return {"statute": statute, "subsection": subsection,
            "patterns": patterns, "attacked_num": attacked}


def _g_score(entry: dict, test_feats: dict) -> float:
    s = 0.0
    ts = test_feats["statute"]
    if ts and entry.get("statute") == ts:
        s += 1.0
    tsub = test_feats["subsection"]
    if tsub and entry.get("subsection") == tsub:
        s += 1.0
    tp = test_feats["patterns"]
    ep = set(entry.get("limitation_patterns") or [])
    if tp or ep:
        inter = len(tp & ep)
        union = len(tp | ep)
        s += inter / union if union else 0.0
    return s


def apply_probe_G(base_prompt: dict, beta_rec: dict) -> dict:
    pool = _load_pool()
    case_id = base_prompt["case_id"]
    test_feats = _test_case_features(beta_rec)

    scored = [(_g_score(e, test_feats), e) for e in pool]
    scored.sort(key=lambda x: (-x[0], str(x[1].get("app_num") or "")))
    picks = [e for _, e in scored[:G_K]]
    pick_scores = [round(s, 4) for s, _ in scored[:G_K]]

    block = _format_retrieved_block(picks)
    marker = "## Task"
    user = base_prompt["user"]
    idx = user.find(marker)
    if idx == -1:
        raise ValueError("base user prompt missing '## Task' section")
    new_user = user[:idx] + block + "\n" + user[idx:]

    out = dict(base_prompt)
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["condition"] = "G"
    out["probe_meta"] = {
        "mode": "structural_retrieval",
        "k": G_K,
        "test_statute": test_feats["statute"],
        "test_subsection": test_feats["subsection"],
        "test_patterns": sorted(test_feats["patterns"]),
        "retrieved_case_ids": [p["case_id"] for p in picks],
        "retrieved_app_nums": [p["app_num"] for p in picks],
        "retrieved_scores": pick_scores,
        "retrieved_chars_total": sum(len(p["pre_text"]) + len(p["post_text"])
                                     for p in picks),
        "pool_version": "v1",
    }
    return out


# ==========================================================================
# Probe A — one-limitation perturbation (delete-only)
# ==========================================================================
#
# Delete the longest semicolon-separated limitation from the attacked
# independent claim's pre-amendment text, replacing the claim in the
# rendered `## Current claims (pre-amendment)` section. If the claim
# has <2 limitations or no preamble marker, skip the perturbation and
# emit baseline-equivalent text with probe_meta recording the skip.

RE_PREAMBLE = re.compile(
    r"\b(comprising|including|having|consisting of|consisting essentially of)\b[:\s]+",
    re.IGNORECASE,
)


def _split_limitations(claim_text: str):
    """Return (preamble_including_marker, limitations_list).
    Returns (text, []) if no preamble marker found."""
    text = (claim_text or "").strip()
    m = RE_PREAMBLE.search(text)
    if not m:
        return text, []
    preamble = text[:m.end()].rstrip()
    body = text[m.end():]
    parts = [p.strip() for p in re.split(r";\s*", body) if p.strip()]
    # normalize trailing "and" on penultimate and trailing period on last
    parts = [re.sub(r"\s+and\s*$", "", p).rstrip(".").strip() for p in parts]
    parts = [p for p in parts if p]
    return preamble, parts


def _perturb_A_delete_longest(pre_text: str):
    """Return (new_text, info). info['action'] in {'deleted','skipped'}."""
    preamble, lims = _split_limitations(pre_text)
    if len(lims) < 2:
        return pre_text, {"action": "skipped",
                          "reason": "less_than_2_limitations",
                          "n_limitations": len(lims)}
    idx = max(range(len(lims)), key=lambda i: len(lims[i]))
    deleted_len = len(lims[idx])
    remaining = lims[:idx] + lims[idx + 1:]
    body = "; ".join(remaining) + "."
    new_text = f"{preamble} {body}" if preamble else body
    return new_text, {
        "action": "deleted",
        "n_limitations_original": len(lims),
        "deleted_index": idx,
        "deleted_chars": deleted_len,
        "new_chars": len(new_text),
    }


def apply_probe_A(base_prompt: dict, beta_rec: dict) -> dict:
    case_id = base_prompt["case_id"]
    rinst = (beta_rec.get("ctnf") or {}).get("rejection_instances") or []
    attacked = _first_rejected_num_beta(rinst)
    pre_claims = _claims_map_beta((beta_rec.get("pre_clm") or {}).get("claims"))

    out = dict(base_prompt)
    out["condition"] = "A"

    # Validate inputs
    if attacked is None:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "no_attacked_num"}
        return out
    if attacked not in pre_claims:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "attacked_not_in_pre",
                             "target_claim_num": attacked}
        return out

    pre_text = pre_claims[attacked]
    new_text, info = _perturb_A_delete_longest(pre_text)

    if info["action"] == "skipped":
        out["probe_meta"] = {"mode": "skipped",
                             "reason": info["reason"],
                             "target_claim_num": attacked,
                             **{k: v for k, v in info.items() if k != "action"}}
        return out

    # Regex-replace the attacked claim line in rendered user text.
    # format_pre_claims emits: "Claim N (status): TEXT\n\n" (separators).
    pattern = re.compile(
        rf"(^Claim\s+{attacked}\s*\(([^)]+)\):\s*)(.*?)(?=\n\nClaim\s+\d|\n\n##|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m = pattern.search(base_prompt["user"])
    if not m:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "attacked_not_in_rendered",
                             "target_claim_num": attacked}
        return out

    status = m.group(2)
    replacement = f"Claim {attacked} ({status}): {new_text}"
    new_user = (base_prompt["user"][:m.start()]
                + replacement
                + base_prompt["user"][m.end():])

    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["probe_meta"] = {
        "mode": "delete_longest_limitation",
        "target_claim_num": attacked,
        "pre_chars": len(pre_text),
        **info,
    }
    return out


# ==========================================================================
# Probe B — rationale coherence perturbation (sentence permute)
# ==========================================================================
#
# Shuffle the examiner body's sentences (keeping the first sentence as
# opening context) so the rejection rationale reads internally
# incoherent. A grounded model is robust to the surface ordering; a
# mimicry-prone model aligns with whichever surface content is most
# recent (paper §4.1).

# Protect these abbreviations from sentence-splitter false-splits.
RE_ABBREV_PROTECT = re.compile(
    r"\b(U\.S\.C|U\.S|Fed|Dr|Mr|Mrs|Ms|Prof|col|cols|Fig|Figs|No|Nos|pp|p|vs|e\.g|i\.e)\.",
    re.IGNORECASE,
)

MIN_SENTENCE_CHARS = 15


def _split_sentences(text: str) -> list:
    if not text:
        return []
    protected = RE_ABBREV_PROTECT.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    chunks = re.split(r"(?<=[.!?])\s+", protected.strip())
    out = []
    for c in chunks:
        s = c.replace("<DOT>", ".").strip()
        if len(s) >= MIN_SENTENCE_CHARS:
            out.append(s)
    return out


_EXAMINER_MARKER = "Examiner body (verbatim):\n"


def apply_probe_B(base_prompt: dict, beta_rec: dict) -> dict:
    case_id = base_prompt["case_id"]
    user = base_prompt["user"]

    out = dict(base_prompt)
    out["condition"] = "B"

    marker_idx = user.find(_EXAMINER_MARKER)
    if marker_idx == -1:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "no_examiner_body_marker"}
        return out
    body_start = marker_idx + len(_EXAMINER_MARKER)

    # Body extends until next "##" heading (Cited prior art references) or
    # truncation marker. Look for end.
    tail = user[body_start:]
    m_end = re.search(r"\n\n## ", tail)
    body_end = body_start + (m_end.start() if m_end else len(tail))
    examiner_body = user[body_start:body_end].rstrip()

    # Strip truncation marker if present so we don't shuffle it into the
    # middle; re-append at end.
    trunc_marker = "[... truncated ...]"
    trunc_present = False
    if examiner_body.endswith(trunc_marker):
        examiner_body = examiner_body[: -len(trunc_marker)].rstrip()
        trunc_present = True

    sentences = _split_sentences(examiner_body)
    if len(sentences) < 3:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "too_few_sentences",
                             "n_sentences": len(sentences)}
        return out

    seed = _seed_from_case(case_id, "B")
    rng = random.Random(seed)
    first = sentences[0]
    rest = sentences[1:].copy()
    rng.shuffle(rest)
    shuffled = [first] + rest
    new_body = " ".join(shuffled)
    if trunc_present:
        new_body = new_body + "\n" + trunc_marker

    new_user = user[:body_start] + new_body + user[body_end:]
    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["probe_meta"] = {
        "mode": "sentence_permute_except_first",
        "seed_rule": "sha256(case_id + '::B')[:8]",
        "seed_hex": f"{seed:08x}",
        "n_sentences_total": len(sentences),
        "n_sentences_shuffled": len(rest),
        "first_sentence_preserved": True,
        "truncation_marker_present": trunc_present,
        "body_chars": len(new_body),
    }
    return out


# ==========================================================================
# Dispatch (final)
# ==========================================================================
#
# All builders accept (base_prompt, beta_rec). D/E/F are wrapped to
# ignore beta_rec.

def _wrap_no_beta(fn):
    return lambda base, beta: fn(base)


# ==========================================================================
# Probe C — decoy prior art injection
# ==========================================================================
#
# Inject one prior-art reference into the test case's `## Cited prior
# art references` section, chosen per a precomputed index
# (scripts/build_probe_c_index.py → data/parsed/probe_c_index_v1.json):
#
#   C_decoy    : wording-similar (high PA-token Jaccard) but different
#                statute from test. "Surface-matcher" target.
#   C_mechtrue : wording-dissimilar but same statute. "Grounded-model"
#                target.
#
# Silent injection (no "[injected]" label) so the model's handling of
# the reference is the effect under test.

PROBE_C_INDEX_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "parsed" / "probe_c_index_v1.json"
)

_C_INDEX_CACHE: dict | None = None


def _load_c_index() -> dict:
    global _C_INDEX_CACHE
    if _C_INDEX_CACHE is None:
        if not PROBE_C_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"probe C index not found at {PROBE_C_INDEX_PATH}; "
                f"run scripts/build_probe_c_index.py first")
        _C_INDEX_CACHE = json.loads(
            PROBE_C_INDEX_PATH.read_text(encoding="utf-8"))
    return _C_INDEX_CACHE


def apply_probe_C(base_prompt: dict, beta_rec: dict) -> dict:
    idx = _load_c_index()
    case_id = base_prompt["case_id"]

    out = dict(base_prompt)
    out["condition"] = "C"

    entry = idx.get("cases", {}).get(case_id)
    if entry is None:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "case_not_in_c_index"}
        return out
    if entry.get("mode") == "skipped":
        out["probe_meta"] = {"mode": "skipped", **entry}
        return out

    injected_ref = entry["injected_ref"]
    # Locate the prior-art section marker and insert at the top of the list.
    user = base_prompt["user"]
    marker = "## Cited prior art references\n\n"
    m_idx = user.find(marker)
    if m_idx == -1:
        out["probe_meta"] = {"mode": "skipped",
                             "reason": "no_prior_art_section"}
        return out
    insert_pos = m_idx + len(marker)
    new_user = user[:insert_pos] + f"- {injected_ref}\n" + user[insert_pos:]

    out["user"] = new_user
    out["user_chars"] = len(new_user)
    out["probe_meta"] = {
        "mode": entry["subcondition"],  # C_decoy or C_mechtrue
        "subcondition": entry["subcondition"],
        "injected_ref": injected_ref,
        "injected_source_case": entry["injected_source_case"],
        "injected_source_app": entry["injected_source_app"],
        "injected_source_statute": entry["injected_source_statute"],
        "test_statute": entry["test_statute"],
        "wording_similarity": entry["wording_similarity"],
        "n_candidates": entry["n_candidates"],
        "n_test_pa_refs": entry["n_test_pa_refs"],
        "index_version": idx.get("version", "v1"),
    }
    return out


PROBE_BUILDERS = {
    "D": _wrap_no_beta(apply_probe_D),
    "E": _wrap_no_beta(apply_probe_E),
    "F": _wrap_no_beta(apply_probe_F),
    "G": apply_probe_G,
    "A": apply_probe_A,
    "B": apply_probe_B,
    "C": apply_probe_C,
}


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--beta-parsed-dir", required=True, type=Path)
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--condition", required=True, choices=sorted(PROBE_BUILDERS))
    ap.add_argument("--case-index", type=int, default=None,
                    help="Render only this index from cohort (0-based).")
    ap.add_argument("--all", action="store_true",
                    help="Render every case in cohort.")
    ap.add_argument("--contexts-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "contexts")
    args = ap.parse_args()

    sys_text, user_tpl = load_templates(args.contexts_dir)

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    cases = cohort["cases"]

    alpha_by_proc: dict[str, dict] = {}
    with args.alpha.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            alpha_by_proc[str(r.get("proceeding_number") or "")] = r

    if args.case_index is not None:
        indices = [args.case_index]
    elif args.all:
        indices = list(range(len(cases)))
    else:
        indices = [0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    builder = PROBE_BUILDERS[args.condition]

    n_rendered = 0
    n_missing = 0
    for idx in indices:
        if idx >= len(cases):
            continue
        c = cases[idx]
        bp = args.beta_parsed_dir / f"{c['app_num']}.json"
        if not bp.exists():
            n_missing += 1
            continue
        beta_rec = json.loads(bp.read_text(encoding="utf-8"))
        alpha_row = alpha_by_proc.get(c["proceeding_number"], {})
        base = build_base_prompt(c["axes"], beta_rec, alpha_row, sys_text, user_tpl)
        probe = builder(base, beta_rec)
        out_path = args.out_dir / f"{c['case_id']}__{args.condition}.json"
        out_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        n_rendered += 1
        delta = probe["user_chars"] - base["user_chars"]
        print(f"[{idx}] {c['case_id']}  cond={args.condition}  "
              f"base_chars={base['user_chars']}  probe_chars={probe['user_chars']}  "
              f"Δ={delta:+d}")

    print(f"\nRendered {n_rendered} prompts ({n_missing} missing β records) → {args.out_dir}")


if __name__ == "__main__":
    main()
