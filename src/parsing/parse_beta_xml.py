"""
Parse β corpus XML archives into structured per-case JSON.

Handles two CLM XML formats observed in USPTO ODP:
  (1) Legacy DTD  : root <us-patent-application>, claims via <claim><claim-text>
                    status via claim-type attribute (may be absent on pre-filing)
                    deletion markup: [[text]] inline brackets (short); strikethrough
                                     for long deletions is not explicitly tagged
  (2) NS ClaimsDocument : root <pat:ClaimsDocument>, claims via <pat:Claim>
                          status via <pat:ClaimStatusCategory>
                          deletion via <pat:DeletedText>; addition often via
                          <pat:U> or <com:SpanFormat style="underline">
                          inline [[text]] brackets for short deletions

CTNF parsed from <uspat:OutgoingDocument>, extracting:
  - rejection sections by FormParagraphNumber (USC_101/102/103/112 headings)
  - per-rejection tuple: affected claims, statute section, rejection type, prior-art refs
  - examiner's reasoning paragraphs (the DataField-wrapped "GeneralText")

OCR confidence tags are stripped, inner text preserved.

Usage:
  python3 scripts/parse_beta_xml.py \
      --manifest data/parsed/beta_100_manifest.jsonl \
      --beta-root outputs/beta \
      --out-dir  data/parsed/beta_parsed \
      --stats    data/parsed/beta_parsed_stats.json \
      --limit 5   # smoke; 0 = all
"""

import argparse
import json
import re
import sys
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "pat": "urn:us:gov:doc:uspto:patent",
    "uspat": "urn:us:gov:doc:uspto:patent",
    "uscom": "urn:us:gov:doc:uspto:common",
    "com": "http://www.wipo.int/standards/XMLSchema/ST96/Common",
}

# Strip OCR confidence noise (keeps inner text) -----------------------------
# Both legacy <confidence value="N">TEXT</confidence>
# and namespace <pat:OCRConfidenceData ...>TEXT</pat:OCRConfidenceData>
RE_CONF_LEGACY = re.compile(r"<confidence[^>]*>([^<]*)</confidence>")
RE_CONF_NS = re.compile(r"<pat:OCRConfidenceData[^>]*>([^<]*)</pat:OCRConfidenceData>")
RE_BOUNDARY_DATA = re.compile(
    r"<boundary-data[^>]*>.*?</boundary-data>", re.DOTALL
)
RE_PAGE_BREAK = re.compile(r"<\?PageStart[^>]*\?>")
RE_MULTISPACE = re.compile(r"\s+")


def clean_xml_text(raw: str) -> str:
    """Strip OCR confidence wrappers (keep inner text) prior to ET parsing."""
    s = RE_CONF_LEGACY.sub(r"\1", raw)
    s = RE_CONF_NS.sub(r"\1", s)
    return s


def element_text(el: ET.Element) -> str:
    """Concatenate all descendant text, ignoring tags. Preserves [[...]] brackets."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(element_text(child))
        if child.tail:
            parts.append(child.tail)
    out = "".join(parts)
    out = RE_MULTISPACE.sub(" ", out).strip()
    return out


# ---------- CLM DTD legacy ----------

def parse_clm_dtd(root: ET.Element) -> list:
    """Extract claims from legacy <us-patent-application> / <claims>.

    Handles the canonical structure (each <claim num="N"> separate) and also
    the inline-embedded variant where a single <claim num="UNKNOWN"> wraps
    many <claim-text> elements whose leading text is "[Claim N] ..."
    or "N. (Status) ..." — in which case we split into per-num records.
    """
    claims = []
    for claim in root.iter("claim"):
        num = claim.get("num") or ""
        status = claim.get("claim-type") or ""
        ct_texts = [element_text(ct) for ct in claim.findall("claim-text")]
        nonempty = [t for t in ct_texts if t]
        # Skip junk header-only claims.
        if num == "UNKNOWN" and not nonempty:
            continue
        full = " ".join(nonempty).strip()

        # Wrapper case: num=UNKNOWN and multiple ClaimText children with
        # inline headers — split into per-claim records.
        if num in ("", "UNKNOWN") and len(nonempty) > 1:
            headers = [_parse_inline_header(t)[0] for t in nonempty]
            if sum(1 for h in headers if h) >= 2:
                current = None
                for text, hdr in zip(nonempty, headers):
                    if hdr:
                        if current is not None:
                            claims.append(current)
                        num_raw, status_in, _rest = _parse_inline_header(text)
                        nums = _expand_inline_range(num_raw)
                        if not nums:
                            current = None
                            continue
                        if len(nums) > 1:
                            for n in nums:
                                claims.append(_claim_record(
                                    n, status_in or status,
                                    f"(range {num_raw}, {status_in or status}) claim {n}"))
                            current = None
                        else:
                            current = _claim_record(
                                nums[0], status_in or status, text)
                    else:
                        if current is not None:
                            current["text"] = (current["text"] + " " + text).strip()
                if current is not None:
                    claims.append(current)
                continue

        # Single-claim case: if num is UNKNOWN, try inline header on body.
        if num in ("", "UNKNOWN") and full:
            num_raw, status_in, _ = _parse_inline_header(full)
            if num_raw:
                nums = _expand_inline_range(num_raw)
                if len(nums) > 1:
                    for n in nums:
                        claims.append(_claim_record(
                            n, status_in or status,
                            f"(range {num_raw}, {status_in or status}) claim {n}"))
                    continue
                if len(nums) == 1:
                    claims.append(_claim_record(
                        nums[0], status_in or status, full))
                    continue

        if not full:
            continue
        claims.append(_claim_record(num, status or "unlabeled", full))
    return claims


# ---------- CLM NS ClaimsDocument ----------

# Namespace URIs observed in USPTO ODP CLM XMLs.
# In ST96 V2 documents, pat: prefix unexpectedly maps to the ST96 Patent URI
# while sibling tags (Claim, ClaimText, ClaimStatusCategory) stay on the
# legacy USPTO Patent URI. We therefore try both URIs for each element.
USPAT_URI = "{urn:us:gov:doc:uspto:patent}"
ST96_PAT_URI = "{http://www.wipo.int/standards/XMLSchema/ST96/Patent}"
ST96_COM_URI = "{http://www.wipo.int/standards/XMLSchema/ST96/Common}"


def _find_in_ns(parent, local_name,
                uris=(USPAT_URI, ST96_PAT_URI, ST96_COM_URI)):
    """Find a direct child by local name, trying multiple namespace URIs."""
    for uri in uris:
        el = parent.find(uri + local_name)
        if el is not None:
            return el
    return None


def _iter_in_ns(parent, local_name,
                uris=(USPAT_URI, ST96_PAT_URI, ST96_COM_URI)):
    """Iterate descendants by local name across multiple namespace URIs."""
    for uri in uris:
        for el in parent.iter(uri + local_name):
            yield el


# Inline claim header parsed from ClaimText content when the structural
# ClaimNumber is absent or empty. Matches forms we observed in real filings:
#   "1. (Currently Amended) A device..."
#   "1 (Original) A method..."
#   "1-10 (Canceled)"     <- range form (handled by separate expansion)
#   "[Claim 1] A method ..."   <- legacy-DTD bracket prefix form
#   "[Claim 1-10] (Canceled)"
INLINE_HEADER_RE_PAREN = re.compile(
    r"^\s*(\d+(?:\s*-\s*\d+)?)\s*[.\s]\s*\(\s*([^)\n]+?)\s*\)\s*",
    re.DOTALL,
)
INLINE_HEADER_RE_BRACKET = re.compile(
    r"^\s*\[\s*Claim\s+(\d+(?:\s*-\s*\d+)?)\s*\]\s*"
    r"(?:\(\s*([^)\n]+?)\s*\))?\s*",
    re.IGNORECASE | re.DOTALL,
)
# "Claim N (status):" or "Claim N (status)" word prefix (no brackets/dot).
# Allow small leading noise (page-number artifacts like "2 Claim 4 (...)")
INLINE_HEADER_RE_WORD = re.compile(
    r"^\s*(?:\d+\s+)?"
    r"(?:LISTING\s+OF\s+THE\s+CLAIMS[:\s]+)?"
    r"Claim\s+(\d+(?:\s*-\s*\d+)?)\s*"
    r"\(\s*([^)\n]+?)\s*\)\s*:?\s*",
    re.IGNORECASE | re.DOTALL,
)


def _parse_inline_header(text):
    """Return (num_or_range, status, remainder) or (None, None, text) on no match.

    Tries paren form first ("1. (Status) ..."), then bracket form
    ("[Claim 1] ..."). num_or_range preserves the "N" or "N-M" string;
    caller decides whether to expand a range."""
    if not text:
        return None, None, text
    for rx in (INLINE_HEADER_RE_PAREN,
               INLINE_HEADER_RE_BRACKET,
               INLINE_HEADER_RE_WORD):
        m = rx.match(text)
        if m:
            num_raw = m.group(1).replace(" ", "")
            status = (m.group(2) or "").strip() or None
            remainder = text[m.end():].strip()
            return num_raw, status, remainder
    return None, None, text


def _expand_inline_range(num_raw):
    """'1-10' -> [1..10]; '5' -> [5]. Returns empty list on bad input."""
    if not num_raw:
        return []
    if "-" in num_raw:
        try:
            a, b = num_raw.split("-", 1)
            a, b = int(a), int(b)
            if a > b or b - a > 200:
                return []
            return list(range(a, b + 1))
        except ValueError:
            return []
    try:
        return [int(num_raw)]
    except ValueError:
        return []


def _claim_record(num, status, text, deletions=None, additions=None):
    return {
        "num": str(num),
        "status": status or "unlabeled",
        "text": text,
        "explicit_deletions": deletions or [],
        "explicit_additions": additions or [],
        "bracket_deletions": [m.group(0) for m in
                              re.finditer(r"\[\[[^\[\]]{1,200}\]\]", text or "")],
    }


def parse_clm_ns(root: ET.Element) -> list:
    """Extract claims from <...:ClaimsDocument> (both v1.3 and ST96 V2 layouts).

    Handles three structural variants observed in USPTO ODP post_clm:
      (1) Canonical: each <Claim> has <ClaimNumber> + <ClaimText>.
      (2) Inline-wrapped: <Claim> has empty/missing <ClaimNumber>;
          num + status embedded in ClaimText leading text.
      (3) Multi-claim wrapper: a single <Claim> has >1 <ClaimText> siblings
          each of which is itself a separate inline-headered claim.
    Also expands inline ranges like "1-10 (Canceled)" into per-num records.
    """
    claims = []

    for claim in _iter_in_ns(root, "Claim"):
        num_el = _find_in_ns(claim, "ClaimNumber")
        struct_num = (num_el.text or "").strip() if num_el is not None else ""

        status_el = _find_in_ns(claim, "ClaimStatusCategory")
        struct_status = (status_el.text or "").strip() if status_el is not None else "unlabeled"

        # Collect per-ClaimText bodies with their deletion/addition markup.
        ct_records = []  # (text, deletions, additions)
        for ct in _iter_in_ns(claim, "ClaimText"):
            dels = [element_text(dt) for dt in _iter_in_ns(ct, "DeletedText")]
            adds = []
            for tag_name in ("Ins", "U"):
                for ut in _iter_in_ns(ct, tag_name):
                    adds.append(element_text(ut))
            ct_records.append((element_text(ct), dels, adds))

        nonempty_ct = [r for r in ct_records if r[0] or r[1]]
        if not nonempty_ct:
            continue

        # Case 3: wrapper with multiple inline-headered ClaimText children and
        # no structural num. Split each ClaimText into its own claim.
        if not struct_num and len(nonempty_ct) > 1:
            headers = [_parse_inline_header(t)[0] for t, _, _ in nonempty_ct]
            inline_count = sum(1 for h in headers if h)
            if inline_count >= 2:
                # Each inline-headed ClaimText becomes its own claim; any
                # following header-less ClaimText is a continuation appended
                # to the previous claim.
                current = None
                for (text, dels, adds), hdr in zip(nonempty_ct, headers):
                    if hdr:
                        if current is not None:
                            claims.append(current)
                        num_raw, status_in, rest = _parse_inline_header(text)
                        nums = _expand_inline_range(num_raw)
                        if not nums:
                            continue
                        # For ranges, emit all nums with same status (text as stub)
                        if len(nums) > 1:
                            for n in nums:
                                claims.append(_claim_record(
                                    n, status_in,
                                    f"(range {num_raw}, {status_in}) claim {n}",
                                    [], []))
                            current = None
                        else:
                            current = _claim_record(
                                nums[0], status_in, text, dels, adds)
                    else:
                        if current is not None:
                            current["text"] = (current["text"] + " " + text).strip()
                            current["explicit_deletions"].extend(dels)
                            current["explicit_additions"].extend(adds)
                            current["bracket_deletions"].extend(
                                m.group(0) for m in
                                re.finditer(r"\[\[[^\[\]]{1,200}\]\]", text))
                if current is not None:
                    claims.append(current)
                continue  # handled this wrapper, don't emit as single claim

        # Cases 1 and 2: collapse all ClaimText into one claim body.
        texts = [t for t, _, _ in ct_records if t]
        deleted_fragments = [d for _, dels, _ in ct_records for d in dels]
        addition_fragments = [a for _, _, adds in ct_records for a in adds]
        full = " ".join(texts).strip()
        if not full and not deleted_fragments:
            continue

        # Case 2: structural num missing — parse from inline header.
        if not struct_num:
            num_raw, status_in, _ = _parse_inline_header(full)
            if num_raw:
                nums = _expand_inline_range(num_raw)
                if len(nums) > 1:
                    # Inline range like "1-10 (Canceled)" — expand; drop body
                    # (it's just the range summary line).
                    for n in nums:
                        claims.append(_claim_record(
                            n, status_in or struct_status,
                            f"(range {num_raw}, {status_in or struct_status}) claim {n}",
                            [], []))
                    continue
                if len(nums) == 1:
                    num = str(nums[0])
                    status = status_in or struct_status
                    claims.append(_claim_record(
                        num, status, full,
                        deleted_fragments, addition_fragments))
                    continue
            # No inline header either — keep as anonymous (skip-on-filter below)
            num = ""
            status = struct_status
        else:
            num = struct_num
            status = struct_status

        claims.append(_claim_record(
            num, status, full, deleted_fragments, addition_fragments))

    # <ImplicitClaim> with <ClaimNumberRange> encodes range-cancellations like
    # "Claims 1-15 (Canceled)" as a single element. Expand into individual
    # claim records so downstream (similarity, C1/C2 metrics) see them.
    for ic in _iter_in_ns(root, "ImplicitClaim"):
        status_el = _find_in_ns(ic, "ClaimStatusCategory")
        status = (status_el.text or "").strip() if status_el is not None else "unlabeled"
        rng = _find_in_ns(ic, "ClaimNumberRange")
        if rng is None:
            continue
        begin_el = _find_in_ns(rng, "BeginRangeNumber")
        end_el = _find_in_ns(rng, "EndRangeNumber")
        try:
            begin = int((begin_el.text or "").strip()) if begin_el is not None else None
            end = int((end_el.text or "").strip()) if end_el is not None else None
        except (ValueError, TypeError):
            continue
        if begin is None or end is None or begin > end:
            continue
        for n in range(begin, end + 1):
            claims.append({
                "num": str(n),
                "status": status,
                "text": f"(implicit, {status}) claim {n}",
                "explicit_deletions": [],
                "explicit_additions": [],
                "bracket_deletions": [],
            })

    return claims


# ---------- CTNF OutgoingDocument ----------

PAT_USC = "{urn:us:gov:doc:uspto:patent}"
COM_USC = "{urn:us:gov:doc:uspto:common}"
COM_ST96 = "{http://www.wipo.int/standards/XMLSchema/ST96/Common}"


# Regex fallback — for CTNF variants where FormParagraphNumber is not
# semantic (e.g. "fpn-001" under OutgoingDocument_V7_1) and DataField-based
# extraction yields 0 rejection instances. We scan prose for standard
# USPTO rejection boilerplate.
RE_REJECTION_SENTENCE = re.compile(
    r"Claim(?:s)?\s+"                                 # Claim(s)
    r"([0-9][0-9,\s\-\u2013and]{0,80})\s+"           # claim numbers (digits, dashes, 'and')
    r"(?:is|are)\s+rejected\s+under\s+"
    r"35\s*U\.?\s*S\.?\s*C\.?\s*(?:§)?\s*"
    r"(\d+)\s*"                                       # statute section
    r"(?:\(([a-z0-9]+)\))?"                           # optional subsection
    r"\s*(?:as\s+(?:being\s+)?([^.]{3,400}))?",      # rejection type/clause
    re.IGNORECASE,
)

# Match "over Smith", "over Smith et al.", "over Smith (1234567)",
# "over Smith (5,295,698) in view of Jones"
RE_PRIOR_ART = re.compile(
    r"(?:over|by)\s+"
    r"([A-Z][a-zA-Z][a-zA-Z\-\.]{1,40}"
    r"(?:\s+et\s+al\.?)?"
    r"(?:\s*\([0-9,]{5,20}\))?)"
)


def fallback_regex_ctnf(full_text: str) -> dict:
    """Regex-based fallback when DataField extraction yields nothing."""
    rejection_instances = []
    prior_art = []
    seen_pa = set()
    for m in RE_REJECTION_SENTENCE.finditer(full_text or ""):
        claims_raw = (m.group(1) or "").strip()
        statute = (m.group(2) or "").strip()
        subsection = (m.group(3) or "").strip()
        rtype = (m.group(4) or "").strip()

        # Extract prior-art names from the rejection clause (rtype)
        arts = []
        for am in RE_PRIOR_ART.finditer(rtype):
            ref = am.group(1).strip()
            if ref and ref not in seen_pa:
                arts.append(ref)
                seen_pa.add(ref)
                prior_art.append(ref)
        rejection_instances.append({
            "source": "regex_fallback",
            "claims_raw": [claims_raw] if claims_raw else [],
            "statute_section": statute,
            "statute_subsection": subsection,
            "rejection_clause": rtype[:300],
            "prior_art_refs": arts,
        })
    return {
        "rejection_instances": rejection_instances,
        "prior_art_refs": prior_art,
    }


def parse_ctnf(root: ET.Element) -> dict:
    """Pull out rejection sections from <uspat:OutgoingDocument>.
    Returns dict with keys: sections (list of {heading, form_para_num, body}),
    prior_art_refs (deduped list), rejection_instances (structured)."""

    sections = []
    rejection_instances = []
    prior_art = []
    last_statute = ""            # Track section heading seen most recently
    post_conclusion = False      # Flip when we cross into Conclusion/OacsClose footer

    # FormParagraph numbers that mark the end of rejection content
    CLOSING_FP = {"conclusion_heading", "OacsClose",
                  "examiner_contact", "conclusion",
                  "signature", "signature_block"}

    # Acceptable tokens for DataField#3 (rejection type) — lowercase substring match
    REJ_TYPE_ALLOW = (
        "anticipated", "rejected", "obvious", "being", "unpatentable",
        "indefinite", "ineligible", "rendered", "lack", "fail", "taught",
        "disclos",  # discloses/disclosed
    )

    # Patterns that disqualify a DataField#4 as a prior-art reference
    import re as _re
    BAD_PRIOR_ART_PATTERNS = [
        _re.compile(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}"),   # US phone
        _re.compile(r"\bfax\b", _re.IGNORECASE),
        _re.compile(r"\bPAIR\b"),
        _re.compile(r"Information\s+regarding\s+the\s+status",
                    _re.IGNORECASE),
        _re.compile(r"^\s*\d+[-.\s]", _re.IGNORECASE),   # starts with digits
    ]

    # Walk FormParagraph blocks in order
    for fp in root.iter(COM_USC + "FormParagraph"):
        fpn_el = fp.find(COM_USC + "FormParagraphNumber")
        fpn = (fpn_el.text or "").strip() if fpn_el is not None else ""

        # Collect all <uscom:P> under this FormParagraph as body
        body_pieces = []
        for p in fp.iter(COM_USC + "P"):
            body_pieces.append(element_text(p))
        body = " ".join(b for b in body_pieces if b).strip()

        # Classify by heading
        heading_map = {
            "detailed_action": "DETAILED ACTION",
            "USC_101_heading": "35 USC § 101",
            "USC_102_heading": "35 USC § 102",
            "USC_103_heading": "35 USC § 103",
            "USC_112_heading": "35 USC § 112",
            "conclusion_heading": "Conclusion",
        }
        heading = heading_map.get(fpn, "")
        # If this FormParagraph is a section heading, remember for subsequent rejections
        if fpn.startswith("USC_") and fpn.endswith("_heading"):
            m = re.search(r"USC_(\d+)", fpn)
            if m:
                last_statute = m.group(1)
        # Detect crossing into the footer (conclusion / examiner signature)
        if fpn in CLOSING_FP or fpn.startswith("conclusion"):
            post_conclusion = True

        sections.append({"form_para_num": fpn, "heading": heading, "body": body})

        # Footer sections don't carry rejection content — skip rejection_instance
        # extraction from here onward (but keep sections record for completeness).
        if post_conclusion:
            continue

        # Rejection instance detection via DataField slots.
        # Typical pattern: dataFieldNumber 1=claims, 2=statute subsection,
        # 3=rejection type, 4=prior art reference
        dfs = list(fp.iter(COM_USC + "DataField"))
        by_num = {}
        for df in dfs:
            n = df.get(COM_USC + "dataFieldNumber") or df.get("dataFieldNumber") or ""
            text = element_text(df)
            if n and text:
                by_num.setdefault(n, []).append(text)
        if "1" in by_num and "4" in by_num:
            # --- Validate DataField#3 is a real rejection type -----------
            raw_type = by_num.get("3", [])
            valid_type = False
            for t in raw_type:
                tl = (t or "").lower()
                if any(tok in tl for tok in REJ_TYPE_ALLOW):
                    valid_type = True
                    break
            if not valid_type:
                # DataField slots present but not a rejection — examiner
                # metadata block or similar. Skip.
                continue

            # --- Clean DataField#4 prior-art refs -----------------------
            ref_candidates = [r.strip() for r in by_num.get("4", [])
                              if r and len(r.strip()) <= 200]
            clean_refs = []
            for r in ref_candidates:
                if r.startswith("Re:") or r.startswith("["):
                    continue
                if not re.search(r"[A-Z][a-z]{2,}", r):
                    continue
                # Blacklist: phone numbers, PAIR/fax boilerplate, etc.
                if any(pat.search(r) for pat in BAD_PRIOR_ART_PATTERNS):
                    continue
                clean_refs.append(r)
            # If no clean refs survived, still record the rejection
            # (the examiner text may reference the prior art elsewhere)

            rejection_instances.append({
                "form_para_num": fpn,
                "statute_section": last_statute,
                "claims_raw": by_num.get("1", []),
                "statute_subsection": by_num.get("2", []),
                "rejection_type": raw_type,
                "prior_art_refs": clean_refs,
            })
            for ref in clean_refs:
                prior_art.append(ref)

    # Also capture free <uscom:P> siblings not under FormParagraph (some docs)
    # Top-level <uscom:P> tend to carry examiner reasoning
    body_outer = []
    for p in root.findall(".//" + COM_USC + "P"):
        # Skip ones that are already inside FormParagraph
        # (we already collected them; duplicate OK for recall)
        body_outer.append(element_text(p))
    full_text = "\n".join(t for t in body_outer if t)

    # Dedupe prior art list while preserving order
    seen = set()
    prior_art_unique = []
    for ref in prior_art:
        if ref not in seen:
            seen.add(ref)
            prior_art_unique.append(ref)

    # Fallback: if DataField-based extraction found nothing, try regex.
    fallback_used = False
    if not rejection_instances:
        fb = fallback_regex_ctnf(full_text)
        if fb["rejection_instances"]:
            rejection_instances = fb["rejection_instances"]
            for ref in fb["prior_art_refs"]:
                if ref not in seen:
                    seen.add(ref)
                    prior_art_unique.append(ref)
            fallback_used = True

    return {
        "sections": sections,
        "rejection_instances": rejection_instances,
        "prior_art_refs": prior_art_unique,
        "full_text": full_text,
        "regex_fallback_used": fallback_used,
    }


# ---------- Pre/post claim diff ----------

def _claims_by_num(claims):
    """Map claim_num -> text, skip unnumbered or empty."""
    out = {}
    for c in claims or []:
        num = str(c.get("num", "")).strip()
        if not num or num.upper() == "UNKNOWN":
            continue
        out[num] = c.get("text", "") or ""
    return out


def compute_claim_diff(pre_claims, post_claims):
    """Produce per-claim diff records.

    For each claim_num present in pre or post:
      - status: 'kept' (identical) / 'modified' / 'new' / 'cancelled'
      - ratio: SequenceMatcher ratio (for 'modified')
      - added_spans: text added in post vs pre
      - removed_spans: text removed from pre vs post
    Also returns summary stats.
    """
    from difflib import SequenceMatcher

    pre_map = _claims_by_num(pre_claims)
    post_map = _claims_by_num(post_claims)
    all_nums = sorted(set(pre_map) | set(post_map), key=lambda x: (len(x), x))

    per_claim = []
    counts = {"kept": 0, "modified": 0, "new": 0, "cancelled": 0}
    for num in all_nums:
        pre = pre_map.get(num)
        post = post_map.get(num)
        if pre is None:
            entry = {"num": num, "status": "new",
                     "ratio": 0.0, "added_spans": [post or ""], "removed_spans": []}
            counts["new"] += 1
        elif post is None:
            entry = {"num": num, "status": "cancelled",
                     "ratio": 0.0, "added_spans": [], "removed_spans": [pre]}
            counts["cancelled"] += 1
        elif pre.strip() == post.strip():
            entry = {"num": num, "status": "kept", "ratio": 1.0,
                     "added_spans": [], "removed_spans": []}
            counts["kept"] += 1
        else:
            sm = SequenceMatcher(a=pre, b=post, autojunk=False)
            added = []
            removed = []
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "insert":
                    added.append(post[j1:j2])
                elif op == "delete":
                    removed.append(pre[i1:i2])
                elif op == "replace":
                    removed.append(pre[i1:i2])
                    added.append(post[j1:j2])
            entry = {"num": num, "status": "modified",
                     "ratio": round(sm.ratio(), 3),
                     "added_spans": added, "removed_spans": removed}
            counts["modified"] += 1
        per_claim.append(entry)
    return {"per_claim": per_claim, "summary": counts}


# ---------- Archive helpers ----------

def iter_xmls_in_tar(tar_path: Path):
    """Yield (member_name, xml_bytes) for every .xml file inside a tar."""
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            if not m.name.lower().endswith(".xml"):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            yield m.name, f.read()


def detect_root(xml_bytes: bytes) -> str:
    head = xml_bytes[:3000].decode("utf-8", errors="replace")
    # Claims document variants: pat:ClaimsDocument (v1.3), uspat:ClaimsDocument (ST96 V2)
    if re.search(r"<[A-Za-z]+:?ClaimsDocument\b", head):
        return "ns_claims"
    if "<us-patent-application" in head:
        return "dtd_legacy"
    if re.search(r"<[A-Za-z]+:?OutgoingDocument\b", head):
        return "outgoing"
    if re.search(r"<[A-Za-z]+:?IncomingDocument\b", head):
        return "incoming"
    return "unknown"


def parse_clm_archive(tar_path: Path) -> dict:
    """Process a CLM tar archive. Returns {format, claims, n_xmls}."""
    results = {"format": None, "claims": [], "n_xmls": 0}
    for name, body in iter_xmls_in_tar(tar_path):
        results["n_xmls"] += 1
        cleaned = clean_xml_text(body.decode("utf-8", errors="replace"))
        fmt = detect_root(cleaned.encode())
        try:
            root = ET.fromstring(cleaned)
        except ET.ParseError as e:
            results.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {e}")
            continue
        if fmt == "dtd_legacy":
            results["format"] = "dtd_legacy"
            results["claims"] = parse_clm_dtd(root)
        elif fmt == "ns_claims":
            results["format"] = "ns_claims"
            results["claims"] = parse_clm_ns(root)
        else:
            results["format"] = fmt
    return results


def parse_ctnf_archive(tar_path: Path) -> dict:
    for name, body in iter_xmls_in_tar(tar_path):
        cleaned = clean_xml_text(body.decode("utf-8", errors="replace"))
        try:
            root = ET.fromstring(cleaned)
        except ET.ParseError as e:
            return {"error": f"parse: {e}"}
        return parse_ctnf(root)
    return {"error": "no_xml_in_tar"}


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--beta-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") != "complete_3_of_3":
                continue
            cases.append(r)
            if args.limit and len(cases) >= args.limit:
                break
    print(f"Parsing {len(cases)} complete_3_of_3 cases")

    fmt_counter = Counter()
    claim_counts = Counter()
    pa_ref_counts = []
    rejection_inst_counts = []
    parse_errors = 0

    for i, case in enumerate(cases):
        app = case.get("app_num")
        case_dir = args.beta_root / str(app)
        pre = parse_clm_archive(case_dir / "pre_clm.tar")
        post = parse_clm_archive(case_dir / "post_clm.tar")
        ctnf = parse_ctnf_archive(case_dir / "ctnf.tar")

        fmt_counter[f"pre:{pre.get('format')}"] += 1
        fmt_counter[f"post:{post.get('format')}"] += 1
        claim_counts[f"pre_n={len(pre.get('claims', []))}"] += 1
        claim_counts[f"post_n={len(post.get('claims', []))}"] += 1

        pa_refs = ctnf.get("prior_art_refs", []) if isinstance(ctnf, dict) else []
        rej_inst = ctnf.get("rejection_instances", []) if isinstance(ctnf, dict) else []
        pa_ref_counts.append(len(pa_refs))
        rejection_inst_counts.append(len(rej_inst))

        # Track regex fallback usage
        if isinstance(ctnf, dict) and ctnf.get("regex_fallback_used"):
            fmt_counter["ctnf:regex_fallback"] += 1

        if "error" in ctnf or pre.get("errors") or post.get("errors"):
            parse_errors += 1

        claim_diff = compute_claim_diff(pre.get("claims"), post.get("claims"))

        record = {
            "case_id": case.get("case_id"),
            "proceeding_number": case.get("proceeding_number"),
            "app_num": app,
            "pre_clm": pre,
            "post_clm": post,
            "ctnf": ctnf,
            "claim_diff": claim_diff,
        }
        out_path = args.out_dir / f"{app}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(cases)}")

    def pct_summary(lst):
        if not lst:
            return {}
        lst_sorted = sorted(lst)
        n = len(lst_sorted)
        return {
            "min": lst_sorted[0], "median": lst_sorted[n // 2],
            "max": lst_sorted[-1], "mean": round(sum(lst_sorted) / n, 1),
        }

    # Aggregate claim_diff summary across all cases
    diff_summary_agg = Counter()
    modified_ratio_samples = []
    for p in args.out_dir.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cd = rec.get("claim_diff", {})
        for k, v in (cd.get("summary") or {}).items():
            diff_summary_agg[k] += v
        for c in (cd.get("per_claim") or []):
            if c.get("status") == "modified":
                modified_ratio_samples.append(c.get("ratio", 0))

    stats = {
        "n_cases_parsed": len(cases),
        "format_counts": dict(fmt_counter.most_common()),
        "prior_art_refs_per_case": pct_summary(pa_ref_counts),
        "rejection_instances_per_case": pct_summary(rejection_inst_counts),
        "claim_count_distribution": dict(claim_counts.most_common(30)),
        "parse_errors": parse_errors,
        "claim_diff_total": dict(diff_summary_agg),
        "modified_claim_ratio_stats": pct_summary([round(r, 3) for r in modified_ratio_samples]),
    }
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== PARSE STATS ===")
    print(json.dumps(stats, indent=2))
    print(f"\nPer-case JSONs → {args.out_dir}")


if __name__ == "__main__":
    main()
