"""
Build β corpus by querying USPTO ODP API for XML-grade 4-tuple:
  pre-amendment claim (CLM XML before first CTNF)
  rejection rationale (CTNF XML, full OutgoingDocument schema)
  post-amendment claim (CLM XML filed in response to CTNF)
  prior-art references (parsed from CTNF body)

Input: alpha_corpus.jsonl (produced by build_alpha_corpus.py)
Uses each case's published.publication filename to resolve app_num via
ODP search endpoint, then walks documents timeline to locate pre/post CLM
around the first CTNF.

API key: read from env PATENT_CLIENT_ODP_API_KEY (do NOT hardcode).

Usage:
  export PATENT_CLIENT_ODP_API_KEY=$(cat /data2/hsm2026/vPClaimGeneration/.env | grep PATENT | cut -d= -f2)
  python3 scripts/build_beta_corpus.py \
      --alpha data/parsed/alpha_corpus.jsonl \
      --out-root outputs/beta \
      --manifest data/parsed/beta_manifest.jsonl \
      --stats data/parsed/beta_stats.json \
      --limit 5    # smoke; then --limit 100
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

ODP_BASE = "https://api.uspto.gov/api/v1"
ENV_KEY_NAME = "PATENT_CLIENT_ODP_API_KEY"

# Rate limit safety: ODP is 60/min/key. We use 4 calls/case → target 12 cases/min.
INTER_CALL_DELAY = 0.35  # seconds between API calls
RETRY_BACKOFF = [2, 5, 15]  # seconds


def pub_filename_to_query(pub_filename: str):
    """Given a publication filename like 'US20070067890A1' (from the applicant-XML JSON
    filename stem) return it unchanged — already normalized."""
    if not pub_filename:
        return None
    # Strip any trailing .* or leading non-alnum
    s = re.sub(r"\.\w+$", "", pub_filename)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    if not s.startswith("US"):
        return None
    return s


class ODPClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key, "Accept": "application/json"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.ncalls = 0

    def close(self):
        self.session.close()

    def _request(self, method: str, url: str, **kw):
        self.ncalls += 1
        time.sleep(INTER_CALL_DELAY)
        last_exc = None
        kw.setdefault("timeout", 60)
        for backoff in [0] + RETRY_BACKOFF:
            if backoff:
                time.sleep(backoff)
            try:
                resp = self.session.request(method, url, **kw)
                if resp.status_code == 429:
                    continue
                return resp
            except requests.RequestException as e:
                last_exc = e
        if last_exc:
            raise last_exc
        return resp

    def search_by_pub(self, pub_number: str):
        url = f"{ODP_BASE}/patent/applications/search"
        params = {
            "q": f"applicationMetaData.earliestPublicationNumber:{pub_number}",
            "limit": 1,
        }
        r = self._request("GET", url, params=params)
        if r.status_code != 200:
            return None
        d = r.json()
        bag = d.get("patentFileWrapperDataBag") or []
        if not bag:
            return None
        return bag[0].get("applicationNumberText")

    def list_documents(self, app_num: str):
        url = f"{ODP_BASE}/patent/applications/{app_num}/documents"
        r = self._request("GET", url)
        if r.status_code != 200:
            return None
        return r.json().get("documentBag", [])

    def download_xml_archive(self, url: str, out_path: Path):
        r = self._request("GET", url,
                          headers={"Accept": "application/octet-stream"})
        if r.status_code != 200:
            return False, r.status_code
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        return True, 200


def pick_pre_post_around_ctnf(docs: list):
    """Given a prosecution document bag, locate the FIRST CTNF and the CLM
    documents (incoming) directly before / after it.
    Returns dict with pre_clm, ctnf, post_clm entries (each may be None).
    """
    docs_sorted = sorted(
        docs, key=lambda x: (x.get("officialDate") or "", x.get("documentIdentifier") or "")
    )
    first_ctnf = None
    for d in docs_sorted:
        if d.get("documentCode") == "CTNF":
            first_ctnf = d
            break
    if not first_ctnf:
        return None
    ctnf_date = first_ctnf.get("officialDate", "")

    pre_clm = None
    post_clm = None
    for d in docs_sorted:
        if d.get("documentCode") != "CLM":
            continue
        if d.get("directionCategory") != "INCOMING":
            continue
        date = d.get("officialDate", "")
        if date <= ctnf_date:
            pre_clm = d  # keep walking; last one before CTNF wins
        elif date > ctnf_date and post_clm is None:
            post_clm = d  # first CLM after CTNF
            break
    return {"pre_clm": pre_clm, "ctnf": first_ctnf, "post_clm": post_clm}


def xml_archive_url(doc):
    for opt in doc.get("downloadOptionBag", []) or []:
        if opt.get("mimeTypeIdentifier") == "XML":
            return opt.get("downloadUrl")
    return None


def process_case(case: dict, client: ODPClient, out_root: Path, skip_existing: bool = True):
    case_id = case.get("case_id")
    pub_data = case.get("published") or {}
    pub_filename = pub_data.get("publication")
    result = {
        "case_id": case_id,
        "proceeding_number": case.get("proceeding_number"),
        "pub_filename": pub_filename,
        "status": "unknown",
    }

    # Early cache hit: if we already have all 3 tar archives, skip API calls.
    if skip_existing:
        probe_pub = pub_filename_to_query(pub_filename)
        # We can't know app_num without the search; but we can scan out_root for
        # any case_dir whose 3 tar files exist and whose parent is tagged. Use
        # a dedicated cache probe file to remember previous success.
        cache_marker = out_root / ".cache" / f"{case_id}.json"
        if cache_marker.exists():
            try:
                cached = json.loads(cache_marker.read_text())
                cached["cached"] = True
                return cached
            except Exception:
                pass

    pub_q = pub_filename_to_query(pub_filename)
    if not pub_q:
        result["status"] = "no_pub_filename"
        return result

    app_num = client.search_by_pub(pub_q)
    if not app_num:
        result["status"] = "pub_not_in_odp"
        return result
    result["app_num"] = app_num

    docs = client.list_documents(app_num)
    if docs is None:
        result["status"] = "documents_404"
        return result
    result["n_docs"] = len(docs)

    picked = pick_pre_post_around_ctnf(docs)
    if not picked:
        result["status"] = "no_ctnf"
        return result

    case_dir = out_root / app_num
    case_dir.mkdir(parents=True, exist_ok=True)

    downloads = {}
    for label in ("pre_clm", "ctnf", "post_clm"):
        doc = picked[label]
        if not doc:
            downloads[label] = {"status": "missing"}
            continue
        url = xml_archive_url(doc)
        if not url:
            downloads[label] = {"status": "no_xml_option",
                                "document_code": doc.get("documentCode"),
                                "mimes": [o.get("mimeTypeIdentifier")
                                          for o in (doc.get("downloadOptionBag") or [])]}
            continue
        out_path = case_dir / f"{label}.tar"
        ok, code = client.download_xml_archive(url, out_path)
        downloads[label] = {
            "status": "ok" if ok else f"http_{code}",
            "doc_identifier": doc.get("documentIdentifier"),
            "doc_date": doc.get("officialDate"),
            "document_code": doc.get("documentCode"),
            "size_bytes": out_path.stat().st_size if ok else 0,
            "local_path": str(out_path.relative_to(out_root.parent.parent))
                           if ok else None,
        }

    result["downloads"] = downloads
    ok_count = sum(1 for v in downloads.values() if v.get("status") == "ok")
    result["status"] = f"complete_{ok_count}_of_3"

    # Write cache marker for full successes so future runs skip
    if skip_existing and ok_count == 3:
        cache_dir = out_root / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{case_id}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--require-quote", action="store_true",
                    help="Only process cases with appeal-time claim quote present")
    args = ap.parse_args()

    key = os.environ.get(ENV_KEY_NAME, "").strip()
    if not key:
        print(f"ERROR: env var {ENV_KEY_NAME} not set.", file=sys.stderr)
        sys.exit(2)

    # Stream alpha, pick eligible cases up to limit
    eligible = []
    with args.alpha.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            flags = row.get("flags") or {}
            if not flags.get("has_published_claims"):
                continue
            if row.get("published_status") != "ok":
                continue
            if args.require_quote and not flags.get("has_any_claim_quote"):
                continue
            eligible.append(row)
            if args.limit and len(eligible) >= args.limit:
                break
    print(f"Eligible cases selected: {len(eligible)}")

    client = ODPClient(key)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with args.manifest.open("w", encoding="utf-8") as mh:
        for i, case in enumerate(eligible):
            try:
                r = process_case(case, client, args.out_root)
            except Exception as e:
                r = {"case_id": case.get("case_id"), "status": f"exception:{type(e).__name__}",
                     "error": str(e)[:400]}
            results.append(r)
            mh.write(json.dumps(r, ensure_ascii=False) + "\n")
            mh.flush()
            sys.stdout.write(f"  [{i+1}/{len(eligible)}] {r.get('case_id','?')} -> {r['status']}\n")
            sys.stdout.flush()

    client.close()

    # Stats
    from collections import Counter
    status_counts = Counter(r["status"] for r in results)
    dl_counts = Counter()
    for r in results:
        for lbl, v in (r.get("downloads") or {}).items():
            dl_counts[f"{lbl}:{v.get('status','?')}"] += 1
    stats = {
        "n": len(results),
        "api_calls_total": client.ncalls,
        "status_counts": dict(status_counts),
        "per_label_download_status": dict(dl_counts),
        "n_fully_ok": sum(1 for r in results if r["status"] == "complete_3_of_3"),
    }
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n=== STATS ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
