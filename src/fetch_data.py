"""Download clinical trial records from ClinicalTrials.gov (API v2) and the NLM MeSH descriptor file.

Usage: python -m src.fetch_data [--n 50000]
"""
import argparse
import gzip
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:  # python.org builds on macOS ship without CA certificates; certifi comes with requests/streamlit
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = None

DATA = Path(__file__).resolve().parent.parent / "data"
TRIALS = DATA / "trials.jsonl.gz"
MESH = DATA / "desc2026.gz"

API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,Condition,EligibilityCriteria"
MESH_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ml-mini-project"})
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as r:
                return r.read()
        except OSError as e:
            if i == tries - 1:
                raise
            print(f"  request failed ({e}), retrying")
            time.sleep(2 ** i)


def flatten(study):
    """Keep only the six text sections the paper extracts keywords from."""
    p = study["protocolSection"]
    ident = p.get("identificationModule", {})
    desc = p.get("descriptionModule", {})
    return {
        "nct_id": ident["nctId"],
        "brief_title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "brief_summary": desc.get("briefSummary", ""),
        "detailed_description": desc.get("detailedDescription", ""),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "eligibility": p.get("eligibilityModule", {}).get("eligibilityCriteria", ""),
    }


def fetch_trials(n):
    part = TRIALS.with_suffix(".part")
    token, done = None, 0
    with gzip.open(part, "wt", encoding="utf-8") as f:
        while done < n:
            query = {"pageSize": 1000, "fields": FIELDS, "format": "json"}
            if token:
                query["pageToken"] = token
            page = json.loads(get(API + "?" + urllib.parse.urlencode(query)))
            for study in page["studies"][: n - done]:
                f.write(json.dumps(flatten(study)) + "\n")
                done += 1
            print(f"  {done}/{n} trials")
            token = page.get("nextPageToken")
            if not token:
                break
    part.rename(TRIALS)  # only a complete download gets the real name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50000, help="number of trial records to download")
    ap.add_argument("--force", action="store_true", help="download again even if files exist")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    if args.force or not MESH.exists():
        print("Downloading MeSH 2026 descriptors ...")
        MESH.write_bytes(get(MESH_URL))
    if args.force or not TRIALS.exists():
        print(f"Downloading {args.n} trial records from ClinicalTrials.gov ...")
        fetch_trials(args.n)
    print(f"Done. Files in {DATA}")


if __name__ == "__main__":
    main()
