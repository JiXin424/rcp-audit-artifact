#!/usr/bin/env python3
"""Create/publish the Zenodo deposit for the RCP audit artifact (LRE submission).

Workflow:
  1. ZENODO_TOKEN=<token> python3 scripts/zenodo_deposit.py reserve
     -> creates a draft deposit with anonymous metadata, prints the reserved
        DOI and the deposit URL. Write the DOI into main_lre.tex
        (Data availability) at this point.
  2. Build the artifact snapshot zip (see --zip). After the final artifact
     repo state is committed, run:
     ZENODO_TOKEN=<token> python3 scripts/zenodo_deposit.py upload <zip>
     -> uploads the zip as the deposit's file.
  3. ZENODO_TOKEN=<token> python3 scripts/zenodo_deposit.py publish
     -> publishes (freezes) the deposit and prints the final DOI.
     NOTE: publishing is irreversible (the DOI cannot be unpublished).
"""
import json
import os
import sys
import urllib.request

API = "https://zenodo.org/api"
HEADERS_BASE = {"Content-Type": "application/json"}

METADATA = {
    "title": (
        "Artifact: Auditing and Partially Reproducing the SLRTP2025 "
        "Back-Translation Evaluator"
    ),
    "description": (
        "Audit artifact for the LRE submission \"Auditing and Partially "
        "Reproducing the SLRTP2025 Back-Translation Evaluator\". Contains "
        "paper sources (main_lre.tex/supplementary.tex), all per-checkpoint "
        "results, decoded hypotheses, analysis scripts, the claim-to-file "
        "manifest, and the L1 audit environment definition "
        "(requirements-audit.txt). Round 35 adds: the verbatim candidate-"
        "upstream source archive with per-file SHA-256 hashes "
        "(third_party/signjoey-signidd-slt-249d3cd8fc2/, Apache-2.0; full "
        "commit 249d3cd8fc249b1a06eba39f84cb2d289ed37bce) and update-level "
        "gradient diagnostics (results/grad_diag_*.json: dual lockstep "
        "clip1.0/unclipped run, full-horizon clipped run, and an update-"
        "norm-matched unclipped control that still enters the memorisation "
        "regime). Checkpoints are hosted separately (ModelScope "
        "J1Xin424/rcp-audit-checkpoints); the released SLRTP2025 evaluator "
        "is not redistributed. The git commit this archive was built from "
        "is recorded in the Data availability section of the paper."
    ),
    "creators": [{"name": "Anonymous"}],
    "upload_type": "software",
    "access_right": "open",
    "license": "MIT",
    "version": "LRE-submission-2026-08-r36",
    "keywords": [
        "sign language translation",
        "back-translation",
        "reproducibility audit",
        "SLRTP2025",
        "PHOENIX-2014T",
        "DGS",
    ],
    "related_identifiers": [
        {
            "relation": "isSupplementTo",
            "identifier": "https://anonymous.4open.science/r/rcp-audit-artifact-B314/",
            "resource_type": "other",
        }
    ],
}


def auth_headers(token):
    h = dict(HEADERS_BASE)
    h["Authorization"] = "Bearer " + token
    return h


def request(method, url, token=None, data=None, content_type=None):
    body = None
    headers = auth_headers(token) if token else dict(HEADERS_BASE)
    if content_type:
        headers["Content-Type"] = content_type
    if data is not None:
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def state_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", ".zenodo_deposit_state.json")


def main():
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("Set ZENODO_TOKEN.", file=sys.stderr)
        sys.exit(1)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    st = state_file()

    if cmd == "reserve":
        st_code, dep = request("POST", API + "/deposit/depositions", token,
                               {"metadata": METADATA})
        if st_code != 201:
            print("reserve failed:", st_code, json.dumps(dep)[:500],
                  file=sys.stderr)
            sys.exit(1)
        with open(st, "w") as f:
            json.dump({"id": dep["id"], "doi": dep["metadata"]["prereserve_doi"]["doi"],
                       "bucket": dep["links"]["bucket"], "deposit_url": dep["links"]["html"]},
                      f, indent=2)
        print("RESERVED DOI:", dep["metadata"]["prereserve_doi"]["doi"])
        print("deposit:", dep["links"]["html"])
        print("state ->", st)

    elif cmd == "upload":
        with open(st) as f:
            s = json.load(f)
        zpath = sys.argv[2]
        fname = os.path.basename(zpath)
        with open(zpath, "rb") as f:
            data = f.read()
        st_code, resp = request(
            "PUT", s["bucket"] + "/" + fname, token, data,
            content_type="application/octet-stream")
        if st_code not in (200, 201):
            print("upload failed:", st_code, json.dumps(resp)[:500],
                  file=sys.stderr)
            sys.exit(1)
        print("uploaded", fname, "->", resp["links"]["self"])

    elif cmd == "update-meta":
        with open(st) as f:
            s = json.load(f)
        st_code, dep = request("PUT",
                               API + "/deposit/depositions/%s" % s["id"], token,
                               {"metadata": METADATA})
        if st_code not in (200, 201):
            print("update-meta failed:", st_code, json.dumps(dep)[:500],
                  file=sys.stderr)
            sys.exit(1)
        print("metadata updated for deposit", s["id"])

    elif cmd == "publish":
        with open(st) as f:
            s = json.load(f)
        st_code, dep = request("POST",
                               API + "/deposit/depositions/%s/actions/publish" % s["id"],
                               token)
        if st_code not in (200, 202):
            print("publish failed:", st_code, json.dumps(dep)[:500],
                  file=sys.stderr)
            sys.exit(1)
        print("PUBLISHED DOI:", dep["doi"])
        print("record:", dep["links"]["record_html"])

    elif cmd == "newversion":
        # create a new-version draft of the (already published) deposit
        import time
        with open(st) as f:
            s0 = json.load(f)
        st_code, dep = request("POST",
                               API + "/deposit/depositions/%s/actions/newversion" % s0["id"],
                               token)
        if st_code != 201:
            print("newversion failed:", st_code, json.dumps(dep)[:500], file=sys.stderr)
            sys.exit(1)
        nv_url = dep["links"]["new_version"]
        draft = None
        for _ in range(10):
            time.sleep(2)
            code, draft = request("GET", nv_url, token)
            if code == 200 and not draft.get("locked", True):
                break
            draft = None
        if draft is None:
            print("new-version draft not ready/locked", file=sys.stderr)
            sys.exit(1)
        s0.update({"id": draft["id"], "doi": draft.get("doi") or s0["doi"],
                   "bucket": draft["links"]["bucket"],
                   "deposit_url": draft["links"]["html"],
                   "original_id": s0.get("original_id", s0["id"])})
        with open(st, "w") as f:
            json.dump(s0, f, indent=2)
        print("new-version draft:", draft["id"], "->", draft["links"]["html"])

    elif cmd == "status":
        if os.path.exists(st):
            print(open(st).read())
        else:
            print("no deposit state yet; run 'reserve' first.")

    else:
        print("usage: reserve | newversion | update-meta | upload <zip> | publish | status", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
