#!/usr/bin/env python3
"""Round 5 E3b/E3c: caliper sensitivity and signer-strict matched systems.

Builds three SEEN-MATCHED variants on the canonical 641 queries:
  SEEN-MATCHED-T005: Jaccard tolerance 0.05 (tighter caliper)
  SEEN-MATCHED-T020: Jaccard tolerance 0.20 (looser caliper)
  SEEN-MATCHED-SIGNER: same-signer constraint + Jaccard tolerance 0.10
Selection: minimize |Jaccard - target_unseen_jaccard| among eligible train donors,
canonical tie-breaks (min Levenshtein, min SHA-256).
Outputs registries + 25 fps pose bundles for decoding.
"""
from __future__ import annotations
import hashlib, json, re, sys, unicodedata
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parents[1] / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data"
TEST_PT, TRAIN_PT = DATA_ROOT / "data/test.pt", DATA_ROOT / "data/train.pt"
R3_REGISTRY = ROOT / "revision_20260728_round3/results/e2_1_exposure_registry.jsonl"
OUT_DIR = ROOT / "revision_20260729_round5/results"
POSE_DIR = ROOT / "revision_20260729_round5/outputs"


def sha256_str(s): return hashlib.sha256(s.encode()).hexdigest()


def norm_text(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip().lower()


def jaccard(a, b): return len(a & b) / len(a | b) if a | b else 0.0


def lev(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = cur
    return dp[n]


def build(name, tol, same_signer, test, train, train_tok, target_j):
    rows = []
    poses = {}
    train_ids = sorted(train)
    train_norm = {k: norm_text(v["text"]) for k, v in train.items()}
    train_signer = {k: v.get("speaker") for k, v in train.items()}
    for qid in sorted(test):
        if qid not in target_j:
            continue
        qt = set(norm_text(test[qid]["text"]).split())
        qn = norm_text(test[qid]["text"])
        qsigner = test[qid].get("speaker")
        tj = target_j[qid]
        # stage 1: cheap filter by |dJ| <= tol
        cand = []
        for did in train_ids:
            if did == qid or train_norm[did] == qn:
                continue
            if same_signer and train_signer[did] != qsigner:
                continue
            j = jaccard(qt, train_tok[did])
            dj = abs(j - tj)
            if dj <= tol:
                cand.append((dj, did, j))
        if not cand:
            continue
        # stage 2: min |dJ|, then Levenshtein only among near-ties (|dJ| within 1e-12 of min)
        min_dj = min(c[0] for c in cand)
        tied = [c for c in cand if c[0] <= min_dj + 1e-12]
        best = min(tied, key=lambda c: (lev(qn, train_norm[c[1]]), sha256_str(c[1])))
        _, did, j = best
        rows.append({"query_id": qid, "system": name, "donor_id": did, "jaccard": j,
                     "target_unseen_jaccard": tj, "jaccard_abs_diff": abs(j - tj),
                     "same_signer": same_signer, "tolerance": tol, "n_in_band": len(cand)})
        poses[qid] = train[did]["poses_3d"]
    with open(OUT_DIR / f"e3_{name}_registry.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    torch.save(poses, POSE_DIR / f"{name}.pt")
    dj = [r["jaccard_abs_diff"] for r in rows]
    print(f"{name}: {len(rows)}/641 matched, mean|dJ|={sum(dj)/max(len(dj),1):.4f} max={max(dj) if dj else 0:.4f}")


def main():
    test = torch.load(TEST_PT, map_location="cpu", weights_only=True)
    train = torch.load(TRAIN_PT, map_location="cpu", weights_only=True)
    train_tok = {k: set(norm_text(v["text"]).split()) for k, v in train.items()}
    target_j = {}
    for line in open(R3_REGISTRY):
        r = json.loads(line)
        if r["system"] == "UNSEEN-PURE-v1":
            target_j[r["query_id"]] = r["jaccard"]
    build("SEEN-MATCHED-T005", 0.05, False, test, train, train_tok, target_j)
    build("SEEN-MATCHED-T020", 0.20, False, test, train, train_tok, target_j)
    build("SEEN-MATCHED-SIGNER", 0.10, True, test, train, train_tok, target_j)


if __name__ == "__main__":
    main()
