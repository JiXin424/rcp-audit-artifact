#!/usr/bin/env python3
"""Round 5 E2: build SEEN-RAND640-MATCHED-v1, the missing cell for path-sensitivity analysis.

Design (2x2 within-train factorial + origin contrast):
  size:      7060 full pool  vs 640 per-query random sub-pool (same frozen sub-pools as
             SEEN-PURE-RAND640-v1, reproduced by the deterministic per-query seed rule)
  selection: max-Jaccard (canonical) vs matched-to-UNSEEN-Jaccard (|dJ| minimised)

Cells:
  S1 = SEEN-PURE-v1          (7060, max)     [exists]
  S2 = SEEN-PURE-RAND640-v1  (640,  max)     [exists]
  S3 = SEEN-PURE-MATCHED-v1  (7060, match)   [exists]
  S4 = SEEN-RAND640-MATCHED  (640,  match)   [this script]
  U  = UNSEEN-PURE-v1        (test 640, max) [exists]

Path 1 (size -> selection -> origin): S1 -S2- S2 -S4- U
Path 2 (selection -> size -> origin): S1 -S3- S4 -U   (S3->S4 size, S4->U origin)
Shapley-style average over the two admissible orderings for size and selection;
origin contrast is path-invariant (S4 - U).
"""
from __future__ import annotations
import hashlib, json, random, re, sys, unicodedata
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parents[1] / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data"
TEST_PT, TRAIN_PT = DATA_ROOT / "data/test.pt", DATA_ROOT / "data/train.pt"
R3_REGISTRY = ROOT / "revision_20260728_round3/results/e2_1_exposure_registry.jsonl"
OUT_REG = ROOT / "revision_20260729_round5/results/e2_seen_rand640_matched_registry.jsonl"
OUT_PT = ROOT / "revision_20260729_round5/outputs/SEEN-RAND640-MATCHED-v1.pt"
OUT_PT.parent.mkdir(parents=True, exist_ok=True)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def tokens(t: str) -> list[str]:
    return norm_text(t).split()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def lev(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = cur
    return dp[n]


def main():
    test = torch.load(TEST_PT, map_location="cpu", weights_only=True)
    train = torch.load(TRAIN_PT, map_location="cpu", weights_only=True)
    train_ids = sorted(train)
    # target unseen Jaccard per query
    target_j = {}
    for line in open(R3_REGISTRY):
        r = json.loads(line)
        if r["system"] == "UNSEEN-PURE-v1":
            target_j[r["query_id"]] = r["jaccard"]

    train_tok = {k: set(tokens(v["text"])) for k, v in train.items()}
    rows = []
    poses = {}
    for qid in sorted(test):
        if qid not in target_j:
            continue
        qt = set(tokens(test[qid]["text"]))
        per_q_rng = random.Random(int(sha256_str(qid), 16) % (2**31))
        sub_pool = per_q_rng.sample(train_ids, min(640, len(train_ids)))
        tj = target_j[qid]
        cand = []
        for did in sub_pool:
            if did == qid or norm_text(train[did]["text"]) == norm_text(test[qid]["text"]):
                continue
            j = jaccard(qt, train_tok[did])
            cand.append((abs(j - tj), did, j))
        min_dj = min(c[0] for c in cand)
        tied = [c for c in cand if c[0] <= min_dj + 1e-12]
        best = min(tied, key=lambda c: (lev(norm_text(test[qid]["text"]), norm_text(train[c[1]]["text"])),
                                        sha256_str(c[1])))
        _, did, j = best
        rows.append({"query_id": qid, "system": "SEEN-RAND640-MATCHED-v1", "donor_id": did,
                     "donor_pool": "train_random_640", "jaccard": j,
                     "target_unseen_jaccard": tj, "jaccard_abs_diff": abs(j - tj),
                     "n_candidates": len(sub_pool)})
        poses[qid] = train[did]["poses_3d"]
    with open(OUT_REG, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    torch.save(poses, OUT_PT)
    dj = [r["jaccard_abs_diff"] for r in rows]
    print(f"built {len(rows)} rows; mean |dJ|={sum(dj)/len(dj):.4f}, max={max(dj):.4f}")
    print(f"wrote {OUT_REG} and {OUT_PT}")


if __name__ == "__main__":
    main()
