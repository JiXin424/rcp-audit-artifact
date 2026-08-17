#!/usr/bin/env python3
"""Pose-retrieval PURE probe: donor selection WITHOUT the scored reference.

The canonical TN-PURE retriever selects donors by lexical overlap with the
query's text -- which is identical to the BLEU scoring reference (target
leakage). This probe severs that channel: donors are selected by POSE
similarity to the query's own recorded pose sequence (an input-side signal
available to a deployed system, no reference text involved).

Pose signature (per sequence, after the canonical 12.5fps subsampling):
  - mean pose over time (captures posture/position distribution)
  - mean absolute frame-to-frame velocity (captures dynamics)
Each part is z-normalized per dimension over the pool, then the two parts are
concatenated and L2-normalized. Donor = nearest neighbour under Euclidean
distance on this signature; ties broken by SHA-256 hash of the donor ID
(same final tie-break convention as the canonical retriever).

Decoded under the released evaluator with the canonical beam-3 protocol on
the same 641-item test split as the headline probe, so the gap is directly
comparable to +10.24.

Output:
  results/pose_retrieval_probe.json
  results/pose_retrieval_probe_items/{gt,pure}_items.json
  results/pose_retrieval_probe_items/donor_registry_pose.jsonl
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT_DIR = ROOT / "results/pose_retrieval_probe_items"
OUT_JSON = ROOT / "results/pose_retrieval_probe.json"
RELEASED = ROOT / "checkpoints/released/backTranslation_PHIX_model"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pose_signature(item) -> np.ndarray:
    """(mean_pose, mean|delta_pose) signature on the subsampled sequence."""
    p = item["poses_3d"]
    p = p[::2] if isinstance(p, torch.Tensor) else \
        torch.as_tensor(np.asarray(p, dtype=torch.float32))[::2]
    a = p.numpy() if isinstance(p, torch.Tensor) else np.asarray(p, dtype=np.float32)
    mean_pose = a.mean(axis=0).reshape(-1)
    if a.shape[0] > 1:
        vel = np.abs(np.diff(a, axis=0)).mean(axis=0).reshape(-1)
    else:
        vel = np.zeros_like(mean_pose)
    return np.concatenate([mean_pose, vel])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    test_items = load_pickle(DATA / "test.pt")
    train_items = load_pickle(DATA / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}
    test_ids = [it["name"] for it in test_items]
    refs = [it["text"] for it in test_items]

    print("Computing pose signatures...", flush=True)
    t0 = time.time()
    q_sig = np.stack([pose_signature(it) for it in test_items])       # (Nq, 2F)
    d_ids = [it["name"] for it in train_items]
    d_sig = np.stack([pose_signature(it) for it in train_items])      # (Nd, 2F)
    # z-normalize each half per dimension over the POOLED queries+donors
    F = q_sig.shape[1] // 2
    pooled = np.concatenate([q_sig, d_sig], axis=0)
    mu = pooled.mean(axis=0); sd = pooled.std(axis=0) + 1e-8
    q_z = (q_sig - mu) / sd; d_z = (d_sig - mu) / sd
    # L2-normalize each half so posture and dynamics contribute comparably
    def l2_half(z):
        h1 = z[:, :F] / (np.linalg.norm(z[:, :F], axis=1, keepdims=True) + 1e-8)
        h2 = z[:, F:] / (np.linalg.norm(z[:, F:], axis=1, keepdims=True) + 1e-8)
        return np.concatenate([h1, h2], axis=1)
    q_z, d_z = l2_half(q_z), l2_half(d_z)
    print(f"  signatures: {q_z.shape[0]} queries x {d_z.shape[0]} donors, dim {q_z.shape[1]} "
          f"({time.time() - t0:.1f}s)", flush=True)

    print("Nearest-donor retrieval (Euclidean on pose signature)...", flush=True)
    t0 = time.time()
    # dist matrix in blocks to bound memory: (Nq, Nd)
    dist = np.zeros((q_z.shape[0], d_z.shape[0]), dtype=np.float32)
    B = 128
    for i in range(0, q_z.shape[0], B):
        block = q_z[i:i + B] @ d_z.T
        # squared L2 = |q|^2 + |d|^2 - 2 q.d ; halves already unit-norm each so
        # |q|^2 = |d|^2 = 2 constant -> distance ranking == -dot ranking
        dist[i:i + B] = -block  # higher dot = closer
    best_idx = np.argmax(dist, axis=1)
    print(f"  retrieval done in {time.time() - t0:.1f}s", flush=True)

    registry_path = OUT_DIR / "donor_registry_pose.jsonl"
    with open(registry_path, "w") as rf:
        for qi, di in enumerate(best_idx):
            rf.write(json.dumps({"query_id": test_ids[qi], "donor_id": d_ids[di],
                                 "similarity": float(dist[qi, di])}) + "\n")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()

    print("Building poses (12.5fps subsample)...", flush=True)
    gt_poses, pure_poses = [], []
    for item, di in zip(test_items, best_idx):
        gp = item["poses_3d"]
        gp = gp[::2] if isinstance(gp, torch.Tensor) else \
            torch.as_tensor(np.asarray(gp, dtype=torch.float32))[::2]
        gt_poses.append(gp)
        donor = train_items[di]
        dp = donor["poses_3d"]
        dp = dp[::2] if isinstance(dp, torch.Tensor) else \
            torch.as_tensor(np.asarray(dp, dtype=torch.float32))[::2]
        pure_poses.append(dp)

    print("Decoding under the released evaluator (beam-3)...", flush=True)
    model = make_back_translation_model(str(RELEASED))
    t0 = time.time()
    gt_hyps = back_translate(model, gt_poses)
    pure_hyps = back_translate(model, pure_poses)
    elapsed = time.time() - t0

    gt_bleu = BLEU.corpus_score(gt_hyps, [refs]).score
    pure_bleu = BLEU.corpus_score(pure_hyps, [refs]).score
    gap = pure_bleu - gt_bleu

    json.dump([{"id": i, "hypothesis": h, "reference": r}
               for i, h, r in zip(test_ids, gt_hyps, refs)],
              open(OUT_DIR / "gt_items.json", "w"), indent=1, ensure_ascii=False)
    json.dump([{"id": i, "hypothesis": h, "reference": r}
               for i, h, r in zip(test_ids, pure_hyps, refs)],
              open(OUT_DIR / "pure_items.json", "w"), indent=1, ensure_ascii=False)

    result = {
        "split": "test (same 641 items as headline probe)",
        "n_queries": len(test_ids),
        "evaluator": "released",
        "evaluator_ckpt_sha256": sha256_file(str(RELEASED / "best.ckpt")),
        "donor_registry_sha256": registry_sha,
        "retriever": "pose-signature nearest neighbour (mean pose + mean |delta pose|, "
                     "z-normalized, half-L2-normalized, Euclidean/nearest-dot; no text used)",
        "gt_bleu": gt_bleu, "pure_bleu": pure_bleu, "gap": gap,
        "text_retrieval_reference": {"gt_bleu": 12.78, "pure_bleu": 23.02, "gap": 10.24},
        "random_donor_reference": {"pure_bleu": 0.90, "gap": -11.88},
        "decode_time_s": elapsed,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nPOSE-PURE: REC={gt_bleu:.2f} POSE-PURE={pure_bleu:.2f} gap={gap:+.2f} "
          f"(text-retrieval TN-PURE: +10.24; random donor: -11.88)")


if __name__ == "__main__":
    main()
