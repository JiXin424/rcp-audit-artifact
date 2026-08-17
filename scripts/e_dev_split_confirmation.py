#!/usr/bin/env python3
"""Frozen-split confirmation of the headline TN-PURE probe on the DEV split.

The canonical 641-query probe panel (results/gap_43_canonical_beam3_items) was
constructed and explored on the PHOENIX-2014T *test* split; the reviewer
asked for confirmation on a split that was never used for probe exploration.
This script rebuilds the identical TN-PURE-v1 construction (NFKC + token-set
Jaccard + Levenshtein tie-break + SHA-256 final tie-break + exact-normalized-
text exclusion, donors from the train pool) with DEV items as queries, decodes
both REC (recorded dev poses) and PURE (donor poses) under the released
evaluator with the canonical beam-3 protocol, and reports the dev-split gap.

Sanity anchor: the released evaluator's dev REC BLEU should be close to the
13.38 logged in the released validations.txt.

Output:
  results/dev_split_confirmation.json
  results/dev_split_confirmation_items/{gt,pure}_items.json
  results/dev_split_confirmation_items/donor_registry_dev.jsonl
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
from src.evaluation.retrieval import build_text_nearest_registry
from src.models import make_back_translation_model, back_translate

DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT_DIR = ROOT / "results/dev_split_confirmation_items"
OUT_JSON = ROOT / "results/dev_split_confirmation.json"
RELEASED = ROOT / "checkpoints/released/backTranslation_PHIX_model"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    dev_items = load_pickle(DATA / "dev.pt")
    train_items = load_pickle(DATA / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}
    dev_ids = [it["name"] for it in dev_items]
    refs = [it["text"] for it in dev_items]

    print("Building DEV-split TN-PURE-v1 donor registry (identical algorithm)...", flush=True)
    t0 = time.time()
    queries = [{"id": it["name"], "text": it["text"]} for it in dev_items]
    donors = [{"id": it["name"], "text": it["text"]} for it in train_items]
    registry = build_text_nearest_registry(queries, donors, tie_break="levenshtein")
    print(f"  {len(registry)} entries in {time.time() - t0:.1f}s", flush=True)
    registry_path = OUT_DIR / "donor_registry_dev.jsonl"
    with open(registry_path, "w") as rf:
        for qid in dev_ids:
            r = registry[qid]
            rf.write(json.dumps({"query_id": qid, "donor_id": r["donor_id"],
                                 "jaccard": r["jaccard"], "levenshtein": r["levenshtein"],
                                 "sha256_tb": r.get("sha256_tb")}) + "\n")
    registry_sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()

    print("Building poses (12.5fps subsample)...", flush=True)
    gt_poses, pure_poses = [], []
    for item in dev_items:
        gp = item["poses_3d"]
        gp = gp[::2] if isinstance(gp, torch.Tensor) else \
            torch.as_tensor(np.asarray(gp, dtype=torch.float32))[::2]
        gt_poses.append(gp)
        donor = train_by_id[registry[item["name"]]["donor_id"]]
        dp = donor["poses_3d"]
        dp = dp[::2] if isinstance(dp, torch.Tensor) else \
            torch.as_tensor(np.asarray(dp, dtype=torch.float32))[::2]
        pure_poses.append(dp)
    print(f"  {len(gt_poses)} dev REC poses, {len(pure_poses)} PURE poses", flush=True)

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
               for i, h, r in zip(dev_ids, gt_hyps, refs)],
              open(OUT_DIR / "gt_items.json", "w"), indent=1, ensure_ascii=False)
    json.dump([{"id": i, "hypothesis": h, "reference": r}
               for i, h, r in zip(dev_ids, pure_hyps, refs)],
              open(OUT_DIR / "pure_items.json", "w"), indent=1, ensure_ascii=False)

    result = {
        "split": "dev",
        "n_queries": len(dev_ids),
        "evaluator": "released",
        "evaluator_ckpt_sha256": sha256_file(str(RELEASED / "best.ckpt")),
        "donor_registry_sha256": registry_sha,
        "construction": "TN-PURE-v1 (NFKC + Jaccard + Levenshtein tb + SHA-256 tb + exact-norm-text exclusion)",
        "gt_bleu": gt_bleu, "pure_bleu": pure_bleu, "gap": gap,
        "test_split_reference": {"gt_bleu": 12.78, "pure_bleu": 23.02, "gap": 10.24},
        "released_dev_bleu_logged": 13.38,
        "decode_time_s": elapsed,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nDEV-split: REC={gt_bleu:.2f} PURE={pure_bleu:.2f} gap={gap:+.2f} "
          f"(test-split canonical: +10.24)")


if __name__ == "__main__":
    main()
