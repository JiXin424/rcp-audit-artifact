#!/usr/bin/env python3
"""Rebuild canonical TN-PURE-v1 donor registry and decode all 30 checkpoints.

Reproduces the 30-checkpoint beam-3 panel in results/gap_43_canonical_beam3.json.

Algorithm (matches paper §2.2):
  1. Unicode NFKC + whitespace + lowercase normalization (src.evaluation.retrieval.normalize_text)
  2. Token-set Jaccard similarity (jaccard)
  3. Character-level Levenshtein tie-breaking (char_levenshtein)
  4. SHA-256 hash final tie-break (build_text_nearest_registry, tie_break="levenshtein")

Evaluators decoded (30 total):
  - released: checkpoints/released/backTranslation_PHIX_model
  - 14 reconstructions: checkpoints/reconstructions/seed_{101..1405}
  - 15 distillation: checkpoints/distillation/alpha_{0.0,0.25,0.5,0.75,1.0}_seed_{101,202,303}

Output: results/gap_43_canonical_beam3.json
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
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
OUT = ROOT / "results/gap_43_canonical_beam3.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

CHECKPOINTS = OrderedDict([
    # Released evaluator
    ("released", ROOT / "checkpoints/released/backTranslation_PHIX_model"),
])
# 14 reconstructions
for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405]:
    CHECKPOINTS[f"reco_{seed}"] = ROOT / f"checkpoints/reconstructions/seed_{seed}"
# 15 distillation
for alpha in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
    for seed in ["101", "202", "303"]:
        CHECKPOINTS[f"distill_a{alpha}_{seed}"] = ROOT / f"checkpoints/distillation/alpha_{alpha}_seed_{seed}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    print("Loading data...", flush=True)
    test_items = load_pickle(DATA / "test.pt")
    train_items = load_pickle(DATA / "train.pt")
    test_ids = [it["name"] for it in test_items]
    refs = [it["text"] for it in test_items]

    # ---- Build canonical donor registry (NFKC + Levenshtein + SHA-256) ----
    print("Building canonical TN-PURE-v1 donor registry "
          "(NFKC + Levenshtein + SHA-256)...", flush=True)
    t0 = time.time()
    queries = [{"id": it["name"], "text": it["text"]} for it in test_items]
    donors = [{"id": it["name"], "text": it["text"]} for it in train_items]
    registry = build_text_nearest_registry(queries, donors, tie_break="levenshtein")
    print(f"  Registry: {len(registry)} entries in {time.time() - t0:.1f}s", flush=True)

    # ---- Build poses ----
    train_by_id = {it["name"]: it for it in train_items}
    gt_poses = []
    pure_poses = []
    for item in test_items:
        gp = item["poses_3d"]
        gp = gp[::2] if isinstance(gp, torch.Tensor) else \
            torch.as_tensor(np.asarray(gp, dtype=np.float32))[::2]
        gt_poses.append(gp)

        donor = train_by_id[registry[item["name"]]["donor_id"]]
        dp = donor["poses_3d"]
        dp = dp[::2] if isinstance(dp, torch.Tensor) else \
            torch.as_tensor(np.asarray(dp, dtype=np.float32))[::2]
        pure_poses.append(dp)

    # Record donor IDs for reproducibility
    donor_ids = {qid: registry[qid]["donor_id"] for qid in test_ids}

    # ---- Decode all 30 checkpoints ----
    results = OrderedDict()
    n_skipped = 0
    for name, ckpt_dir in CHECKPOINTS.items():
        ckpt_path = ckpt_dir / "best.ckpt"
        if not ckpt_path.exists():
            print(f"SKIP {name}: no {ckpt_path}", flush=True)
            n_skipped += 1
            continue
        model = make_back_translation_model(str(ckpt_dir))
        t0 = time.time()
        gh = back_translate(model, gt_poses)
        ph = back_translate(model, pure_poses)
        elapsed = time.time() - t0
        gb = BLEU.corpus_score(gh, [refs]).score
        pb = BLEU.corpus_score(ph, [refs]).score
        results[name] = {"gt_bleu": gb, "pure_bleu": pb, "gap": pb - gb,
                          "time_s": elapsed}
        print(f"  {name}: GT={gb:.2f} PURE={pb:.2f} gap={pb - gb:+.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- Summary ----
    non_rel = {k: v for k, v in results.items() if k != "released"}
    gaps = [v["gap"] for v in non_rel.values()]
    summary = {
        "note": "All 30 checkpoints decoded with a single canonical TN-PURE-v1 "
                "donor registry (NFKC + Levenshtein + SHA-256; see paper §2.2).",
        "n_total": len(results) + n_skipped,
        "n_decoded": len(results),
        "n_skipped": n_skipped,
        "algorithm": "NFKC + token-set Jaccard + char-Levenshtein + SHA-256 hash",
        "released": results.get("released", {}),
        "non_released_gap_range": [min(gaps), max(gaps)] if gaps else None,
        "results": results,
    }
    OUT.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nDecoded: {len(results)}/{len(CHECKPOINTS)} checkpoints "
          f"({n_skipped} skipped)", flush=True)
    if results:
        r = results["released"]
        print(f"Released: GT={r['gt_bleu']:.2f} PURE={r['pure_bleu']:.2f} "
              f"gap={r['gap']:+.2f}", flush=True)
    if gaps:
        print(f"Non-released gap range: [{min(gaps):.2f}, {max(gaps):.2f}]",
              flush=True)
    print(f"Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
