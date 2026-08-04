#!/usr/bin/env python3
"""Rebuild canonical TN-PURE-v1 donor registry and decode all 30 checkpoints.

Reproduces results/gap_43_canonical_beam3.json using the deterministic algorithm
from paper §2.2: NFKC normalization, token-set Jaccard, character-level Levenshtein
tie-break, SHA-256 hash final tie-break, exact-normalized-text exclusion.

Evaluators (30 total):
  - released: checkpoints/released/backTranslation_PHIX_model
  - 14 reconstructions: seed 101-606 (primary) + 707-1405 (extension)
  - 15 distillation: alpha {0.0,0.25,0.5,0.75,1.0} x seed {101,202,303}

Output (flat structure, one key per evaluator):
  results/gap_43_canonical_beam3.json
    { "<name>": { "gt_bleu": ..., "pure_bleu": ..., "gap": ...,
      "gt_items": [{"id":...,"hypothesis":...,"reference":...}, ...],
      "pure_items": [{"id":...,"hypothesis":...,"reference":...}, ...],
      "checkpoint_sha256": "...", "donor_registry_sha256": "..." },
      "_meta": {...} }
"""
import argparse
import hashlib
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

CHECKPOINTS = OrderedDict()
CHECKPOINTS["released"] = ROOT / "checkpoints/released/backTranslation_PHIX_model"
for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405]:
    CHECKPOINTS[f"reco_{seed}"] = ROOT / f"checkpoints/reconstructions/seed_{seed}"
for alpha in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
    for seed in ["101", "202", "303"]:
        CHECKPOINTS[f"distill_a{alpha}_{seed}"] = ROOT / f"checkpoints/distillation/alpha_{alpha}_seed_{seed}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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

    # ---- Build canonical donor registry ----
    print("Building canonical TN-PURE-v1 donor registry "
          "(NFKC + excl exact-norm-text + Levenshtein + SHA-256)...", flush=True)
    t0 = time.time()
    queries = [{"id": it["name"], "text": it["text"]} for it in test_items]
    donors = [{"id": it["name"], "text": it["text"]} for it in train_items]
    registry = build_text_nearest_registry(queries, donors, tie_break="levenshtein")
    print(f"  {len(registry)} entries in {time.time() - t0:.1f}s", flush=True)

    # Save donor registry for reproducibility
    donor_registry_json = json.dumps({qid: registry[qid] for qid in test_ids},
                                     indent=1, ensure_ascii=False, sort_keys=True)
    donor_registry_sha256 = hashlib.sha256(donor_registry_json.encode()).hexdigest()

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

    # ---- Decode all 30 checkpoints ----
    results = OrderedDict()
    n_skipped = 0
    for name, ckpt_dir in CHECKPOINTS.items():
        ckpt_path = ckpt_dir / "best.ckpt"
        if not ckpt_path.exists():
            print(f"SKIP {name}: no {ckpt_path}", flush=True)
            n_skipped += 1
            continue
        ckpt_hash = sha256_file(str(ckpt_path))
        model = make_back_translation_model(str(ckpt_dir))
        t0 = time.time()

        # Decode GT
        gt_hyps = back_translate(model, gt_poses)
        # Decode PURE (canonical donors)
        pure_hyps = back_translate(model, pure_poses)
        elapsed = time.time() - t0

        gb = BLEU.corpus_score(gt_hyps, [refs]).score
        pb = BLEU.corpus_score(pure_hyps, [refs]).score

        # Per-item hypotheses
        gt_items = [{"id": tid, "hypothesis": hyp, "reference": ref}
                     for tid, hyp, ref in zip(test_ids, gt_hyps, refs)]
        pure_items = [{"id": tid, "hypothesis": hyp, "reference": ref}
                       for tid, hyp, ref in zip(test_ids, pure_hyps, refs)]
        gt_items_sha = hashlib.sha256(
            json.dumps(gt_items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        pure_items_sha = hashlib.sha256(
            json.dumps(pure_items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

        results[name] = {
            "gt_bleu": gb, "pure_bleu": pb, "gap": pb - gb,
            "checkpoint_sha256": ckpt_hash,
            "donor_registry_sha256": donor_registry_sha256,
            "gt_items_sha256": gt_items_sha,
            "pure_items_sha256": pure_items_sha,
            "gt_items": gt_items,
            "pure_items": pure_items,
            "time_s": elapsed,
        }
        print(f"  {name}: GT={gb:.2f} PURE={pb:.2f} gap={pb - gb:+.2f} "
              f"({elapsed:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- Summary ----
    non_rel = {k: v for k, v in results.items() if k != "released"}
    gaps = [v["gap"] for v in non_rel.values()]

    results["_meta"] = {
        "schema": "canonical-beam3-panel-v2",
        "generated_by": "scripts/build_canonical_panel.py",
        "algorithm": "NFKC normalization + exact-normalized-text exclusion + "
                     "token-set Jaccard + character-level Levenshtein tie-break + "
                     "SHA-256 hash final tie-break (paper §2.2)",
        "n_total_checkpoints_with_model_files": len(CHECKPOINTS),
        "n_decoded": len(results) - 1,  # exclude _meta
        "n_skipped": n_skipped,
        "decoder": "model.run_batch(translation_beam_size=3, "
                   "translation_beam_alpha=-1, translation_max_output_length=30)",
        "sacrebleu_signature": "BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1",
        "fps": "12.5 (skeleton_subsample=2 applied before decoding)",
        "released_gap": results["released"]["gap"],
        "non_released_gap_range": [min(gaps), max(gaps)] if gaps else None,
        "donor_registry_sha256": donor_registry_sha256,
    }
    # Move _meta last
    results.move_to_end("_meta")

    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nDecoded: {len(results) - 1}/{len(CHECKPOINTS)} ({n_skipped} skipped)", flush=True)
    r = results["released"]
    print(f"Released: GT={r['gt_bleu']:.2f} PURE={r['pure_bleu']:.2f} "
          f"gap={r['gap']:+.2f}", flush=True)
    if gaps:
        print(f"Non-released gap range: [{min(gaps):.2f}, {max(gaps):.2f}]",
              flush=True)
    print(f"Donor registry SHA-256: {donor_registry_sha256[:16]}...", flush=True)
    print(f"Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
