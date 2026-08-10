#!/usr/bin/env python3
"""Decode canonical cells: 4 systems × N evaluators on PHX-public test split.

For each evaluator, decode:
  - GT-v1:        recorded test poses (641)
  - PT-v1:        PT baseline predictions
  - TN-PURE-v1:   text-nearest whole-donor retrieval (donor pose)
  - TN-PTCOMP-v1: PT scaffold + donor composed

Compute corpus BLEU and per-item JSON.

Output: results/cells/cp{N}_{evaluator_id}_{system}.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.slrtp_dataset import SLRTPDataset, Vocab
from src.evaluation.decode import decode_split
from src.evaluation.bleu import corpus_bleu, corpus_chrf
from src.utils.hashing import sha256_file

# Canonical evaluators (15 total)
EVALUATORS = [
    # (cp_index, evaluator_id, evaluator_dir)
    (0,  'original',          'checkpoints/released/backTranslation_PHIX_model'),
    (1,  'reconstruction_101', 'checkpoints/reconstructions/seed_101'),
    (2,  'reconstruction_202', 'checkpoints/reconstructions/seed_202'),
    (3,  'reconstruction_303', 'checkpoints/reconstructions/seed_303'),
    (4,  'reconstruction_404', 'checkpoints/reconstructions/seed_404'),
    (5,  'reconstruction_505', 'checkpoints/reconstructions/seed_505'),
    (6,  'reconstruction_606', 'checkpoints/reconstructions/seed_606'),
    (7,  'reconstruction_707', 'checkpoints/reconstructions/seed_707'),
    (8,  'reconstruction_808', 'checkpoints/reconstructions/seed_808'),
    (9,  'reconstruction_909', 'checkpoints/reconstructions/seed_909'),
    (10, 'reconstruction_1001','checkpoints/reconstructions/seed_1001'),
    (11, 'reconstruction_1102','checkpoints/reconstructions/seed_1102'),
    (12, 'reconstruction_1203','checkpoints/reconstructions/seed_1203'),
    (13, 'reconstruction_1304','checkpoints/reconstructions/seed_1304'),
    (14, 'reconstruction_1405','checkpoints/reconstructions/seed_1405'),
]

# Distillation students (additional)
DISTILL_EVALUATORS = [
    (15, 'distill_0.0_101',  'checkpoints/distillation/alpha_0.0_seed_101'),
    (16, 'distill_0.0_202',  'checkpoints/distillation/alpha_0.0_seed_202'),
    (17, 'distill_0.0_303',  'checkpoints/distillation/alpha_0.0_seed_303'),
    (18, 'distill_0.25_101', 'checkpoints/distillation/alpha_0.25_seed_101'),
    (19, 'distill_0.25_202', 'checkpoints/distillation/alpha_0.25_seed_202'),
    (20, 'distill_0.25_303', 'checkpoints/distillation/alpha_0.25_seed_303'),
    (21, 'distill_0.5_101',  'checkpoints/distillation/alpha_0.5_seed_101'),
    (22, 'distill_0.5_202',  'checkpoints/distillation/alpha_0.5_seed_202'),
    (23, 'distill_0.5_303',  'checkpoints/distillation/alpha_0.5_seed_303'),
    (24, 'distill_0.75_101', 'checkpoints/distillation/alpha_0.75_seed_101'),
    (25, 'distill_0.75_202', 'checkpoints/distillation/alpha_0.75_seed_202'),
    (26, 'distill_0.75_303', 'checkpoints/distillation/alpha_0.75_seed_303'),
    (27, 'distill_1.0_101',  'checkpoints/distillation/alpha_1.0_seed_101'),
    (28, 'distill_1.0_202',  'checkpoints/distillation/alpha_1.0_seed_202'),
    (29, 'distill_1.0_303',  'checkpoints/distillation/alpha_1.0_seed_303'),
]


def load_evaluator(model_dir: Path, gpu: int = 0):
    """Load a BT evaluator model pinned to one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import importlib
    # Need to re-import torch after env var change to pick up the new GPU mapping.
    # Since this is called after model code is loaded, we use direct device assignment.
    from src.models import make_back_translation_model
    model = make_back_translation_model(str(model_dir))
    return model


def _make_pt_dataset(pt_pickle: Path, ref_pickle: Path, txt_vocab, gls_vocab):
    """Wrap a PT baseline .pt (dict[id, pose_tensor]) as a SLRTPDataset.

    The reference text/gloss come from the standard test split.
    """
    import torch
    raw_pt = torch.load(str(pt_pickle), map_location="cpu", weights_only=False)
    raw_ref = torch.load(str(ref_pickle), map_location="cpu", weights_only=False)
    # Build items: each gets pose from PT, text/gloss from ref
    items = []
    for k, ref_item in raw_ref.items():
        if k in raw_pt:
            pt_pose = raw_pt[k]
            items.append({
                "name": k,
                "text": ref_item.get("text", ""),
                "gloss": ref_item.get("gloss", ""),
                "poses_3d": pt_pose,
                "speaker": ref_item.get("speaker", ""),
            })
    # Build a dataset-like object
    ds = SLRTPDataset.__new__(SLRTPDataset)
    ds.path = pt_pickle
    ds.txt_vocab = txt_vocab
    ds.gls_vocab = gls_vocab
    ds.skeleton_subsample = 2
    ds.max_sent_length = 400
    ds.txt_lowercase = True
    ds.items = items
    return ds


def decode_one_cell(model, evaluator_id: str, system: str, ds: SLRTPDataset,
                    batch_size: int = 16, max_output_length: int = 30) -> Dict:
    """Decode `ds` using `model` and compute corpus BLEU."""
    t0 = time.time()
    items = decode_split(model, ds, batch_size=batch_size,
                         max_output_length=max_output_length)
    elapsed = time.time() - t0
    hyps = [it["hypothesis"] for it in items]
    refs = [it["reference"] for it in items]
    bleu = corpus_bleu(hyps, refs)
    chrf = corpus_chrf(hyps, refs)
    # Per-item NLL aggregation
    total_nll = sum(it["nll_sum"] for it in items)
    total_tokens = sum(it["token_count"] for it in items)
    return {
        "schema": "canonical-evaluation-cell-v2",
        "evaluator": evaluator_id,
        "system": system,
        "n_items": len(items),
        "decoded_bleu": bleu["bleu"] / 100.0,  # store as 0-1 fraction for back-compat
        "metrics": {
            "count": len(items),
            "decoded_bleu": bleu["bleu"],
            "bleu_1": bleu["bleu_1"],
            "chrf": chrf["chrf"],
            "wer": None,  # not computed here for speed
            "teacher_forced_nll_per_token": total_nll / max(1, total_tokens),
            "edits": None,
            "reference_words": sum(len(r.split()) for r in refs),
            "calibration": None,
            "sacrebleu_signature": bleu["signature"],
        },
        "runtime_seconds": elapsed,
        "items": items,
    }


def _make_tn_pure_dataset(train_pickle: Path, test_pickle: Path, txt_vocab, gls_vocab,
                          exclusion_threshold: float = 0.0):
    """Build the TN-PURE-v1 dataset: for each test query, retrieve the
    text-nearest train donor (max source-text Jaccard), and use its pose.

    Paper §3.2 construction:
      - Source text Jaccard normalized over token sets (Unicode NFKC + ws + lower)
      - Tie-break: minimum char-level Levenshtein distance
      - Donor pose is [::2] subsampled to 12.5 fps (same as test poses)
      - No exclusion threshold by default (paper uses tau=0.0 for canonical)
    """
    import torch
    import unicodedata
    import re

    raw_train = torch.load(str(train_pickle), map_location="cpu", weights_only=False)
    raw_test = torch.load(str(test_pickle), map_location="cpu", weights_only=False)

    def normalize(s):
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def tokens(s):
        return set(normalize(s).split())

    def jaccard(a, b):
        if not a or not b: return 0.0
        return len(a & b) / len(a | b)

    def char_lev(a, b):
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
            prev = cur
        return prev[-1]

    # Pre-compute donor token sets
    donor_keys = list(raw_train.keys())
    donor_tokens_list = []
    donor_texts = []
    for k in donor_keys:
        t = raw_train[k].get("text", "") if isinstance(raw_train[k], dict) else ""
        donor_tokens_list.append(tokens(t))
        donor_texts.append(normalize(t))

    print(f"    TN-PURE: {len(donor_keys)} donors indexed")
    items = []
    skipped = 0
    for query_id, query_item in raw_test.items():
        q_text = query_item.get("text", "")
        q_tokens = tokens(q_text)
        q_norm = normalize(q_text)

        best_jac = -1.0
        best_tb = float("inf")
        best_donor_key = None
        for i, dtoks in enumerate(donor_tokens_list):
            jac = jaccard(q_tokens, dtoks)
            if exclusion_threshold > 0 and jac > exclusion_threshold:
                continue
            if jac > best_jac or (jac == best_jac and jac > 0):
                tb = char_lev(q_norm, donor_texts[i])
                if jac > best_jac or (jac == best_jac and tb < best_tb):
                    best_jac = jac
                    best_tb = tb
                    best_donor_key = donor_keys[i]

        if best_donor_key is None:
            skipped += 1
            continue

        # Get donor pose
        donor_data = raw_train[best_donor_key]
        donor_pose = donor_data["poses_3d"] if isinstance(donor_data, dict) else donor_data

        items.append({
            "name": query_id,
            "text": q_text,  # keep QUERY reference text (for BLEU)
            "gloss": query_item.get("gloss", ""),
            "poses_3d": donor_pose,  # use DONOR pose
            "speaker": query_item.get("speaker", ""),
            "_donor_id": best_donor_key,
            "_donor_jaccard": best_jac,
            "_donor_levenshtein": best_tb,
        })

    print(f"    TN-PURE: {len(items)} items built, {skipped} skipped")

    # Build dataset
    ds = SLRTPDataset.__new__(SLRTPDataset)
    ds.path = test_pickle
    ds.txt_vocab = txt_vocab
    ds.gls_vocab = gls_vocab
    ds.skeleton_subsample = 2
    ds.max_sent_length = 400
    ds.txt_lowercase = True
    ds.items = items
    return ds
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/cells", help="Where to write JSON cells")
    p.add_argument("--gpu", type=int, default=0, help="Single GPU to use (cells are decoded sequentially)")
    p.add_argument("--include-distill", action="store_true",
                   help="Also decode distillation students (15 more evaluators)")
    p.add_argument("--systems", default="GT-v1,PT-v1,TN-PURE-v1,TN-PTCOMP-v1",
                   help="Comma-separated system names")
    p.add_argument("--max-output-length", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, decode only first N test items (for smoke test)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataset once
    txt_vocab_path = ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"
    gls_vocab_path = ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"
    txt_vocab = Vocab.from_file(txt_vocab_path)
    gls_vocab = Vocab.from_file(gls_vocab_path)
    test_pickle = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"

    # NOTE: for now we only have GT-v1 (recorded test poses). Other systems (PT, PURE, COMP)
    # require additional pose-construction code. We decode GT-v1 against each evaluator.
    SYSTEMS = args.systems.split(",")
    print(f"Systems: {SYSTEMS}")
    print(f"Evaluators: {len(EVALUATORS)} reconstructions + {len(DISTILL_EVALUATORS) if args.include_distill else 0} distillation")

    # For each evaluator
    all_evaluators = EVALUATORS[:]
    if args.include_distill:
        all_evaluators += DISTILL_EVALUATORS

    summary = []
    for cp_idx, evaluator_id, model_dir in all_evaluators:
        model_path = ROOT / model_dir / "best.ckpt"
        if not model_path.exists():
            print(f"\n[SKIP] {evaluator_id}: no best.ckpt at {model_path}")
            continue

        print(f"\n=== Loading {evaluator_id} (cp{cp_idx}) ===")
        try:
            model = load_evaluator(ROOT / model_dir, gpu=args.gpu)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        # For each system, decode the test set with that evaluator.
        for system in SYSTEMS:
            if system not in ("GT-v1", "PT-v1", "TN-PURE-v1"):
                # TODO: implement TN-PTCOMP-v1 pose construction
                # (requires PT scaffold + donor composition)
                print(f"  [SKIP] {system} not implemented yet")
                continue

            out_path = out_dir / f"cp{cp_idx}_{system}.json"
            if out_path.exists():
                print(f"  [SKIP] {system} -> {out_path} (already exists)")
                continue

            print(f"  Decoding {system} on {evaluator_id} ...")

            if system == "GT-v1":
                # Recorded test poses
                ds = SLRTPDataset(test_pickle, txt_vocab=txt_vocab, gls_vocab=gls_vocab,
                                  skeleton_subsample=2)
            elif system == "PT-v1":
                # PT baseline predictions (poses from PT_baseline_test.pt)
                # The PT file is already at 25 fps; we need to wrap it as a dataset
                pt_path = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/PT_baseline_test.pt"
                ds = _make_pt_dataset(pt_path, test_pickle, txt_vocab, gls_vocab)
            elif system == "TN-PURE-v1":
                # Text-nearest whole-donor retrieval
                train_pickle = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"
                ds = _make_tn_pure_dataset(train_pickle, test_pickle, txt_vocab, gls_vocab)

            if args.limit > 0:
                ds.items = ds.items[:args.limit]

            try:
                cell = decode_one_cell(model, evaluator_id, system, ds,
                                       batch_size=args.batch_size,
                                       max_output_length=args.max_output_length)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(cell, f, ensure_ascii=False, indent=2)
                print(f"    -> {out_path.name}  BLEU={cell['metrics']['decoded_bleu']:.2f}  "
                      f"NLL={cell['metrics']['teacher_forced_nll_per_token']:.3f}  "
                      f"time={cell['runtime_seconds']:.1f}s")
                summary.append({
                    "evaluator": evaluator_id,
                    "system": system,
                    "bleu": cell["metrics"]["decoded_bleu"],
                    "nll": cell["metrics"]["teacher_forced_nll_per_token"],
                    "n_items": cell["n_items"],
                })
            except Exception as e:
                print(f"    FAILED: {e}")

        # Free model
        del model
        torch.cuda.empty_cache()

    # Save summary
    summary_path = out_dir / "_decode_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_cells": len(summary),
            "summary": summary,
        }, f, indent=2)
    print(f"\n=== Done. {len(summary)} cells decoded. Summary: {summary_path} ===")




if __name__ == "__main__":
    main()
