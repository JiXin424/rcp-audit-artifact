#!/usr/bin/env python3
"""Uniform full-pool readout for the reconstruction family [reviewer R2-5].

Decodes the FULL 7,060-item training pool, the 515-item dev split, and the
641-item test split for every checkpoint passed on the command line, using an
identical protocol for all checkpoints:

  * beam = 3, length penalty alpha = -1 (model config, identical across all)
  * [::2] subsampling to 12.5 fps at input time (same as training)
  * sacreBLEU corpus BLEU (tokenize=13a, smooth=exp, effective_order=False)
  * exact match: case-insensitive, stripped
  * references: the released train.pt / dev.pt / test.pt texts

This replaces the mixed protocol where the released evaluator's readout was
computed on the full pool but students were measured on a 1,000-item prefix.

Usage:
  python scripts/e_full_readout.py --gpu 0 --ckpt-dir PATH [--ckpt-dir PATH ...]

Output: results/full_readout/<ckpt_id>.json (per-item hypotheses included),
plus a printed summary line per checkpoint.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sacrebleu
from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT_DIR = ROOT / "results/full_readout"


def compute_em(hyps, refs):
    matches = sum(1 for h, r in zip(hyps, refs)
                  if h.strip().lower() == r.strip().lower())
    return matches / len(hyps) if hyps else 0.0


def decode_split(model, items, subsample=2):
    poses, ids, refs = [], [], []
    for item in items:
        pose = item["poses_3d"]
        if not isinstance(pose, torch.Tensor):
            pose = torch.as_tensor(np.asarray(pose, dtype=np.float32))
        if subsample and subsample > 1:
            pose = pose[::subsample]
        poses.append(pose)
        ids.append(item.get("name", ""))
        refs.append(item.get("text", ""))
    hyps = back_translate(model, poses)
    return ids, hyps, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--ckpt-dir", action="append", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "dev", "test"):
        splits[name] = load_pickle(DATA_DIR / f"{name}.pt")

    for ckpt_dir in args.ckpt_dir:
        ckpt_dir = Path(ckpt_dir)
        ckpt_id = ckpt_dir.name
        out_file = OUT_DIR / f"{ckpt_id}.json"
        if out_file.exists():
            print(f"[skip] {ckpt_id} already done", flush=True)
            continue
        t0 = time.time()
        model = make_back_translation_model(str(ckpt_dir))
        assert model.beam_size == 3, f"beam mismatch: {model.beam_size}"
        record = {
            "schema": "full-readout-v1",
            "ckpt_dir": str(ckpt_dir),
            "beam_size": model.beam_size,
            "beam_alpha": model.beam_alpha,
            "protocol": "full-pool, beam=3, alpha=-1, [::2] subsample, "
                        "sacrebleu 13a/exp/effective_order=False",
            "splits": {},
        }
        for split_name, items in splits.items():
            ids, hyps, refs = decode_split(model, items)
            bleu = BLEU.corpus_score(hyps, [refs]).score
            em = compute_em(hyps, refs)
            record["splits"][split_name] = {
                "n": len(items),
                "bleu": bleu,
                "em": em,
                "n_empty": sum(1 for h in hyps if not h.strip()),
                "per_item": [
                    {"id": i, "hyp": h, "ref": r}
                    for i, h, r in zip(ids, hyps, refs)
                ],
            }
            print(f"  {ckpt_id} {split_name}: BLEU={bleu:.2f} "
                  f"EM={em*100:.1f}% n={len(items)}", flush=True)
        record["elapsed_s"] = time.time() - t0
        out_file.write_text(json.dumps(record, ensure_ascii=False))
        print(f"[done] {ckpt_id} in {record['elapsed_s']:.0f}s -> {out_file}",
              flush=True)


if __name__ == "__main__":
    main()
