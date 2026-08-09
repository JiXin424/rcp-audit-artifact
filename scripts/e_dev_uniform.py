#!/usr/bin/env python3
"""Uniform-protocol dev-set decode (reviewer C1 fix).

Decodes the FULL 515-item dev split under the identical protocol used by
e_full_readout.py (beam=3, alpha=-1, [::2] subsample, sacreBLEU 13a/exp/
effective_order=False), plus official normalized WER (jiwer 3.1.0), for any
checkpoint passed on the command line. This closes the protocol gap between
the training-log validation values (13.38 for the released evaluator) and
the uniform full-pool free-decode protocol.

Usage:
  python scripts/e_dev_uniform.py --gpu 0 --ckpt-dir PATH [--ckpt-dir PATH ...]

Output: results/dev_uniform/<ckpt_id>.json (per-item hypotheses included)
"""
from __future__ import annotations

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
from src.evaluation.bleu import corpus_wer

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT_DIR = ROOT / "results/dev_uniform"


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

    dev_items = load_pickle(DATA_DIR / "dev.pt")
    print(f"dev split: {len(dev_items)} items", flush=True)

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
        ids, hyps, refs = decode_split(model, dev_items)
        bleu = BLEU.corpus_score(hyps, [refs]).score
        em = compute_em(hyps, refs)
        wer = corpus_wer(hyps, refs)["wer"]
        record = {
            "schema": "dev-uniform-v1",
            "ckpt_dir": str(ckpt_dir),
            "beam_size": model.beam_size,
            "beam_alpha": model.beam_alpha,
            "protocol": "dev full-pool, beam=3, alpha=-1, [::2] subsample, "
                        "sacrebleu 13a/exp/effective_order=False, "
                        "official jiwer 3.1.0 normalized WER",
            "n": len(ids),
            "bleu": bleu,
            "em": em,
            "wer": wer,
            "n_empty": sum(1 for h in hyps if not h.strip()),
            "per_item": [
                {"id": i, "hyp": h, "ref": r}
                for i, h, r in zip(ids, hyps, refs)
            ],
        }
        out_file.write_text(json.dumps(record, ensure_ascii=False))
        print(f"[done] {ckpt_id}: dev BLEU={bleu:.2f} EM={em*100:.1f}% "
              f"WER={wer:.2f} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
