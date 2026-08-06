#!/usr/bin/env python3
"""Experiment D: epoch-checkpoint decoupling of readout vs gap [reviewer R2-1].

The 27 non-degenerate checkpoints in Sup.~D all pair (low readout, negative
gap) and the released evaluator pairs (high readout, positive gap), leaving
readout and gap perfectly confounded. This script searches the available
intermediate epoch checkpoints (rescue/seed_202_wd0, long_schedule/seed_202)
for a decoupling point --- e.g. a checkpoint with elevated train-pool readout
but gap still ≤ 0, or vice versa --- using the uniform full-7060 protocol.

Output: results/epoch_decouple.json
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

DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
REGISTRY = ROOT / "results/gap_43_canonical_beam3_items/donor_registry.jsonl"
OUT = ROOT / "results/epoch_decouple.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

# (label, model_dir_for_config, ckpt_path)
TARGETS = [
    ("rescue_wd0_best", "checkpoints/rescue/seed_202_wd0",
     "checkpoints/rescue/seed_202_wd0/best.ckpt"),
    ("rescue_wd0_ep25", "checkpoints/rescue/seed_202_wd0",
     "checkpoints/rescue/seed_202_wd0/epoch_0025.ckpt"),
    ("rescue_wd0_ep50", "checkpoints/rescue/seed_202_wd0",
     "checkpoints/rescue/seed_202_wd0/epoch_0050.ckpt"),
    ("longsched_s202_best", "checkpoints/long_schedule/seed_202",
     "checkpoints/long_schedule/seed_202/best.ckpt"),
    ("longsched_s202_ep25", "checkpoints/long_schedule/seed_202",
     "checkpoints/long_schedule/seed_202/epoch_0025.ckpt"),
    ("longsched_s202_ep50", "checkpoints/long_schedule/seed_202",
     "checkpoints/long_schedule/seed_202/epoch_0050.ckpt"),
]


def em(hyps, refs):
    return sum(1 for h, r in zip(hyps, refs)
               if h.strip().lower() == r.strip().lower()) / max(1, len(hyps))


def pose_of(item):
    p = item["poses_3d"]
    p = p if isinstance(p, torch.Tensor) else torch.as_tensor(np.asarray(p, dtype=torch.float32))
    return p[::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test_items = load_pickle(DATA / "test.pt")
    dev_items = load_pickle(DATA / "dev.pt")
    train_items = load_pickle(DATA / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}
    registry = {json.loads(l)["query_id"]: json.loads(l)["donor_id"]
                for l in open(REGISTRY)}

    gt_poses = [pose_of(it) for it in test_items]
    refs = [it["text"] for it in test_items]
    pure_poses = [pose_of(train_by_id[registry[it["name"]]]) for it in test_items]
    dev_poses = [pose_of(it) for it in dev_items]
    dev_refs = [it["text"] for it in dev_items]
    train_poses = [pose_of(it) for it in train_items]
    train_refs = [it["text"] for it in train_items]

    out = {"schema": "epoch-decouple-D-v1", "records": []}

    for label, model_dir, ckpt_path in TARGETS:
        if not Path(ckpt_path).exists():
            print(f"[skip] {label}: {ckpt_path} missing", flush=True)
            continue
        t0 = time.time()
        model = make_back_translation_model(str(ROOT / model_dir))
        alt = torch.load(str(ROOT / ckpt_path), map_location="cpu", weights_only=False)
        sd = alt["model_state"] if "model_state" in alt else alt
        model.load_state_dict(sd)
        model.eval()

        gt_h = back_translate(model, gt_poses)
        pure_h = back_translate(model, pure_poses)
        dev_h = back_translate(model, dev_poses)
        tr_h = back_translate(model, train_poses)

        rec = {
            "label": label,
            "ckpt": ckpt_path,
            "gt_bleu": BLEU.corpus_score(gt_h, [refs]).score,
            "pure_bleu": BLEU.corpus_score(pure_h, [refs]).score,
            "dev_bleu": BLEU.corpus_score(dev_h, [dev_refs]).score,
            "train_bleu": BLEU.corpus_score(tr_h, [train_refs]).score,
            "train_em": em(tr_h, train_refs),
            "elapsed_s": time.time() - t0,
        }
        rec["gap"] = rec["pure_bleu"] - rec["gt_bleu"]
        out["records"].append(rec)
        print(f"{label}: gap={rec['gap']:+.2f} gt={rec['gt_bleu']:.2f} "
              f"pure={rec['pure_bleu']:.2f} dev={rec['dev_bleu']:.2f} "
              f"train={rec['train_bleu']:.2f} EM={rec['train_em']*100:.1f}% "
              f"({rec['elapsed_s']:.0f}s)", flush=True)
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
