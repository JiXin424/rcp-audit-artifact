#!/usr/bin/env python3
"""Experiment A1: released-checkpoint local weight perturbation [reviewer R2-1].

The reviewer's competence confound (released dev 13.38 vs reconstructions ≤10.9)
means our 14-seed non-reproduction cannot separate checkpoint identity from
competence. This experiment operates AT competence parity by starting FROM the
released checkpoint and adding i.i.d. Gaussian weight noise at increasing scale
sigma. Small sigma perturbs identity while preserving competence; we trace the
PURE--REC gap, dev BLEU-4, and (at selected sigma) the full-pool training-pool
readout as functions of sigma.

Output: results/released_perturbation.json
"""
import argparse
import copy
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
MODEL_DIR = ROOT / "checkpoints/released/backTranslation_PHIX_model"
REGISTRY = ROOT / "results/gap_43_canonical_beam3_items/donor_registry.jsonl"
OUT = ROOT / "results/released_perturbation.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

SIGMAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
SEEDS = [101, 202]
READOUT_SIGMAS = [0.0]  # only baseline gets full-pool readout


def em(hyps, refs):
    return sum(1 for h, r in zip(hyps, refs)
               if h.strip().lower() == r.strip().lower()) / max(1, len(hyps))


def pose_of(item):
    p = item["poses_3d"]
    p = p if isinstance(p, torch.Tensor) else torch.as_tensor(np.asarray(p, dtype=np.float32))
    return p[::2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test_items = load_pickle(DATA / "test.pt")
    dev_items = load_pickle(DATA / "dev.pt")
    train_items = load_pickle(DATA / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}
    registry = {}
    for line in open(REGISTRY):
        r = json.loads(line)
        registry[r["query_id"]] = r["donor_id"]

    gt_poses = [pose_of(it) for it in test_items]
    refs = [it["text"] for it in test_items]
    pure_poses = [pose_of(train_by_id[registry[it["name"]]]) for it in test_items]
    dev_poses = [pose_of(it) for it in dev_items]
    dev_refs = [it["text"] for it in dev_items]
    train_poses = [pose_of(it) for it in train_items]
    train_refs = [it["text"] for it in train_items]

    print(f"loaded: test {len(gt_poses)}, dev {len(dev_poses)}, "
          f"train {len(train_poses)}", flush=True)

    print("loading released model...", flush=True)
    model = make_back_translation_model(str(MODEL_DIR))
    # Only perturb parameters (weights/biases), NOT buffers (running_var,
    # running_mean, num_batches_tracked for BatchNorm layers). Perturbing
    # running_var (must stay positive) collapses the model with NaNs.
    param_keys = {n for n, _ in model.named_parameters()}
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in param_keys}

    def decode_gap_dev():
        gt_h = back_translate(model, gt_poses)
        pure_h = back_translate(model, pure_poses)
        dev_h = back_translate(model, dev_poses)
        return {
            "gt_bleu": BLEU.corpus_score(gt_h, [refs]).score,
            "pure_bleu": BLEU.corpus_score(pure_h, [refs]).score,
            "dev_bleu": BLEU.corpus_score(dev_h, [dev_refs]).score,
        }

    out = {"schema": "released-perturbation-A1-v1",
           "sigmas": SIGMAS, "seeds": SEEDS, "records": []}

    for sigma in SIGMAS:
        seeds_here = [0] if sigma == 0.0 else SEEDS
        for sd in seeds_here:
            if sigma == 0.0:
                # baseline: load the original param state (idempotent)
                model.load_state_dict(base_state, strict=False)
            else:
                state = {k: v.clone() for k, v in base_state.items()}
                g = torch.Generator(device="cpu").manual_seed(sd)
                for k, v in state.items():
                    noise = torch.randn(v.shape, generator=g, dtype=v.dtype)
                    state[k] = v + noise.to(v.device) * sigma
                model.load_state_dict(state, strict=False)
            model.eval()
            t0 = time.time()
            rec = decode_gap_dev()
            rec["gap"] = rec["pure_bleu"] - rec["gt_bleu"]
            rec["sigma"] = sigma
            rec["seed"] = sd
            rec["elapsed_s"] = time.time() - t0
            if sigma in READOUT_SIGMAS:
                tr_h = back_translate(model, train_poses)
                rec["train_bleu"] = BLEU.corpus_score(tr_h, [train_refs]).score
                rec["train_em"] = em(tr_h, train_refs)
            out["records"].append(rec)
            print(f"sigma={sigma:.0e} seed={sd}: gap={rec['gap']:+.2f} "
                  f"dev={rec['dev_bleu']:.2f} gt={rec['gt_bleu']:.2f} "
                  f"pure={rec['pure_bleu']:.2f}"
                  + (f" train={rec.get('train_bleu',float('nan')):.2f}"
                     f" EM={rec.get('train_em',float('nan'))*100:.1f}%"
                     if 'train_bleu' in rec else "")
                  + f"  ({rec['elapsed_s']:.0f}s)", flush=True)
            OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
