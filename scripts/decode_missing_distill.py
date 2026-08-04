#!/usr/bin/env python3
"""Decode the 2 missing distillation seeds (alpha=0.5 seed 303, alpha=0.75 seed 303)
on full PHX-public with beam=3. This eliminates the '2 OOM' caveat."""
import json, sys, os, time, re
from pathlib import Path
import numpy as np
import torch
import sacrebleu

ROOT = Path("/ssd/xkb4/RCP")
sys.path.insert(0, str(ROOT))
from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT = ROOT / "results/missing_distill_beam3.json"
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def normalize(s):
    return set(re.sub(r'\s+', ' ', s.strip().lower()).split())


def build_donor(test_items, train_items):
    """Build text-nearest donor retrieval."""
    train_texts = {it["name"]: it["text"] for it in train_items}
    train_norm = [(it["name"], normalize(it["text"])) for it in train_items]
    donors = []
    for item in test_items:
        q_norm = normalize(item["text"])
        best_jac, best_donor = -1, None
        for did, d_norm in train_norm:
            if item["text"].strip().lower() == train_texts[did].strip().lower():
                continue
            inter = len(q_norm & d_norm)
            union = len(q_norm | d_norm)
            jac = inter / union if union else 0
            if jac > best_jac:
                best_jac = jac
                best_donor = did
        donor_item = next(it for it in train_items if it["name"] == best_donor)
        donors.append(donor_item)
    return donors


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
    print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    test_items = load_pickle(DATA / "test.pt")
    train_items = load_pickle(DATA / "train.pt")
    print(f"Loaded {len(test_items)} test, {len(train_items)} train items", flush=True)

    print("Building donor registry...", flush=True)
    t0 = time.time()
    donors = build_donor(test_items, train_items)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    # Prepare poses (subsampled)
    gt_poses = []
    pure_poses = []
    for item, donor in zip(test_items, donors):
        gp = item["poses_3d"]
        if not isinstance(gp, torch.Tensor):
            gp = torch.as_tensor(np.asarray(gp, dtype=torch.float32))
        gt_poses.append(gp[::2])

        dp = donor["poses_3d"]
        if not isinstance(dp, torch.Tensor):
            dp = torch.as_tensor(np.asarray(dp, dtype=torch.float32))
        pure_poses.append(dp[::2])

    refs = [it["text"] for it in test_items]
    ids = [it["name"] for it in test_items]

    seeds = [
        ("alpha_0.5_seed_303", "checkpoints/distillation/alpha_0.5_seed_303"),
        ("alpha_0.75_seed_303", "checkpoints/distillation/alpha_0.75_seed_303"),
    ]

    results = {}
    for name, ckpt_dir in seeds:
        print(f"\n=== {name} ===", flush=True)
        model = make_back_translation_model(ckpt_dir)

        # GT decode
        print("  Decoding GT...", flush=True)
        t0 = time.time()
        gt_hyps = back_translate(model, gt_poses)
        gt_bleu = BLEU.corpus_score(gt_hyps, [refs]).score
        print(f"  GT BLEU={gt_bleu:.2f} ({time.time()-t0:.1f}s)", flush=True)

        # PURE decode
        print("  Decoding PURE...", flush=True)
        t0 = time.time()
        pure_hyps = back_translate(model, pure_poses)
        pure_bleu = BLEU.corpus_score(pure_hyps, [refs]).score
        gap = pure_bleu - gt_bleu
        print(f"  PURE BLEU={pure_bleu:.2f}, gap={gap:+.2f} ({time.time()-t0:.1f}s)", flush=True)

        results[name] = {
            "gt_bleu": gt_bleu, "pure_bleu": pure_bleu, "gap": gap,
            "ckpt_dir": ckpt_dir,
        }
        del model
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"{name}: GT={r['gt_bleu']:.2f}, PURE={r['pure_bleu']:.2f}, gap={r['gap']:+.2f}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
