#!/usr/bin/env python3
"""Input-side pose permutation test [E:E0006].

The label-permutation test (shuffling reference labels AFTER decoding) confirms
output-reference alignment but cannot exclude ID/cache leakage: a hypothetical
ID-based or cache-based retrieval would also produce low BLEU when labels are
shuffled.

This test shuffles POSE TENSORS before decoding while keeping item IDs and text
metadata fixed. If outputs follow IDs/cache, BLEU stays high. If outputs follow
pose content, BLEU collapses.

Additional controls:
  - Zero poses: replace all pose tensors with zeros
  - Random noise poses: replace with Gaussian noise
  - Same pose for all items: replace every item with the first item's pose
  - Batch order change: shuffle batch order, check per-item determinism
"""
import json, sys, os, time
from pathlib import Path
import numpy as np
import torch
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
MODEL_DIR = ROOT / "checkpoints/released/backTranslation_PHIX_model"
OUT = ROOT / "results/input_permutation.json"
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def compute_em(hyps, refs):
    return sum(1 for h, r in zip(hyps, refs)
               if h.strip().lower() == r.strip().lower()) / len(hyps)


def decode_and_score(model, poses, ids, refs, label, subsample=2):
    """Decode poses and score against refs."""
    # Apply subsampling
    poses_sub = [p[::subsample] if isinstance(p, torch.Tensor)
                 else torch.as_tensor(np.asarray(p, dtype=np.float32))[::subsample]
                 for p in poses]
    t0 = time.time()
    hyps = back_translate(model, poses_sub)
    elapsed = time.time() - t0
    bleu = BLEU.corpus_score(hyps, [refs]).score
    em = compute_em(hyps, refs)
    print(f"  {label}: BLEU={bleu:.2f}, EM={em:.4f} ({em*100:.1f}%), "
          f"time={elapsed:.1f}s", flush=True)
    # Save first 5 hypothesis-reference pairs for inspection
    sample = [{"id": ids[i], "hyp": hyps[i][:100], "ref": refs[i][:100]}
              for i in range(min(5, len(hyps)))]
    return {"label": label, "n": len(hyps), "bleu": bleu, "em": em,
            "time_s": elapsed, "sample": sample}


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    print(f"Using GPU {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    N = 200  # number of items for the test

    print("Loading model...", flush=True)
    model = make_back_translation_model(str(MODEL_DIR))

    print(f"Loading first {N} training items...", flush=True)
    train_items = load_pickle(DATA_DIR / "train.pt")[:N]

    ids = [it["name"] for it in train_items]
    refs = [it["text"] for it in train_items]
    poses_orig = []
    for it in train_items:
        p = it["poses_3d"]
        if not isinstance(p, torch.Tensor):
            p = torch.as_tensor(np.asarray(p, dtype=np.float32))
        poses_orig.append(p)

    results = {"n_items": N, "tests": []}

    # 1. Original (correct pairing)
    print("\n=== Test 1: Original pairing ===", flush=True)
    r = decode_and_score(model, poses_orig, ids, refs, "original")
    results["tests"].append(r)
    orig_bleu = r["bleu"]
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # 2. Shuffle pose tensors (keeping IDs/refs fixed)
    print("\n=== Test 2: Shuffled poses (input-side permutation) ===", flush=True)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    poses_shuffled = [poses_orig[perm[i]] for i in range(N)]
    r = decode_and_score(model, poses_shuffled, ids, refs, "shuffled_poses")
    r["perm"] = perm.tolist()
    results["tests"].append(r)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # 3. Zero poses
    print("\n=== Test 3: Zero poses ===", flush=True)
    zero_pose = torch.zeros_like(poses_orig[0])
    poses_zero = [zero_pose.clone() for _ in range(N)]
    r = decode_and_score(model, poses_zero, ids, refs, "zero_poses")
    results["tests"].append(r)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # 4. Random noise poses
    print("\n=== Test 4: Random noise poses ===", flush=True)
    poses_noise = [torch.randn_like(poses_orig[0]) for _ in range(N)]
    r = decode_and_score(model, poses_noise, ids, refs, "noise_poses")
    results["tests"].append(r)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # 5. Same pose for all items
    print("\n=== Test 5: Same pose (first item) for all ===", flush=True)
    same_pose = poses_orig[0]
    poses_same = [same_pose.clone() for _ in range(N)]
    r = decode_and_score(model, poses_same, ids, refs, "same_pose")
    results["tests"].append(r)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # 6. Batch order change (verify per-item determinism)
    print("\n=== Test 6: Original pairing, reversed order ===", flush=True)
    rev_idx = list(range(N - 1, -1, -1))
    poses_rev = [poses_orig[i] for i in rev_idx]
    ids_rev = [ids[i] for i in rev_idx]
    refs_rev = [refs[i] for i in rev_idx]
    r = decode_and_score(model, poses_rev, ids_rev, refs_rev, "reversed_order")
    results["tests"].append(r)
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Test':<25} {'BLEU':<10} {'EM':<10} {'Δ BLEU vs orig':<15}")
    for t in results["tests"]:
        delta = t["bleu"] - orig_bleu
        print(f"{t['label']:<25} {t['bleu']:<10.2f} {t['em']:<10.4f} {delta:<+15.2f}")

    print(f"\nInterpretation: If BLEU collapses under shuffled/same/zero poses,")
    print(f"outputs follow pose content (excludes ID/cache leakage).")
    print(f"If BLEU stays high, outputs follow IDs/cache (not poses).")

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
