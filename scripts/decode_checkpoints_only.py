#!/usr/bin/env python3
"""Decode a list of checkpoints under the canonical beam-3 protocol.

Reuses the existing donor registry at
results/gap_43_canonical_beam3_items/donor_registry.jsonl (no rebuild).
For each checkpoint, writes <prefix>_gt.json and <prefix>_pure.json per-item
files into results/gap_43_canonical_beam3_items/, overwriting any stale
files from prior protocol variants.

Usage:
    # Decode all 10 step_faithful re-trained seeds across 8 GPUs
    python3 scripts/decode_checkpoints_only.py \\
        --checkpoints sf_1701:checkpoints/step_faithful/seed_1701 \\
                      sf_1702:checkpoints/step_faithful/seed_1702 \\
        --gpu 0

Or one at a time per GPU for parallelism:
    CUDA_VISIBLE_DEVICES=N python3 scripts/decode_checkpoints_only.py \\
        --checkpoints sf_X:checkpoints/step_faithful/seed_X --gpu 0
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
ITEMS_DIR = ROOT / "results/gap_43_canonical_beam3_items"
DONOR_REGISTRY = ITEMS_DIR / "donor_registry.jsonl"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checkpoint_arg(spec):
    """Parse 'prefix:path' or just 'path' (derives prefix from path)."""
    if ":" in spec:
        prefix, path = spec.split(":", 1)
        return prefix, path
    # Derive prefix from path: checkpoints/step_faithful/seed_1701 -> sf_1701
    p = Path(spec)
    fam = p.parent.name
    seed = p.name.replace("seed_", "")
    fam_prefix = {
        "faithful": "faithful",
        "step_faithful": "sf",
        "config_faithful": "cf",
        "reconstructions": "reco",
        "reconstructions_v2": "sf",  # joint-loss greedy
        "reconstructions_v3": "sf",  # joint-loss beam-3
        "confirmation": "conf",
        "long_schedule": "ls",
        "distillation": None,  # special
        "ladder": None,
        "rescue": None,
    }.get(fam, "x")
    if fam_prefix is None:
        # fallback
        return f"{fam}_{seed}", spec
    return f"{fam_prefix}_{seed}", spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="Checkpoints to decode, as 'prefix:path' or just 'path'.")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    if not DONOR_REGISTRY.exists():
        print(f"ERROR: donor registry not found at {DONOR_REGISTRY}", file=sys.stderr)
        print("Run scripts/build_canonical_panel.py first to build the canonical registry.",
              file=sys.stderr)
        sys.exit(2)

    print("Loading donor registry...", flush=True)
    donor_map = {}
    with open(DONOR_REGISTRY) as f:
        for line in f:
            r = json.loads(line)
            donor_map[r["query_id"]] = r["donor_id"]
    print(f"  {len(donor_map)} entries", flush=True)

    print("Loading test/train pickles...", flush=True)
    test_items = load_pickle(DATA / "test.pt")
    train_items = load_pickle(DATA / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}
    test_ids = [it["name"] for it in test_items]
    refs = [it["text"] for it in test_items]

    # Build pose lists (with 12.5fps subsampling)
    gt_poses = []
    pure_poses = []
    for item in test_items:
        gp = item["poses_3d"]
        gp = gp[::2] if isinstance(gp, torch.Tensor) else \
            torch.as_tensor(np.asarray(gp, dtype=np.float32))[::2]
        gt_poses.append(gp)
        donor_id = donor_map.get(item["name"])
        if donor_id is None or donor_id not in train_by_id:
            print(f"  ERROR: no donor for {item['name']}", file=sys.stderr)
            sys.exit(2)
        donor = train_by_id[donor_id]
        dp = donor["poses_3d"]
        dp = dp[::2] if isinstance(dp, torch.Tensor) else \
            torch.as_tensor(np.asarray(dp, dtype=np.float32))[::2]
        pure_poses.append(dp)
    print(f"  {len(gt_poses)} test poses, {len(pure_poses)} PURE poses", flush=True)

    # Decode each requested checkpoint
    results = OrderedDict()
    for spec in args.checkpoints:
        prefix, path = parse_checkpoint_arg(spec)
        ckpt_dir = ROOT / path
        ckpt_path = ckpt_dir / "best.ckpt"
        if not ckpt_path.exists():
            print(f"SKIP {prefix}: no {ckpt_path}", flush=True)
            continue
        ckpt_hash = sha256_file(str(ckpt_path))
        print(f"\nDecoding {prefix} from {ckpt_dir}...", flush=True)
        model = make_back_translation_model(str(ckpt_dir))
        t0 = time.time()
        gt_hyps = back_translate(model, gt_poses)
        pure_hyps = back_translate(model, pure_poses)
        elapsed = time.time() - t0

        gb = BLEU.corpus_score(gt_hyps, [refs]).score
        pb = BLEU.corpus_score(pure_hyps, [refs]).score
        gap = pb - gb

        # Write per-item files (overwrites stale)
        json.dump([{"id": tid, "hypothesis": hyp, "reference": ref}
                    for tid, hyp, ref in zip(test_ids, gt_hyps, refs)],
                  open(ITEMS_DIR / f"{prefix}_gt.json", "w"), indent=1, ensure_ascii=False)
        json.dump([{"id": tid, "hypothesis": hyp, "reference": ref}
                    for tid, hyp, ref in zip(test_ids, pure_hyps, refs)],
                  open(ITEMS_DIR / f"{prefix}_pure.json", "w"), indent=1, ensure_ascii=False)

        results[prefix] = {"gt_bleu": gb, "pure_bleu": pb, "gap": gap,
                           "checkpoint_sha256": ckpt_hash, "time_s": elapsed}
        print(f"  {prefix}: GT={gb:.2f} PURE={pb:.2f} gap={gap:+.4f} ({elapsed:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n=== Summary ===")
    for prefix, r in results.items():
        print(f"  {prefix}: gap={r['gap']:+.4f}")


if __name__ == "__main__":
    main()
