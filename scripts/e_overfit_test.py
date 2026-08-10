#!/usr/bin/env python3
"""Small-sample overfit unit test (reviewer M2 sanity check).

Train a reconstruction model on N=50 training samples for many epochs;
free-decode the same 50 samples and verify near-perfect corpus BLEU.
This rules out a fundamental code-path bug in the reconstruction pipeline.

Usage: python scripts/e_overfit_test.py --gpu 0 --n 50 --epochs 200
Output: results/overfit_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=50, help="training subset size")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. Create small training subset
    train_path = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"
    train_data = torch.load(str(train_path), map_location="cpu", weights_only=False)
    all_keys = sorted(train_data.keys())
    rng = random.Random(args.seed)
    subset = rng.sample(all_keys, args.n)
    subset_dict = {k: train_data[k] for k in subset}

    smoke_dir = ROOT / "tmp_overfit"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)
    smoke_pt = smoke_dir / "train_smoke.pt"
    torch.save(subset_dict, str(smoke_pt))
    print(f"Created {smoke_pt} with {args.n} samples")

    # 2. Train model on subset with many epochs
    out_dir = smoke_dir / "ckpt"
    out_dir.mkdir()
    cmd = [
        sys.executable, "scripts/train_reconstruction.py",
        "--seed", str(args.seed),
        "--gpu", str(args.gpu),
        "--epochs", str(args.epochs),
        "--batch-size", "16",
        "--selection", "nll",
        "--output", str(out_dir),
    ]
    print(f"Training: {' '.join(cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Override train pickle to use smoke subset
    proc = subprocess.run(
        [sys.executable, "-m", "src.training.train_matched",
         "--config", str(ROOT / "configs/released.yaml"),
         "--train-pickle", str(smoke_pt),
         "--dev-pickle", str(smoke_pt),
         "--txt-vocab", str(ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"),
         "--gls-vocab", str(ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"),
         "--seed", str(args.seed),
         "--gpu", "0",
         "--epochs", str(args.epochs),
         "--batch-size", "16",
         "--workers", "0",
         "--selection", "nll",
         "--output", str(out_dir),
        ],
        cwd=str(ROOT), env=env,
    )
    if proc.returncode != 0:
        print(f"Training failed with code {proc.returncode}", file=sys.stderr)
        sys.exit(1)
    print("Training done")

    # 3. Copy config + vocab to checkpoint dir (required by make_back_translation_model)
    import shutil as _shutil
    _shutil.copy(str(ROOT / "configs/released.yaml"), str(out_dir / "config.yaml"))
    for v in ("txt.vocab", "gls.vocab"):
        _shutil.copy(str(ROOT / "checkpoints/released/backTranslation_PHIX_model" / v),
                     str(out_dir / v))
    # Copy tokenizer metadata if available
    subsample_dir = ROOT / "checkpoints/released/backTranslation_PHIX_model/tokens"
    if subsample_dir.exists():
        _shutil.copytree(str(subsample_dir), str(out_dir / "tokens"), dirs_exist_ok=True)

    from src.models.back_translate import make_back_translation_model, back_translate
    import sacrebleu

    BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                                  effective_order=False, force=True)

    model = make_back_translation_model(str(out_dir))
    keys = sorted(subset_dict.keys())
    refs = [subset_dict[k]["text"] for k in keys]
    poses = [subset_dict[k]["poses_3d"][::2] for k in keys]
    hyps = back_translate(model, poses)

    b = BLEU.corpus_score(hyps, [refs]).score
    print(f"Decoded {len(hyps)} items: sacreBLEU-4 = {b:.2f}")

    # Also compute train NLL from the log
    log_file = out_dir / "training_log.json"
    train_nll_best = None
    if log_file.exists():
        import json as _json
        log = _json.load(open(log_file))
        epochs = log.get("epochs", []) if isinstance(log, dict) else []
        if isinstance(epochs, list):
            nlls = [e.get("dev_nll") for e in epochs if isinstance(e, dict) and "dev_nll" in e]
            if nlls:
                train_nll_best = min(nlls)

    out = {
        "n": args.n, "epochs": args.epochs, "seed": args.seed,
        "train_nll_best": train_nll_best,
        "free_decode_bleu": round(b, 2),
        "verdict": "PASS" if b > 60 else ("MARGINAL" if b > 30 else "FAIL"),
    }
    out_path = ROOT / "results/overfit_test.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))

    # Cleanup
    shutil.rmtree(smoke_dir)


if __name__ == "__main__":
    main()
