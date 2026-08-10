#!/usr/bin/env python3
"""CLI wrapper for src.training.train_matched.

Sets CUDA_VISIBLE_DEVICES BEFORE importing torch so the --gpu flag actually
pins the process to one physical GPU.

Example:
    python scripts/train_reconstruction.py \
        --seed 101 --gpu 0 --epochs 300 \
        --output checkpoints/reconstructions/seed_101
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--config", default=str(ROOT / "configs" / "released.yaml"))
    p.add_argument("--output", required=True)
    p.add_argument("--selection", choices=["nll", "bleu"], default="nll")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    # CRITICAL: set CUDA_VISIBLE_DEVICES BEFORE Python interpreter starts the worker
    # so that torch.cuda only sees one physical device.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        sys.executable, "-m", "src.training.train_matched",
        "--config", args.config,
        "--train-pickle", str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"),
        "--dev-pickle", str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt"),
        "--txt-vocab", str(ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"),
        "--gls-vocab", str(ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"),
        "--seed", str(args.seed),
        "--gpu", "0",  # always 0 inside the process (CUDA_VISIBLE_DEVICES pins the physical GPU)
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--selection", args.selection,
        "--output", args.output,
    ]
    if args.smoke:
        cmd.append("--smoke")
    print(f"Running: CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
