#!/usr/bin/env python3
"""CLI wrapper for src.training.train_distillation (sets CUDA_VISIBLE_DEVICES early)."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--config", default=str(ROOT / "configs" / "released.yaml"))
    p.add_argument("--teacher", default=str(ROOT / "checkpoints" / "released" / "backTranslation_PHIX_model"))
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = [
        sys.executable, "-m", "src.training.train_distillation",
        "--config", args.config,
        "--teacher", args.teacher,
        "--train-pickle", str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"),
        "--dev-pickle", str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt"),
        "--txt-vocab", str(ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"),
        "--gls-vocab", str(ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"),
        "--alpha", str(args.alpha),
        "--temperature", str(args.temperature),
        "--seed", str(args.seed),
        "--gpu", "0",  # always 0 inside the process
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--output", args.output,
    ]
    if args.smoke:
        cmd.append("--smoke")
    print(f"Running: CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
