#!/usr/bin/env python3
"""Train ladder checkpoints on subsampled training pools."""
import sys, os, json, random, argparse
sys.path.insert(0, '.')
import torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    DATA = Path("data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data")

    # Load and subsample training data
    from src.data.slrtp_dataset import load_pickle
    train = load_pickle(DATA / "train.pt")
    rng = random.Random(20260729)
    indices = list(range(len(train)))
    rng.shuffle(indices)
    k = max(1, int(len(train) * args.frac))
    subset_indices = sorted(indices[:k])

    # Save subsampled train data
    subsample_path = DATA / f"train_ladder_{int(args.frac*1000)}.pt"
    subsample_dict = {train[i]["name"]: train[i] for i in subset_indices}
    torch.save(subsample_dict, str(subsample_path))
    print(f"Subsample: frac={args.frac}, {k} items → {subsample_path}", flush=True)

    # Train using train_matched
    from src.training.train_matched import main as train_main
    sys.argv = [
        "train_matched",
        "--config", "configs/released.yaml",
        "--train-pickle", str(subsample_path),
        "--dev-pickle", str(DATA / "dev.pt"),
        "--txt-vocab", "checkpoints/released/backTranslation_PHIX_model/txt.vocab",
        "--gls-vocab", "checkpoints/released/backTranslation_PHIX_model/gls.vocab",
        "--seed", str(args.seed), "--gpu", "0",
        "--epochs", str(args.epochs),
        "--batch-size", "256", "--grad-accum", "8",
        "--selection", "nll",
        "--output", args.output,
    ]
    train_main()

if __name__ == "__main__":
    main()
