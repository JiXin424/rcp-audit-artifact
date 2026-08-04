#!/usr/bin/env python3
"""Train a recipe-conditioned reconstruction of the released SLRTP2025 BT evaluator.

This is the "reconstruction" / "matched" training that produces seeds 101-1405
in the paper. It uses the released config (architecture, optimizer, LR schedule,
selection rule) with a fresh seed.

Usage (matches paper §3.4 recipe):
    python -m src.training.train_matched \
        --config configs/released.yaml \
        --train-manifest manifests/available_train.jsonl \
        --dev-manifest manifests/available_dev.jsonl \
        --data-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/.../train.pickle \
        --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
        --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
        --seed 101 --gpu 0 --epochs 300 --output checkpoints/reconstructions/seed_101

Provenance: rewritten 2026-08-02 to replace the lost
revision_20260728_round3/src/train_matched.py. The training loop is fresh,
but it uses the SignModel architecture copied from SignDiff/SLRTP2025_eval.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.slrtp_dataset import (
    SLRTPDataset, Vocab, build_dataloader, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
)
from src.models import build_model, SignModel
from src.utils.seed import set_seed
from src.utils.hashing import sha256_file


# -------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="Train a reconstruction BT evaluator.")
    p.add_argument("--config", required=True, help="Path to released.yaml or a derivative.")
    p.add_argument("--train-pickle", required=True, help="train.pickle path.")
    p.add_argument("--dev-pickle", required=True, help="dev.pickle path.")
    p.add_argument("--txt-vocab", required=True)
    p.add_argument("--gls-vocab", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (use >1 to avoid OOM).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Save a checkpoint every N epochs (in addition to best).")
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stop patience (in validation steps).")
    p.add_argument("--selection", choices=["nll", "bleu"], default="nll",
                   help="Checkpoint selection objective.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 2 epochs, no checkpointing.")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint to resume from.")
    return p.parse_args()


# -------------------------------------------------------------------- loop
def train_one_epoch(
    model: SignModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    device: torch.device,
    grad_accum: int = 1,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train one epoch. Returns {'loss': mean_loss, 'nll': mean_nll, 'tokens': total_tokens}."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_nll = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        sgn = batch["sgn"].to(device, non_blocking=True)
        sgn_mask = batch["sgn_mask"].to(device, non_blocking=True)
        sgn_lengths = batch["sgn_lengths"].to(device, non_blocking=True)
        txt_input = batch["txt_input"].to(device, non_blocking=True)
        txt_output = batch["txt_output"].to(device, non_blocking=True)
        txt_mask = batch["txt_mask"].to(device, non_blocking=True)

        # SignModel.forward returns (gloss_probs, decoder_outputs, att_probs, ...)
        out = model(sgn=sgn, sgn_mask=sgn_mask, sgn_lengths=sgn_lengths,
                    txt_input=txt_input, txt_mask=txt_mask)
        decoder_outputs = out[0][0] if isinstance(out, tuple) else out["decoder_outputs"]

        # Cross-entropy over text tokens, ignoring padding
        logits = decoder_outputs.reshape(-1, decoder_outputs.size(-1))
        targets = txt_output.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=model.txt_pad_index,
                               reduction="mean")
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # token-count for NLL normalization
        n_tokens = (~txt_mask).reshape(txt_mask.shape[0], -1).sum().item()
        total_loss += loss.item() * n_tokens
        total_nll += loss.item() * n_tokens
        total_tokens += n_tokens

        if (step + 1) % log_every == 0:
            print(f"    step {step+1}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  n_tokens={n_tokens}",
                  flush=True)

    return {
        "loss": total_loss / max(1, total_tokens),
        "nll": total_nll / max(1, total_tokens),
        "tokens": total_tokens,
    }


# -------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate_dev(
    model: SignModel,
    dev_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute dev NLL (per token)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in dev_loader:
        sgn = batch["sgn"].to(device, non_blocking=True)
        sgn_mask = batch["sgn_mask"].to(device, non_blocking=True)
        sgn_lengths = batch["sgn_lengths"].to(device, non_blocking=True)
        txt_input = batch["txt_input"].to(device, non_blocking=True)
        txt_output = batch["txt_output"].to(device, non_blocking=True)
        txt_mask = batch["txt_mask"].to(device, non_blocking=True)

        out = model(sgn=sgn, sgn_mask=sgn_mask, sgn_lengths=sgn_lengths,
                    txt_input=txt_input, txt_mask=txt_mask)
        decoder_outputs = out[0][0] if isinstance(out, tuple) else out["decoder_outputs"]
        logits = decoder_outputs.reshape(-1, decoder_outputs.size(-1))
        targets = txt_output.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=model.txt_pad_index,
                               reduction="sum")
        n_tokens = (~txt_mask).reshape(txt_mask.shape[0], -1).sum().item()
        total_nll += loss.item()
        total_tokens += n_tokens
    return {"nll": total_nll / max(1, total_tokens), "tokens": total_tokens}


# -------------------------------------------------------------------- main
def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load config
    cfg = yaml.safe_load(open(args.config))
    feat_size = cfg["data"]["feature_size"]
    if isinstance(feat_size, list):
        feat_size = sum(feat_size)
    subsample = cfg["data"].get("skeleton_subsample", 2)
    max_sent_length = cfg["data"].get("max_sent_length", 400)
    txt_lowercase = cfg["data"].get("txt_lowercase", True)

    # --- vocabularies
    txt_vocab = Vocab.from_file(args.txt_vocab)
    gls_vocab = Vocab.from_file(args.gls_vocab)

    # --- data
    pad_idx = txt_vocab.stoi[PAD_TOKEN]
    train_ds = SLRTPDataset(args.train_pickle, txt_vocab=txt_vocab, gls_vocab=gls_vocab,
                            skeleton_subsample=subsample, max_sent_length=max_sent_length,
                            txt_lowercase=txt_lowercase)
    dev_ds = SLRTPDataset(args.dev_pickle, txt_vocab=txt_vocab, gls_vocab=gls_vocab,
                          skeleton_subsample=subsample, max_sent_length=max_sent_length,
                          txt_lowercase=txt_lowercase)
    print(f"Train: {len(train_ds)} items | Dev: {len(dev_ds)} items")

    train_loader = build_dataloader(train_ds, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.workers, pad_idx=pad_idx)
    dev_loader = build_dataloader(dev_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.workers, pad_idx=pad_idx)

    # --- model
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=feat_size,
        do_recognition=cfg["training"].get("recognition_loss_weight", 1.0) > 0.0,
        do_translation=cfg["training"].get("translation_loss_weight", 1.0) > 0.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: SignModel  params={n_params:,}  device={device}")

    # --- optimizer / scheduler
    opt_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.998])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.001)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(opt_cfg.get("decrease_factor", 0.8)),
        patience=int(opt_cfg.get("patience", 15)),
        min_lr=float(opt_cfg.get("learning_rate_min", 1e-8)),
    ) if args.selection == "nll" else None

    # --- training log
    log = {
        "seed": args.seed,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "txt_vocab": str(args.txt_vocab),
        "txt_vocab_sha256": sha256_file(args.txt_vocab),
        "gls_vocab": str(args.gls_vocab),
        "gls_vocab_sha256": sha256_file(args.gls_vocab),
        "train_pickle": str(args.train_pickle),
        "dev_pickle": str(args.dev_pickle),
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "feat_size": feat_size,
        "skeleton_subsample": subsample,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "selection": args.selection,
        "device": str(device),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs_log": [],
        "best": None,
    }

    best_metric = float("inf") if args.selection == "nll" else -1.0
    best_epoch = -1
    no_improve = 0

    # --- main loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, optimizer, scheduler, train_loader, device,
            grad_accum=args.grad_accum,
        )
        dev_stats = evaluate_dev(model, dev_loader, device)
        elapsed = time.time() - t0

        metric = dev_stats["nll"]
        improved = metric < best_metric - 1e-6 if args.selection == "nll" else False
        if improved:
            best_metric = metric
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "dev_nll": metric,
                        "config": cfg},
                       out_dir / "best.ckpt")
        else:
            no_improve += 1

        log["epochs_log"].append({
            "epoch": epoch,
            "train_nll": train_stats["nll"],
            "dev_nll": dev_stats["nll"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": elapsed,
            "improved": improved,
        })

        if scheduler is not None and args.selection == "nll":
            scheduler.step(dev_stats["nll"])

        print(f"  epoch {epoch}/{args.epochs}  "
              f"train_nll={train_stats['nll']:.4f}  dev_nll={dev_stats['nll']:.4f}  "
              f"best={best_metric:.4f}@{best_epoch}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"elapsed={elapsed:.1f}s  {'' if improved else '(no improve)'}",
              flush=True)

        # periodic checkpoint (for trajectory analysis)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "dev_nll": dev_stats["nll"]},
                       out_dir / f"epoch_{epoch:04d}.ckpt")

        # early stop (only on NLL selection; plateau scheduler handles LR decay)
        if args.patience > 0 and no_improve >= args.patience:
            print(f"  early stop at epoch {epoch} (no improvement for {no_improve} validations)")
            break

        if args.smoke and epoch >= 2:
            print("  smoke test: stopping after 2 epochs")
            break

    log["best"] = {"epoch": best_epoch, "dev_nll": best_metric}
    log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nDone. Best: epoch={best_epoch} dev_nll={best_metric:.4f}")
    print(f"  output: {out_dir}/best.ckpt")
    print(f"  log: {out_dir}/training_log.json")


if __name__ == "__main__":
    main()
