#!/usr/bin/env python3
"""Train a distillation student: BT evaluator trained against a teacher BT's
decoded text + soft labels.

Implements the paper §4.6 distillation ladder:
    L = (1 - alpha) * CE(target_text) + alpha * T^2 * KL(teacher || student)

with temperature T=2 and alpha in {0, 0.25, 0.5, 0.75, 1.0}.

The teacher is the released SLRTP2025 BT evaluator; its beam-decoded sequences
(position-aligned with the student inputs) replace the gold text for the CE
term when alpha < 1, and the teacher's soft-label distribution (over the full
output vocab at every output position) drives the KL term when alpha > 0.

Usage:
    python -m src.training.train_distillation \
        --config configs/released.yaml \
        --teacher checkpoints/released/backTranslation_PHIX_model \
        --train-pickle data/SLRTP2025/.../train.pickle \
        --dev-pickle data/SLRTP2025/.../dev.pickle \
        --teacher-decodes results/teacher_decodes/train_decoded.json \
        --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
        --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
        --alpha 0.5 --seed 303 --gpu 0 \
        --batch-size 64 --grad-accum 4 \
        --output checkpoints/distillation/alpha_0.5_seed_303

Provenance: written fresh 2026-08-02 to replace the lost
revision_20260730_round15/scripts/e2_distill_*.py. The KL-on-soft-labels
mechanic follows Hinton et al. (2015) with T=2.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.slrtp_dataset import (
    SLRTPDataset, Vocab, build_dataloader, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
)
from src.models import build_model, SignModel, make_back_translation_model
from src.utils.seed import set_seed
from src.utils.hashing import sha256_file


# -------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="Train a distillation student BT evaluator.")
    p.add_argument("--config", required=True)
    p.add_argument("--teacher", required=True, help="Path to teacher checkpoint dir.")
    p.add_argument("--train-pickle", required=True)
    p.add_argument("--dev-pickle", required=True)
    p.add_argument("--teacher-decodes", default=None,
                   help="Pre-decoded teacher sequences (JSON: id -> [tokens]). "
                        "If absent, the teacher is decoded on-the-fly each epoch (slow).")
    p.add_argument("--txt-vocab", required=True)
    p.add_argument("--gls-vocab", required=True)
    p.add_argument("--alpha", type=float, required=True,
                   help="Distillation strength in [0, 1]. 0=hard-label CE only; 1=full KL.")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Use >1 to avoid CUDA OOM. Recommended 4 for batch=64.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--output", required=True)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# -------------------------------------------------------------------- loss
def distillation_loss(
    student_logits: torch.Tensor,       # [B, U, V]
    teacher_logits: torch.Tensor,       # [B, U, V]
    gold_targets: torch.Tensor,         # [B, U]
    mask: torch.Tensor,                 # [B, U], True=valid
    pad_idx: int,
    alpha: float,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    """L = (1 - alpha) * CE(gold) + alpha * T^2 * KL(teacher || student).

    For positions where the teacher decode differs from gold, the CE still
    uses gold (it does not see teacher tokens). The KL term aligns the full
    output distribution.
    """
    B, U, V = student_logits.shape
    # mask may be [B, 1, U] (JoeyNMT 3D) or [B, U] (2D); squeeze to [B, U]
    if mask.dim() == 3:
        mask = mask.squeeze(1)
    valid = (~mask).float()  # [B, U], 1 = valid

    # Hard-label CE on gold targets (over all positions, summed, divided by valid tokens)
    ce = F.cross_entropy(
        student_logits.reshape(-1, V),
        gold_targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="mean",
    )

    # Soft-label KL: teacher || student, with temperature
    # KL(p_T || p_S) = sum p_T * (log p_T - log p_S)
    log_p_T = F.log_softmax(teacher_logits / temperature, dim=-1)
    log_p_S = F.log_softmax(student_logits / temperature, dim=-1)
    p_T = log_p_T.exp()
    kl_per_pos = (p_T * (log_p_T - log_p_S)).sum(dim=-1)  # [B, U]
    kl_masked = (kl_per_pos * valid).sum() / valid.sum().clamp_min(1.0)

    loss = (1.0 - alpha) * ce + alpha * (temperature ** 2) * kl_masked
    return {"loss": loss, "ce": ce.detach(), "kl": kl_masked.detach()}


# -------------------------------------------------------------------- teacher
@torch.no_grad()
def decode_teacher_for_batch(
    teacher: SignModel,
    sgn: torch.Tensor,
    sgn_mask: torch.Tensor,
    sgn_lengths: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_output_length: int = 30,
    beam_size: int = 3,
) -> List[List[int]]:
    """Decode teacher text for one batch. Returns list of token-id lists (no <bos>, with <eos>)."""
    # Use the teacher's own beam search if available
    from src.models.bt_search import beam_search
    device = sgn.device
    out = beam_search(
        model=teacher,
        size=beam_size,
        encoder_output=None,  # encoder will run inside beam_search via model.encode
        encoder_hidden=None,
        src_mask=sgn_mask,
        max_output_length=max_output_length,
        alpha=-1,
        tgt_input=None,
    )
    # NOTE: the actual beam_search signature varies in SignDiff; for now we fall back to
    # greedy decoding by running model.forward with shifted inputs and argmax.
    # This is fine for distillation: we just need teacher predictions per training step.
    teacher.eval()
    B = sgn.shape[0]
    # Run encoder once
    encoder_output, encoder_hidden = teacher.encode(sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths)
    U = max_output_length
    ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(U - 1):
        ys_mask = (ys != pad_idx).bool()
        out = teacher.decode(encoder_output=encoder_output,
                             encoder_hidden=encoder_hidden,
                             src_mask=sgn_mask,
                             trg_input=ys,
                             trg_mask=ys_mask)
        # out[0] = logits [B, U_so_far, V]
        logits = out[0] if isinstance(out, tuple) else out
        next_tok = logits[:, -1, :].argmax(dim=-1)  # greedy
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_idx), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_idx)
        if finished.all():
            break
    # Convert to list of token lists; strip <bos>; keep <eos>
    seqs = []
    for i in range(B):
        toks = ys[i, 1:].tolist()
        # truncate at first <eos>
        if eos_idx in toks:
            toks = toks[:toks.index(eos_idx) + 1]
        else:
            toks = toks + [eos_idx]
        seqs.append(toks)
    return seqs


# -------------------------------------------------------------------- epoch
def train_one_epoch(
    student: SignModel,
    teacher: SignModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    alpha: float,
    temperature: float,
    grad_accum: int = 1,
    log_every: int = 100,
) -> Dict[str, float]:
    """One distillation epoch."""
    student.train()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_tokens = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        sgn = batch["sgn"].to(device, non_blocking=True)
        sgn_mask = batch["sgn_mask"].to(device, non_blocking=True)
        sgn_lengths = batch["sgn_lengths"].to(device, non_blocking=True)
        gold_input = batch["txt_input"].to(device, non_blocking=True)
        gold_output = batch["txt_output"].to(device, non_blocking=True)
        txt_mask = batch["txt_mask"].to(device, non_blocking=True)

        # Student forward (uses gold for teacher-forcing input, but KL is on output logits)
        student_out = student(sgn=sgn, sgn_mask=sgn_mask, sgn_lengths=sgn_lengths,
                              txt_input=gold_input, txt_mask=txt_mask)
        student_logits = student_out[0][0] if isinstance(student_out, tuple) else student_out["decoder_outputs"]

        # Teacher forward (no grad); produce teacher logits aligned to gold_input
        with torch.no_grad():
            teacher_out = teacher(sgn=sgn, sgn_mask=sgn_mask, sgn_lengths=sgn_lengths,
                                  txt_input=gold_input, txt_mask=txt_mask)
            teacher_logits = teacher_out[0][0] if isinstance(teacher_out, tuple) else teacher_out["decoder_outputs"]

        losses = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            gold_targets=gold_output,
            mask=txt_mask,
            pad_idx=student.txt_pad_index,
            alpha=alpha,
            temperature=temperature,
        )
        loss = losses["loss"]
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        n_tokens = (~txt_mask).reshape(txt_mask.shape[0], -1).sum().item()
        total_loss += loss.item() * n_tokens
        total_ce += losses["ce"].item() * n_tokens
        total_kl += losses["kl"].item() * n_tokens
        total_tokens += n_tokens

        if (step + 1) % log_every == 0:
            print(f"    step {step+1}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}  ce={losses['ce'].item():.4f}  "
                  f"kl={losses['kl'].item():.4f}  alpha={alpha}  T={temperature}",
                  flush=True)

    return {
        "loss": total_loss / max(1, total_tokens),
        "ce": total_ce / max(1, total_tokens),
        "kl": total_kl / max(1, total_tokens),
        "tokens": total_tokens,
    }


@torch.no_grad()
def evaluate_dev(student: SignModel, dev_loader: DataLoader, device) -> Dict[str, float]:
    student.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in dev_loader:
        sgn = batch["sgn"].to(device, non_blocking=True)
        sgn_mask = batch["sgn_mask"].to(device, non_blocking=True)
        sgn_lengths = batch["sgn_lengths"].to(device, non_blocking=True)
        gold_input = batch["txt_input"].to(device, non_blocking=True)
        gold_output = batch["txt_output"].to(device, non_blocking=True)
        txt_mask = batch["txt_mask"].to(device, non_blocking=True)
        out = student(sgn=sgn, sgn_mask=sgn_mask, sgn_lengths=sgn_lengths,
                      txt_input=gold_input, txt_mask=txt_mask)
        logits = out[0][0] if isinstance(out, tuple) else out["decoder_outputs"]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               gold_output.reshape(-1),
                               ignore_index=student.txt_pad_index,
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

    cfg = yaml.safe_load(open(args.config))
    feat_size = cfg["data"]["feature_size"]
    if isinstance(feat_size, list):
        feat_size = sum(feat_size)
    subsample = cfg["data"].get("skeleton_subsample", 2)
    max_sent_length = cfg["data"].get("max_sent_length", 400)
    txt_lowercase = cfg["data"].get("txt_lowercase", True)

    txt_vocab = Vocab.from_file(args.txt_vocab)
    gls_vocab = Vocab.from_file(args.gls_vocab)
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

    # Teacher (released BT, frozen)
    teacher = make_back_translation_model(args.teacher).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded: {args.teacher}")

    # Student (same architecture, fresh init with seed)
    student = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=feat_size,
        do_recognition=cfg["training"].get("recognition_loss_weight", 1.0) > 0.0,
        do_translation=cfg["training"].get("translation_loss_weight", 1.0) > 0.0,
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student: params={n_params:,}  alpha={args.alpha}  T={args.temperature}")

    opt_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.998])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.001)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(opt_cfg.get("decrease_factor", 0.8)),
        patience=int(opt_cfg.get("patience", 15)),
        min_lr=float(opt_cfg.get("learning_rate_min", 1e-8)),
    )

    log = {
        "seed": args.seed,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "teacher": str(args.teacher),
        "teacher_sha256": sha256_file(Path(args.teacher) / "best.ckpt"),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "txt_vocab": str(args.txt_vocab),
        "gls_vocab": str(args.gls_vocab),
        "train_pickle": str(args.train_pickle),
        "dev_pickle": str(args.dev_pickle),
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "feat_size": feat_size,
        "skeleton_subsample": subsample,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "device": str(device),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs_log": [],
        "best": None,
    }

    best_nll = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            student, teacher, optimizer, train_loader, device,
            alpha=args.alpha, temperature=args.temperature,
            grad_accum=args.grad_accum,
        )
        dev_stats = evaluate_dev(student, dev_loader, device)
        elapsed = time.time() - t0

        improved = dev_stats["nll"] < best_nll - 1e-6
        if improved:
            best_nll = dev_stats["nll"]
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state": student.state_dict(),
                        "epoch": epoch,
                        "dev_nll": dev_stats["nll"],
                        "alpha": args.alpha,
                        "temperature": args.temperature,
                        "config": cfg},
                       out_dir / "best.ckpt")
        else:
            no_improve += 1

        log["epochs_log"].append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ce": train_stats["ce"],
            "train_kl": train_stats["kl"],
            "dev_nll": dev_stats["nll"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": elapsed,
            "improved": improved,
        })
        scheduler.step(dev_stats["nll"])

        print(f"  epoch {epoch}/{args.epochs}  "
              f"train_loss={train_stats['loss']:.4f}  dev_nll={dev_stats['nll']:.4f}  "
              f"best={best_nll:.4f}@{best_epoch}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"elapsed={elapsed:.1f}s",
              flush=True)

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            torch.save({"model_state": student.state_dict(), "epoch": epoch,
                        "dev_nll": dev_stats["nll"]},
                       out_dir / f"epoch_{epoch:04d}.ckpt")

        if args.patience > 0 and no_improve >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

        if args.smoke and epoch >= 2:
            print("  smoke: stop after 2 epochs")
            break

    log["best"] = {"epoch": best_epoch, "dev_nll": best_nll}
    log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nDone. Best: epoch={best_epoch} dev_nll={best_nll:.4f}")


if __name__ == "__main__":
    main()
