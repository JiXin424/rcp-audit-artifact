#!/usr/bin/env python3
"""Experiment A2: fine-tune released checkpoint (not random perturbation).

Key difference from A1 (Gaussian noise): this moves weights along real gradient
directions, producing genuinely different variants that explore the loss
landscape near the released checkpoint.  Gaussian noise only probes the
isotropic weight-norm ball.

Design: load released checkpoint, continue training with very small LR
(1e-5, 5e-6, 1e-4) for 30 epochs, 3 seeds each.  After fine-tuning, decode
REC / PURE / dev (beam-3) and full-pool train readout.

Output: results/released_finetune.json
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sacrebleu

# reuse the perturbation script's loading helpers
from src.data.slrtp_dataset import load_pickle  # noqa: E402

# --- direct imports from src (same module space as train_matched.py) ---
from src.data.slrtp_dataset import (
    SLRTPDataset, Vocab, build_dataloader, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
)
from src.models import make_back_translation_model, back_translate
from src.utils.seed import set_seed

# -------------------------------------------------------------------- paths
DATA = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
MODEL_DIR = ROOT / "checkpoints/released/backTranslation_PHIX_model"
REGISTRY = ROOT / "results/gap_43_canonical_beam3_items/donor_registry.jsonl"
OUT = ROOT / "results/released_finetune.json"
OUTDIR = ROOT / "checkpoints/finetune_released"

# -------------------------------------------------------------------- config
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)
SEEDS = [42, 123, 789]
LRS = [5e-6, 1e-5, 3e-5]
EPOCHS = 60
BATCH_SIZE = 64  # smaller to avoid OOM with released checkpoint
GRAD_ACCUM = 4   # effective batch 256
LOG_EVERY = 50


# -------------------------------------------------------------------- helpers
def em(hyps, refs):
    return sum(1 for h, r in zip(hyps, refs)
               if h.strip().lower() == r.strip().lower()) / max(1, len(hyps))


def corpus_gap(hyps_pure, hyps_rec, refs):
    b_p = BLEU.corpus_score(hyps_pure, [refs]).score
    b_r = BLEU.corpus_score(hyps_rec, [refs]).score
    return b_p - b_r, b_p, b_r


def pose_of(item):
    p = item["poses_3d"]
    p = p if isinstance(p, torch.Tensor) else torch.as_tensor(np.asarray(p, dtype=np.float32))
    return p[::2]


# -------------------------------------------------------------------- data
def load_data():
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
    train_poses_7k = [pose_of(it) for it in train_items]
    train_refs_7k = [it["text"] for it in train_items]
    return (test_items, dev_items, train_items,
            gt_poses, refs, pure_poses, dev_poses, dev_refs,
            train_poses_7k, train_refs_7k)


def build_loaders(txt_vocab_path, gls_vocab_path, batch_size, seed):
    txt_vocab = Vocab.from_file(txt_vocab_path)
    gls_vocab = Vocab.from_file(gls_vocab_path)
    train_ds = SLRTPDataset(str(DATA / "train.pt"), txt_vocab=txt_vocab,
                            gls_vocab=gls_vocab)
    dev_ds = SLRTPDataset(str(DATA / "dev.pt"), txt_vocab=txt_vocab,
                          gls_vocab=gls_vocab)
    pad_idx = txt_vocab.stoi.get(PAD_TOKEN, 0)
    train_loader = build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pad_idx=pad_idx)
    dev_loader = build_dataloader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pad_idx=pad_idx)
    return train_loader, dev_loader, txt_vocab, gls_vocab


# -------------------------------------------------------------------- decode
@torch.no_grad()
def decode_all(model, gt_poses, refs, pure_poses, dev_poses, dev_refs,
               train_poses_7k, train_refs_7k, readout=False):
    model.eval()
    gt_h = back_translate(model, gt_poses)
    pure_h = back_translate(model, pure_poses)
    dev_h = back_translate(model, dev_poses)
    gap, p, r = corpus_gap(pure_h, gt_h, refs)
    rec = {
        "gap": gap, "pure_bleu": p, "gt_bleu": r,
        "dev_bleu": BLEU.corpus_score(dev_h, [dev_refs]).score,
    }
    if readout:
        tr_h = back_translate(model, train_poses_7k)
        rec["train_bleu"] = BLEU.corpus_score(tr_h, [train_refs_7k]).score
        rec["train_em"] = em(tr_h, train_refs_7k)
    return rec


# -------------------------------------------------------------------- training loop
def train_one_epoch(model, optimizer, train_loader, device, grad_accum):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
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
        loss = F.cross_entropy(logits, targets,
                               ignore_index=getattr(model, "txt_pad_index", 0),
                               reduction="mean")
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        n_tokens = (~txt_mask).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(1, total_tokens), total_tokens


@torch.no_grad()
def dev_nll(model, dev_loader, device):
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
        loss = F.cross_entropy(logits, targets,
                               ignore_index=getattr(model, "txt_pad_index", 0),
                               reduction="mean")
        n_tokens = (~txt_mask).sum().item()
        total_nll += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_nll / max(1, total_tokens)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None,
                    help="Single LR override; if set, only this LR is run.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Single seed override.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs only, no full-pool readout.")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    (test_items, dev_items, train_items,
     gt_poses, refs, pure_poses, dev_poses, dev_refs,
     train_poses_7k, train_refs_7k) = load_data()
    txt_vocab_p = str(MODEL_DIR / "txt.vocab")
    gls_vocab_p = str(MODEL_DIR / "gls.vocab")
    epochs = 2 if args.smoke else EPOCHS

    lrs = [args.lr] if args.lr else LRS
    seeds = [args.seed] if args.seed else SEEDS
    records = []

    # -- baseline (no fine-tuning) --
    print("loading released model for baseline...", flush=True)
    base_model = make_back_translation_model(str(MODEL_DIR)).to(device)
    base_model.eval()
    base_rec = decode_all(base_model, gt_poses, refs, pure_poses,
                          dev_poses, dev_refs, train_poses_7k, train_refs_7k,
                          readout=True)
    base_rec["lr"] = None
    base_rec["seed"] = "baseline"
    base_rec["best_epoch"] = None
    records.append(base_rec)
    print(f"baseline: gap={base_rec['gap']:+.2f} dev={base_rec['dev_bleu']:.2f} "
          f"train={base_rec.get('train_bleu',0):.2f} EM={base_rec.get('train_em',0)*100:.1f}%",
          flush=True)

    # -- fine-tune runs --
    for lr in lrs:
        for sd in seeds:
            tag = f"lr{lr:.0e}_seed{sd}"
            ckpt_dir = OUTDIR / tag
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== fine-tune lr={lr:.1e} seed={sd} epochs={epochs} ===", flush=True)

            set_seed(sd)
            model = make_back_translation_model(str(MODEL_DIR)).to(device)
            train_loader, dev_loader, _, _ = build_loaders(
                txt_vocab_p, gls_vocab_p, BATCH_SIZE, sd)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         betas=(0.9, 0.998), weight_decay=0.0)
            # warmup + cosine schedule: small lr → mostly flat
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, epochs=epochs,
                steps_per_epoch=len(train_loader), pct_start=0.1)

            best_dev_nll = float("inf")
            best_epoch = 0
            best_state = None
            log = []

            for ep in range(1, epochs + 1):
                t0 = time.time()
                train_loss, train_tok = train_one_epoch(
                    model, optimizer, train_loader, device, GRAD_ACCUM)
                scheduler.step()
                d_nll = dev_nll(model, dev_loader, device)
                dt = time.time() - t0
                log.append({"epoch": ep, "train_loss": round(train_loss, 4),
                            "dev_nll": round(d_nll, 4), "elapsed_s": round(dt, 1)})
                print(f"  ep {ep:2d}/{epochs}: loss={train_loss:.3f} "
                      f"dev_nll={d_nll:.3f}  ({dt:.0f}s)", flush=True)
                if d_nll < best_dev_nll:
                    best_dev_nll = d_nll
                    best_epoch = ep
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    print(f"    -> best so far", flush=True)

            # load best and decode
            if best_state is not None:
                model.load_state_dict(best_state)
            rec = decode_all(model, gt_poses, refs, pure_poses,
                             dev_poses, dev_refs, train_poses_7k, train_refs_7k,
                             readout=(not args.smoke))
            rec["lr"] = lr
            rec["seed"] = sd
            rec["best_epoch"] = best_epoch
            rec["dev_nll"] = round(best_dev_nll, 4)
            rec["log"] = log
            records.append(rec)
            print(f"  done: gap={rec['gap']:+.2f} dev={rec['dev_bleu']:.2f} "
                  f"train={rec.get('train_bleu',0):.2f} EM={rec.get('train_em',0)*100:.1f}%  "
                  f"best_ep={best_epoch}", flush=True)
            # save checkpoint
            torch.save(model.state_dict(), ckpt_dir / "best.ckpt")
            OUT.write_text(json.dumps(records, indent=1, ensure_ascii=False))

    # -- summary --
    print(f"\n=== summary ===", flush=True)
    for rec in records:
        lr_s = f"lr={rec['lr']:.0e}" if rec['lr'] is not None else "baseline"
        print(f"  {lr_s:>12s} s={str(rec['seed']):>8s}  "
              f"gap={rec['gap']:+.2f}  dev={rec['dev_bleu']:.2f}  "
              f"pure={rec['pure_bleu']:.2f}  rec_bleu={rec['gt_bleu']:.2f}  "
              f"train={rec.get('train_bleu',0):.1f}  EM={rec.get('train_em',0)*100:.1f}%  "
              f"best_ep={rec['best_epoch']}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
