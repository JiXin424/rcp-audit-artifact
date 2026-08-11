#!/usr/bin/env python3
"""Corrected reconstruction training (v2) — fixes reviewer round-9 issues.

Differences from train_matched.py (v1):
  1. Joint loss: translation CE + recognition CTC (weight from config, was
     omitted in v1 despite recognition_loss_weight=1.0).
  2. Step-level validation: validate every `validation_freq` optimizer steps
     (JoeyNMT semantics), not every epoch.
  3. --selection bleu actually works: greedy-decode dev, compute sacreBLEU,
     improved=True when BLEU rises; scheduler enabled in both branches.
  4. Custom collate supplies gls (gloss id tensor) + gls_lengths for CTC.

Usage (matches released config semantics):
    python -m src.training.train_matched_v2 \
        --config configs/released.yaml \
        --train-pickle .../train.pt --dev-pickle .../dev.pt \
        --txt-vocab .../txt.vocab --gls-vocab .../gls.vocab \
        --seed 101 --gpu 0 --epochs 300 --batch-size 256 \
        --output checkpoints/reconstructions_v2/seed_101
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml, sacrebleu
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.slrtp_dataset import SLRTPDataset, Vocab, PAD_TOKEN
from src.models import build_model, SignModel
from src.utils.seed import set_seed
from src.utils.hashing import sha256_file

GlsBlank = 0   # gls_vocab[0] = '<si>' (silence) -> CTC blank
GlsPad = 2     # gls_vocab[2] = '<pad>'


def collate_v2(batch: List[dict], txt_pad: int, gls_vocab, gls_pad: int = GlsPad) -> dict:
    B = len(batch)
    feat_dim = batch[0]["sgn"].shape[-1]
    T_max = max(it["sgn"].shape[0] for it in batch)
    U_max = max(it["txt"].shape[0] for it in batch)
    G_max = max(1, max(len(it["gls"]) for it in batch))
    sgn = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
    sgn_mask = torch.zeros(B, 1, T_max, dtype=torch.bool)
    sgn_lengths = torch.zeros(B, dtype=torch.long)
    txt_input = torch.full((B, U_max), txt_pad, dtype=torch.long)
    txt_output = torch.full((B, U_max), txt_pad, dtype=torch.long)
    txt_mask = torch.zeros(B, 1, U_max, dtype=torch.bool)
    gls = torch.full((B, G_max), gls_pad, dtype=torch.long)
    gls_lengths = torch.zeros(B, dtype=torch.long)
    ids = []
    for i, it in enumerate(batch):
        T = it["sgn"].shape[0]
        sgn[i, :T] = it["sgn"]
        sgn_mask[i, 0, :T] = True
        sgn_lengths[i] = T
        ids.append(it["id"])
        full = it["txt"]
        if full.numel() > 1:
            U = full.numel() - 1
            txt_input[i, :U] = full[:-1]
            txt_output[i, :U] = full[1:]
            txt_mask[i, 0, :U] = True
        gtoks = it["gls"][:G_max] if len(it["gls"]) > G_max else it["gls"]
        gtok_ids = [gls_vocab.stoi.get(t, 1) for t in gtoks]  # unk=1
        gls[i, :len(gtok_ids)] = torch.tensor(gtok_ids, dtype=torch.long)
        gls_lengths[i] = len(gtok_ids)
    return {"sgn": sgn, "sgn_mask": sgn_mask, "sgn_lengths": sgn_lengths,
            "txt_input": txt_input, "txt_output": txt_output, "txt_mask": txt_mask,
            "gls": gls, "gls_lengths": gls_lengths, "ids": ids}


def compute_loss(model, batch, device, rec_weight, trans_weight, do_rec, do_tr):
    out = model(sgn=batch["sgn"], sgn_mask=batch["sgn_mask"],
                sgn_lengths=batch["sgn_lengths"],
                txt_input=batch["txt_input"], txt_mask=batch["txt_mask"])
    decoder_outputs, gloss_probs = out
    word_outputs = decoder_outputs[0]
    logits = word_outputs.reshape(-1, word_outputs.size(-1))
    targets = batch["txt_output"].reshape(-1)
    trans_loss = F.cross_entropy(logits, targets,
                                 ignore_index=model.txt_pad_index, reduction="mean")
    loss = trans_weight * trans_loss
    rec_loss_v = 0.0
    if do_rec and gloss_probs is not None:
        rec_loss = F.ctc_loss(gloss_probs, batch["gls"],
                              batch["sgn_lengths"].long(), batch["gls_lengths"].long(),
                              blank=GlsBlank, reduction="mean", zero_infinity=True)
        loss = loss + rec_weight * rec_loss
        rec_loss_v = rec_loss.item()
    return loss, trans_loss.item(), rec_loss_v


@torch.no_grad()
def evaluate_dev(model, dev_loader, device, selection, gls_vocab_len):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    hyps, refs = [], []
    for batch in dev_loader:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        out = model(sgn=batch["sgn"], sgn_mask=batch["sgn_mask"],
                    sgn_lengths=batch["sgn_lengths"],
                    txt_input=batch["txt_input"], txt_mask=batch["txt_mask"])
        decoder_outputs, _ = out
        word_outputs = decoder_outputs[0]
        logits = word_outputs.reshape(-1, word_outputs.size(-1))
        targets = batch["txt_output"].reshape(-1)
        nll = F.cross_entropy(logits, targets, ignore_index=model.txt_pad_index,
                              reduction="sum")
        n_tok = (~batch["txt_mask"]).reshape(batch["txt_mask"].shape[0], -1).sum().item()
        total_nll += nll.item(); total_tokens += n_tok
        if selection == "bleu":
            logp = F.log_softmax(word_outputs, dim=-1)
            pred = logp.argmax(dim=-1)  # (B, U)
            for j in range(pred.size(0)):
                U = (~batch["txt_mask"][j, 0]).sum().item()
                toks = [model.txt_vocab.itos[t] for t in pred[j, :U].tolist()
                        if t != model.txt_pad_index]
                hyps.append(" ".join(toks))
                ref = [model.txt_vocab.itos[t] for t in batch["txt_output"][j, :U].tolist()
                       if t != model.txt_pad_index]
                refs.append(" ".join(ref))
    nll_per_tok = total_nll / max(1, total_tokens)
    bleu = None
    if selection == "bleu" and hyps:
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    model.train()
    return {"nll": nll_per_tok, "bleu": bleu}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-pickle", required=True)
    ap.add_argument("--dev-pickle", required=True)
    ap.add_argument("--txt-vocab", required=True)
    ap.add_argument("--gls-vocab", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--selection", choices=["nll", "bleu"], default="bleu",
                    help="Checkpoint selection (config eval_metric=bleu).")
    ap.add_argument("--rec-weight", type=float, default=None,
                    help="Override config recognition_loss_weight. Set to 0 to produce "
                         "the v1.5 translation-only-step-val-bleu protocol (same val/selection "
                         "as v2 but translation-only loss).")
    ap.add_argument("--protocol-name", default=None,
                    help="Override the protocol string written to training_log.json. "
                         "Defaults: 'v2-joint-ctc-translation-step-val' when rec_weight>0, "
                         "'v1.5-trans-only-step-val-bleu' when rec_weight==0.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    # CUDA_VISIBLE_DEVICES must be set externally BEFORE Python starts so torch
    # enumerates only the pinned GPU. Setting it here is too late: torch is
    # imported at the top of this module and has already enumerated all devices.
    # The launcher (scripts/launch_reconstructions.sh or equivalent) is
    # responsible for `CUDA_VISIBLE_DEVICES=N python -m src.training.train_matched_v2 --gpu 0 ...`.
    if args.gpu != 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(f"WARNING: --gpu {args.gpu} passed but CUDA_VISIBLE_DEVICES not set externally; "
              f"torch will use device 0 of however many it sees. Set "
              f"CUDA_VISIBLE_DEVICES={args.gpu} before invoking Python.", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    feat_size = cfg["data"]["feature_size"]
    if isinstance(feat_size, list): feat_size = sum(feat_size)
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
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers,
                              collate_fn=lambda b: collate_v2(b, pad_idx, gls_vocab))
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers,
                            collate_fn=lambda b: collate_v2(b, pad_idx, gls_vocab))

    rec_weight = float(cfg["training"].get("recognition_loss_weight", 1.0))
    trans_weight = float(cfg["training"].get("translation_loss_weight", 1.0))
    val_freq = int(cfg["training"].get("validation_freq", 14))
    if args.rec_weight is not None:
        rec_weight = args.rec_weight
    do_rec = rec_weight > 0; do_tr = trans_weight > 0
    if args.protocol_name is not None:
        protocol_name = args.protocol_name
    else:
        protocol_name = "v2-joint-ctc-translation-step-val" if do_rec else "v1.5-trans-only-step-val-bleu"
    print(f"loss weights: rec={rec_weight} trans={trans_weight} | val_freq={val_freq} steps | selection={args.selection} | protocol={protocol_name}")

    model = build_model(cfg=cfg["model"], gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                        sgn_dim=feat_size, do_recognition=do_rec,
                        do_translation=do_tr).to(device)
    model.txt_pad_index = pad_idx
    model.txt_vocab = txt_vocab
    print(f"SignModel params={sum(p.numel() for p in model.parameters()):,}")

    opt_cfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(opt_cfg["learning_rate"]),
                                 betas=tuple(opt_cfg.get("betas", [0.9, 0.998])),
                                 weight_decay=float(opt_cfg.get("weight_decay", 0.001)))
    sched_mode = "min" if args.selection == "nll" else "max"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode,
        factor=float(opt_cfg.get("decrease_factor", 0.8)),
        patience=int(opt_cfg.get("patience", 15)),
        min_lr=float(opt_cfg.get("learning_rate_min", 1e-8)))

    log = {"seed": args.seed, "config": str(args.config),
           "config_sha256": sha256_file(args.config),
           "protocol": protocol_name,
           "rec_weight": rec_weight, "trans_weight": trans_weight,
           "validation_freq_steps": val_freq, "selection": args.selection,
           "batch_size": args.batch_size, "grad_accum": args.grad_accum,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "validations": [], "best": None}

    best_metric = float("inf") if args.selection == "nll" else -1.0
    best_step = -1; no_improve = 0; global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            for k in batch:
                if torch.is_tensor(batch[k]): batch[k] = batch[k].to(device)
            loss, tl, rl = compute_loss(model, batch, device, rec_weight, trans_weight,
                                        do_rec, do_tr)
            (loss / args.grad_accum).backward()
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % val_freq == 0:
                dev = evaluate_dev(model, dev_loader, device, args.selection, len(gls_vocab))
                metric = dev["nll"] if args.selection == "nll" else dev["bleu"]
                better = (metric < best_metric - 1e-6) if args.selection == "nll" \
                         else (metric is not None and metric > best_metric + 1e-6)
                if better:
                    best_metric = metric; best_step = global_step; no_improve = 0
                    torch.save({"model_state": model.state_dict(), "step": global_step,
                                "epoch": epoch, "config": cfg,
                                "dev_nll": dev["nll"], "dev_bleu": dev["bleu"]},
                               out_dir / "best.ckpt")
                else:
                    no_improve += 1
                scheduler.step(metric if metric is not None else dev["nll"])
                log["validations"].append({
                    "step": global_step, "epoch": epoch,
                    "train_loss": loss.item(), "trans_loss": tl, "rec_loss": rl,
                    "dev_nll": dev["nll"], "dev_bleu": dev["bleu"],
                    "lr": float(optimizer.param_groups[0]["lr"]), "improved": better})
                print(f"  step {global_step} (ep{epoch}) trans={tl:.4f} rec={rl:.4f} "
                      f"dev_nll={dev['nll']:.4f} dev_bleu={(dev['bleu'] or 0):.2f} "
                      f"best={best_metric:.4f}@{best_step} lr={optimizer.param_groups[0]['lr']:.2e}"
                      f"{' *' if better else ''}", flush=True)
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"  early stop at step {global_step}"); break
            if args.smoke and global_step >= val_freq * 2:
                print("  smoke: stop"); break
        if args.patience > 0 and no_improve >= args.patience: break
        if args.smoke and global_step >= val_freq * 2: break

    log["best"] = {"step": best_step, "dev_metric": best_metric, "selection": args.selection}
    log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))

    # Copy config.yaml and vocab files so make_back_translation_model can load this dir directly
    # Override recognition_loss_weight in the saved config so it matches the actual training
    # protocol (matters when --rec-weight was passed on the CLI).
    import shutil
    cfg_dest = out_dir / "config.yaml"
    if not cfg_dest.exists():
        shutil.copy(args.config, cfg_dest)
    # Override rec_weight in the saved config
    try:
        import yaml
        with open(cfg_dest) as f: cfg_copy = yaml.safe_load(f)
        if cfg_copy and "training" in cfg_copy:
            cfg_copy["training"]["recognition_loss_weight"] = rec_weight
            with open(cfg_dest, "w") as f: yaml.safe_dump(cfg_copy, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"  WARN: could not override rec_weight in saved config: {e}", flush=True)
    for vocab_name in ["txt.vocab", "gls.vocab"]:
        vocab_src = Path(args.txt_vocab if "txt" in vocab_name else args.gls_vocab)
        vocab_dest = out_dir / vocab_name
        if not vocab_dest.exists() and vocab_src.exists():
            shutil.copy(vocab_src, vocab_dest)

    print(f"\nDone. best {args.selection}={best_metric:.4f} @ step {best_step}")
    print(f"  {out_dir}/best.ckpt  +  {out_dir}/training_log.json")


if __name__ == "__main__":
    main()
