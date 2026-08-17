#!/usr/bin/env python3
"""Faithful reconstruction of the released SLRTP2025 BT evaluator training loop.

Fixes vs train_matched_v3.py (each disclosed in the SI config-mapping table):
1. translation_normalization: batch -> CE(reduction="sum") / n_sentences.
   Verified against the released validations.txt: step-14 Translation Loss
   43,440.80078 = 7,813 dev target tokens * ln(PPL 259.84015).
2. CTC loss likewise sum / n_sentences (released Recognition Loss 17.28 at
   step 14 = dev CTC sum / 515 sentences).
3. Mask bug fixed: token counts use txt_mask.sum() (True = valid position),
   not (~txt_mask).sum() which counted padding.
4. Config-driven epochs (3000), patience (5), random_seed (42),
   validation_freq (14 optimizer steps) -- no CLI defaults that override them.
5. Stop rule: LR floor (learning_rate_min = 1e-8) by default. The released
   run improved last at validation #130 (step 1820, BLEU 13.38) and then ran
   72 further non-improving validations to step 2828 (= 101 epochs exactly),
   so patience-5 early stopping is provably NOT the released stopping rule;
   the LR-floor rule is the only stopping condition reproducible from the
   released config. (--stop-rule patience kept for comparison, default off.)
6. Plateau scheduler: ReduceLROnPlateau(mode="max", factor=0.8, patience=5,
   min_lr=1e-8) stepped once per validation on dev BLEU; LR logged AFTER the
   step. The released log's 20 decays each fire on the 6th consecutive
   non-improving validation, matching this wiring exactly.
7. best initialised to -inf so the first validation (BLEU 0.00) counts as an
   improvement, matching the released log's first `*` marker.
8. Joint CTC+translation always on (recognition_loss_weight 1.0); no
   --rec-weight override exists in this script.
9. Gradient clipping exposed as --clip (default 1.0, kept from v1-v3;
   JoeyNMT-framework default clipping threshold, documented as author
   inference in the mapping table). Pre-clip grad norms logged for the
   first 10 optimizer steps.
10. Emits validations.txt in the released line format (WER/CHRF/ROUGE
    columns omitted; we do not compute them) for line-by-line diffing.

Usage (CUDA_VISIBLE_DEVICES pins the physical GPU BEFORE torch starts):
    CUDA_VISIBLE_DEVICES=0 python -m src.training.train_faithful \
        --output checkpoints/faithful/seed_42
"""
from __future__ import annotations
import argparse, json, math, os, shutil, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml, sacrebleu
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.slrtp_dataset import SLRTPDataset, Vocab, PAD_TOKEN, EOS_TOKEN, load_pickle
from src.models import build_model
from src.models.back_translate import back_translate
from src.utils.seed import set_seed
from src.utils.hashing import sha256_file

GlsBlank = 0; GlsPad = 2
PROTOCOL = "faithful-joint-ctc-batchnorm-stepval14-plateau5-bleu-beam3"


def collate_v2(batch, txt_pad, gls_vocab, gls_pad=GlsPad):
    B = len(batch); feat_dim = batch[0]["sgn"].shape[-1]
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
        sgn[i, :T] = it["sgn"]; sgn_mask[i, 0, :T] = True; sgn_lengths[i] = T
        ids.append(it["id"])
        full = it["txt"]
        if full.numel() > 1:
            U = full.numel() - 1
            txt_input[i, :U] = full[:-1]; txt_output[i, :U] = full[1:]; txt_mask[i, 0, :U] = True
        gtoks = it["gls"][:G_max] if len(it["gls"]) > G_max else it["gls"]
        gtok_ids = [gls_vocab.stoi.get(t, 1) for t in gtoks]
        gls[i, :len(gtok_ids)] = torch.tensor(gtok_ids, dtype=torch.long); gls_lengths[i] = len(gtok_ids)
    return {"sgn": sgn, "sgn_mask": sgn_mask, "sgn_lengths": sgn_lengths,
            "txt_input": txt_input, "txt_output": txt_output, "txt_mask": txt_mask,
            "gls": gls, "gls_lengths": gls_lengths, "ids": ids}


def compute_loss_faithful(model, batch, device, rec_weight, trans_weight):
    """translation_normalization: batch -> token-sum CE / n_sentences (JoeyNMT).

    The bt_model.get_loss_for_batch docstring ("sum of losses over non-pad
    elements", divided by batch size for 'batch' normalization) and the
    released validations.txt magnitudes both pin this semantics.
    """
    B = batch["sgn"].size(0)
    out = model(sgn=batch["sgn"], sgn_mask=batch["sgn_mask"], sgn_lengths=batch["sgn_lengths"],
                txt_input=batch["txt_input"], txt_mask=batch["txt_mask"])
    decoder_outputs, gloss_probs = out
    word_outputs = decoder_outputs[0]
    logits = word_outputs.reshape(-1, word_outputs.size(-1))
    targets = batch["txt_output"].reshape(-1)
    trans_loss = F.cross_entropy(logits, targets, ignore_index=model.txt_pad_index,
                                 reduction="sum") / B
    loss = trans_weight * trans_loss
    rec_loss_v = 0.0
    if gloss_probs is not None:
        rec_loss = F.ctc_loss(gloss_probs, batch["gls"], batch["sgn_lengths"].long(),
                              batch["gls_lengths"].long(), blank=GlsBlank, reduction="sum",
                              zero_infinity=True) / B
        loss = loss + rec_weight * rec_loss
        rec_loss_v = rec_loss.item()
    return loss, trans_loss.item(), rec_loss_v


@torch.no_grad()
def evaluate_dev_faithful(model, dev_items_raw, dev_loader, device, subsample):
    """Dev NLL (token-sum, mask-fixed), dev CTC (per-sentence), beam-3 BLEU.

    Returns released-comparable quantities:
      nll_sum      : total token-sum CE over dev (matches released
                     "Translation Loss" column, e.g. 43,440.8 at step 14)
      ppl          : exp(nll_sum / n_valid_tokens) (released PPL 259.84)
      rec_per_sent : dev CTC sum / n_sentences (released Recognition Loss 17.28)
      bleu         : sacreBLEU of autoregressive beam-3 decodes
    """
    saved_do_rec = model.do_recognition
    model.do_recognition = True
    model.eval()
    total_nll, total_tokens, total_rec, total_sents = 0.0, 0, 0.0, 0
    for batch in dev_loader:
        for k in batch:
            if torch.is_tensor(batch[k]): batch[k] = batch[k].to(device)
        out = model(sgn=batch["sgn"], sgn_mask=batch["sgn_mask"], sgn_lengths=batch["sgn_lengths"],
                    txt_input=batch["txt_input"], txt_mask=batch["txt_mask"])
        decoder_outputs, gloss_probs = out
        logits = decoder_outputs[0].reshape(-1, decoder_outputs[0].size(-1))
        nll = F.cross_entropy(logits, batch["txt_output"].reshape(-1),
                              ignore_index=model.txt_pad_index, reduction="sum")
        # FIX: txt_mask True = valid token; count valid targets, not padding.
        n_tok = batch["txt_mask"].reshape(batch["txt_mask"].shape[0], -1).sum().item()
        total_nll += nll.item(); total_tokens += n_tok
        total_sents += batch["sgn"].size(0)
        if gloss_probs is not None:
            rec = F.ctc_loss(gloss_probs, batch["gls"], batch["sgn_lengths"].long(),
                             batch["gls_lengths"].long(), blank=GlsBlank, reduction="sum",
                             zero_infinity=True)
            total_rec += rec.item()
    nll_per_tok = total_nll / max(1, total_tokens)
    ppl = math.exp(min(nll_per_tok, 20.0))

    # Autoregressive beam-3 decode (matches eval_translation_beam_size=3).
    poses = []
    for it in dev_items_raw:
        p = it["poses_3d"]
        if not isinstance(p, torch.Tensor): p = torch.as_tensor(np.asarray(p, dtype=torch.float32))
        if subsample and subsample > 1: p = p[::subsample]
        poses.append(p)
    if not hasattr(model.txt_vocab, 'arrays_to_sentences'):
        _eos = model.txt_vocab.stoi.get(EOS_TOKEN, -1)
        _pad = model.txt_pad_index
        def _a2s(arrays, _eos=_eos, _pad=_pad):
            out = []
            for arr in arrays:
                toks = []
                for idx in arr:
                    idx = int(idx)
                    if idx == _pad: continue
                    if _eos >= 0 and idx == _eos: break
                    toks.append(model.txt_vocab.itos[idx])
                out.append(toks)
            return out
        model.txt_vocab.arrays_to_sentences = _a2s
    hyps = back_translate(model, poses)
    refs = [it.get("text", "") for it in dev_items_raw]
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    model.do_recognition = saved_do_rec  # restore for next train epoch (CTC loss)
    model.train()
    return {"nll_sum": total_nll, "n_tok": total_tokens, "nll_per_tok": nll_per_tok,
            "ppl": ppl, "rec_per_sent": total_rec / max(1, total_sents),
            "n_sents": total_sents, "bleu": bleu}


def format_validation_line(step, dev, lr, improved):
    """Released validations.txt line format (WER/CHRF/ROUGE columns omitted;
    we do not compute them)."""
    return (f"Steps: {step}\tRecognition Loss: {dev['rec_per_sent']:.5f}\t"
            f"Translation Loss: {dev['nll_sum']:.5f}\tPPL: {dev['ppl']:.5f}\t"
            f"Eval Metric: bleu\tBLEU-4 {dev['bleu']:.2f}\t"
            f"LR: {lr:.8f}\t{'*' if improved else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/released.yaml")
    ap.add_argument("--train-pickle",
                    default="data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
    ap.add_argument("--dev-pickle",
                    default="data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
    ap.add_argument("--txt-vocab", default="checkpoints/released/backTranslation_PHIX_model/txt.vocab")
    ap.add_argument("--gls-vocab", default="checkpoints/released/backTranslation_PHIX_model/gls.vocab")
    ap.add_argument("--seed", type=int, default=None,
                    help="Default: cfg.training.random_seed (42) -- the faithful default.")
    ap.add_argument("--gpu", type=int, default=0,
                    help="Informational; set CUDA_VISIBLE_DEVICES externally BEFORE Python starts.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Default: cfg.training.epochs (3000).")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Default: cfg.training.batch_size (256).")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=None,
                    help="Default: cfg.training.patience (5). Only used when --stop-rule patience.")
    ap.add_argument("--stop-rule", choices=["lr_floor", "patience", "none"], default="lr_floor",
                    help="lr_floor (default): stop when the plateau scheduler reaches "
                         "learning_rate_min. The released run ran 72 consecutive "
                         "non-improving validations before ending, so patience-5 early "
                         "stopping is provably not the released rule.")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="Grad-norm clip (v1-v3 hardcoded 1.0; JoeyNMT-framework default, "
                         "author inference).")
    ap.add_argument("--max-validations", type=int, default=1200,
                    help="Hard safety cap (LR floor expected at ~450 validations).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="Stop after 4 validations (56 steps).")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = args.seed if args.seed is not None else int(cfg["training"]["random_seed"])
    epochs = args.epochs if args.epochs is not None else int(cfg["training"]["epochs"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["training"]["batch_size"])
    patience = args.patience if args.patience is not None else int(cfg["training"]["patience"])
    min_lr = float(cfg["training"].get("learning_rate_min", 1e-8))

    # Seed immediately: before dataset/model/loader construction.
    set_seed(seed)
    if "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.device_count() > 1:
        print(f"WARNING: CUDA_VISIBLE_DEVICES not set externally; torch sees "
              f"{torch.cuda.device_count()} devices and will use device 0.", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

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
    dev_items_raw = load_pickle(args.dev_pickle)
    print(f"Faithful: seed={seed} epochs={epochs} batch={batch_size} patience={patience} "
          f"stop_rule={args.stop_rule} clip={args.clip} | Train: {len(train_ds)} | Dev: {len(dev_ds)}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=lambda b: collate_v2(b, pad_idx, gls_vocab))
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=lambda b: collate_v2(b, pad_idx, gls_vocab))

    rec_weight = float(cfg["training"].get("recognition_loss_weight", 1.0))
    trans_weight = float(cfg["training"].get("translation_loss_weight", 1.0))
    val_freq = int(cfg["training"].get("validation_freq", 14))
    assert rec_weight > 0 and trans_weight > 0, \
        "faithful script requires the released joint CTC+translation loss (weights 1.0/1.0)"
    do_rec, do_tr = True, True

    model = build_model(cfg=cfg["model"], gls_vocab=gls_vocab, txt_vocab=txt_vocab,
                        sgn_dim=feat_size, do_recognition=do_rec, do_translation=do_tr).to(device)
    model.txt_pad_index = pad_idx
    model.txt_vocab = txt_vocab
    model.gls_vocab = gls_vocab
    model.beam_size = int(cfg["testing"].get("eval_translation_beam_size", 3))
    model.beam_alpha = float(cfg["testing"].get("eval_translation_beam_alpha", -1))
    model.max_output_len = int(cfg["testing"].get("translation_max_output_length", 30))
    print(f"SignModel params={sum(p.numel() for p in model.parameters()):,} "
          f"beam={model.beam_size} val_freq={val_freq}steps selection=bleu (beam-3)", flush=True)

    opt_cfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(opt_cfg["learning_rate"]),
                                 betas=tuple(opt_cfg.get("betas", [0.9, 0.998])),
                                 weight_decay=float(opt_cfg.get("weight_decay", 0.001)))
    # mode=max on dev BLEU; scheduler patience = config patience (5). The released
    # log decays on the 6th consecutive non-improving validation, exactly
    # ReduceLROnPlateau(patience=5) semantics stepped once per validation.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(opt_cfg.get("decrease_factor", 0.8)),
        patience=patience, min_lr=min_lr)

    log = {"seed": seed, "config": str(args.config), "config_sha256": sha256_file(args.config),
           "protocol": PROTOCOL, "rec_weight": rec_weight, "trans_weight": trans_weight,
           "validation_freq_steps": val_freq, "selection": "bleu", "beam_size": model.beam_size,
           "batch_size": batch_size, "grad_accum": args.grad_accum,
           "translation_normalization": "batch", "loss_reduction": "token-sum / n_sentences",
           "stop_rule": args.stop_rule, "patience": patience, "clip": args.clip,
           "max_validations": args.max_validations,
           "stop_rule_note": ("released run improved last at val#130 (step 1820) then ran 72 "
                              "non-improving validations to step 2828; patience-5 early stop is "
                              "provably not the released rule; lr_floor is the only stopping "
                              "condition reproducible from the released config"),
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "validations": [], "first_grad_norms": [], "best": None, "stop_reason": None}

    best_metric = float("-inf")  # first validation (BLEU 0.00) counts as improved, as in released log
    best_step = -1; no_improve = 0; global_step = 0; stop_reason = None
    val_fh = open(out_dir / "validations.txt", "w")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            for k in batch:
                if torch.is_tensor(batch[k]): batch[k] = batch[k].to(device)
            loss, tl, rl = compute_loss_faithful(model, batch, device, rec_weight, trans_weight)
            (loss / args.grad_accum).backward()
            if (global_step + 1) % args.grad_accum == 0:
                if len(log["first_grad_norms"]) < 10:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                    log["first_grad_norms"].append(round(float(gn), 4))
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                optimizer.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % val_freq == 0:
                dev = evaluate_dev_faithful(model, dev_items_raw, dev_loader, device, subsample)
                metric = dev["bleu"]
                better = metric > best_metric  # strict, matches released `*` marks
                if better:
                    best_metric = metric; best_step = global_step; no_improve = 0
                    torch.save({"model_state": model.state_dict(), "step": global_step,
                                "epoch": epoch, "config": cfg,
                                "dev_nll": dev["nll_per_tok"], "dev_nll_sum": dev["nll_sum"],
                                "dev_ppl": dev["ppl"], "dev_bleu": dev["bleu"],
                                "lr": float(optimizer.param_groups[0]["lr"])},
                               out_dir / "best.ckpt")
                else:
                    no_improve += 1
                scheduler.step(metric)
                lr = float(optimizer.param_groups[0]["lr"])  # post-step, as in released log
                log["validations"].append({"step": global_step, "epoch": epoch,
                                           "trans_loss_per_sent": tl, "rec_loss_per_sent": rl,
                                           "dev_nll_sum": dev["nll_sum"],
                                           "dev_nll_per_tok": dev["nll_per_tok"],
                                           "dev_ppl": dev["ppl"],
                                           "dev_rec_per_sent": dev["rec_per_sent"],
                                           "dev_bleu": dev["bleu"], "lr": lr, "improved": better})
                val_fh.write(format_validation_line(global_step, dev, lr, better) + "\n")
                val_fh.flush()
                print(f"  step {global_step} (ep{epoch}) trans/sent={tl:.2f} rec/sent={rl:.2f} "
                      f"devNLLsum={dev['nll_sum']:.1f} ppl={dev['ppl']:.2f} "
                      f"bleu={dev['bleu']:.2f} best={best_metric:.2f}@{best_step} "
                      f"lr={lr:.2e}{' *' if better else ''}", flush=True)

                n_vals = len(log["validations"])
                if args.smoke and n_vals >= 4:
                    stop_reason = "smoke"; break
                if args.stop_rule == "lr_floor" and lr <= min_lr * (1.0 + 1e-9):
                    stop_reason = "lr_floor"; print(f"  LR floor {min_lr} reached; stop", flush=True)
                    break
                if args.stop_rule == "patience" and no_improve >= patience:
                    stop_reason = f"patience({patience})"; print(f"  early stop at step {global_step}", flush=True)
                    break
                if n_vals >= args.max_validations:
                    stop_reason = "max_validations"; print(f"  max validations reached; stop", flush=True)
                    break
        if stop_reason: break

    val_fh.close()
    log["best"] = {"step": best_step, "dev_metric": best_metric, "selection": "bleu"}
    log["stop_reason"] = stop_reason or f"epochs({epochs})"
    log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))

    # Verbatim config copy (no overrides -- faithful trains exactly what the
    # released config specifies) + vocab copies so make_back_translation_model
    # can load this directory directly.
    cfg_dest = out_dir / "config.yaml"
    if not cfg_dest.exists():
        shutil.copy(args.config, cfg_dest)
    for vocab_name, vocab_src in [("txt.vocab", Path(args.txt_vocab)), ("gls.vocab", Path(args.gls_vocab))]:
        vocab_dest = out_dir / vocab_name
        if not vocab_dest.exists() and vocab_src.exists():
            shutil.copy(vocab_src, vocab_dest)

    print(f"\nDone (stop={log['stop_reason']}). best bleu={best_metric:.4f} @ step {best_step}")
    print(f"  {out_dir}/best.ckpt + training_log.json + validations.txt")


if __name__ == "__main__":
    main()
