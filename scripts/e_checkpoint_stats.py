#!/usr/bin/env python3
"""Per-checkpoint teacher-forced stats for reviewer M2 sanity checks.

Computes for each evaluator checkpoint:
  - Train/dev teacher-forced NLL per token
  - Train/dev token accuracy (argmax of teacher-forced logits vs target, ignore pad)
  - Train/dev free-decode BLEU (subset)
  - Sequence length statistics

Usage: python scripts/e_checkpoint_stats.py --gpu 0 --only released
       bash /tmp/run_stats.sh  # parallel across 8 GPUs
Output: results/checkpoint_stats.json (incremental)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.back_translate import make_back_translation_model
from src.data.slrtp_dataset import SLRTPDataset, Vocab, PAD_TOKEN, collate_batch

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

CHECKPOINTS = {
    "released": "checkpoints/released/backTranslation_PHIX_model",
    **{f"reco_{s}": f"checkpoints/reconstructions/seed_{s}"
       for s in ["101", "202", "303", "404", "505", "606",
                 "707", "808", "909", "1001", "1102", "1203", "1304", "1405"]},
}

TRAIN_PATH = str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt")
DEV_PATH = str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt")
TXT_VOCAB_PATH = str(ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab")
GLS_VOCAB_PATH = str(ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab")


def token_accuracy_batched(model, dataset, device, max_samples=2000, batch_size=16):
    """Teacher-forced token accuracy and NLL, batched via collate_batch."""
    model.eval()
    pad_id = model.txt_pad_index
    correct = 0
    total = 0
    total_nll = 0.0
    with torch.no_grad():
        indices = list(range(len(dataset)))
        if len(indices) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(indices, max_samples, replace=False).tolist()
        for start in range(0, len(indices), batch_size):
            batch_items = [dataset[i] for i in indices[start:start + batch_size]]
            batch = collate_batch(batch_items, pad_id)
            sgn = batch["sgn"].to(device)
            sgn_mask = batch["sgn_mask"].to(device)
            sgn_len = batch["sgn_lengths"].to(device)
            txt_in = batch["txt_input"].to(device)
            txt_out = batch["txt_output"].to(device)

            txt_mask = batch["txt_mask"].to(device)
            decoder_outputs, _ = model(sgn, sgn_mask, sgn_len, txt_in, txt_mask)
            logits = decoder_outputs[0]  # (outputs, hidden, att_probs, att_vectors)
            pred = logits.argmax(dim=-1)
            non_pad = (txt_out != pad_id)
            correct += (pred[non_pad] == txt_out[non_pad]).sum().item()
            total += non_pad.sum().item()

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), txt_out.reshape(-1),
                ignore_index=pad_id, reduction="sum")
            total_nll += loss.item()
    return correct / max(total, 1), total_nll / max(total, 1), total


def free_decode_bleu_batched(model, dataset, device, max_samples=500, batch_size=8):
    """Free-decode BLEU on a subset, batched via collate_batch."""
    model.eval()
    pad_id = model.txt_pad_index
    hyps, refs = [], []
    indices = list(range(len(dataset)))
    if len(indices) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, max_samples, replace=False).tolist()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_items = [dataset[i] for i in indices[start:start + batch_size]]
            batch = collate_batch(batch_items, pad_id)
            # Extract raw text references from dataset raw items (list of dicts)
            for idx in indices[start:start + batch_size]:
                raw_item = dataset.items[idx]
                refs.append(raw_item.get("text", ""))
            sgn = batch["sgn"].to(device)
            sgn_mask = batch["sgn_mask"].to(device)
            sgn_len = batch["sgn_lengths"].to(device)
            batch_obj = type('B', (), {'sgn': sgn, 'sgn_mask': sgn_mask,
                                        'sgn_lengths': sgn_len, 'make_cuda': lambda: None})()
            decoded = model.run_batch(
                batch=batch_obj, recognition_beam_size=None, translation_beam_size=3,
                translation_beam_alpha=-1, translation_max_output_length=50,
            )[1]
            hyps.extend(decoded)
    # Decode token arrays to strings
    decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=hyps)
    decoded_txt = [' '.join(t) for t in decoded_txt]
    return BLEU.corpus_score(decoded_txt, [refs]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--train-samples", type=int, default=2000)
    ap.add_argument("--decode-samples", type=int, default=500)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    out_path = ROOT / "results/checkpoint_stats.json"
    prev = json.loads(out_path.read_text()) if out_path.exists() else {}

    txt_vocab = Vocab.from_file(Path(TXT_VOCAB_PATH))
    gls_vocab = Vocab.from_file(Path(GLS_VOCAB_PATH))
    train_ds = SLRTPDataset(TRAIN_PATH, txt_vocab=txt_vocab, gls_vocab=gls_vocab, skeleton_subsample=2)
    dev_ds = SLRTPDataset(DEV_PATH, txt_vocab=txt_vocab, gls_vocab=gls_vocab, skeleton_subsample=2)

    keys = [args.only] if args.only else list(CHECKPOINTS)
    for name in keys:
        if name in prev:
            print(f"  {name}: already done, skip")
            continue
        ckpt_dir = str(ROOT / CHECKPOINTS[name])
        if not Path(ckpt_dir, "best.ckpt").exists():
            print(f"  {name}: no best.ckpt, skip")
            continue

        print(f"\n{'='*60}\n{name}\n{'='*60}")
        model = make_back_translation_model(ckpt_dir)

        print(f"  Train token acc ({args.train_samples} samples)...")
        t_acc, t_nll, t_tokens = token_accuracy_batched(model, train_ds, device, args.train_samples)
        print(f"  Dev token acc (all {len(dev_ds)} samples)...")
        d_acc, d_nll, d_tokens = token_accuracy_batched(model, dev_ds, device, max_samples=10000)
        print(f"  Free-decode train ({args.decode_samples} samples)...")
        t_bleu = free_decode_bleu_batched(model, train_ds, device, args.decode_samples)
        print(f"  Free-decode dev ({args.decode_samples} samples)...")
        d_bleu = free_decode_bleu_batched(model, dev_ds, device, args.decode_samples)

        # Extract training_log NLL for train
        log_path = Path(ckpt_dir) / "training_log.json"
        train_nll_from_log = None
        if log_path.exists():
            log = json.loads(log_path.read_text())
            if isinstance(log, dict) and "best" in log and isinstance(log["best"], dict):
                train_nll_from_log = log["best"].get("dev_nll")

        result = {
            "train_nll_per_token": round(t_nll, 4),
            "train_token_accuracy": round(t_acc, 4),
            "train_tokens_evaluated": t_tokens,
            "train_free_decode_bleu_subset": round(t_bleu, 2),
            "dev_nll_per_token": round(d_nll, 4),
            "dev_token_accuracy": round(d_acc, 4),
            "dev_tokens_evaluated": d_tokens,
            "dev_free_decode_bleu_subset": round(d_bleu, 2),
            "train_nll_from_training_log": train_nll_from_log,
        }
        prev[name] = result
        out_path.write_text(json.dumps(prev, indent=1))
        print(f"  => train_acc={t_acc:.4f} dev_acc={d_acc:.4f} dev_bleu={d_bleu:.2f}")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
