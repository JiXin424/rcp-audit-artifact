#!/usr/bin/env python3
"""Noisy-gloss retrieval on the SLRTP2025 PHX-public test set.

Recovered from the round-3 reviewer-revision pipeline (2026-07-21): predict a
gloss for each test sentence with a text->gloss NMT (weak word-level BLEU-4
1.15 or strong BPE BLEU-4 13.17), then retrieve the nearest training donor by
token-set Jaccard over the *predicted* (noisy) gloss, excluding exact gloss
matches. This is the deployment-realistic analogue of the oracle TN-PURE-v1
retrieval and underlies the SI noisy-gloss retrieval numbers (Sup. P).

Outputs (results/noisy_gloss/):
  noisy_gloss_public.pt        names, texts, oracle gloss, predicted gloss
  NG_public_donor_copy.pt      donor-copy poses keyed by test key

Usage: python3 scripts/predict_noisy_gloss.py --mode strong --gpu 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_text_to_gloss_nmt import NMT, WeakNMT, greedy_decode

CKPTS = {
    # weak: word-level, shared vocab in checkpoint
    "weak": "/ssd/SignDiff_checkpoints/text_to_gloss_nmt_20260721_round3/best.pt",
    # strong: SentencePiece BPE, shared vocab, config in checkpoint
    "strong": "/ssd/SignDiff_checkpoints/text_to_gloss_nmt_20260721_round4_strong/best.pt",
}


def text_jaccard(a, b):
    sa = set(str(a).split())
    sb = set(str(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weak", "strong"], default="strong")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="override NMT checkpoint path")
    ap.add_argument("--out", default=str(ROOT / "results/noisy_gloss"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(CKPTS[args.mode])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if not cfg:
        # weak checkpoint predates config; infer from checkpoint content
        src_vocab = dict(ckpt["src_vocab"])
        tgt_vocab = dict(ckpt["tgt_vocab"])
        cfg = {"d_model": 256, "n_head": 4, "n_layer": 4, "ff_dim": 1024,
               "dropout": 0.1, "max_len": 60}
        model = WeakNMT(src_vocab, tgt_vocab)
        model.load_state_dict(ckpt["model"])
        sp = None
    else:
        vocab_size = ckpt["vocab_size"]
        model = NMT(vocab_size, cfg["d_model"], cfg["n_head"], cfg["n_layer"],
                    cfg["ff_dim"], cfg["dropout"], cfg["max_len"])
        model.load_state_dict(ckpt["model"])
        import sentencepiece as spm
        bpe_model = cfg.get("bpe_model") or str(ckpt_path.parent / "bpe.model")
        sp = spm.SentencePieceProcessor(model_file=bpe_model)
        src_vocab = tgt_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    model.to(device)
    model.eval()
    print(f"[{args.mode}] loaded NMT epoch={ckpt['epoch']} dev_bleu4={ckpt['bleu']:.2f}",
          flush=True)

    test_data = torch.load(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt",
                           map_location="cpu", weights_only=False)
    train_data = torch.load(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt",
                            map_location="cpu", weights_only=False)
    test_keys = list(test_data.keys())
    test_texts = [test_data[k]["text"] for k in test_keys]
    test_gloss_oracle = [test_data[k].get("gloss", "") for k in test_keys]
    print(f"#test: {len(test_keys)}", flush=True)

    # ---- predict noisy gloss (greedy, batched) ----
    BOS, EOS = tgt_vocab["<bos>"], tgt_vocab["<eos>"]
    IV_TGT = {v: k for k, v in tgt_vocab.items()}
    max_len = cfg["max_len"]
    pred_gloss = []
    with torch.no_grad():
        for start in range(0, len(test_texts), 32):
            batch_texts = test_texts[start:start + 32]
            src_tensors = []
            for t in batch_texts:
                t = t.lower()
                if sp is not None:
                    ids = sp.encode(t) + [EOS]  # source carries no BOS (released convention)
                    ids = ids[:max_len]
                else:
                    toks = t.split()
                    ids = [src_vocab.get(tk, src_vocab["<unk>"]) for tk in toks][:max_len - 1] + [EOS]
                src_tensors.append(torch.tensor(ids, dtype=torch.long))
            src = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=0).to(device)
            dec = greedy_decode(model, src, (src == 0), max_len, BOS, EOS)
            for i in range(src.size(0)):
                toks = []
                for x in dec[i, 1:].tolist():
                    if x == EOS:
                        break
                    toks.append(sp.IdToPiece(int(x)) if sp is not None else IV_TGT.get(x, "<unk>"))
                pred_gloss.append(sp.DecodePieces(toks) if sp is not None else " ".join(toks))

    for i in range(3):
        print(f"  test[{i}] text={test_texts[i][:60]!r}", flush=True)
        print(f"         oracle={test_gloss_oracle[i][:60]!r}", flush=True)
        print(f"         pred  ={pred_gloss[i][:60]!r}", flush=True)

    torch.save({"names": test_keys, "texts": test_texts,
                "oracle_gloss": test_gloss_oracle, "pred_gloss": pred_gloss},
               out_dir / f"noisy_gloss_public_{args.mode}.pt")
    print("saved noisy_gloss_public", flush=True)

    mean_jacc = sum(text_jaccard(p, o) for p, o in zip(pred_gloss, test_gloss_oracle)) / len(test_keys)
    print(f"mean Jaccard(predicted, oracle) gloss: {mean_jacc:.4f}", flush=True)

    # ---- retrieve nearest donor by predicted-gloss Jaccard ----
    train_keys = list(train_data.keys())
    train_gloss = [train_data[k].get("gloss", "") for k in train_keys]
    train_poses = [train_data[k]["poses_3d"] for k in train_keys]
    print(f"#train donors: {len(train_keys)}", flush=True)

    out_dict = {}
    jacc_vs_donor = []
    for i, tname in enumerate(test_keys):
        qg = pred_gloss[i]
        best_j, best_jac = 0, -1.0
        for j, tg in enumerate(train_gloss):
            if tg == qg:
                continue  # exact-gloss exclusion
            jac = text_jaccard(qg, tg)
            if jac > best_jac:
                best_jac, best_j = jac, j
        out_dict[tname] = train_poses[best_j].clone()
        jacc_vs_donor.append(best_jac)
    print(f"mean Jaccard(predicted query, retrieved donor): "
          f"{sum(jacc_vs_donor) / len(jacc_vs_donor):.4f}", flush=True)

    out_path = out_dir / f"NG_public_donor_copy_{args.mode}.pt"
    torch.save(out_dict, out_path)
    print(f"saved {len(out_dict)} noisy-gloss donor-copy poses -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
