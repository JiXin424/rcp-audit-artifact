#!/usr/bin/env python3
"""Train a small Transformer NMT: PHOENIX-2014T German text -> DGS gloss.

Recovered from the round-3/round-4 reviewer-revision pipeline (2026-07-21),
which produced the two released noisy-gloss text->gloss models used for the
noisy-gloss retrieval proxy in SI (Sup. P, noisy-gloss retrieval):

  --mode weak   4-layer word-level Transformer, shared text/gloss vocab.
                dev BLEU-4 = 1.15 (the "weak NMT" of the SI table).
  --mode strong 6-layer BPE (SentencePiece, shared vocab size 4000).
                dev BLEU-4 = 13.17 (the "strong NMT" of the SI table).

The released checkpoints are mirrored in the artifact under
  model/text_to_gloss_nmt_20260721_round3/     (weak,  best.pt)
  model/text_to_gloss_nmt_20260721_round4_strong/ (strong, best.pt)
This script documents and can reproduce their training recipe. Both use the
same PHOENIX-2014T manual annotations (German text -> orthographic gloss).

Usage:
  python3 scripts/train_text_to_gloss_nmt.py --mode strong --gpu 0
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_PHOENIX_ROOT = Path("/ssd/data/public_datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual")

# ---- weak model hyper-parameters (as originally run) ----
WEAK = dict(d_model=256, n_head=4, n_layer=4, ff_dim=1024, dropout=0.1,
            batch=64, epochs=30, lr=3e-4, warmup=1000, max_len=60, seed=42)
# ---- strong model hyper-parameters (as originally run, from checkpoint config) ----
STRONG = dict(d_model=512, n_head=8, n_layer=6, ff_dim=2048, dropout=0.2,
              batch=64, epochs=300, lr=3e-4, warmup=150_000, max_len=100,
              seed=42, label_smooth=0.1, bpe_vocab=4000)


class NMT(nn.Module):
    """Strong (released) architecture: shared embedding + learned positions,
    pre-norm (norm_first=True) transformer. Verified to reproduce the released
    checkpoint's dev BLEU-4 = 13.17 exactly with the released BPE model."""
    def __init__(self, vocab_size, d, h, n, ff, do, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        self.tf = nn.Transformer(d, h, n, n, ff, do, batch_first=True,
                                 norm_first=True)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, src, tgt, src_kpm, tgt_kpm, tgt_mask=None):
        s = self.embed(src) + self.pos(torch.arange(src.size(1), device=src.device))
        t = self.embed(tgt) + self.pos(torch.arange(tgt.size(1), device=tgt.device))
        mem = self.tf.encoder(s, src_key_padding_mask=src_kpm)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        h = self.tf.decoder(t, mem, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_kpm,
                            memory_key_padding_mask=src_kpm)
        return self.out(h)


class WeakNMT(nn.Module):
    """Weak (original round-3) architecture: separate src/tgt embeddings and
    a fixed position ramp (no learned positions)."""
    def __init__(self, src_v, tgt_v, d=256, h=4, n=4, ff=1024, do=0.1):
        super().__init__()
        self.se = nn.Embedding(len(src_v), d, padding_idx=0)
        self.te = nn.Embedding(len(tgt_v), d, padding_idx=0)
        self.tf = nn.Transformer(d, h, n, n, ff, do, batch_first=True)
        self.out = nn.Linear(d, len(tgt_v))

    def forward(self, src, tgt, src_kpm, tgt_kpm, tgt_mask=None):
        src_pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        s = self.se(src) + src_pos * 0.1
        t = self.te(tgt) + tgt_pos * 0.1
        mem = self.tf.encoder(s, src_key_padding_mask=src_kpm)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        h = self.tf.decoder(t, mem, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_kpm,
                            memory_key_padding_mask=src_kpm)
        return self.out(h)


@torch.no_grad()
def greedy_decode(model, src, src_kpm, max_len, bos, eos):
    b = src.size(0)
    tgt = torch.full((b, 1), bos, device=src.device, dtype=torch.long)
    for _ in range(max_len):
        tgt_kpm = (tgt == 0)
        out = model(src, tgt, src_kpm, tgt_kpm)
        nxt = out[:, -1].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, nxt], dim=1)
        if (nxt == eos).all():
            break
    return tgt


def load_phoenix_split(split: str, root: Path):
    import csv
    path = root / f"PHOENIX-2014-T.{split}.corpus.csv"
    pairs = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            text = row["translation"].strip().lower()
            gloss = row["orth"].strip().upper()
            pairs.append((text, gloss))
    return pairs


def build_vocab(seqs, min_freq=2):
    from collections import Counter
    cnt = Counter(tok for s in seqs for tok in s.split())
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok, c in cnt.most_common():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode_word(s, vocab, max_len):
    toks = s.split()
    ids = [vocab.get(t, vocab["<unk>"]) for t in toks][:max_len - 1] + [vocab["<eos>"]]
    return ids


class PairsDS(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len, sp=None):
        self.pairs = pairs
        self.sv = src_vocab
        self.tv = tgt_vocab
        self.max_len = max_len
        self.sp = sp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        text, gloss = self.pairs[i]
        if self.sp is not None:
            # source carries no <bos> (matches released training convention)
            s = self.sp.encode(text) + [self.sv["<eos>"]]
            t = [self.tv["<bos>"]] + self.sp.encode(gloss) + [self.tv["<eos>"]]
            return s[:self.max_len], t[:self.max_len]
        return encode_word(text, self.sv, self.max_len), encode_word(gloss, self.tv, self.max_len)


def collate(batch, pad_id):
    src = [torch.tensor(s) for s, _ in batch]
    tgt = [torch.tensor(t) for _, t in batch]
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    return src, tgt


def train_model(pairs_train, pairs_dev, cfg, out_dir, sp=None, weak_arch=False):
    """Train one NMT run; returns (model, src_vocab, tgt_vocab, best_bleu)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sp is not None:
        src_vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        tgt_vocab = src_vocab
    else:
        src_vocab = build_vocab([t for t, _ in pairs_train])
        tgt_vocab = build_vocab([g for _, g in pairs_train])

    tr_dl = DataLoader(PairsDS(pairs_train, src_vocab, tgt_vocab, cfg["max_len"], sp),
                       cfg["batch"], shuffle=True,
                       collate_fn=lambda b: collate(b, 0), num_workers=2)
    dv_dl = DataLoader(PairsDS(pairs_dev, src_vocab, tgt_vocab, cfg["max_len"], sp),
                       cfg["batch"], shuffle=False,
                       collate_fn=lambda b: collate(b, 0), num_workers=2)

    if weak_arch:
        model = WeakNMT(src_vocab, tgt_vocab).to(device)
    else:
        model = NMT(len(tgt_vocab), cfg["d_model"], cfg["n_head"], cfg["n_layer"],
                    cfg["ff_dim"], cfg["dropout"], cfg["max_len"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) ** -0.5 * (cfg["warmup"] ** 0.5),
                           (s + 1) * cfg["warmup"] ** -1.5))
    ls = cfg.get("label_smooth", 0.0)

    best_bleu = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot = 0.0
        n_b = 0
        t0 = time.time()
        for src, tgt in tr_dl:
            src, tgt = src.to(device), tgt.to(device)
            src_kpm = (src == 0)
            tgt_kpm = (tgt == 0)
            logits = model(src, tgt[:, :-1], src_kpm, tgt_kpm[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                ignore_index=0, label_smoothing=ls)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n_b += 1
        # dev BLEU-4 (greedy)
        model.eval()
        hyps, refs = [], []
        iv_t = {v: k for k, v in tgt_vocab.items()}
        with torch.no_grad():
            for src, tgt in dv_dl:
                src, tgt = src.to(device), tgt.to(device)
                dec = greedy_decode(model, src, (src == 0), cfg["max_len"],
                                    tgt_vocab["<bos>"], tgt_vocab["<eos>"])
                for i in range(src.size(0)):
                    toks = []
                    for x in dec[i, 1:].tolist():
                        if x == tgt_vocab["<eos>"]:
                            break
                        toks.append(iv_t.get(x, "<unk>"))
                    hyp = sp.DecodePieces(toks) if sp is not None else " ".join(toks)
                    ref = []
                    for x in tgt[i].tolist():
                        if x == tgt_vocab["<eos>"]:
                            break
                        if x in (0, 2, 3):
                            continue
                        ref.append(iv_t.get(x, "<unk>"))
                    refs.append(sp.DecodePieces(ref) if sp is not None else " ".join(ref))
                    hyps.append(hyp)
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        elapsed = time.time() - t0
        print(f"epoch {epoch} train_loss={tot / max(1, n_b):.3f} "
              f"dev_bleu4={bleu:.2f} lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.0f}s",
              flush=True)
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save({"model": model.state_dict(), "epoch": epoch, "bleu": bleu,
                        "src_vocab": src_vocab, "tgt_vocab": tgt_vocab,
                        "vocab_size": len(tgt_vocab),
                        "config": dict(d_model=cfg["d_model"], n_head=cfg["n_head"],
                                       n_layer=cfg["n_layer"], ff_dim=cfg["ff_dim"],
                                       dropout=cfg["dropout"], max_len=cfg["max_len"],
                                       label_smooth=cfg.get("label_smooth", 0.0))},
                       out_dir / "best.pt")
    print(f"best dev BLEU = {best_bleu:.2f}", flush=True)
    return model, src_vocab, tgt_vocab, best_bleu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["weak", "strong"], default="weak")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--phoenix-root", type=Path, default=DEFAULT_PHOENIX_ROOT)
    ap.add_argument("--out", type=Path,
                    default=Path("/ssd/SignDiff_checkpoints/text_to_gloss_nmt"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = dict(WEAK if args.mode == "weak" else STRONG)
    out_dir = args.out / f"text_to_gloss_nmt_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_phoenix_split("train", args.phoenix_root)
    dev = load_phoenix_split("dev", args.phoenix_root)
    print(f"mode={args.mode} train={len(train)} dev={len(dev)}", flush=True)

    sp = None
    if args.mode == "strong":
        # SentencePiece BPE-4000 shared over text(lower)/gloss(upper), as released
        import sentencepiece as spm
        corpus = out_dir / "bpe_corpus.txt"
        if not corpus.exists():
            with corpus.open("w", encoding="utf-8") as f:
                for t, g in train:
                    f.write(t + "\n")
                    f.write(g + "\n")
        bpe_model = out_dir / "bpe.model"
        if not bpe_model.exists():
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(out_dir / "bpe"),
                vocab_size=cfg["bpe_vocab"], model_type="bpe",
                pad_id=0, unk_id=1, bos_id=2, eos_id=3)
        sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
        (out_dir / "vocab.json").write_text(json.dumps(
            {"vocab_size": cfg["bpe_vocab"], "pad_id": 0, "unk_id": 1,
             "bos_id": 2, "eos_id": 3, "max_len": cfg["max_len"],
             "bpe_model": str(bpe_model), "type": "sentencepiece_bpe",
             "shared_text_gloss": True}, indent=1))

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_model(train, dev, cfg, out_dir, sp, weak_arch=(args.mode == "weak"))


if __name__ == "__main__":
    main()
