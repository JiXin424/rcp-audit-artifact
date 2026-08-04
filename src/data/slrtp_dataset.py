#!/usr/bin/env python3
"""SLRTP2025 dataset loader.

Loads the SLRTP2025 BT-evaluator training data (JoeyNMT-style pickles) and
produces torch tensors ready for SignModel.forward(sgn, sgn_mask, sgn_lengths,
txt_input, txt_mask).

Provenance: written fresh in 2026-08-02 to replace the lost
`revision_20260728_major/src/data.py`. The pickle schema follows
SignDiff/SLRTP2025_eval/back_translation/load_pickle_file().
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Vocabulary special tokens (must match SignModel's expectations)
SIL_TOKEN = "<si>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


# -------------------------------------------------------------------- vocab
class Vocab:
    """Minimal vocab matching JoeyNMT's stoi/itos interface."""

    def __init__(self, stoi: Dict[str, int], itos: List[str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def from_file(cls, path: Path) -> "Vocab":
        """Load a JoeyNMT-style vocabulary file.

        File format: one token per line; line number (1-indexed) is the index.
        JoeyNMT writes special tokens explicitly so we read them as-is.
        """
        itos: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n")
                if not tok:
                    continue
                itos.append(tok)
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)


# -------------------------------------------------------------------- pickle
def load_pickle(path: Path | str) -> dict:
    """Load a JoeyNMT-style data pickle produced by SLRTP2025 preprocessing.

    SLRTP2025 stores splits as torch.save'd .pt files, each containing a dict
    keyed by sequence id; values are dicts with keys
    {'name', 'text', 'gloss', 'poses_3d', 'speaker'}.

    This loader accepts both .pt (torch.save) and .pickle (pickle.load) and
    normalizes the result to a list of item dicts (preserving original key names).
    """
    import torch
    p = Path(path)
    # Try torch.load first (SLRTP2025 .pt format), fall back to pickle.load
    try:
        raw = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception:
        with open(p, "rb") as f:
            raw = pickle.load(f)

    if isinstance(raw, dict):
        # Could be {seq_id: item_dict} or {"data": [...]}
        if "data" in raw and isinstance(raw["data"], list):
            return raw["data"]
        # Standard SLRTP2025 .pt: dict keyed by sequence id
        return list(raw.values())
    elif isinstance(raw, list):
        return raw
    else:
        raise ValueError(f"Unrecognized data format in {p}: {type(raw).__name__}")


# -------------------------------------------------------------------- dataset
class SLRTPDataset(Dataset):
    """A torch Dataset over a single SLRTP2025 split (train/dev/test).

    Each item returns a dict with keys:
      - id: str (sequence ID like '21September_2010_Tuesday_tagesschau-950')
      - sgn: FloatTensor [T, feat_dim]   (pose frame features, already [::2] subsampled)
      - gls: List[str]                    (gloss tokens, ignored by BT decoder)
      - txt: List[int]                    (text token ids, with <bos>/<eos>)

    Accepts either legacy 'sign' key (JoeyNMT pickle) or 'poses_3d' key
    (SLRTP2025 .pt). For 'poses_3d' with shape [T, J, C], frames are flattened
    to [T, J*C] to match the released model's expected input size (e.g., 178*3=534).
    """

    def __init__(
        self,
        pickle_path: Path | str,
        txt_vocab: Vocab,
        gls_vocab: Optional[Vocab] = None,
        skeleton_subsample: int = 2,
        max_sent_length: int = 400,
        txt_lowercase: bool = True,
    ):
        super().__init__()
        self.path = Path(pickle_path)
        self.txt_vocab = txt_vocab
        self.gls_vocab = gls_vocab
        self.skeleton_subsample = skeleton_subsample
        self.max_sent_length = max_sent_length
        self.txt_lowercase = txt_lowercase

        self.items = load_pickle(self.path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        # SLRTP2025 keys:
        #   'name'      -> sequence id string
        #   'poses_3d'  -> FloatArray [T, J=178, C=3] (or legacy 'sign' [T, feat_dim])
        #   'gloss'     -> str (space-separated gloss tokens)
        #   'text'      -> str (spoken-language target sentence)
        seq_id = item.get("name", str(idx))

        # Find pose tensor (support both 'poses_3d' and legacy 'sign')
        if "poses_3d" in item:
            pose = item["poses_3d"]
            if not isinstance(pose, torch.Tensor):
                pose = torch.as_tensor(np.asarray(pose, dtype=np.float32))
            # Flatten [T, J, C] -> [T, J*C] if 3D
            if pose.dim() == 3:
                pose = pose.reshape(pose.shape[0], -1)
        elif "sign" in item:
            pose = item["sign"]
            if not isinstance(pose, torch.Tensor):
                pose = torch.as_tensor(np.asarray(pose, dtype=np.float32))
        else:
            raise KeyError(f"Item {seq_id} has neither 'poses_3d' nor 'sign' key")

        if self.skeleton_subsample and self.skeleton_subsample > 1:
            pose = pose[:: self.skeleton_subsample]
        sgn = pose.float()  # [T, feat_dim]

        gls_str = item.get("gloss", "") or ""
        gls_tokens = gls_str.split()

        txt_str = item.get("text", "") or ""
        if self.txt_lowercase:
            txt_str = txt_str.lower()
        txt_tokens = txt_str.split()
        # Encode with <bos> ... <eos>
        stoi = self.txt_vocab.stoi
        ids = [stoi[BOS_TOKEN]]
        for tok in txt_tokens:
            ids.append(stoi.get(tok, stoi[UNK_TOKEN]))
            if len(ids) >= self.max_sent_length:
                break
        ids.append(stoi[EOS_TOKEN])

        return {
            "id": seq_id,
            "sgn": sgn,
            "gls": gls_tokens,
            "txt": torch.tensor(ids, dtype=torch.long),
        }


# -------------------------------------------------------------------- collate
def collate_batch(batch: List[dict], pad_idx: int) -> dict:
    """Pad a batch of variable-length sign/text into aligned tensors.

    Mask convention (JoeyNMT, matches SignModel expectations):
      sgn_mask: [B, 1, T_max]   True = VALID position (False = padding)
      txt_mask: [B, 1, U_max]   True = VALID position (False = padding)

    This is the opposite of the more common "True=padding" convention,
    but matches the masked_fill(~mask, -inf) idiom used inside
    MultiHeadedAttention. The leading `1` is needed so that
    `txt_mask & subsequent_mask(U_max)` broadcasts to (B, U_max, U_max)
    inside the transformer decoder.

    Returns:
      sgn:        [B, T_max, feat_dim]
      sgn_mask:   [B, 1, T_max]   (True = VALID)
      sgn_lengths:[B]              (original lengths before padding)
      txt_input:  [B, U_max]      (text shifted right: <bos> t1 t2 ...)
      txt_output: [B, U_max]      (text shifted left:  t1 t2 ... <eos>)
      txt_mask:   [B, 1, U_max]   (True = VALID)
      ids:        List[str]
    """
    B = len(batch)
    feat_dim = batch[0]["sgn"].shape[-1]
    T_max = max(it["sgn"].shape[0] for it in batch)
    U_max = max(it["txt"].shape[0] for it in batch)

    sgn = torch.zeros(B, T_max, feat_dim, dtype=torch.float32)
    sgn_mask = torch.zeros(B, 1, T_max, dtype=torch.bool)  # False=pad, JoeyNMT 3D convention (True=valid)
    sgn_lengths = torch.zeros(B, dtype=torch.long)
    txt_input = torch.full((B, U_max), pad_idx, dtype=torch.long)
    txt_output = torch.full((B, U_max), pad_idx, dtype=torch.long)
    txt_mask = torch.zeros(B, 1, U_max, dtype=torch.bool)
    ids = []

    for i, it in enumerate(batch):
        T = it["sgn"].shape[0]
        sgn[i, :T] = it["sgn"]
        sgn_mask[i, 0, :T] = True   # valid positions
        sgn_lengths[i] = T
        ids.append(it["id"])

        # For teacher-forced training: input = <bos> t1 ... t_{n-1}
        # output = t1 t2 ... <eos>; i.e. shifted by one
        full = it["txt"]  # [bos, t1, ..., tn, eos]
        if full.numel() > 1:
            U = full.numel() - 1
            txt_input[i, :U] = full[:-1]
            txt_output[i, :U] = full[1:]
            txt_mask[i, 0, :U] = True   # valid positions

    return {
        "ids": ids,
        "sgn": sgn,
        "sgn_mask": sgn_mask,
        "sgn_lengths": sgn_lengths,
        "txt_input": txt_input,
        "txt_output": txt_output,
        "txt_mask": txt_mask,
    }


# -------------------------------------------------------------------- helpers
def make_collate(pad_idx: int):
    def _c(batch):
        return collate_batch(batch, pad_idx)
    return _c


def build_dataloader(
    dataset: SLRTPDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_idx: Optional[int] = None,
) -> DataLoader:
    if pad_idx is None:
        pad_idx = dataset.txt_vocab.stoi[PAD_TOKEN]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate(pad_idx),
        pin_memory=True,
        drop_last=False,
    )
