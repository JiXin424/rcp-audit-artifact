#!/usr/bin/env python3
"""Rebuild USTC-MoE (SLRTP2025 winner) fixed test outputs from the official
CVPRW-SLP-2025 pipeline intermediates.

Faithful to official main.py:
  1. text2gloss predictions (align_hyp, '*' tokens dropped) per test sentence;
  2. if the sentence text matches a training text -> copy that train pose
     (truncated/padded to the target frame count);
  3. otherwise concatenate per-gloss poses from the released gloss->pose
     dictionary (longest variant first), truncated/padded to target length.

Target frame counts come from the official frame_lengths.csv. Only the 500
challenging-set sentences have released text2gloss outputs; the remaining 141
PHX-public test sentences have no official fixed output and are excluded.

Outputs:
  results/ustc_moe/SLP_test_500.pt   keyed by PHX-public test key -> poses_3d
  results/ustc_moe/rebuild_meta.json per-key branch + hyp glosses + mapping
"""
import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SLRTP = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
USTC = Path("/ssd/xkb4/SignDiff/external/ustc_moe_slrtp2025")


def clean_toks(s):
    return [t for t in s.split() if not t.startswith("*")]


def norm_text(s):
    return re.sub(r"[^a-zäöüß ]", "", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/ustc_moe"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    test = torch.load(SLRTP / "test.pt", map_location="cpu", weights_only=False)
    train = torch.load(SLRTP / "train.pt", map_location="cpu", weights_only=False)
    tkeys = list(test.keys())

    with open(USTC / "phoenix_text2gloss_results.pkl", "rb") as f:
        t2g = pickle.load(f)
    with open(USTC / "phoenix_gloss2pose_results.pkl", "rb") as f:
        g2p = pickle.load(f)
    fl = list(csv.DictReader(open(USTC / "frame_lengths.csv")))

    # 1) map each alignment (order = official prediction order) to a test key
    #    via exact gloss match (after dropping '*' tokens, uppercasing)
    gloss_map = {}
    for k in tkeys:
        gloss_map[tuple(t.upper() for t in test[k]["gloss"].split())] = k
    al = t2g["wer_list"]["alignment"]
    mapping = {}  # ai -> key
    for ai, a in enumerate(al):
        ref = tuple(t.upper() for t in clean_toks(a["align_ref"]))
        k = gloss_map.get(ref)
        if k is not None:
            mapping[ai] = k
    print(f"mapped {len(mapping)}/500 text2gloss alignments to test keys",
          flush=True)

    # 2) training text index for the copy branch (normalized)
    train_text = {norm_text(v["text"]): k for k, v in train.items()}

    target_len = {ai: int(fl[ai]["num_frames"]) for ai in range(len(fl))}
    hyp_gloss = {ai: clean_toks(a["align_hyp"]) for ai, a in enumerate(al)}

    poses = {}
    meta = []
    n_copy, n_retrieval, n_empty = 0, 0, 0
    for ai in sorted(mapping):
        key = mapping[ai]
        T = target_len[ai]
        out_pose = torch.zeros((T, 178, 3))
        src = "retrieval"
        # copy branch: sentence text present in training set
        normt = norm_text(test[key]["text"])
        if normt in train_text:
            tp = train[train_text[normt]]["poses_3d"]
            n = min(T, tp.shape[0])
            out_pose[:n] = tp[:n]
            src = "copy"
            n_copy += 1
        else:
            segs = []
            for gls in hyp_gloss[ai]:
                if gls in g2p:
                    segs.append(torch.as_tensor(np.asarray(g2p[gls][0], np.float32)))
            if not segs:
                n_empty += 1
            else:
                cat = torch.concat(segs, dim=0)
                n = min(T, cat.shape[0])
                out_pose[:n] = cat[:n]
                n_retrieval += 1
        poses[key] = out_pose
        meta.append({
            "ai": ai, "key": key, "branch": src,
            "target_frames": T, "hyp_gloss": hyp_gloss[ai],
        })

    torch.save(poses, out / "SLP_test_500.pt")
    (out / "rebuild_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"copy branch: {n_copy}, retrieval: {n_retrieval}, empty hyp: {n_empty}, "
          f"total: {len(poses)}", flush=True)


if __name__ == "__main__":
    main()
