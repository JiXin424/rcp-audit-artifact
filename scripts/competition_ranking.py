#!/usr/bin/env python3
"""Real competitive-system ranking under released + reconstruction evaluators.

Reviewer R2-3 (option a) asks for SLRTP2025 competitive-system outputs (at least
retrieval, diffusion, transformer) ranked under the released evaluator and under
reconstruction evaluators, to test leaderboard-rank stability on real systems
rather than only on constructed probes.

This script consumes a directory of fixed system outputs and reports per-system
corpus sacreBLEU + rankings under each evaluator. It is ready to run once the
SLRTP2025 competition outputs are placed in the systems directory.

Expected input layout (one of):
  --systems-dir DIR   where each subdirectory is a system name containing
                      per-item pose tensors, OR
  --manifest JSON     mapping system_name -> list of {"id","poses_3d",...} dicts,
                      or a path to a .pt/.npy archive loadable by the same loader
                      used for the released test set.

Each system must cover the same 641 PHX-public test IDs (or a documented
subset, which will be intersected across all systems and evaluators).

Output: results/competition_ranking.json + a printed rank table.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sacrebleu
from src.data.slrtp_dataset import load_pickle
from src.models import make_back_translation_model, back_translate

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
OUT = ROOT / "results/competition_ranking.json"
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

DEFAULT_EVALUATORS = {
    "released": "checkpoints/released/backTranslation_PHIX_model",
    "reco_seed_101": "checkpoints/reconstructions/seed_101",
    "reco_seed_303": "checkpoints/reconstructions/seed_303",
    "reco_seed_505": "checkpoints/reconstructions/seed_505",
}


def load_system_poses(systems_dir):
    """Each subdir = one system; loads poses from a per-system .pt or .npy file
    keyed by test ID. Accepts either:
      systems_dir/<sys>/poses.pt   -> dict id -> tensor/dict
      systems_dir/<sys>.pt         -> dict id -> tensor
    Returns {system: {id: pose_tensor}} and the reference map.
    """
    test_items = load_pickle(DATA_DIR / "test.pt")
    refs = {k: v["text"] for k, v in test_items.items()}
    systems = {}
    sd = Path(systems_dir)
    candidates = sorted([p for p in sd.iterdir()
                         if p.suffix in (".pt", ".npy")]) \
        if sd.exists() else []
    for c in candidates:
        sys_name = c.stem
        if c.suffix == ".pt":
            data = torch.load(str(c), map_location="cpu", weights_only=False)
        else:
            data = np.load(str(c), allow_pickle=True).item()
        systems[sys_name] = {k: v if isinstance(v, torch.Tensor)
                             else torch.as_tensor(np.asarray(v["poses_3d"]
                                                             if isinstance(v, dict)
                                                             else v, dtype=np.float32))
                             for k, v in data.items()}
    return systems, refs


def decode_evaluator(model_dir, systems, ids):
    model = make_back_translation_model(str(model_dir))
    out = {}
    for sys_name, pose_map in systems.items():
        poses = []
        for i in ids:
            p = pose_map[i]
            if not isinstance(p, torch.Tensor):
                p = torch.as_tensor(np.asarray(p, dtype=np.float32))
            poses.append(p[::2])
        hyps = back_translate(model, poses)
        out[sys_name] = hyps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems-dir", default="data/competition_systems")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    systems, refs = load_system_poses(args.systems_dir)
    if not systems:
        print(f"[pending] No system outputs found in {args.systems_dir}. "
              f"Place SLRTP2025 competition outputs (retrieval/diffusion/"
              f"transformer) there and re-run.", flush=True)
        OUT.write_text(json.dumps({
            "schema": "competition-ranking-v1",
            "status": "pending-outputs",
            "systems_dir": str(args.systems_dir),
            "note": "Run after placing fixed system outputs to address "
                    "reviewer R2-3 option (a)."}, indent=1))
        return

    ids = sorted(set.intersection(*[set(s) for s in systems.values()]) & set(refs))
    print(f"Evaluating {len(systems)} systems on {len(ids)} common IDs", flush=True)

    ranking = {}
    for ev_name, ev_dir in DEFAULT_EVALUATORS.items():
        hyps_per = decode_evaluator(ev_dir, systems, ids)
        scores = {}
        for sys_name, hyps in hyps_per.items():
            r = [refs[i] for i in ids]
            scores[sys_name] = BLEU.corpus_score(hyps, [r]).score
        ranking[ev_name] = scores
        order = sorted(scores, key=lambda s: -scores[s])
        print(f"\n=== {ev_name} ===", flush=True)
        for rank, s in enumerate(order, 1):
            print(f"  {rank}. {s}: {scores[s]:.2f}", flush=True)

    OUT.write_text(json.dumps({
        "schema": "competition-ranking-v1",
        "status": "complete",
        "n_systems": len(systems),
        "n_common_ids": len(ids),
        "evaluators": DEFAULT_EVALUATORS,
        "ranking": ranking}, indent=1, ensure_ascii=False))
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
