#!/usr/bin/env python3
"""E-A': competence-matched alternative-architecture BT evaluators.

The released evaluator (dev BLEU-4 13.38) sits far above every same-architecture
reconstruction (max 9.9). The reviewer asks: does a competence-matched, differently
trained evaluator show the reversal? We train four larger-capacity variants on the
identical released train data with the documented optimizer recipe (Adam 1e-3,
betas .9/.998, wd .001, batch 256, grad clip 1.0, NLL selection, legacy counter schedule):

  A1: 4 layers, hidden 512, ff 1024
  A2: 6 layers, hidden 512, ff 1024
  A3: 4 layers, hidden 512, ff 2048
  A4: 6 layers, hidden 384, ff 768

If a variant reaches dev BLEU ~12-13.4, we decode GT/PURE and measure gap + pass-through
on it: reversal reappears -> competence explains; absent -> released checkpoint is special.

Usage: python e12a_bigarch_train.py --variant A1 --seed 101 --gpu 0 --output DIR
"""
from __future__ import annotations
import argparse, copy, json, sys, types
from pathlib import Path

ROUND3 = Path("/ssd/xkb4/RCP/revision_20260728_round3")
sys.path.insert(0, str(ROUND3 / "src"))
import train_matched
import yaml
import torch

VARIANTS = {
    "A1": {"num_layers": 4, "hidden_size": 512, "ff_size": 1024},
    "A2": {"num_layers": 6, "hidden_size": 512, "ff_size": 1024, "batch_size": 128},
    "A3": {"num_layers": 4, "hidden_size": 512, "ff_size": 2048},
    "A4": {"num_layers": 6, "hidden_size": 384, "ff_size": 768, "batch_size": 192},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(ROUND3 / "config/experiment.yaml"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    spec = VARIANTS[args.variant]

    legacy = yaml.safe_load(train_matched.LEGACY_CONFIG.read_text())
    cfg = copy.deepcopy(legacy)
    for side in ("encoder", "decoder"):
        cfg["model"][side]["num_layers"] = spec["num_layers"]
        cfg["model"][side]["hidden_size"] = spec["hidden_size"]
        cfg["model"][side]["ff_size"] = spec["ff_size"]
        cfg["model"][side]["embeddings"]["embedding_dim"] = spec["hidden_size"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if "batch_size" in spec:
        cfg["training"]["batch_size"] = spec["batch_size"]
    tmp_cfg = out / "model_config.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg))
    train_matched.LEGACY_CONFIG = tmp_cfg
    train_matched.require_hash = lambda *a, **k: None

    t_args = types.SimpleNamespace(
        seed=args.seed, gpu=args.gpu, epochs=args.epochs, workers=args.workers,
        config=args.config, output=args.output, smoke=False, resume=None, checkpoint_every=25,
    )
    result = train_matched.run(t_args)
    info = {"variant": args.variant, "spec": spec, "seed": args.seed,
            "best_val_loss": result.get("best_val_loss"), "best_epoch": result.get("best_epoch"),
            "model_config": str(tmp_cfg)}
    (out / "variant_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
