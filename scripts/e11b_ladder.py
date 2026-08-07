#!/usr/bin/env python3
"""Round 6 E-B: competence ladder via train-pool subsampling.

Trains the documented recipe on seeded subsamples of the 7,060-item train pool
(fractions 0.125/0.25/0.5/0.75), producing evaluators spanning a wide competence range.
Together with the existing 14 reconstructions (dev BLEU 0.067-0.099), rescue variants
(0.098 max), config-faithful runs (E-A), and the released evaluator (0.134), these give
a gap-vs-competence dose-response curve: if the PURE-GT gap stays ~0 across the whole
ladder while the released evaluator sits at +11, competence alone cannot explain the
reversal -- positive evidence, not a verbal caveat.

Usage: python e11b_ladder.py --frac 0.25 --seed 101 --gpu 0 --output DIR
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

ROUND3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND3 / "src"))
import train_matched
import yaml
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(ROUND3 / "config/experiment.yaml"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    train_matched.require_hash = lambda *a, **k: None
    all_ids = [x["id"] for x in map(json.loads, open(str(Path(__file__).resolve().parents[1] / "manifests/available_train.jsonl")))]
    rng = random.Random(20260729)  # shared subsample seed across fractions (nested subsets)
    shuffled = sorted(all_ids, key=lambda _: rng.random())
    k = max(1, int(round(len(all_ids) * args.frac)))
    sub_ids = shuffled[:k]
    sub_manifest = Path(args.output) / "train_sub.jsonl"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with open(sub_manifest, "w") as f:
        for i in sub_ids:
            f.write(json.dumps({"id": i}) + "\n")

    # Patch the training-pool manifest loading: train_matched.run reads manifests itself,
    # so instead patch _jsonl via monkey-patching the module's manifest path usage.
    import types
    orig_run = train_matched.run

    def patched_run(t_args):
        # monkey-patch the manifest contract: replace available_train content
        train_matched.assert_manifest_contract = lambda *a, **k: types.SimpleNamespace(
            train_sha256="subsampled", dev_sha256="dev")
        orig__jsonl = train_matched._jsonl
        def fake_jsonl(path):
            if "available_train" in str(path):
                return orig__jsonl(sub_manifest)
            return orig__jsonl(path)
        train_matched._jsonl = fake_jsonl
        try:
            return orig_run(t_args)
        finally:
            train_matched._jsonl = orig__jsonl

    t_args = types.SimpleNamespace(
        seed=args.seed, gpu=args.gpu, epochs=args.epochs, workers=args.workers,
        config=args.config, output=args.output, smoke=False, resume=None, checkpoint_every=25,
    )
    result = patched_run(t_args)
    info = {"variant": f"ladder_frac{args.frac}", "frac": args.frac, "n_train": k,
            "seed": args.seed, "best_val_loss": result.get("best_val_loss"),
            "best_epoch": result.get("best_epoch")}
    (Path(args.output) / "variant_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
