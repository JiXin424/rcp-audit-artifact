#!/usr/bin/env python3
"""Evaluate arbitrary checkpoints of the round-33 faithful runs (E-A/E-B).

For each requested checkpoint file (e.g. step_1820.ckpt, step_2828.ckpt,
best.ckpt, final.ckpt) inside a faithful run directory, this script
  1. builds a temporary eval dir (config.yaml + vocabs + best.ckpt symlink),
  2. loads the evaluator via the standard pipeline,
  3. decodes the 641 recorded test poses (REC),
  4. decodes the 641 canonical PURE donor poses (donor_registry.jsonl,
     SHA-256 9170a530... -- no retrieval rebuild),
  5. decodes the full 7,060-item training pool (readout; beam-3, [::2]).

Accumulates results into results/faithful_steps_eval.json (one row per
run/ckpt: dev_bleu read from validations.txt where applicable, plus
rec/pure/gap/train-readout/EM).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_step_ckpts.py \
      --run-dir checkpoints/faithful_steps/seed_42 \
      --ckpts step_1820 step_2828 best final
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.slrtp_dataset import SLRTPDataset, Vocab, load_pickle  # noqa: E402
from src.models import make_back_translation_model, back_translate  # noqa: E402

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
REGISTRY = ROOT / "results/gap_43_canonical_beam3_items/donor_registry.jsonl"
OUT = ROOT / "results/faithful_steps_eval.json"
EVAL_ROOT = ROOT / "checkpoints/faithful_steps/eval"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)
TXT_VOCAB = ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"
GLS_VOCAB = ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"


def poses_of(item, subsample=2):
    p = item["poses_3d"]
    if not isinstance(p, torch.Tensor):
        p = torch.as_tensor(np.asarray(p, dtype=torch.float32))
    return p[::subsample] if subsample and subsample > 1 else p


def decode(model, items):
    return back_translate(model, [poses_of(it) for it in items])


def dev_bleu_at_step(run_dir: Path, step: int):
    """Look up the training-time dev BLEU at the nearest validation step."""
    vf = run_dir / "validations.txt"
    best = None
    for line in vf.read_text().splitlines():
        if line.startswith("Steps:"):
            s = int(line.split()[1])
            b = float(line.split("BLEU-4")[1].split()[0])
            if best is None or abs(s - step) < abs(best[0] - step):
                best = (s, b)
    return {"val_step": best[0], "dev_bleu": best[1]} if best else None


def best_and_last_from_validations(run_dir: Path):
    """Best (max BLEU-4 among '*' lines) and last validation from
    validations.txt -- used when the run was terminated externally and
    training_log.json has not been finalized."""
    vf = run_dir / "validations.txt"
    best, last = None, None
    for line in vf.read_text().splitlines():
        if not line.startswith("Steps:"):
            continue
        s = int(line.split()[1])
        b = float(line.split("BLEU-4")[1].split()[0])
        last = {"val_step": s, "dev_bleu": b}
        if line.rstrip().endswith("*"):
            if best is None or b > best["dev_bleu"]:
                best = {"val_step": s, "dev_bleu": b}
    return best, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpts", default="step_1820,step_2828,best,final")
    ap.add_argument("--train-pool", action="store_true", default=True,
                    help="Decode full 7,060-item train pool (readout).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_id = f"{run_dir.parent.name}/{run_dir.name}"

    test_items = load_pickle(DATA_DIR / "test.pt")
    train_items = load_pickle(DATA_DIR / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}

    # Canonical donor poses in test order
    donor_of = {}
    for line in REGISTRY.read_text().splitlines():
        r = json.loads(line)
        donor_of[r["query_id"]] = r["donor_id"]
    pure_items = [train_by_id[donor_of[it["name"]]] for it in test_items]
    print(f"Donor registry loaded: {len(donor_of)} entries", flush=True)

    results = json.load(open(OUT)) if OUT.exists() else {"rows": []}

    for name in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        ckpt_file = run_dir / f"{name}.ckpt"
        if not ckpt_file.exists():
            print(f"[skip] {ckpt_file} not found", flush=True)
            continue
        eval_dir = EVAL_ROOT / f"{run_dir.name}_{name}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        # train_faithful copies config/vocabs at END of training; fall back to
        # the original faithful run dir (same seed) or the released bundle.
        seed_num = run_dir.name.split("_")[-1].replace("seed", "")
        fallback = ROOT / f"checkpoints/faithful/seed_{seed_num}"
        src_config = run_dir / "config.yaml"
        if not src_config.exists():
            src_config = fallback / "config.yaml"
        for fname, src in [("config.yaml", src_config),
                           ("txt.vocab", TXT_VOCAB),
                           ("gls.vocab", GLS_VOCAB),
                           ("best.ckpt", ckpt_file)]:
            dst = eval_dir / fname
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        if any(r["ckpt"] == f"{run_id}/{name}" for r in results["rows"]):
            print(f"[skip] {run_id}/{name} already evaluated", flush=True)
            continue

        t0 = time.time()
        model = make_back_translation_model(str(eval_dir))
        assert model.beam_size == 3

        rec_h = decode(model, test_items)
        rec_refs = [it["text"] for it in test_items]
        rec_bleu = BLEU.corpus_score(rec_h, [rec_refs]).score

        pure_h = decode(model, pure_items)
        pure_bleu = BLEU.corpus_score(pure_h, [rec_refs]).score
        gap = pure_bleu - rec_bleu

        row = {"run": run_id, "ckpt": f"{run_id}/{name}",
               "rec_bleu": rec_bleu, "pure_bleu": pure_bleu, "gap": gap}

        if name.startswith("step_"):
            row.update(dev_bleu_at_step(run_dir, int(name.split("_")[1])))
        elif name in ("best", "final"):
            best_v, last_v = best_and_last_from_validations(run_dir)
            if (run_dir / "training_log.json").exists():
                log = json.load(open(run_dir / "training_log.json"))
                if name == "best" and log.get("best"):
                    row.update({"val_step": log["best"]["step"],
                                "dev_bleu": log["best"]["dev_metric"]})
                elif name == "final" and log.get("validations"):
                    last = log["validations"][-1]
                    row.update({"val_step": last["step"],
                                "dev_bleu": last["dev_bleu"]})
            elif name == "best" and best_v:
                row.update(best_v)
            elif name == "final" and last_v:
                row.update(last_v)

        if args.train_pool:
            tr_h = decode(model, train_items)
            tr_refs = [it["text"] for it in train_items]
            tr_bleu = BLEU.corpus_score(tr_h, [tr_refs]).score
            em = sum(1 for h, r in zip(tr_h, tr_refs)
                     if h.strip().lower() == r.strip().lower()) / len(tr_h)
            row["train_readout_bleu"] = tr_bleu
            row["train_readout_em"] = em

        row["elapsed_s"] = time.time() - t0
        results["rows"].append(row)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[done] {row['ckpt']}: REC={rec_bleu:.2f} PURE={pure_bleu:.2f} "
              f"gap={gap:+.2f} readout={row.get('train_readout_bleu', float('nan')):.2f} "
              f"EM={100*row.get('train_readout_em', float('nan')):.1f}% "
              f"dev={row.get('dev_bleu', float('nan')):.2f} "
              f"({row['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
