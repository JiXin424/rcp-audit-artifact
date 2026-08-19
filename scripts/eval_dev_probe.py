#!/usr/bin/env python3
"""Frozen-dev-probe evaluation of arbitrary reconstruction checkpoints.

Round-34 (advisor comment): the +8.10 dev-split probe response was verified
only under the RELEASED evaluator; the unclipped reconstructions that
"reproduce the probe response" were never evaluated on the frozen dev probe.
This script decodes dev REC + dev PURE (TN-PURE-v1 registry over dev queries,
frozen in results/dev_split_confirmation_items/donor_registry_dev.jsonl,
SHA-256 2d4780a6...) under any requested checkpoint and records the dev gap.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/eval_dev_probe.py \
      --run-dir checkpoints/ctc_clip_sens/clipinf_seed42 --ckpts best
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.slrtp_dataset import load_pickle  # noqa: E402
from src.models import make_back_translation_model, back_translate  # noqa: E402

DATA_DIR = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data"
DEV_REGISTRY = ROOT / "results/dev_split_confirmation_items/donor_registry_dev.jsonl"
OUT = ROOT / "results/dev_probe_eval.json"
EVAL_ROOT = ROOT / "checkpoints/faithful_steps/eval_dev"
TXT_VOCAB = ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"
GLS_VOCAB = ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def poses_of(item, subsample=2):
    p = item["poses_3d"]
    if not isinstance(p, torch.Tensor):
        p = torch.as_tensor(np.asarray(p, dtype=torch.float32))
    return p[::subsample] if subsample and subsample > 1 else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpts", default="best")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_id = f"{run_dir.parent.name}/{run_dir.name}"

    dev_items = load_pickle(DATA_DIR / "dev.pt")
    train_items = load_pickle(DATA_DIR / "train.pt")
    train_by_id = {it["name"]: it for it in train_items}
    refs = [it["text"] for it in dev_items]

    donor_of = {}
    for line in DEV_REGISTRY.read_text().splitlines():
        r = json.loads(line)
        donor_of[r["query_id"]] = r["donor_id"]
    assert len(donor_of) == len(dev_items), "dev donor registry mismatch"
    pure_items = [train_by_id[donor_of[it["name"]]] for it in dev_items]
    registry_sha = hashlib.sha256(DEV_REGISTRY.read_bytes()).hexdigest()[:12]

    results = json.load(open(OUT)) if OUT.exists() else {"registry_sha256_prefix": registry_sha, "rows": []}

    for name in [c.strip() for c in args.ckpts.split(",") if c.strip()]:
        ckpt_file = run_dir / f"{name}.ckpt"
        if not ckpt_file.exists():
            print(f"[skip] {ckpt_file} not found", flush=True)
            continue
        if any(r["ckpt"] == f"{run_id}/{name}" for r in results["rows"]):
            print(f"[skip] {run_id}/{name} already evaluated", flush=True)
            continue
        eval_dir = EVAL_ROOT / f"{run_dir.name}_{name}"
        eval_dir.mkdir(parents=True, exist_ok=True)
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

        t0 = time.time()
        model = make_back_translation_model(str(eval_dir))
        assert model.beam_size == 3

        rec_h = back_translate(model, [poses_of(it) for it in dev_items])
        rec_bleu = BLEU.corpus_score(rec_h, [refs]).score
        pure_h = back_translate(model, [poses_of(it) for it in pure_items])
        pure_bleu = BLEU.corpus_score(pure_h, [refs]).score
        gap = pure_bleu - rec_bleu

        ckpt_sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()[:12]
        row = {"run": run_id, "ckpt": f"{run_id}/{name}", "ckpt_sha256_prefix": ckpt_sha,
               "split": "dev", "n_queries": len(dev_items),
               "rec_bleu": rec_bleu, "pure_bleu": pure_bleu, "gap": gap,
               "released_dev_reference": {"rec": 13.379, "pure": 21.474, "gap": 8.096},
               "elapsed_s": time.time() - t0}
        results["rows"].append(row)
        OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"[done] {row['ckpt']}: dev REC={rec_bleu:.2f} PURE={pure_bleu:.2f} "
              f"gap={gap:+.2f} ({row['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
