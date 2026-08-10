#!/usr/bin/env python3
"""Decode REC (GT-v1) + PURE (TN-PURE-v1) gap panel for new step-corrected seeds.

Reuses decode_cells.py functions. Writes per-item JSON to
results/gap_43_canonical_beam3_items/{prefix}_{gt|pure}.json matching the
existing format (list of {id, hypothesis, reference}), and prints corpus BLEU
and the PURE-REC gap.

Usage: python3 scripts/decode_step_faithful.py --ckpt-dir checkpoints/step_faithful/seed_1701 --gpu 0
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load decode_cells as a module (scripts/ is not a package)
_spec = importlib.util.spec_from_file_location("decode_cells",
                                                ROOT / "scripts/decode_cells.py")
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

from src.data.slrtp_dataset import SLRTPDataset, Vocab  # noqa: E402

OUT_DIR = ROOT / "results/gap_43_canonical_beam3_items"
TXT_VOCAB = ROOT / "checkpoints/released/backTranslation_PHIX_model/txt.vocab"
GLS_VOCAB = ROOT / "checkpoints/released/backTranslation_PHIX_model/gls.vocab"
TEST_PT = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"
TRAIN_PT = ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    seed_field = ckpt_dir.name  # seed_1701
    prefix = "sf_" + seed_field.split("_")[1]  # sf_1701

    txt_vocab = Vocab.from_file(TXT_VOCAB)
    gls_vocab = Vocab.from_file(GLS_VOCAB)

    print(f"[{prefix}] loading model from {ckpt_dir} on GPU {args.gpu}", flush=True)
    model = dc.load_evaluator(ckpt_dir, gpu=args.gpu)

    # --- REC (GT-v1): recorded test poses ---
    ds_gt = SLRTPDataset(TEST_PT, txt_vocab=txt_vocab, gls_vocab=gls_vocab,
                         skeleton_subsample=2)
    cell_gt = dc.decode_one_cell(model, prefix, "GT-v1", ds_gt)
    gt_items = [{"id": it["id"], "hypothesis": it["hypothesis"],
                 "reference": it["reference"]} for it in cell_gt["items"]]
    json.dump(gt_items, open(OUT_DIR / f"{prefix}_gt.json", "w"),
              ensure_ascii=False)
    gt_bleu = cell_gt["decoded_bleu"] * 100.0

    # --- PURE (TN-PURE-v1): text-nearest whole-donor retrieval ---
    ds_pure = dc._make_tn_pure_dataset(TRAIN_PT, TEST_PT, txt_vocab, gls_vocab)
    cell_pure = dc.decode_one_cell(model, prefix, "TN-PURE-v1", ds_pure)
    pure_items = [{"id": it["id"], "hypothesis": it["hypothesis"],
                   "reference": it["reference"]} for it in cell_pure["items"]]
    json.dump(pure_items, open(OUT_DIR / f"{prefix}_pure.json", "w"),
              ensure_ascii=False)
    pure_bleu = cell_pure["decoded_bleu"] * 100.0

    gap = pure_bleu - gt_bleu
    print(f"[{prefix}] REC={gt_bleu:.2f} PURE={pure_bleu:.2f} "
          f"gap={gap:+.2f}", flush=True)


if __name__ == "__main__":
    main()
