#!/usr/bin/env python3
"""Cross-evaluator decode of noisy-gloss retrieval outputs (strong-NMT query).

Decodes the donor-copy poses retrieved via strong-NMT noisy-gloss queries
(results/noisy_gloss/NG_public_donor_copy_strong.pt, 641 PHX-public sentences)
under every evaluator checkpoint (released + all 43 non-released), together
with the recorded test poses (REC) as baseline, corpus sacreBLEU-4.

This is the SI noisy-gloss retrieval analogue of decode_ustc_moe_cross_eval.py
(a real-system fixed output): the strong-NMT noisy-gloss proxy should yield
~12.7 BLEU under the released evaluator (SI Sup. P).

Usage: python scripts/e_noisy_gloss_cross_eval.py [--gpu N] [--only released]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.back_translate import make_back_translation_model, back_translate

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

EVA = {
    "released": "checkpoints/released/backTranslation_PHIX_model",
    **{f"reco_{s}": f"checkpoints/reconstructions/seed_{s}"
       for s in ["101", "202", "303", "404", "505", "606", "707", "808", "909",
                 "1001", "1102", "1203", "1304", "1405"]},
    **{f"distill_a{a}_{s}": f"checkpoints/distillation/alpha_{a}_seed_{s}"
       for a in ["0.0", "0.25", "0.5", "0.75", "1.0"] for s in ["101", "202", "303"]},
    "cf_101": "checkpoints/config_faithful/seed_101",
    "cf_202": "checkpoints/config_faithful/seed_202",
    "cf_303": "checkpoints/config_faithful/seed_303",
    "cf_404": "checkpoints/config_faithful/seed_404",
    "sf_505": "checkpoints/step_faithful/seed_505",
    "sf_606": "checkpoints/step_faithful/seed_606",
    "conf_1506": "checkpoints/confirmation/seed_1506",
    "conf_1607": "checkpoints/confirmation/seed_1607",
    "ls_202": "checkpoints/long_schedule/seed_202",
    "rescue_wd0": "checkpoints/rescue/seed_202_wd0",
    "ladder_0125": "checkpoints/ladder/frac_0125",
    "ladder_025": "checkpoints/ladder/frac_025",
    "ladder_05": "checkpoints/ladder/frac_05",
    "ladder_075": "checkpoints/ladder/frac_075",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only", default=None, help="evaluator name to decode only")
    ap.add_argument("--out", default=None,
                    help="output json path (default: results/noisy_gloss/cross_eval_bleu.json)")
    ap.add_argument("--mode", choices=["strong", "weak"], default="strong")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test = torch.load(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt",
                      map_location="cpu", weights_only=False)
    ng = torch.load(ROOT / f"results/noisy_gloss/NG_public_donor_copy_{args.mode}.pt",
                    map_location="cpu", weights_only=False)
    keys = sorted(test.keys())
    assert set(ng.keys()) == set(keys), "noisy-gloss file must cover all PHX-public keys"
    refs = [test[k]["text"] for k in keys]
    print(f"{len(keys)} noisy-gloss donor-copy outputs (mode={args.mode})", flush=True)

    # NOTE: sacrebleu 2.x zip-transposes references; flat [r1, ..., rN] is the
    # correct corpus form ([[r1], ..., [rN]] would collapse to one document).
    names = [args.only] if args.only else list(EVA)
    results = {}
    for name in names:
        model = make_back_translation_model(ROOT / EVA[name])
        hyp_ng = back_translate(model, [ng[k][::2] for k in keys])
        hyp_rec = back_translate(model, [test[k]["poses_3d"][::2] for k in keys])
        b_ng = BLEU.corpus_score(hyp_ng, [refs]).score
        b_rec = BLEU.corpus_score(hyp_rec, [refs]).score
        results[name] = {"noisy_gloss": float(b_ng), "rec": float(b_rec),
                         "gap": float(b_ng - b_rec), "n": len(keys)}
        print(f"{name}: noisy-gloss {b_ng:.2f}  REC {b_rec:.2f}  gap {b_ng - b_rec:+.2f}",
              flush=True)

    out = Path(args.out) if args.out else ROOT / f"results/noisy_gloss/cross_eval_bleu_{args.mode}.json"
    if args.only:
        prev = json.loads(out.read_text()) if out.exists() else {}
        prev.update(results)
        out.write_text(json.dumps(prev, indent=1))
    else:
        out.write_text(json.dumps(results, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
