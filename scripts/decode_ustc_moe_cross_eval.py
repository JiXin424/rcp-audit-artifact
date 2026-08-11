#!/usr/bin/env python3
"""Decode the rebuilt USTC-MoE fixed outputs (and same-subset REC poses) under
every evaluator checkpoint (released + all 43 non-released), corpus sacreBLEU-4.

Usage: python scripts/decode_ustc_moe_cross_eval.py [--gpu N] [--only released]
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
    # Step-corrected (re-trained Round-26: translation-only + step-val + decoded-BLEU)
    **{f"sf_{s}": f"checkpoints/step_faithful/seed_{s}"
       for s in ["1701", "1702", "1703", "1704", "1705", "1706", "1707", "1708",
                 "505", "606"]},
    # Joint-loss greedy (1801-1808)
    **{f"sf_{s}": f"checkpoints/reconstructions_v2/seed_{s}"
       for s in ["1801", "1802", "1803", "1804", "1805", "1806", "1807", "1808"]},
    # Joint-loss beam-3 (1901-1908)
    **{f"sf_{s}": f"checkpoints/reconstructions_v3/seed_{s}"
       for s in ["1901", "1902", "1903", "1904", "1905", "1906", "1907", "1908"]},
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
    ap.add_argument("--out", default=None, help="output json path (default: results/ustc_moe/cross_eval_bleu.json)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    test = torch.load(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt",
                      map_location="cpu", weights_only=False)
    slp = torch.load(ROOT / "results/ustc_moe/SLP_test_500.pt", map_location="cpu",
                     weights_only=False)
    keys = sorted(slp.keys())
    refs = [test[k]["text"] for k in keys]
    print(f"{len(keys)} rebuilt USTC-MoE outputs", flush=True)

    # NOTE: sacrebleu 2.x zip-transposes references, so the flat form
    # [r1, ..., rN] is the correct corpus form; [[r1], ..., [rN]] would
    # collapse to a single reference document over the whole corpus.
    names = [args.only] if args.only else list(EVA)
    results = {}
    for name in names:
        model = make_back_translation_model(ROOT / EVA[name])
        hyp_ustc = back_translate(model, [slp[k][::2] for k in keys])
        hyp_rec = back_translate(model, [test[k]["poses_3d"][::2] for k in keys])
        b_ustc = BLEU.corpus_score(hyp_ustc, [refs]).score
        b_rec = BLEU.corpus_score(hyp_rec, [refs]).score
        results[name] = {"ustc_moe": float(b_ustc), "rec": float(b_rec),
                         "gap": float(b_ustc - b_rec), "n": len(keys)}
        print(f"{name}: USTC-MoE {b_ustc:.2f}  REC {b_rec:.2f}  gap {b_ustc - b_rec:+.2f}",
              flush=True)

    out = Path(args.out) if args.out else ROOT / "results/ustc_moe/cross_eval_bleu.json"
    if args.only:
        prev = json.loads(out.read_text()) if out.exists() else {}
        prev.update(results)
        out.write_text(json.dumps(prev, indent=1))
    else:
        out.write_text(json.dumps(results, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
