#!/usr/bin/env python3
"""Readout-vs-overfit association analysis (reviewer C2/R3-M2).

Quantifies, across the family of checkpoints with uniform full-pool readouts,
how the training-pool free-decode readout relates to (i) decoded competence
(dev BLEU), (ii) train-minus-dev BLEU (an overfit indicator), and (iii) the
PURE-REC replay gap. This anchors the Discussion paragraph on the
"overfitting interpretation" of the released evaluator's readout-with-
generalization signature.

Output: results/readout_overfit.json
"""
from __future__ import annotations

import json
import glob
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# full_readout file stem -> gap-table id (gap_43_canonical_beam3.json)
GAP_ID_MAP = {
    "released": "released",
    **{f"seed_{s}": f"reco_{s}" for s in
       [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203,
        1304, 1405]},
    **{f"alpha_{a}_seed_{s}": f"distill_a{a}_{s}"
       for a in [0.0, 0.25, 0.5, 0.75, 1.0] for s in [101, 202, 303]},
    "seed_202_wd0": "rescue_wd0",
}


def main():
    rows = []

    # 30 checkpoints with uniform full-pool readouts (train/dev/test);
    # the released evaluator (backTranslation_PHIX_model) is appended below
    for f in sorted(glob.glob(str(ROOT / "results/full_readout/*.json"))):
        d = json.load(open(f))
        ckpt_id = Path(f).stem
        if ckpt_id == "backTranslation_PHIX_model":
            continue
        if ckpt_id == "seed_202_wd0":
            fam = "rescue"
        elif ckpt_id.startswith("alpha_"):
            fam = "distillation"
        else:
            fam = "reconstructions"
        tr = d["splits"]["train"]
        dv = d["splits"]["dev"]
        rows.append({
            "id": ckpt_id, "family": fam,
            "train_bleu": tr["bleu"], "train_em": tr["em"],
            "dev_bleu": dv["bleu"], "dev_em": dv["em"],
            "train_minus_dev_bleu": tr["bleu"] - dv["bleu"],
        })

    # Released evaluator: full-pool readout (leakage sanity, beam-3) + dev
    # uniform value (dev_uniform/backTranslation_PHIX_model.json or full_readout)
    rel_fr = ROOT / "results/full_readout/backTranslation_PHIX_model.json"
    dev_f = ROOT / "results/dev_uniform/backTranslation_PHIX_model.json"
    if rel_fr.exists():
        d = json.load(open(rel_fr))
        rows.append({
            "id": "released", "family": "released",
            "train_bleu": d["splits"]["train"]["bleu"],
            "train_em": d["splits"]["train"]["em"],
            "dev_bleu": d["splits"]["dev"]["bleu"],
            "dev_em": d["splits"]["dev"]["em"],
            "train_minus_dev_bleu": d["splits"]["train"]["bleu"] - d["splits"]["dev"]["bleu"],
        })
    elif dev_f.exists():
        d = json.load(open(dev_f))
        leak = json.load(open(ROOT / "results/leakage_sanity.json"))
        tr_bleu = leak["experiment_a_free_decode"]["bleu"]
        tr_em = leak["experiment_a_free_decode"]["em_rate"]
        rows.append({
            "id": "released", "family": "released",
            "train_bleu": tr_bleu, "train_em": tr_em,
            "dev_bleu": d["bleu"], "dev_em": d["em"],
            "train_minus_dev_bleu": tr_bleu - d["bleu"],
        })

    # Gaps (PURE-REC) from the canonical gap table
    gap_data = json.load(open(ROOT / "results/gap_43_canonical_beam3.json"))
    for r in rows:
        gkey = GAP_ID_MAP.get(r["id"])
        r["gap"] = gap_data[gkey]["gap"] if gkey and gkey in gap_data else None

    # Teacher-forced stats (released + 14 reconstructions, keyed by paper id)
    ck = json.load(open(ROOT / "results/checkpoint_stats.json"))
    for r in rows:
        key = GAP_ID_MAP.get(r["id"], r["id"])
        if key in ck:
            r["train_nll"] = ck[key]["train_nll_per_token"]
            r["train_tok_acc"] = ck[key]["train_token_accuracy"]
            r["dev_nll"] = ck[key]["dev_nll_per_token"]

    def spearman(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 3 or len(b) < 3:
            return None
        rho, p = stats.spearmanr(a, b)
        return {"rho": float(rho), "p": float(p), "n": int(len(a))}

    valid = [r for r in rows if r["gap"] is not None]
    all31 = [r for r in rows if r["dev_bleu"] is not None]

    corr = {
        "readout_vs_dev_bleu": spearman([r["train_bleu"] for r in all31],
                                        [r["dev_bleu"] for r in all31]),
        "readout_vs_train_minus_dev": spearman([r["train_bleu"] for r in all31],
                                               [r["train_minus_dev_bleu"] for r in all31]),
        "readout_vs_gap": spearman([r["train_bleu"] for r in valid],
                                   [r["gap"] for r in valid]),
        "gap_vs_dev_bleu": spearman([r["gap"] for r in valid],
                                    [r["dev_bleu"] for r in valid]),
        "readout_vs_train_nll": spearman(
            [r["train_bleu"] for r in all31 if "train_nll" in r],
            [r["train_nll"] for r in all31 if "train_nll" in r]),
        "readout_vs_train_tok_acc": spearman(
            [r["train_bleu"] for r in all31 if "train_tok_acc" in r],
            [r["train_tok_acc"] for r in all31 if "train_tok_acc" in r]),
        "train_minus_dev_vs_gap": spearman([r["train_minus_dev_bleu"] for r in valid],
                                           [r["gap"] for r in valid]),
    }

    # Sensitivity: the three alpha=1.0 distillation students decode to empty
    # output (train_bleu 0.0, degenerate); recompute the headline correlation
    # excluding them.
    nd = [r for r in rows if r["id"] != "released" and r["train_bleu"] > 0]
    corr["non_degenerate_readout_vs_dev_bleu"] = spearman(
        [r["train_bleu"] for r in nd], [r["dev_bleu"] for r in nd])

    released = next(r for r in rows if r["id"] == "released")
    family = [r for r in rows if r["id"] != "released"]

    summary = {
        "n_total": len(rows),
        "released": released,
        "family": {
            "n": len(family),
            "train_bleu_range": [min(r["train_bleu"] for r in family),
                                 max(r["train_bleu"] for r in family)],
            "dev_bleu_range": [min(r["dev_bleu"] for r in family),
                               max(r["dev_bleu"] for r in family)],
            "train_minus_dev_range": [min(r["train_minus_dev_bleu"] for r in family),
                                      max(r["train_minus_dev_bleu"] for r in family)],
        },
        "correlations": corr,
    }

    out = ROOT / "results/readout_overfit.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:2500])


if __name__ == "__main__":
    main()
