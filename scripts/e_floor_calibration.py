#!/usr/bin/env python3
"""BLEU floor calibration under reference permutation (reviewer R1-M3).

Scores the released evaluator's REC/PURE hypotheses against permuted
references (references of *other* items in the same subset). A nonzero floor
quantifies how much of an observed gap could be obtained by reference
mismatch alone (template register shared across items), and bounds the
reference-replacement asymmetry (R1-M3: "random-reference floor calibration").

Conditions:
  - matched 461 confidence=1 subset: same-item (verify 14.94/24.98, 7.35/8.30)
    and permuted floors for both original and human references
  - full 641: same-item and permuted floor for original references

Output: results/floor_calibration.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

N_PERM = 30
SEED = 42


def load_hypotheses(kind: str):
    """kind: 'gt' (REC) or 'pure' (PURE); returns dict id -> hypothesis."""
    f = ROOT / f"results/gap_43_canonical_beam3_items/released_{kind}.json"
    items = json.load(open(f))
    return {it["id"]: it["hypothesis"] for it in items}


def load_original_refs():
    """Official reference texts from the released test.pt materialization."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.slrtp_dataset import load_pickle
    items = load_pickle(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt")
    return {item["name"]: item["text"] for item in items}


def load_human_refs():
    """Returns dict official_id -> (back_translation, confidence)."""
    f = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
    refs = {}
    with open(f, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            refs[row["name"].strip()] = (
                row["back translation"].strip(),
                float(row["translation confidence"]),
            )
    return refs


def floor_bleu(hyps: list, refs: list, rng: np.random.Generator):
    """Mean corpus BLEU of hyps against permuted (other-item) references."""
    n = len(refs)
    scores = []
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        while np.any(perm == np.arange(n)):
            perm = rng.permutation(n)
        scores.append(BLEU.corpus_score(hyps, [[refs[i] for i in perm]]).score)
    return {
        "mean": float(np.mean(scores)),
        "max": float(np.max(scores)),
        "min": float(np.min(scores)),
    }


def main():
    rng = np.random.default_rng(SEED)
    hyps_gt = load_hypotheses("gt")
    hyps_pure = load_hypotheses("pure")
    human = load_human_refs()

    out = {}

    # ---- matched 461 (confidence=1, join with 641 released IDs) ----
    conf1 = {k: v[0] for k, v in human.items() if v[1] == 1.0}
    matched = sorted(set(conf1) & set(hyps_gt))
    print(f"matched subset: {len(matched)}")
    rec_hyps = [hyps_gt[i] for i in matched]
    pure_hyps = [hyps_pure[i] for i in matched]
    orig = load_original_refs()
    orig_refs = [orig[i] for i in matched]
    hum_refs = [conf1[i] for i in matched]

    # same-item values (verification)
    rec_orig = BLEU.corpus_score(rec_hyps, [orig_refs]).score
    pure_orig = BLEU.corpus_score(pure_hyps, [orig_refs]).score
    rec_hum = BLEU.corpus_score(rec_hyps, [hum_refs]).score
    pure_hum = BLEU.corpus_score(pure_hyps, [hum_refs]).score

    out["matched461"] = {
        "n": len(matched),
        "same_item": {
            "orig": {"rec": rec_orig, "pure": pure_orig,
                     "gap": pure_orig - rec_orig},
            "human": {"rec": rec_hum, "pure": pure_hum,
                      "gap": pure_hum - rec_hum},
        },
        "permuted_floor": {
            "orig": {
                "rec": floor_bleu(rec_hyps, orig_refs, rng),
                "pure": floor_bleu(pure_hyps, orig_refs, rng),
            },
            "human": {
                "rec": floor_bleu(rec_hyps, hum_refs, rng),
                "pure": floor_bleu(pure_hyps, hum_refs, rng),
            },
        },
    }

    # ---- full 641, original refs ----
    ids641 = sorted(hyps_gt)
    rec641 = [hyps_gt[i] for i in ids641]
    pure641 = [hyps_pure[i] for i in ids641]
    refs641 = [orig[i] for i in ids641]
    rec641_orig = BLEU.corpus_score(rec641, [refs641]).score
    pure641_orig = BLEU.corpus_score(pure641, [refs641]).score
    out["full641"] = {
        "n": len(ids641),
        "same_item": {
            "orig": {"rec": rec641_orig, "pure": pure641_orig,
                     "gap": pure641_orig - rec641_orig},
        },
        "permuted_floor": {
            "orig": {
                "rec": floor_bleu(rec641, refs641, rng),
                "pure": floor_bleu(pure641, refs641, rng),
            },
        },
    }

    out["meta"] = {
        "n_perm": N_PERM,
        "seed": SEED,
        "note": "floor = corpus BLEU of hypotheses scored against permuted "
                "references (other items in the same subset); quantifies the "
                "score obtainable from in-domain reference mismatch alone.",
    }

    dest = ROOT / "results/floor_calibration.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
