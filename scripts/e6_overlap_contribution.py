#!/usr/bin/env python3
"""E6: Hamidullah-style train-test overlap contribution analysis.

Quantifies how much of corpus BLEU (GT and PURE, original evaluator) is carried by
test items whose reference is highly similar to some train text. For each of the 641
test queries we compute max source-text Jaccard against the 7,060 train texts, then:
  - cumulative share of corpus n-gram hits vs overlap-ranked items
  - corpus BLEU of GT and TN-PURE on overlap strata (top 5%, 10%, 25%, rest)
  - PURE-GT gap per stratum
"""
import json, math
from pathlib import Path
import numpy as np
import sys

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
OUT = ROOT / "revision_20260729_round5/results/e6_overlap_contribution.json"
sys.path.insert(0, str(ROOT / "revision_20260728_major"))
from src import evaluate_checkpoints as ev  # noqa: E402


def norm_tokens(s):
    return " ".join(s.casefold().split()).split()


def corpus_bleu_from(items, idxs):
    c = np.zeros(4); t = np.zeros(4); sysl = 0; refl = 0
    for i in idxs:
        sb = items[i]["segment_bleu"]
        c += np.array(sb["counts"]); t += np.array(sb["totals"])
        sysl += sb["system_length"]; refl += sb["reference_length"]
    p = c / t
    if (p <= 0).any():
        return 0.0
    bp = 1.0 if sysl > refl else math.exp(1 - refl / max(sysl, 1e-9))
    return float(bp * math.exp(np.log(p).mean()) * 100)


def main():
    train = ev.safe_torch_load(ev.TRAIN_PT, ev.PINNED[str(ev.TRAIN_PT)], "train")
    gt = json.load(open(CELLS / "cp0_GT-v1.json"))["metrics"]["items"]
    pure = json.load(open(CELLS / "cp0_TN-PURE-v1.json"))["metrics"]["items"]
    assert [x["id"] for x in gt] == [x["id"] for x in pure]
    train_tok_sets = [set(norm_tokens(v["text"])) for v in train.values()]

    max_j = []
    for it in gt:
        q = set(norm_tokens(it["reference"]))
        max_j.append(max((len(q & ts) / len(q | ts) if q | ts else 0.0) for ts in train_tok_sets))
    max_j = np.array(max_j)
    order = np.argsort(-max_j)
    n = len(gt)

    strata = {}
    for frac in [0.05, 0.10, 0.25, 0.50]:
        k = int(round(n * frac))
        top = order[:k]; rest = order[k:]
        strata[f"top_{int(frac*100)}pct"] = {
            "n": k, "jaccard_min": float(max_j[top].min()), "jaccard_mean": float(max_j[top].mean()),
            "GT": corpus_bleu_from(gt, top), "PURE": corpus_bleu_from(pure, top),
            "gap": corpus_bleu_from(pure, top) - corpus_bleu_from(gt, top),
        }
        strata[f"rest_after_top_{int(frac*100)}pct"] = {
            "n": n - k, "jaccard_max": float(max_j[rest].max()), "jaccard_mean": float(max_j[rest].mean()),
            "GT": corpus_bleu_from(gt, rest), "PURE": corpus_bleu_from(pure, rest),
            "gap": corpus_bleu_from(pure, rest) - corpus_bleu_from(gt, rest),
        }
    # cumulative n-gram-hit share curve (uni-gram hits of GT decode)
    hits = []
    for it in gt:
        sb = it["segment_bleu"]
        hits.append(sb["counts"][0])
    hits = np.array(hits, dtype=float)
    cum = np.cumsum(hits[order]) / hits.sum()
    curve = {"x": [float((i + 1) / n) for i in range(0, n, 25)],
             "cum_unigram_hit_share": [float(cum[i]) for i in range(0, n, 25)]}
    out = {"overlap_jaccard_quantiles": {str(q): float(np.quantile(max_j, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]},
           "exact_or_near": {"j_eq_1": int((max_j == 1.0).sum()), "j_ge_0p8": int((max_j >= 0.8).sum()),
                              "j_ge_0p5": int((max_j >= 0.5).sum())},
           "strata": strata, "hit_concentration": curve,
           "note": "max Jaccard of each test reference against all 7060 train texts (token-set, normalized)"}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["strata"], indent=1))
    print("quantiles:", out["overlap_jaccard_quantiles"])
    print("counts:", out["exact_or_near"])


if __name__ == "__main__":
    main()
