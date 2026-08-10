#!/usr/bin/env python3
"""Compute the matched-461 human-reference residual-gap CI (paired item bootstrap).

Adds `human_gap_bootstrap_ci` to results/canonical_matched_subset.json and
self-checks the recorded attenuation CI (recomputed here from the same per-item
files) against the stored `bootstrap_ci` field.

Uses the same corpus_score-based paired bootstrap as e_beam3_matched_subset.py
(sacrebleu 2.5.1, 13a, exp smoothing; 10,000 resamples, seed 42, percentile CIs).
"""
import json
import sys
from pathlib import Path

import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/gap_43_canonical_beam3_items"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
OUT = ROOT / "results/canonical_matched_subset.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def load_csv(path):
    out = {}
    for line in open(path).read().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            conf = float(parts[2])
        except ValueError:
            conf = 0.0
        out[parts[0].strip()] = (parts[1], conf)
    return out


def main():
    gt_cell = {it["id"]: it for it in json.load(open(ITEMS / "released_gt.json"))}
    pure_cell = {it["id"]: it for it in json.load(open(ITEMS / "released_pure.json"))}
    human_hc = load_csv(CSV_HC)
    ids = sorted(set(gt_cell) & set(human_hc))
    N = len(ids)
    print("matched confidence=1 n:", N, flush=True)

    gt = [gt_cell[i]["hypothesis"] for i in ids]
    pure = [pure_cell[i]["hypothesis"] for i in ids]
    ro = [gt_cell[i]["reference"] for i in ids]
    rh = [human_hc[i][0] for i in ids]

    rng = np.random.RandomState(42)
    gaps_human = np.empty(10000)
    atts = np.empty(10000)
    for b in range(10000):
        idx = rng.randint(0, N, N)
        og = (BLEU.corpus_score([pure[j] for j in idx], [[ro[j] for j in idx]]).score
              - BLEU.corpus_score([gt[j] for j in idx], [[ro[j] for j in idx]]).score)
        hg = (BLEU.corpus_score([pure[j] for j in idx], [[rh[j] for j in idx]]).score
              - BLEU.corpus_score([gt[j] for j in idx], [[rh[j] for j in idx]]).score)
        gaps_human[b] = hg
        atts[b] = og - hg

    human_ci = [float(np.percentile(gaps_human, 2.5)),
                float(np.percentile(gaps_human, 97.5))]
    att_ci = [float(np.percentile(atts, 2.5)), float(np.percentile(atts, 97.5))]
    print(f"human gap CI: {human_ci[0]:.4f}, {human_ci[1]:.4f} "
          f"(mean {gaps_human.mean():.4f})", flush=True)
    print(f"attenuation CI: {att_ci[0]:.4f}, {att_ci[1]:.4f} "
          f"(mean {atts.mean():.4f})", flush=True)

    doc = json.load(open(OUT))
    doc["human_gap_bootstrap_ci"] = human_ci
    doc["human_gap_bootstrap_mean"] = float(gaps_human.mean())
    doc["attenuation_bootstrap_ci_recomputed"] = att_ci
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)

    # Self-check against the stored attenuation CI
    stored = doc.get("bootstrap_ci")
    ok = stored is not None and abs(stored[0] - att_ci[0]) < 1e-6 and abs(stored[1] - att_ci[1]) < 1e-6
    print(f"stored attenuation CI: {stored}")
    print(f"SELF-CHECK {'PASS' if ok else 'FAIL'}: recomputed matches stored")
    if not ok:
        sys.exit(1)
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
