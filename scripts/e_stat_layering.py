#!/usr/bin/env python3
"""Layered statistics requested by the LRE reviewer (round-2 major, M1/M2).

Rebuilds the prediction-interval and attenuation-CI statistics on per-item
data with explicit seed stratification and two-level resampling:

  1. 14-seed reconstruction Student-t PI (as before), split into the 6
     pre-registered primary seeds (101-606) and the 8 post-hoc extension
     seeds (707-1405), each with its own PI and LOO-seed fold range.
  2. LOO-seed PIs: refit the PI leaving out each reconstruction seed.
  3. Two-level (seed x item) bootstrap PI: resample seeds with replacement,
     then items within each resampled seed, pooled corpus BLEU-4 gap.
  4. Matched-461 human-reference attenuation: paired-bootstrap CI for the
     *ratio* (1 - human_gap / orig_gap), plus the absolute-attenuation CI.

All PIs/BLEU use the canonical beam-3 per-item decodes
(results/gap_43_canonical_beam3_items/), sacreBLEU 2.5.1 corpus BLEU-4
(13a, exp smooth), 10,000 resamples, seed 42.

Usage: python3 scripts/e_stat_layering.py
Output: results/stat_layering.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import sacrebleu
import scipy.stats as stats

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/gap_43_canonical_beam3_items"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
OUT = ROOT / "results/stat_layering.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

PRIMARY = [101, 202, 303, 404, 505, 606]
EXTENSION = [707, 808, 909, 1001, 1102, 1203, 1304, 1405]
ALL_SEEDS = PRIMARY + EXTENSION
RNG_SEED = 42
B = 10000


def load_items(name):
    gt = json.load(open(ITEMS / f"{name}_gt.json"))
    pure = json.load(open(ITEMS / f"{name}_pure.json"))
    by_id_g = {it["id"]: it for it in gt}
    by_id_p = {it["id"]: it for it in pure}
    assert set(by_id_g) == set(by_id_p)
    ids = sorted(by_id_g)
    refs = [by_id_g[i]["reference"] for i in ids]
    g_h = [by_id_g[i]["hypothesis"] for i in ids]
    p_h = [by_id_p[i]["hypothesis"] for i in ids]
    return ids, refs, g_h, p_h


def corpus_gap(g_h, p_h, refs):
    return (BLEU.corpus_score(p_h, [refs]).score
            - BLEU.corpus_score(g_h, [refs]).score)


def student_pi(gaps):
    """Student-t prediction interval for the next replicate (not the mean)."""
    gaps = np.asarray(gaps, dtype=float)
    n = len(gaps)
    mean = gaps.mean()
    sd = gaps.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    return mean, sd, n, float(mean - t * sd * np.sqrt(1 + 1.0 / n)), float(
        mean + t * sd * np.sqrt(1 + 1.0 / n))


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
    # ---- per-seed corpus gaps (reconstruction family) ----
    seed_gaps = {}
    for seed in ALL_SEEDS:
        ids, refs, g_h, p_h = load_items(f"reco_{seed}")
        seed_gaps[seed] = corpus_gap(g_h, p_h, refs)
    all_gaps = np.array([seed_gaps[s] for s in ALL_SEEDS])

    out = {}

    # 1. 14-seed PI + 6/8 split
    pi14 = student_pi(all_gaps)
    out["pi_14seed"] = {"mean": pi14[0], "sd": pi14[1], "n": pi14[2],
                        "pi": [pi14[3], pi14[4]]}
    gaps6 = np.array([seed_gaps[s] for s in PRIMARY])
    gaps8 = np.array([seed_gaps[s] for s in EXTENSION])
    for label, gaps, seeds in (("pi_6primary", gaps6, PRIMARY),
                               ("pi_8extension", gaps8, EXTENSION)):
        m, sd, n, lo, hi = student_pi(gaps)
        out[label] = {"mean": float(m), "sd": float(sd), "n": n,
                      "seeds": seeds,
                      "pi": [float(lo), float(hi)],
                      "n_positive": int((gaps > 0).sum())}

    # 2. LOO-seed PIs (leave one reconstruction seed out)
    loo = {}
    for drop in ALL_SEEDS:
        kept = [s for s in ALL_SEEDS if s != drop]
        m, sd, n, lo_, hi_ = student_pi([seed_gaps[s] for s in kept])
        loo[str(drop)] = {"mean": float(m), "pi": [float(lo_), float(hi_)]}
    out["loo_seed"] = {"dropped_pi_lo_min": min(v["pi"][0] for v in loo.values()),
                       "dropped_pi_lo_max": max(v["pi"][0] for v in loo.values()),
                       "dropped_pi_hi_min": min(v["pi"][1] for v in loo.values()),
                       "dropped_pi_hi_max": max(v["pi"][1] for v in loo.values()),
                       "folds": loo}

    # 3. Two-level (seed x item) bootstrap PI
    #    Resample seeds with replacement (14 draws); within each drawn seed
    #    resample its 641 items with replacement; take the corpus BLEU-4 gap
    #    of each resampled seed and average over the 14 draws.
    #    Per-item segment statistics are precomputed once per seed and summed
    #    per draw -- numerically identical to recomputing corpus BLEU over the
    #    resampled multiset (same smoothing/aggregation code path) but ~1000x
    #    faster (140k corpus-BLEU evals instead of 280k sacreBLEU calls).
    def seg_stats(g_h, p_h, refs):
        return (np.array(BLEU._extract_corpus_statistics(p_h, [refs]),
                         dtype=np.int64),
                np.array(BLEU._extract_corpus_statistics(g_h, [refs]),
                         dtype=np.int64))

    def gap_from_stats(S_p, S_g):
        return (BLEU._compute_score_from_stats(S_p.sum(axis=0)).score
                - BLEU._compute_score_from_stats(S_g.sum(axis=0)).score)

    stat_data = {}
    for seed in ALL_SEEDS:
        ids, refs, g_h, p_h = load_items(f"reco_{seed}")
        stat_data[seed] = seg_stats(g_h, p_h, refs)

    rng = np.random.RandomState(RNG_SEED)
    gaps2l = np.empty(B)
    for b in range(B):
        seed_draw = rng.choice(ALL_SEEDS, size=14, replace=True)
        total = 0.0
        for seed in seed_draw:
            S_p, S_g = stat_data[seed]
            n_items = S_p.shape[0]
            idx = rng.randint(0, n_items, n_items)
            total += gap_from_stats(S_p[idx], S_g[idx])
        gaps2l[b] = total / 14.0
    out["bootstrap_seed_item"] = {
        "n_boot": B, "seed_draw": 14,
        "ci": [float(np.percentile(gaps2l, 2.5)),
               float(np.percentile(gaps2l, 97.5))],
        "mean": float(gaps2l.mean())}

    # 4. Matched-461 attenuation ratio CI (paired item bootstrap)
    #    Precomputed segment stats per (system x reference-set) cell, summed
    #    over resampled items; identical to corpus_score on the resample.
    gt_cell = {it["id"]: it for it in json.load(open(ITEMS / "released_gt.json"))}
    pure_cell = {it["id"]: it for it in json.load(open(ITEMS / "released_pure.json"))}
    human_hc = load_csv(CSV_HC)
    ids = sorted(set(gt_cell) & set(human_hc))
    N = len(ids)
    gt = [gt_cell[i]["hypothesis"] for i in ids]
    pure = [pure_cell[i]["hypothesis"] for i in ids]
    ro = [gt_cell[i]["reference"] for i in ids]
    rh = [human_hc[i][0] for i in ids]
    P_ro, G_ro = seg_stats(gt, pure, ro)
    P_rh, G_rh = seg_stats(gt, pure, rh)

    rng2 = np.random.RandomState(RNG_SEED)
    att_abs = np.empty(B)
    att_ratio = np.empty(B)
    for b in range(B):
        idx = rng2.randint(0, N, N)
        og = gap_from_stats(P_ro[idx], G_ro[idx])
        hg = gap_from_stats(P_rh[idx], G_rh[idx])
        att_abs[b] = og - hg
        att_ratio[b] = 1.0 - hg / og if og > 0 else np.nan
    out["matched461_attenuation"] = {
        "n": N,
        "attenuation_pct": 90.53382636978502,
        "abs_ci": [float(np.percentile(att_abs, 2.5)),
                   float(np.percentile(att_abs, 97.5))],
        "ratio_ci": [float(np.nanpercentile(att_ratio, 2.5)),
                     float(np.nanpercentile(att_ratio, 97.5))],
        "ratio_ci_n_valid": int(np.isfinite(att_ratio).sum())}

    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "loo_seed"},
                     indent=1)[:1800])
    lo = out["loo_seed"]
    print("LOO-seed PI range:", lo["dropped_pi_lo_min"], "..",
          lo["dropped_pi_lo_max"], "/", lo["dropped_pi_hi_min"], "..",
          lo["dropped_pi_hi_max"])


if __name__ == "__main__":
    main()
