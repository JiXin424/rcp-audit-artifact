#!/usr/bin/env python3
"""Regenerate all paper numbers from canonical data.

Single-command paper build: reads canonical per-item hypotheses and recomputes
every table/figure value in the paper. Run after any change to the canonical
donor registry or decoded cells.

Usage: make paper PYTHON=/path/to/python3
"""
import json
import sys
from pathlib import Path

import numpy as np
import sacrebleu
from scipy import stats

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/gap_43_canonical_beam3_items"
CANONICAL = ROOT / "results/gap_43_canonical_beam3.json"
OUT = ROOT / "results/paper_numbers.json"


def corpus_bleu(hyps, refs):
    return BLEU.corpus_score(hyps, [refs]).score


def load_items(name):
    gt = json.load(open(ITEMS / f"{name}_gt.json"))
    pure = json.load(open(ITEMS / f"{name}_pure.json"))
    # sort by id so bootstrap positional indices match donor_cluster_bootstrap.py
    gt.sort(key=lambda x: x["id"])
    pure.sort(key=lambda x: x["id"])
    ids = [it["id"] for it in gt]
    refs = [it["reference"] for it in gt]
    gt_h = [it["hypothesis"] for it in gt]
    pure_h = [it["hypothesis"] for it in pure]
    return ids, refs, gt_h, pure_h


def main():
    results = {}

    # 1. Released evaluator headline
    ids, refs, gt_h, pure_h = load_items("released")
    gt_b = corpus_bleu(gt_h, refs)
    pure_b = corpus_bleu(pure_h, refs)
    gap = pure_b - gt_b
    results["headline"] = {"gt": gt_b, "pure": pure_b, "gap": gap}

    # 2. Bootstrap CI (precomputed per-item n-gram statistics for speed)
    import numpy as _np
    def _item_stats(hyps, refs_):
        stats = []
        for h, r in zip(hyps, refs_):
            rk = BLEU._extract_reference_info([r])
            stats.append(BLEU._compute_segment_statistics(h, rk))
        return _np.asarray(stats, dtype=_np.int64)
    def _bleu_idx(stats_arr, idx):
        agg = stats_arr[idx].sum(axis=0).tolist()
        return BLEU._compute_score_from_stats(agg).score
    gt_stats = item_stats_gt = _item_stats(gt_h, refs)
    pure_stats = _item_stats(pure_h, refs)
    rng = _np.random.RandomState(42)
    N = len(ids)
    gaps = _np.empty(10000)
    for b in range(10000):
        idx = rng.randint(0, N, N)
        gaps[b] = _bleu_idx(pure_stats, idx) - _bleu_idx(gt_stats, idx)
    results["headline"]["bootstrap_ci"] = [
        float(_np.percentile(gaps, 2.5)), float(_np.percentile(gaps, 97.5))]

    # 3. 14-seed reconstruction PI
    reco_gaps = []
    for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405]:
        name = f"reco_{seed}"
        try:
            _, _, g, p = load_items(name)
            reco_gaps.append(corpus_bleu(p, refs) - corpus_bleu(g, refs))
        except FileNotFoundError:
            continue
    mean = np.mean(reco_gaps)
    sd = np.std(reco_gaps, ddof=1)
    t_crit = stats.t.ppf(0.975, len(reco_gaps) - 1)
    pi_lo = mean - t_crit * sd * np.sqrt(1 + 1.0 / len(reco_gaps))
    pi_hi = mean + t_crit * sd * np.sqrt(1 + 1.0 / len(reco_gaps))
    results["reco_14seed_PI"] = {"mean": float(mean), "sd": float(sd),
                                  "n": len(reco_gaps),
                                  "n_positive": sum(1 for g in reco_gaps if g > 0),
                                  "PI": [float(pi_lo), float(pi_hi)]}

    # 4. Non-released gap range
    all_gaps = []
    for name in json.load(open(CANONICAL)).keys():
        if name in ("released", "_meta"):
            continue
        d = json.load(open(CANONICAL))
        v = d.get(name, {})
        if isinstance(v, dict) and "gap" in v:
            all_gaps.append(v["gap"])
    results["non_released_gap_range"] = [
        float(min(all_gaps)), float(max(all_gaps))]
    results["n_non_released"] = len(all_gaps)
    results["n_positive"] = sum(1 for g in all_gaps if g > 0)

    # 5. Matched-subset (Czehmann conf=1)
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
            out[parts[0]] = (parts[1], conf)
        return out

    human_hc = load_csv(ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv")
    ids_hc = sorted(set(ids) & set(human_hc.keys()))
    gt_map = dict(zip(ids, gt_h))
    pure_map = dict(zip(ids, pure_h))
    ref_map = dict(zip(ids, refs))

    if ids_hc:
        ro = [ref_map[i] for i in ids_hc]
        rh = [human_hc[i][0] for i in ids_hc]
        gt_o = corpus_bleu([gt_map[i] for i in ids_hc], ro)
        pure_o = corpus_bleu([pure_map[i] for i in ids_hc], ro)
        gt_h2 = corpus_bleu([gt_map[i] for i in ids_hc], rh)
        pure_h2 = corpus_bleu([pure_map[i] for i in ids_hc], rh)
        orig_gap = pure_o - gt_o
        human_gap = pure_h2 - gt_h2
        att = orig_gap - human_gap
        results["matched_subset"] = {
            "n": len(ids_hc),
            "orig_gt": gt_o, "orig_pure": pure_o, "orig_gap": orig_gap,
            "human_gt": gt_h2, "human_pure": pure_h2, "human_gap": human_gap,
            "attenuation": att,
            "attenuation_pct": att / orig_gap * 100 if orig_gap else 0}

    # 6. Full-641 sensitivity
    human_full = load_csv(ROOT / "data/sacrebird/test_full_annotations_sacrebirdphoenix.csv")
    ids_full = sorted(set(ids) & set(human_full.keys()))
    rh_f = []
    if ids_full:
        rh_f = [human_full[i][0] for i in ids_full]
        gt_hf = corpus_bleu([gt_map[i] for i in ids_full], rh_f)
        pure_hf = corpus_bleu([pure_map[i] for i in ids_full], rh_f)
        gt_of = corpus_bleu([gt_map[i] for i in ids_full], [ref_map[i] for i in ids_full])
        pure_of = corpus_bleu([pure_map[i] for i in ids_full], [ref_map[i] for i in ids_full])
        results["full_641_sensitivity"] = {
            "orig_gap": pure_of - gt_of,
            "human_gap": pure_hf - gt_hf,
            "attenuation": (pure_of - gt_of) - (pure_hf - gt_hf)}

    # 7. Cross-metric for released + 6 primary reconstructions
    results["cross_metric"] = {}
    for name in ["released"] + [f"reco_{s}" for s in [101, 202, 303, 404, 505, 606]]:
        try:
            _, _, g, p = load_items(name)
            results["cross_metric"][name] = {
                "gap_bleu4": corpus_bleu(p, refs) - corpus_bleu(g, refs)}
        except FileNotFoundError:
            continue

    # 8. UNSEEN released
    try:
        unseen_items = json.load(open(ITEMS / "released_unseen.json"))
        unseen_h = [it["hypothesis"] for it in unseen_items]
        unseen_refs = [it["reference"] for it in unseen_items]
        gt_b2 = corpus_bleu(gt_h, refs)
        unseen_b = corpus_bleu(unseen_h, unseen_refs)
        results["donor_origin"] = {
            "seen_gap": results["headline"]["gap"],
            "unseen_gap": unseen_b - gt_b2,
            "estimand": results["headline"]["gap"] - (unseen_b - gt_b2)}
    except FileNotFoundError:
        pass

    # 9. Floor effect
    if ids_full and ids_hc:
        results["floor_effect"] = {
            "gt_orig": corpus_bleu([gt_map[i] for i in ids_full], [ref_map[i] for i in ids_full]),
            "gt_human": corpus_bleu([gt_map[i] for i in ids_full], rh_f),
            "pure_orig": corpus_bleu([pure_map[i] for i in ids_full], [ref_map[i] for i in ids_full]),
            "pure_human": corpus_bleu([pure_map[i] for i in ids_full], rh_f)}

    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"Paper numbers written to {OUT}")
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
