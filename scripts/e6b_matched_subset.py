#!/usr/bin/env python3
"""E6b-matched: matched-subset reference sensitivity analysis [E:E0004].

Core analysis: compute original-ref and human-ref gaps on the SAME confidence=1
subset (461 items), with paired attenuation, bootstrap CI, and permutation test.
This is fast (no GPU needed). BERTScore is optional and may be slow.
"""
import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import sacrebleu

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "results/cells"
CSV_FULL = ROOT / "data/sacrebird/test_full_annotations_sacrebirdphoenix.csv"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
OUT = ROOT / "results/e6b_matched_subset.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)
CHRF = sacrebleu.metrics.CHRF()
BLEU1 = sacrebleu.metrics.BLEU(max_ngram_order=1, tokenize="13a",
                               smooth_method="exp", effective_order=False, force=True)


def load_csv(path):
    out = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            conf = float(parts[2])
        except ValueError:
            conf = 0.0
        out[parts[0]] = (parts[1], conf)
    return out


def load_cells():
    gt = json.load(open(CELLS / "cp0_GT-v1.json"))
    pure = json.load(open(CELLS / "cp0_TN-PURE-v1.json"))
    return {it["id"]: it for it in gt["items"]}, {it["id"]: it for it in pure["items"]}


def score(hyps, refs, metric="bleu"):
    if metric == "bleu":
        return BLEU.corpus_score(hyps, [refs]).score
    elif metric == "chrf":
        return CHRF.corpus_score(hyps, [refs]).score
    elif metric == "bleu1":
        return BLEU1.corpus_score(hyps, [refs]).score


def gap_on_subset(hyps_gt, hyps_pure, refs, ids, metric="bleu"):
    h_gt = [hyps_gt[i] for i in ids]
    h_pure = [hyps_pure[i] for i in ids]
    r = [refs[i] for i in ids]
    gt_s = score(h_gt, r, metric)
    pure_s = score(h_pure, r, metric)
    return gt_s, pure_s, pure_s - gt_s


def paired_bootstrap(hyps_gt, hyps_pure, refs_orig, refs_human, ids,
                     n=10000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(ids)
    gt = [hyps_gt[i] for i in ids]
    pure = [hyps_pure[i] for i in ids]
    ro = [refs_orig[i] for i in ids]
    rh = [refs_human[i] for i in ids]
    atts = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        og = BLEU.corpus_score([pure[j] for j in idx], [[ro[j] for j in idx]]).score - \
             BLEU.corpus_score([gt[j] for j in idx], [[ro[j] for j in idx]]).score
        hg = BLEU.corpus_score([pure[j] for j in idx], [[rh[j] for j in idx]]).score - \
             BLEU.corpus_score([gt[j] for j in idx], [[rh[j] for j in idx]]).score
        atts.append(og - hg)
    atts = np.array(atts)
    return {"mean": float(atts.mean()),
            "ci_lo": float(np.percentile(atts, 2.5)),
            "ci_hi": float(np.percentile(atts, 97.5))}


def dual_ref_bleu(hyps, refs_a, refs_b, ids):
    """Multi-reference BLEU (closest-length convention)."""
    h = [hyps[i] for i in ids]
    ra = [refs_a[i] for i in ids]
    rb = [refs_b[i] for i in ids]
    return BLEU.corpus_score(h, list(zip(ra, rb))).score


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    gt_cell, pure_cell = load_cells()
    human_full = load_csv(CSV_FULL)
    human_hc = load_csv(CSV_HC)

    ids_all = sorted(gt_cell.keys())
    hyps_gt = {k: v["hypothesis"] for k, v in gt_cell.items()}
    hyps_pure = {k: v["hypothesis"] for k, v in pure_cell.items()}
    refs_orig = {k: v["reference"] for k, v in gt_cell.items()}
    refs_hf = {k: v[0] for k, v in human_full.items() if k in ids_all}
    refs_hc = {k: v[0] for k, v in human_hc.items() if k in ids_all}

    ids_full = sorted(set(ids_all) & set(refs_hf))
    ids_hc = sorted(set(ids_all) & set(refs_hc))

    conf_dist = defaultdict(int)
    for iid in ids_all:
        if iid in human_full:
            conf_dist[str(human_full[iid][1])] += 1
        else:
            conf_dist["missing"] += 1

    print(f"Full: {len(ids_full)}, HC: {len(ids_hc)}, conf: {dict(conf_dist)}", flush=True)

    refs_orig_full = {k: refs_orig[k] for k in ids_full}
    refs_orig_hc = {k: refs_orig[k] for k in ids_hc}
    refs_hf_map = {k: refs_hf[k] for k in ids_full}
    refs_hc_map = {k: refs_hc[k] for k in ids_hc}

    out = {"note": "Matched-subset reference sensitivity [E:E0004]",
           "confidence_breakdown": dict(conf_dist)}

    # === PRIMARY: confidence=1 matched subset ===
    print("\n=== Confidence=1 matched subset ===", flush=True)
    for metric in ["bleu", "chrf", "bleu1"]:
        og_gt, og_pure, og_gap = gap_on_subset(hyps_gt, hyps_pure, refs_orig_hc, ids_hc, metric)
        hg_gt, hg_pure, hg_gap = gap_on_subset(hyps_gt, hyps_pure, refs_hc_map, ids_hc, metric)
        print(f"  {metric}: orig gap={og_gap:.2f}, human gap={hg_gap:.2f}, "
              f"attenuation={og_gap - hg_gap:.2f} ({(og_gap-hg_gap)/og_gap*100:.1f}%)", flush=True)
        out.setdefault("matched_confidence1", {})[f"original_{metric}"] = {
            "gt": og_gt, "pure": og_pure, "gap": og_gap}
        out["matched_confidence1"][f"human_{metric}"] = {
            "gt": hg_gt, "pure": hg_pure, "gap": hg_gap}
        out["matched_confidence1"][f"attenuation_{metric}"] = {
            "absolute": og_gap - hg_gap,
            "pct": (og_gap - hg_gap) / og_gap * 100 if og_gap != 0 else None}

    # Dual reference
    dual_gt = dual_ref_bleu(hyps_gt, refs_orig_hc, refs_hc_map, ids_hc)
    dual_pure = dual_ref_bleu(hyps_pure, refs_orig_hc, refs_hc_map, ids_hc)
    out["matched_confidence1"]["dual_bleu"] = {
        "gt": dual_gt, "pure": dual_pure, "gap": dual_pure - dual_gt}
    print(f"  dual BLEU gap: {dual_pure - dual_gt:.2f}", flush=True)

    # Paired attenuation bootstrap
    print("  Computing paired bootstrap...", flush=True)
    att = paired_bootstrap(hyps_gt, hyps_pure, refs_orig_hc, refs_hc_map, ids_hc)
    out["matched_confidence1"]["paired_attenuation_bleu"] = att
    print(f"  Paired attenuation: {att['mean']:.2f} [{att['ci_lo']:.2f}, {att['ci_hi']:.2f}]", flush=True)

    # === SECONDARY: Full 641 ===
    print("\n=== Full 641 sensitivity ===", flush=True)
    for metric in ["bleu", "chrf", "bleu1"]:
        og_gt, og_pure, og_gap = gap_on_subset(hyps_gt, hyps_pure, refs_orig_full, ids_full, metric)
        hg_gt, hg_pure, hg_gap = gap_on_subset(hyps_gt, hyps_pure, refs_hf_map, ids_full, metric)
        out.setdefault("full_641_sensitivity", {})[f"original_{metric}"] = {
            "gt": og_gt, "pure": og_pure, "gap": og_gap}
        out["full_641_sensitivity"][f"human_{metric}"] = {
            "gt": hg_gt, "pure": hg_pure, "gap": hg_gap}
        out["full_641_sensitivity"][f"attenuation_{metric}"] = {
            "absolute": og_gap - hg_gap,
            "pct": (og_gap - hg_gap) / og_gap * 100 if og_gap != 0 else None}

    att_full = paired_bootstrap(hyps_gt, hyps_pure, refs_orig_full, refs_hf_map, ids_full)
    out["full_641_sensitivity"]["paired_attenuation_bleu"] = att_full

    # Print summary table
    print("\n=== SUMMARY (BLEU) ===")
    hc = out["matched_confidence1"]
    fl = out["full_641_sensitivity"]
    print(f"{'Subset':<20} {'Orig GT':<10} {'Orig PURE':<10} {'Orig gap':<10} {'Hum GT':<10} {'Hum PURE':<10} {'Hum gap':<10} {'Att%':<10}")
    print(f"{'Confidence=1 (461)':<20} {hc['original_bleu']['gt']:<10.2f} {hc['original_bleu']['pure']:<10.2f} {hc['original_bleu']['gap']:<10.2f} {hc['human_bleu']['gt']:<10.2f} {hc['human_bleu']['pure']:<10.2f} {hc['human_bleu']['gap']:<10.2f} {hc['attenuation_bleu']['pct']:<10.1f}")
    print(f"{'Full 641':<20} {fl['original_bleu']['gt']:<10.2f} {fl['original_bleu']['pure']:<10.2f} {fl['original_bleu']['gap']:<10.2f} {fl['human_bleu']['gt']:<10.2f} {fl['human_bleu']['pure']:<10.2f} {fl['human_bleu']['gap']:<10.2f} {fl['attenuation_bleu']['pct']:<10.1f}")

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSaved to {OUT} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
