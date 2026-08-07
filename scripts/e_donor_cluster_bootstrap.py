#!/usr/bin/env python3
"""Donor-reuse statistics and donor-cluster bootstrap [reviewer R2-4].

The headline PURE-GT gap bootstrap resamples queries, but multiple queries can
reuse the same pose donor, so item-level resampling overstates the effective
sample size. This script:

  1. Reports donor reuse for TN-PURE-v1: unique donors, max reuse, full reuse
     distribution, and the Gini coefficient.
  2. Recomputes all cluster bootstraps on the CANONICAL materialization cells
     (results/gap_43_canonical_beam3_items/released_{gt,pure}.json); the earlier
     Sup. G numbers were computed on the pre-fix cells in results/cells/.
  3. Adds a donor-cluster bootstrap (queries sharing a donor are resampled
     together) and a two-way query x donor bootstrap.

Per-item n-gram sufficient statistics are precomputed once; every resample
re-aggregates them into a corpus BLEU (exact, matches sacrebleu corpus_score).

Output: results/donor_cluster_bootstrap.json
"""
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/gap_43_canonical_beam3_items"
OUT = ROOT / "results/donor_cluster_bootstrap.json"
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)
N_RESAMPLE = 10000
SEED = 42


def item_stats(hyps, refs):
    """Precompute per-item sufficient statistics (list of 10 ints)."""
    stats = []
    for h, r in zip(hyps, refs):
        rk = BLEU._extract_reference_info([r])
        stats.append(BLEU._compute_segment_statistics(h, rk))
    return np.asarray(stats, dtype=np.int64)


def bleu_from_stats(stats_array, idx):
    agg = stats_array[idx].sum(axis=0).tolist()
    return BLEU._compute_score_from_stats(agg).score


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if v.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def bootstrap(labels, ids, stats_gt, stats_pure, rng, two_way=False):
    """Cluster bootstrap. If two_way, also resample items within clusters."""
    cluster_map = defaultdict(list)
    for pos, iid in enumerate(ids):
        cluster_map[labels.get(iid, "singleton_" + iid)].append(pos)
    clusters = sorted(cluster_map)
    members = [np.asarray(cluster_map[c]) for c in clusters]
    n_clusters = len(clusters)
    gaps = np.empty(N_RESAMPLE)
    for b in range(N_RESAMPLE):
        ci = rng.randint(0, n_clusters, n_clusters)
        if two_way:
            idx = np.concatenate([
                members[i][rng.randint(0, len(members[i]), len(members[i]))]
                for i in ci
            ])
        else:
            idx = np.concatenate([members[i] for i in ci])
        gaps[b] = (bleu_from_stats(stats_pure, idx)
                   - bleu_from_stats(stats_gt, idx))
    return {
        "n_clusters": n_clusters,
        "max_cluster_size": int(max(len(m) for m in members)),
        "mean_cluster_size": float(np.mean([len(m) for m in members])),
        "gap_mean": float(gaps.mean()),
        "ci_lo": float(np.percentile(gaps, 2.5)),
        "ci_hi": float(np.percentile(gaps, 97.5)),
    }


def main():
    t0 = time.time()
    gt = json.load(open(ITEMS / "released_gt.json"))
    pure = json.load(open(ITEMS / "released_pure.json"))
    hyps_gt = {x["id"]: x["hypothesis"] for x in gt}
    hyps_pure = {x["id"]: x["hypothesis"] for x in pure}
    refs = {x["id"]: x["reference"] for x in gt}
    ids = sorted(set(hyps_gt) & set(hyps_pure) & set(refs))
    assert len(ids) == 641, len(ids)

    # Donor registry
    donors = {}
    for line in open(ITEMS / "donor_registry.jsonl"):
        r = json.loads(line)
        donors[r["query_id"]] = r["donor_id"]
    assert set(donors) == set(ids), "registry/query mismatch"

    # --- Donor reuse statistics ---
    reuse = Counter(donors[i] for i in ids)
    reuse_counts = sorted(reuse.values(), reverse=True)
    reuse_hist = Counter(reuse.values())
    donor_reuse = {
        "n_queries": len(ids),
        "n_unique_donors": len(reuse),
        "max_reuse": max(reuse_counts),
        "reuse_histogram": {str(k): reuse_hist[k] for k in sorted(reuse_hist)},
        "gini": gini(list(reuse.values())),
        "n_donors_reused_gt1": sum(1 for v in reuse.values() if v > 1),
        "top10_donors": reuse.most_common(10),
    }
    print(json.dumps({k: v for k, v in donor_reuse.items()
                      if k != "top10_donors"}, indent=1), flush=True)

    # --- Precompute stats ---
    h_gt = [hyps_gt[i] for i in ids]
    h_pure = [hyps_pure[i] for i in ids]
    r = [refs[i] for i in ids]
    stats_gt = item_stats(h_gt, r)
    stats_pure = item_stats(h_pure, r)
    base = bleu_from_stats(stats_pure, np.arange(len(ids))) - \
        bleu_from_stats(stats_gt, np.arange(len(ids)))
    print(f"canonical gap (check): {base:.4f}  [expect 10.2433]", flush=True)

    # --- Metadata labels ---
    raw = torch.load(str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"),
                     map_location="cpu", weights_only=False)
    signer, show, date = {}, {}, {}
    for k, v in raw.items():
        if k not in donors:
            continue
        m = re.match(r"\d+\w+_\d+_\w+_(\w+)-\d+", k)
        show[k] = m.group(1) if m else "unknown"
        signer[k] = v.get("speaker", "unknown")
        parts = k.split("_")
        date[k] = "_".join(parts[:3])
    sx = {k: f"{signer[k]}_{show[k]}" for k in signer}

    rng = np.random.RandomState(SEED)
    res = {"schema": "donor-cluster-bootstrap-v1",
           "materialization": "canonical (released_{gt,pure}.json, "
                              "exclusion-fixed donor registry)",
           "n_resample": N_RESAMPLE, "seed": SEED,
           "canonical_gap": float(base),
           "donor_reuse": donor_reuse,
           "bootstraps": {}}

    # Query level
    N = len(ids)
    gaps = np.empty(N_RESAMPLE)
    for b in range(N_RESAMPLE):
        idx = rng.randint(0, N, N)
        gaps[b] = bleu_from_stats(stats_pure, idx) - bleu_from_stats(stats_gt, idx)
    res["bootstraps"]["query_level"] = {
        "n_items": N, "gap_mean": float(gaps.mean()),
        "ci_lo": float(np.percentile(gaps, 2.5)),
        "ci_hi": float(np.percentile(gaps, 97.5))}
    print(f"query: {gaps.mean():.2f} [{np.percentile(gaps,2.5):.2f}, "
          f"{np.percentile(gaps,97.5):.2f}]", flush=True)

    for name, labels in [("donor", donors), ("signer", signer),
                         ("show", show), ("broadcast_date", date),
                         ("signer_x_show", sx)]:
        rb = bootstrap(labels, ids, stats_gt, stats_pure, rng)
        res["bootstraps"][name] = rb
        print(f"{name} ({rb['n_clusters']} clusters): {rb['gap_mean']:.2f} "
              f"[{rb['ci_lo']:.2f}, {rb['ci_hi']:.2f}]", flush=True)

    rb = bootstrap(donors, ids, stats_gt, stats_pure, rng, two_way=True)
    res["bootstraps"]["donor_two_way"] = rb
    print(f"donor two-way: {rb['gap_mean']:.2f} [{rb['ci_lo']:.2f}, "
          f"{rb['ci_hi']:.2f}]", flush=True)

    res["elapsed_s"] = time.time() - t0
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"saved -> {OUT} ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
