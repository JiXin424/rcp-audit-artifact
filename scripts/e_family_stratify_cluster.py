#!/usr/bin/env python3
"""Per-family dose-response stratification + improved cluster bootstrap [E:E0001].

Reviewer concern #8: 41 checkpoints come from heterogeneous generation mechanisms
(reconstruction, ladder, config-faithful, step-faithful, distillation, rescue).
They are NOT IID replicates. The 14 same-recipe seeds can have a PI, but the
heterogeneous families should be analyzed separately.

Also: template-family clusters using first-60-chars are nearly all singletons.
Replace with interpretable clusters: show (from sequence ID), signer, donor ID.
"""
import json, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import sacrebleu

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "results/cells"
OUT = ROOT / "results/family_stratify_cluster.json"
BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def load_dose_response():
    """Load the 30-point dose-response data."""
    d = json.load(open(ROOT / "results/dose_response.json"))
    return d["points"]


def stratify_by_family(points):
    """Group dose-response points by family and compute per-family stats."""
    by_family = defaultdict(list)
    for p in points:
        by_family[p["family"]].append(p)

    result = {}
    for fam, pts in sorted(by_family.items()):
        gaps = [p["gap"] for p in pts if p.get("gap") is not None]
        dev_nlls = [p["dev_nll"] for p in pts if p.get("dev_nll") is not None]
        result[fam] = {
            "n": len(pts),
            "gap_range": [min(gaps), max(gaps)] if gaps else None,
            "gap_mean": float(np.mean(gaps)) if gaps else None,
            "gap_sd": float(np.std(gaps, ddof=1)) if len(gaps) > 1 else None,
            "dev_nll_range": [min(dev_nlls), max(dev_nlls)] if dev_nlls else None,
        }

    # Same-recipe reconstruction PI (14 seeds)
    reco = by_family.get("reconstructions_primary", []) + by_family.get("reconstructions_extension", [])
    reco_gaps = [p["gap"] for p in reco if p.get("gap") is not None]
    if len(reco_gaps) >= 2:
        mean = np.mean(reco_gaps)
        sd = np.std(reco_gaps, ddof=1)
        # Student-t prediction interval for next observation
        from scipy import stats
        t_crit = stats.t.ppf(0.975, len(reco_gaps) - 1)
        pi_lo = mean - t_crit * sd * np.sqrt(1 + 1.0 / len(reco_gaps))
        pi_hi = mean + t_crit * sd * np.sqrt(1 + 1.0 / len(reco_gaps))
        result["_reconstruction_PI_14seed"] = {
            "n": len(reco_gaps),
            "mean": float(mean),
            "sd": float(sd),
            "t_975": float(t_crit),
            "PI_lo": float(pi_lo),
            "PI_hi": float(pi_hi),
            "note": "Same-recipe 14-seed Student-t prediction interval. "
                    "Valid for inference about the reconstruction recipe, "
                    "NOT for the released evaluator.",
        }

    # Heterogeneous family summary (all non-reconstruction, non-released)
    hetero = [p for fam, pts in by_family.items()
              if fam not in ("reconstructions_primary", "reconstructions_extension", "released")
              for p in pts]
    hetero_gaps = [p["gap"] for p in hetero if p.get("gap") is not None]
    if hetero_gaps:
        result["_heterogeneous_summary"] = {
            "n": len(hetero_gaps),
            "families": sorted(set(p["family"] for p in hetero)),
            "gap_range": [min(hetero_gaps), max(hetero_gaps)],
            "gap_mean": float(np.mean(hetero_gaps)),
            "note": "Heterogeneous checkpoints from different generation mechanisms. "
                    "Shown for range exploration only; NOT IID replicates.",
        }

    return result


def load_test_items_with_metadata():
    """Load test items with signer and show metadata for cluster bootstrap."""
    import torch
    raw = torch.load(str(ROOT / "data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/test.pt"),
                     map_location="cpu", weights_only=False)
    items = []
    for k, v in raw.items():
        # Extract show from sequence ID (e.g., "21September_2010_Tuesday_tagesschau-950")
        # Show = the news program name: "heute" or "tagesschau"
        m = re.match(r'\d+\w+_\d+_\w+_(\w+)-\d+', k)
        show = m.group(1) if m else "unknown"
        items.append({
            "id": k,
            "text": v.get("text", ""),
            "speaker": v.get("speaker", "unknown"),
            "show": show,
            "date": k.split("_")[0] + "_" + k.split("_")[1] + "_" + k.split("_")[2],
        })
    return items


def load_donor_registry():
    """Load query→donor mapping for donor-ID clustering."""
    reg_path = ROOT / "results/donor_registry.jsonl"
    if reg_path.exists():
        donors = {}
        for line in reg_path.read_text().splitlines():
            r = json.loads(line)
            donors[r["query_id"]] = r.get("donor_id", r.get("donor", ""))
        return donors
    return {}


def cluster_bootstrap_gap(hyps_gt, hyps_pure, refs, cluster_labels, ids,
                           n_resample=10000, seed=42, label_name="cluster"):
    """Cluster-aware bootstrap: resample clusters (not items).

    All items in a cluster are kept together or dropped together.
    """
    rng = np.random.RandomState(seed)

    # Group items by cluster
    cluster_map = defaultdict(list)
    for iid in ids:
        cl = cluster_labels.get(iid, "singleton_" + iid)
        cluster_map[cl].append(iid)

    clusters = sorted(cluster_map.keys())
    n_clusters = len(clusters)
    sizes = [len(cluster_map[c]) for c in clusters]

    gaps = []
    for _ in range(n_resample):
        # Sample clusters with replacement
        sampled = rng.choice(n_clusters, n_clusters, replace=True)
        sel_ids = []
        for ci in sampled:
            sel_ids.extend(cluster_map[clusters[ci]])
        gt_s = [hyps_gt[i] for i in sel_ids]
        pure_s = [hyps_pure[i] for i in sel_ids]
        r_s = [refs[i] for i in sel_ids]
        gt_b = BLEU.corpus_score(gt_s, [r_s]).score
        pure_b = BLEU.corpus_score(pure_s, [r_s]).score
        gaps.append(pure_b - gt_b)

    gaps = np.array(gaps)
    # Effective sample size (rough): n_clusters
    return {
        "cluster_type": label_name,
        "n_clusters": n_clusters,
        "max_cluster_size": max(sizes),
        "mean_cluster_size": float(np.mean(sizes)),
        "n_items": len(ids),
        "gap_mean": float(np.mean(gaps)),
        "ci_lo": float(np.percentile(gaps, 2.5)),
        "ci_hi": float(np.percentile(gaps, 97.5)),
    }


def main():
    t0 = time.time()
    print("=== Per-family dose-response stratification ===", flush=True)
    points = load_dose_response()
    fam_stats = stratify_by_family(points)
    print(json.dumps(fam_stats, indent=1), flush=True)

    print("\n=== Cluster-aware bootstrap ===", flush=True)
    # Load test items
    test_items = load_test_items_with_metadata()
    test_ids = [it["id"] for it in test_items]
    print(f"Loaded {len(test_items)} test items", flush=True)

    # Show distribution
    shows = Counter(it["show"] for it in test_items)
    signers = Counter(it["speaker"] for it in test_items)
    print(f"Shows: {dict(shows)}", flush=True)
    print(f"Signers: {dict(signers)}", flush=True)

    # Load decoded cells
    gt_cell = json.load(open(CELLS / "cp0_GT-v1.json"))
    pure_cell = json.load(open(CELLS / "cp0_TN-PURE-v1.json"))
    hyps_gt = {it["id"]: it["hypothesis"] for it in gt_cell["items"]}
    hyps_pure = {it["id"]: it["hypothesis"] for it in pure_cell["items"]}
    refs = {it["id"]: it["reference"] for it in gt_cell["items"]}

    ids = sorted(set(hyps_gt.keys()) & set(hyps_pure.keys()) & set(refs.keys()))
    print(f"Common IDs: {len(ids)}", flush=True)

    # Build cluster labels
    signer_labels = {it["id"]: it["speaker"] for it in test_items}
    show_labels = {it["id"]: it["show"] for it in test_items}
    date_labels = {it["id"]: it["date"] for it in test_items}

    # Donor labels
    donor_reg = load_donor_registry()
    donor_labels = {k: v for k, v in donor_reg.items() if k in ids}

    # Combined: signer × show
    combined_labels = {it["id"]: f"{it['speaker']}_{it['show']}" for it in test_items}

    results = {
        "family_stratification": fam_stats,
        "cluster_bootstrap": {},
    }

    # Query-level (baseline)
    print("\nComputing query-level bootstrap...", flush=True)
    rng = np.random.RandomState(42)
    N = len(ids)
    gaps_q = []
    for _ in range(10000):
        idx = rng.randint(0, N, N)
        gt_b = BLEU.corpus_score([hyps_gt[ids[j]] for j in idx],
                                  [[refs[ids[j]] for j in idx]]).score
        pure_b = BLEU.corpus_score([hyps_pure[ids[j]] for j in idx],
                                    [[refs[ids[j]] for j in idx]]).score
        gaps_q.append(pure_b - gt_b)
    gaps_q = np.array(gaps_q)
    results["cluster_bootstrap"]["query_level"] = {
        "n_items": N,
        "gap_mean": float(np.mean(gaps_q)),
        "ci_lo": float(np.percentile(gaps_q, 2.5)),
        "ci_hi": float(np.percentile(gaps_q, 97.5)),
    }
    print(f"  Query: {np.mean(gaps_q):.2f} [{np.percentile(gaps_q, 2.5):.2f}, {np.percentile(gaps_q, 97.5):.2f}]", flush=True)

    # Signer cluster
    print("Computing signer cluster bootstrap...", flush=True)
    r = cluster_bootstrap_gap(hyps_gt, hyps_pure, refs, signer_labels, ids,
                               label_name="signer")
    results["cluster_bootstrap"]["signer"] = r
    print(f"  Signer ({r['n_clusters']} clusters): {r['gap_mean']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", flush=True)

    # Show cluster
    print("Computing show cluster bootstrap...", flush=True)
    r = cluster_bootstrap_gap(hyps_gt, hyps_pure, refs, show_labels, ids,
                               label_name="show")
    results["cluster_bootstrap"]["show"] = r
    print(f"  Show ({r['n_clusters']} clusters): {r['gap_mean']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", flush=True)

    # Date cluster
    print("Computing date cluster bootstrap...", flush=True)
    r = cluster_bootstrap_gap(hyps_gt, hyps_pure, refs, date_labels, ids,
                               label_name="broadcast_date")
    results["cluster_bootstrap"]["broadcast_date"] = r
    print(f"  Date ({r['n_clusters']} clusters): {r['gap_mean']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", flush=True)

    # Donor cluster
    if donor_labels:
        print("Computing donor cluster bootstrap...", flush=True)
        r = cluster_bootstrap_gap(hyps_gt, hyps_pure, refs, donor_labels, ids,
                                   label_name="donor_id")
        results["cluster_bootstrap"]["donor_id"] = r
        print(f"  Donor ({r['n_clusters']} clusters): {r['gap_mean']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", flush=True)

    # Combined signer × show
    print("Computing signer×show cluster bootstrap...", flush=True)
    r = cluster_bootstrap_gap(hyps_gt, hyps_pure, refs, combined_labels, ids,
                               label_name="signer_x_show")
    results["cluster_bootstrap"]["signer_x_show"] = r
    print(f"  Signer×Show ({r['n_clusters']} clusters): {r['gap_mean']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", flush=True)

    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"\nSaved to {OUT} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
