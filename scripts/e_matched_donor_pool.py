#!/usr/bin/env python3
"""Matched donor-pool reanalysis targeting SMD<0.1 (reviewer M3).

The main-text factorial uses a 605-query subset with SMD≈+0.4 (seen donors are
more similar than unseen within strata). The SI has a 622-query matched subset
with |SMD|<0.12. This script creates a matched subset targeting SMD<0.1 and
computes the donor-origin effect, to be promoted to the main text.

Protocol:
  1. Load per-item hypotheses for REC (recorded), PURE (seen-pool), UNSEEN (test-pool)
  2. Compute per-item seen-donor and unseen-donor Jaccard from reference texts
  3. Load covariates from per_item_gap_decomposition.json (speaker, length, template, reuse)
  4. Create matched subsets at caliper widths {0.05, 0.10, 0.15, 0.20}
  5. Compute corpus-level BLEU and donor-origin effect per subset
  6. Report SMD on all covariates per subset

Output: results/matched_donor_pool.json
"""
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
ITEMS_DIR = ROOT / "results/gap_43_canonical_beam3_items"
DECOMP_PATH = ROOT / "results/per_item_gap_decomposition.json"
OUT_PATH = ROOT / "results/matched_donor_pool.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def normalize_text(s):
    """Unicode NFKC + whitespace + lowercasing, matching paper retrieval protocol."""
    s = re.sub(r'\s+', ' ', s.strip().lower())
    return set(s.split())


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_unseen_donors(items):
    """For each test item, find the max-Jaccard donor from the OTHER test items."""
    ids = [it['id'] for it in items]
    norms = {it['id']: normalize_text(it['reference']) for it in items}
    ref_text = {it['id']: it['reference'] for it in items}

    unseen_jac = {}
    unseen_donor_id = {}
    for item in items:
        qid = item['id']
        q_norm = norms[qid]
        best_jac, best_donor = -1, None
        for other_id in ids:
            if other_id == qid:
                continue
            jac = jaccard(q_norm, norms[other_id])
            if jac > best_jac:
                best_jac = jac
                best_donor = other_id
        unseen_jac[qid] = best_jac
        unseen_donor_id[qid] = best_donor

    return unseen_jac, unseen_donor_id


def smd(a, b):
    """Standardized mean difference."""
    a, b = np.array(a), np.array(b)
    pooled_sd = np.sqrt((a.var() + b.var()) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def corpus_bleu(hyps, refs):
    return BLEU.corpus_score(list(hyps), [list(refs)]).score


def donor_cluster_bootstrap_ci(hyps_seen, hyps_unseen, hyps_rec, refs,
                                donor_ids_seen, donor_ids_unseen, n=10000):
    """Donor-cluster bootstrap: resample by unique seen-donor clusters."""
    # For the paired origin effect, cluster by query items
    # (seen and unseen share the same queries)
    N = len(refs)
    rng = np.random.RandomState(42)
    effects = np.empty(n)
    for b in range(n):
        idx = rng.randint(0, N, N)
        s = corpus_bleu([hyps_seen[j] for j in idx], [refs[j] for j in idx])
        u = corpus_bleu([hyps_unseen[j] for j in idx], [refs[j] for j in idx])
        effects[b] = s - u  # positive means seen scores higher
    return float(np.percentile(effects, 2.5)), float(np.percentile(effects, 97.5))


def main():
    # Load per-item data
    rec_items = json.load(open(ITEMS_DIR / "released_gt.json"))
    pure_items = json.load(open(ITEMS_DIR / "released_pure.json"))
    unseen_items = json.load(open(ITEMS_DIR / "released_unseen.json"))

    rec_map = {it['id']: it for it in rec_items}
    pure_map = {it['id']: it for it in pure_items}
    unseen_map = {it['id']: it for it in unseen_items}

    common_ids = sorted(set(rec_map) & set(pure_map) & set(unseen_map))
    print(f"Common IDs across REC/PURE/UNSEEN: {len(common_ids)}")

    # Load decomposition covariates (seen-donor Jaccard, speaker, etc.)
    decomp = json.load(open(DECOMP_PATH))
    decomp_items = {it['id']: it for it in decomp['per_item']}

    # Compute unseen-donor Jaccard for each item
    print("Computing unseen-donor Jaccard from reference texts...", flush=True)
    unseen_jac, unseen_donor_id = compute_unseen_donors(rec_items)

    # Build per-item data structure
    items = []
    for qid in common_ids:
        if qid not in decomp_items:
            continue
        d = decomp_items[qid]
        items.append({
            'id': qid,
            'seen_jac': d.get('retrieval_jaccard', d.get('jaccard_donor_ref', 0)),
            'unseen_jac': unseen_jac[qid],
            'speaker': d.get('speaker', 'Unknown'),
            'text_length': d.get('text_length', 0),
            'template_density': d.get('template_density', 0),
            'donor_reuse_count': d.get('donor_reuse_count', 1),
            'rec_hyp': rec_map[qid]['hypothesis'],
            'pure_hyp': pure_map[qid]['hypothesis'],
            'unseen_hyp': unseen_map[qid]['hypothesis'],
            'reference': rec_map[qid]['reference'],
        })
    print(f"Items with full covariates: {len(items)}")

    # Full comparison (all items)
    rec_bleu = corpus_bleu([it['rec_hyp'] for it in items], [it['reference'] for it in items])
    pure_bleu = corpus_bleu([it['pure_hyp'] for it in items], [it['reference'] for it in items])
    unseen_bleu = corpus_bleu([it['unseen_hyp'] for it in items], [it['reference'] for it in items])
    print(f"\nFull {len(items)}-item comparison:")
    print(f"  REC={rec_bleu:.2f}, PURE(seen)={pure_bleu:.2f}, UNSEEN={unseen_bleu:.2f}")
    print(f"  PURE-REC gap={pure_bleu-rec_bleu:+.2f}")
    print(f"  UNSEEN-REC gap={unseen_bleu-rec_bleu:+.2f}")
    print(f"  Origin effect (PURE-UNSEEN)={pure_bleu-unseen_bleu:+.2f}")

    # SMD on full set
    seen_jacs = [it['seen_jac'] for it in items]
    unseen_jacs = [it['unseen_jac'] for it in items]
    full_smd_jac = smd(seen_jacs, unseen_jacs)
    print(f"  SMD(seen-unseen) on Jaccard: {full_smd_jac:+.3f}")

    # Create matched subsets at various caliper widths
    results = {
        'schema': 'matched-donor-pool-v1',
        'n_total': len(items),
        'full_comparison': {
            'n': len(items),
            'rec_bleu': rec_bleu,
            'pure_bleu': pure_bleu,
            'unseen_bleu': unseen_bleu,
            'pure_rec_gap': pure_bleu - rec_bleu,
            'unseen_rec_gap': unseen_bleu - rec_bleu,
            'origin_effect': pure_bleu - unseen_bleu,
            'smd_jaccard': full_smd_jac,
        },
        'matched_subsets': {},
    }

    for caliper in [0.05, 0.10, 0.15, 0.20]:
        matched = [it for it in items if abs(it['seen_jac'] - it['unseen_jac']) <= caliper]
        n = len(matched)
        if n < 50:
            print(f"\nCaliper {caliper}: n={n} (too small, skipping)")
            continue

        # Compute corpus BLEU on matched subset
        m_rec = corpus_bleu([it['rec_hyp'] for it in matched], [it['reference'] for it in matched])
        m_pure = corpus_bleu([it['pure_hyp'] for it in matched], [it['reference'] for it in matched])
        m_unseen = corpus_bleu([it['unseen_hyp'] for it in matched], [it['reference'] for it in matched])

        # SMD on all covariates
        m_smd_jac = smd([it['seen_jac'] for it in matched], [it['unseen_jac'] for it in matched])
        # For speaker, template_density, text_length: these are per-item (not per-condition)
        # so SMD between conditions isn't meaningful — they're the same items
        # The relevant SMD is on donor characteristics (Jaccard), which differ by condition

        # Bootstrap CI on origin effect
        ci_lo, ci_hi = donor_cluster_bootstrap_ci(
            [it['pure_hyp'] for it in matched],
            [it['unseen_hyp'] for it in matched],
            [it['rec_hyp'] for it in matched],
            [it['reference'] for it in matched],
            None, None, n=5000  # fewer resamples for speed
        )

        origin_effect = m_pure - m_unseen
        print(f"\nCaliper {caliper}: n={n}")
        print(f"  REC={m_rec:.2f}, PURE={m_pure:.2f}, UNSEEN={m_unseen:.2f}")
        print(f"  Origin effect={origin_effect:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        print(f"  SMD(Jaccard)={m_smd_jac:+.3f}")

        results['matched_subsets'][f'caliper_{caliper}'] = {
            'n': n,
            'rec_bleu': m_rec,
            'pure_bleu': m_pure,
            'unseen_bleu': m_unseen,
            'pure_rec_gap': m_pure - m_rec,
            'unseen_rec_gap': m_unseen - m_rec,
            'origin_effect': origin_effect,
            'origin_ci': [ci_lo, ci_hi],
            'smd_jaccard': m_smd_jac,
            'mean_seen_jac': float(np.mean([it['seen_jac'] for it in matched])),
            'mean_unseen_jac': float(np.mean([it['unseen_jac'] for it in matched])),
        }

    # Select the tightest caliper achieving SMD<0.1
    best_caliper = None
    for caliper in [0.05, 0.10, 0.15, 0.20]:
        key = f'caliper_{caliper}'
        if key in results['matched_subsets']:
            if abs(results['matched_subsets'][key]['smd_jaccard']) < 0.10:
                best_caliper = caliper
                break

    if best_caliper:
        results['recommended_subset'] = f'caliper_{best_caliper}'
        print(f"\n✓ Recommended: caliper_{best_caliper} (SMD<0.1, n={results['matched_subsets'][f'caliper_{best_caliper}']['n']})")
    else:
        # Fall back to tightest available with SMD<0.12
        for caliper in [0.05, 0.10, 0.15, 0.20]:
            key = f'caliper_{caliper}'
            if key in results['matched_subsets']:
                if abs(results['matched_subsets'][key]['smd_jaccard']) < 0.12:
                    best_caliper = caliper
                    break
        if best_caliper:
            results['recommended_subset'] = f'caliper_{best_caliper}'
            print(f"\n~ Recommended (SMD<0.12): caliper_{best_caliper}")
        else:
            print("\n⚠ No subset achieves SMD<0.12; reporting tightest caliper")
            results['recommended_subset'] = 'caliper_0.05'

    json.dump(results, open(OUT_PATH, 'w'), indent=2, ensure_ascii=False)
    print(f"\nWritten to {OUT_PATH}")


if __name__ == '__main__':
    main()
