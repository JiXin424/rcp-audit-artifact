#!/usr/bin/env python3
"""Rebuild unified checkpoint registry with REAL training results and decoded gaps.

Reads:
  - checkpoints/reconstructions/seed_*/training_log.json (dev_nll, best_epoch)
  - checkpoints/distillation/alpha_*/training_log.json
  - results/cells/cp*_GT-v1.json (released BLEU on GT)
  - results/cells/cp*_TN-PURE-v1.json (released BLEU on PURE)
  - results/cells/cp*_PT-v1.json (released BLEU on PT)

Writes:
  - results/unified_checkpoint_registry.json (updated with real numbers)
  - results/dose_response.json (data for Fig. 3)
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[1]

# Map evaluator id → cp index
EVAL_TO_CP = {
    'original': 0,
    'reconstruction_101': 1, 'reconstruction_202': 2, 'reconstruction_303': 3,
    'reconstruction_404': 4, 'reconstruction_505': 5, 'reconstruction_606': 6,
    'reconstruction_707': 7, 'reconstruction_808': 8, 'reconstruction_909': 9,
    'reconstruction_1001': 10, 'reconstruction_1102': 11, 'reconstruction_1203': 12,
    'reconstruction_1304': 13, 'reconstruction_1405': 14,
    # Distillation students
    'distill_0.0_101': 15, 'distill_0.0_202': 16, 'distill_0.0_303': 17,
    'distill_0.25_101': 18, 'distill_0.25_202': 19, 'distill_0.25_303': 20,
    'distill_0.5_101': 21, 'distill_0.5_202': 22, 'distill_0.5_303': 23,
    'distill_0.75_101': 24, 'distill_0.75_202': 25, 'distill_0.75_303': 26,
    'distill_1.0_101': 27, 'distill_1.0_202': 28, 'distill_1.0_303': 29,
}


def load_training_log(path: Path) -> dict | None:
    try:
        d = json.load(open(path))
        return {
            'dev_nll': d['best']['dev_nll'],
            'best_epoch': d['best']['epoch'],
            'n_epochs_run': len(d.get('epochs_log', [])),
        }
    except Exception:
        return None


def load_cell(cp_idx: int, system: str) -> dict | None:
    p = ROOT / f'results/cells/cp{cp_idx}_{system}.json'
    if not p.exists():
        return None
    return json.load(open(p))


# Build per-checkpoint data
checkpoints = []

# Released
released_gt = load_cell(0, 'GT-v1')
released_pure = load_cell(0, 'TN-PURE-v1')
released_pt = load_cell(0, 'PT-v1')
checkpoints.append({
    'id': 'released_original',
    'family': 'released',
    'dev_nll': released_gt['metrics']['teacher_forced_nll_per_token'] if released_gt else None,
    'phx_public_GT_bleu': released_gt['metrics']['decoded_bleu'] if released_gt else None,
    'phx_public_PURE_bleu': released_pure['metrics']['decoded_bleu'] if released_pure else None,
    'phx_public_PT_bleu': released_pt['metrics']['decoded_bleu'] if released_pt else None,
    'phx_public_PURE_GT_gap': (released_pure['metrics']['decoded_bleu'] - released_gt['metrics']['decoded_bleu']) if (released_pure and released_gt) else None,
    'note': 'Released SLRTP2025 BT evaluator (audit target)',
})

# 14 reconstructions
for seed in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304, 1405]:
    cp_idx = EVAL_TO_CP[f'reconstruction_{seed}']
    log = load_training_log(ROOT / f'checkpoints/reconstructions/seed_{seed}/training_log.json')
    gt = load_cell(cp_idx, 'GT-v1')
    pure = load_cell(cp_idx, 'TN-PURE-v1')
    pt = load_cell(cp_idx, 'PT-v1')
    gap = (pure['metrics']['decoded_bleu'] - gt['metrics']['decoded_bleu']) if (pure and gt) else None
    family = 'reconstructions_primary' if seed <= 606 else 'reconstructions_extension'
    checkpoints.append({
        'id': f'reco_seed_{seed}',
        'family': family,
        'seed': seed,
        'dev_nll': log['dev_nll'] if log else None,
        'best_epoch': log['best_epoch'] if log else None,
        'n_epochs_run': log['n_epochs_run'] if log else None,
        'phx_public_GT_bleu': gt['metrics']['decoded_bleu'] if gt else None,
        'phx_public_PURE_bleu': pure['metrics']['decoded_bleu'] if pure else None,
        'phx_public_PT_bleu': pt['metrics']['decoded_bleu'] if pt else None,
        'phx_public_PURE_GT_gap': gap,
    })

# Distillation students (with decoded cells now)
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    for seed in [101, 202, 303]:
        cp_idx = EVAL_TO_CP[f'distill_{alpha}_{seed}']
        log = load_training_log(ROOT / f'checkpoints/distillation/alpha_{alpha}_seed_{seed}/training_log.json')
        gt = load_cell(cp_idx, 'GT-v1')
        pure = load_cell(cp_idx, 'TN-PURE-v1')
        gap = (pure['metrics']['decoded_bleu'] - gt['metrics']['decoded_bleu']) if (pure and gt) else None
        checkpoints.append({
            'id': f'distill_alpha_{alpha}_seed_{seed}',
            'family': 'distillation',
            'alpha': alpha,
            'seed': seed,
            'dev_nll': log['dev_nll'] if log else None,
            'best_epoch': log['best_epoch'] if log else None,
            'n_epochs_run': log['n_epochs_run'] if log else None,
            'phx_public_GT_bleu': gt['metrics']['decoded_bleu'] if gt else None,
            'phx_public_PURE_bleu': pure['metrics']['decoded_bleu'] if pure else None,
            'phx_public_PT_bleu': None,
            'phx_public_PURE_GT_gap': gap,
        })

# Summary
trained = [c for c in checkpoints if c['family'] != 'released']
has_gap = [c for c in trained if c.get('phx_public_PURE_GT_gap') is not None]
has_dev = [c for c in trained if c.get('dev_nll') is not None]

print(f'Total checkpoints: {len(checkpoints)} (1 released + {len(trained)} trained)')
print(f'  has_dev: {len(has_dev)}')
print(f'  has_gap (PHX-public PURE-GT): {len(has_gap)}')
print(f'  has_both (dev_nll + gap): {sum(1 for c in has_gap if c.get("dev_nll") is not None)}')

# Compute summary stats
if has_gap:
    gaps = [c['phx_public_PURE_GT_gap'] for c in has_gap]
    print(f'\nGap range: [{min(gaps):.2f}, {max(gaps):.2f}]')
    print(f'Positive gaps: {sum(1 for g in gaps if g > 0)}/{len(gaps)}')

# Write registry
registry = {
    'schema': 'unified-checkpoint-registry-v3-with-real-data',
    'generated_by': 'scripts/build_dose_response_data.py',
    'summary': {
        'total': len(checkpoints),
        'released': 1,
        'trained': len(trained),
        'has_dev': len(has_dev),
        'has_gap': len(has_gap),
        'has_both_dev_and_gap': sum(1 for c in has_gap if c.get("dev_nll") is not None),
    },
    'checkpoints': checkpoints,
}
out = ROOT / 'results/unified_checkpoint_registry.json'
out.write_text(json.dumps(registry, indent=2))
print(f'\nWrote: {out}')

# Build dose-response data (for Fig 3)
# x = dev_nll (proxy for competence; lower NLL = higher competence)
# y = PURE-GT gap
# Both dev_nll AND gap must be present
dose_data = []
for c in checkpoints:
    if c['family'] == 'released':
        continue
    if c.get('dev_nll') is None or c.get('phx_public_PURE_GT_gap') is None:
        # Skip but mark
        continue
    dose_data.append({
        'id': c['id'],
        'family': c['family'],
        'dev_nll': c['dev_nll'],
        'gap': c['phx_public_PURE_GT_gap'],
        'GT_bleu': c.get('phx_public_GT_bleu'),
        'PURE_bleu': c.get('phx_public_PURE_bleu'),
    })

# Add released as the star point
released_cp = checkpoints[0]
dose_data.append({
    'id': 'released_original',
    'family': 'released',
    'dev_nll': released_cp['dev_nll'],
    'gap': released_cp['phx_public_PURE_GT_gap'],
    'GT_bleu': released_cp.get('phx_public_GT_bleu'),
    'PURE_bleu': released_cp.get('phx_public_PURE_bleu'),
})

out2 = ROOT / 'results/dose_response.json'
out2.write_text(json.dumps({
    'n_points': len(dose_data),
    'n_with_both': sum(1 for d in dose_data if d['family'] != 'released'),
    'points': dose_data,
}, indent=2))
print(f'Wrote: {out2} ({len(dose_data)} points, {sum(1 for d in dose_data if d["family"] != "released")} with both dev_nll + gap)')
