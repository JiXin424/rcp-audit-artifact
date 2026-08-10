#!/usr/bin/env python3
"""Build unified checkpoint registry v2.

DEPRECATED (2026-08-06): superseded by results/canonical_checkpoint_registry.json
and results/gap_43_canonical_beam3.json as the single source of truth for paper
numbers. The hardcoded values below predate the canonical donor registry
(SHA-256 9170a530...) and the canonical beam-3 protocol: they contain
pre-exclusion released values (PURE 23.79 / gap +11.01 instead of 23.02 / +10.24)
and legacy training-time dev BLEU-4 values (e.g. seed 404 GT 7.19 vs canonical
9.52; distill dev_bleu4_max 10.51--10.81, of which the alpha=1.0 value is not
reproducible from current checkpoints). Do NOT use this file's numbers in the
paper; regenerate from canonical sources if needed. Kept for provenance only.

Legacy docstring (historical): "Single source of truth for ALL checkpoints in
the paper. All counts in the paper (37 dose-response, 50 gate-eligible, 14
prediction interval, 63 dev-metric, 66 trained, 13 distillation) MUST be
auto-generated from this registry."

Data sources:
- results/checkpoint_registry.json (50 non-distill + released)
- results/e11b_ladder_gaps.json (ladder + config-faithful dev/gap)
- data/cells/cp*.json (per-checkpoint BLEU)
- Paper Tables 4, 11, 12, 13, 14, 18 for gap/dev per family
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter

ARTI = Path(__file__).resolve().parents[1]

# ===================== Original (released) =====================
released = {
    'id': 'released_original',
    'family': 'released',
    'kind': 'released_checkpoint',
    'seed': None,
    'dev_bleu4_official': 13.378651856913775,
    'dev_wer_official': 83.37019018133569,
    'phx_public_GT': 12.78,
    'phx_public_PURE': 23.79,
    'phx_public_PURE_GT_gap': 11.01,
    'phx_public_COMP': 18.86,
    'phx_public_COMP_GT_gap': 6.09,
    'train_pool_readout_bleu_full_7060': 78.40,
    'train_pool_readout_em_full_7060': 0.694,
    'train_pool_readout_bleu_1k_prefix': 79.41,
    'train_pool_readout_em_1k_prefix': 0.698,
    'has_gap': True,
    'has_dev': True,
    'enters_dose_response': False,  # released is the target, not in family cloud
    'enters_gate_eval': False,      # not a trained checkpoint
    'enters_prediction_interval': False,
    'note': 'released SLRTP2025 evaluator; best.ckpt = step 1820 of 2828-step run per bundle validations.txt',
}

# ===================== Reconstructions primary (6) =====================
# (seed, dev_bleu4, dev_wer, GT, PURE, gap, COMP, comp_gap)
reco_primary_data = [
    (101, 8.029, 90.683, 8.57, 8.65, +0.08, 7.92, -0.65),
    (202, 7.731, 88.869, 8.81, 7.83, -0.98, 7.54, -1.28),
    (303, 8.920, 92.142, 9.67, 8.96, -0.71, 8.71, -0.96),
    (404, 6.685, None,  7.19, 6.83, -0.36, 6.38, -0.81),
    (505, 8.961, None,  9.05, 8.16, -0.89, 7.97, -1.08),
    (606, 8.142, None,  8.09, 8.22, +0.13, 7.93, -0.16),
]

# ===================== Reconstructions extension (8) =====================
# (seed, dev_bleu4, dev_wer (from registry), GT, PURE, gap, COMP, comp_gap, WER_table_11)
reco_ext_data = [
    (707,  6.687, 92.997, 7.71, 7.55, -0.16, 7.21, -0.50),
    (808,  7.300, 94.825, 7.71, 6.84, -0.87, 6.90, -0.81),
    (909,  8.965, 88.206, 9.56, 9.09, -0.47, 8.83, -0.73),
    (1001, 9.071, 88.471, 8.63, 8.36, -0.27, 8.19, -0.44),
    (1102, 7.610, None,   8.88, 7.78, -1.10, 7.85, -1.03),
    (1203, 8.140, None,   9.12, 8.53, -0.59, 8.53, -0.59),
    (1304, 8.180, None,   9.41, 8.62, -0.79, 8.58, -0.83),
    (1405, 8.120, None,   8.90, 8.17, -0.73, 8.37, -0.53),
]

# ===================== Rescue lr (8) =====================
# dev BLEU range 7.9-9.7 (from paper); individual values not in artifact
# We record family-level info; per-seed not in main paper tables
rescue_lr_family = {
    'family': 'rescue_lr',
    'family_size': 8,
    'dev_bleu4_range': [7.9, 9.7],
    'has_gap': False, 'has_dev': True,
    'enters_dose_response': False, 'enters_gate_eval': True,
    'note': 'lr 5e-4 / 2e-3, two seeds each; per-seed values in original training logs',
}

# ===================== Rescue expanded (12) =====================
# 6 hyperparameter families × 2 seeds each
# wd0 s202 has gap +0.25 (from line 297), PURE 10.98, GT 10.73
rescue_expanded_summary = [
    {'variant': 'batch_128_512',     'n': 2, 'dev_bleu4_range': [6.8, 9.2], 'has_gap': False},
    {'variant': 'dropout_0.2',       'n': 2, 'dev_bleu4_range': [6.9, 7.5], 'has_gap': False},
    {'variant': 'weight_decay_0',    'n': 2, 'dev_bleu4_range': [9.8, 9.9], 'has_gap_partial': True,
     'note': 'wd0 s202 dev 9.92, gap +0.25 (PURE 10.98 vs GT 10.73); other seed has no PHX-public gap'},
    {'variant': 'label_smoothing_0.1', 'n': 2, 'dev_bleu4_range': [6.8, 7.4], 'has_gap': False},
    {'variant': 'long_600ep_lr5e-4', 'n': 2, 'dev_bleu4_range': [8.2, 8.3], 'has_gap': False},
    # Note: 12 = 5 families above × 2 + 2 from prior rescue2 (config-faithful continuation)
]

# ===================== Ladder (4) =====================
# (frac, dev_bleu4, dev_wer, gap)
ladder_data = [
    (0.125, 4.51, 89.56, +0.05),
    (0.25,  4.56, 89.87, -0.20),
    (0.5,   5.60, 89.56, +0.24),
    (0.75,  5.87, 86.57, +0.19),
]

# ===================== Config-faithful (4) =====================
# (seed, dev_bleu4, dev_wer, gap, pass_through, seen_holdout, ep)
cfaithful_data = [
    (101, 8.91, 86.45, -1.02, 1.24, 7.60, 98),
    (202, 8.59, 87.96, -1.06, 1.22, None, 42),
    (303, 8.67, 88.77, -0.23, 1.21, None, 210),
    (404, 9.51, 86.57, -0.37, 1.24, 7.43, 140),
]

# ===================== Big-arch (4) =====================
big_arch_family = {
    'family': 'big_arch',
    'family_size': 4,
    'dev_bleu4_range': [6.8, 8.0],
    'has_gap': False, 'has_dev': True,
    'enters_dose_response': False, 'enters_gate_eval': True,
    'note': '4-6 layers, 384-512 hidden; overfit fast, dev BLEU 6.8-8.0',
}

# ===================== Step-faithful (2) =====================
# (seed, dev_bleu4, gap, pass_through, seen_holdout, stopped_step)
step_faithful_data = [
    (101, 7.03, +0.02, 1.11, 5.71, 4200),
    (202, 7.47, -0.30, 1.17, 6.59, 4662),
]

# ===================== Confirmation (2) =====================
# (seed, dev_bleu4, gap, pass_through, seen_holdout)
confirmation_data = [
    (1506, 8.1, -1.03, 1.22, 6.8),
    (1607, 7.6, -0.98, 1.29, 7.0),
]

# ===================== Long-schedule (2) =====================
# Long-schedule continuation pushes family max to 10.02; gap -0.45
long_schedule_family = {
    'family': 'long_schedule',
    'family_size': 2,
    'dev_bleu4_max': 10.02,
    'dev_bleu4_range': 'one reaches family-max 10.02',
    'has_gap': True,
    'gap_observed': -0.45,
    'note': '800 epochs, 2 seeds; family-max dev BLEU 10.02; gap -0.45; enters dose-response via long-schedule point',
    'enters_dose_response': True,
    'enters_gate_eval': True,
}

# ===================== BT-retrained holdout (1) =====================
bt_retrained = {
    'id': 'bt_retrained_holdout',
    'family': 'bt_retrained_holdout',
    'kind': 'trained_checkpoint',
    'has_gap': False, 'has_dev': False,
    'enters_dose_response': False, 'enters_gate_eval': False,
    'note': 'BT retrained on unseen holdout; only enters template-holdout series (Table 19)',
}

# ===================== Crossfit (2) =====================
crossfit_family = {
    'family': 'crossfit',
    'family_size': 2,
    'has_gap': False, 'has_dev': False,
    'enters_dose_response': False, 'enters_gate_eval': False,
    'note': 'A/B fold cross-fit; only enters holdout series',
}

# ===================== Distillation (13 of 15 planned; 2 OOM) =====================
# Per-alpha aggregate from Table 14
distill_alpha_summary = [
    {'alpha': 0.0,  'n': 3, 'dev_bleu4_max': 10.51, 'gap_range': [-5.3, -4.2],
     'seeds_completed': [101, 202, 303]},
    {'alpha': 0.25, 'n': 3, 'dev_bleu4_max': 10.75, 'gap_range': [-4.4, -3.5],
     'seeds_completed': [101, 202, 303]},
    {'alpha': 0.5,  'n': 2, 'dev_bleu4_max': 10.41, 'gap_range': [-5.4, -5.1],
     'seeds_completed': [101, 202], 'seeds_oom': [303]},
    {'alpha': 0.75, 'n': 2, 'dev_bleu4_max': 10.93, 'gap_range': [-5.2, -5.0],
     'seeds_completed': [101, 202], 'seeds_oom': [303]},
    {'alpha': 1.0,  'n': 3, 'dev_bleu4_max': 10.81, 'gap_range': [-5.3, -5.2],
     'seeds_completed': [101, 202, 303]},
]

# ===================== Build unified registry =====================
checkpoints = []

# Released
checkpoints.append(released)

# Reconstructions primary
for seed, dev_bleu, dev_wer, GT, PURE, gap, COMP, cg in reco_primary_data:
    checkpoints.append({
        'id': f'reco_primary_seed_{seed}',
        'family': 'reconstructions_primary',
        'kind': 'trained_checkpoint',
        'seed': seed,
        'dev_bleu4_official': dev_bleu,
        'dev_wer_official': dev_wer,
        'phx_public_GT': GT, 'phx_public_PURE': PURE,
        'phx_public_PURE_GT_gap': gap,
        'phx_public_COMP': COMP, 'phx_public_COMP_GT_gap': cg,
        'has_gap': True, 'has_dev': True,
        'enters_dose_response': True,
        'enters_gate_eval': True,
        'enters_prediction_interval': True,
    })

# Reconstructions extension
for seed, dev_bleu, dev_wer, GT, PURE, gap, COMP, cg in reco_ext_data:
    checkpoints.append({
        'id': f'reco_extension_seed_{seed}',
        'family': 'reconstructions_extension',
        'kind': 'trained_checkpoint',
        'seed': seed,
        'dev_bleu4_official': dev_bleu,
        'dev_wer_official': dev_wer,
        'phx_public_GT': GT, 'phx_public_PURE': PURE,
        'phx_public_PURE_GT_gap': gap,
        'phx_public_COMP': COMP, 'phx_public_COMP_GT_gap': cg,
        'has_gap': True, 'has_dev': True,
        'enters_dose_response': True,
        'enters_gate_eval': True,
        'enters_prediction_interval': True,  # extension seeds enter PI per paper
    })

# Ladder
for frac, dev_bleu, dev_wer, gap in ladder_data:
    checkpoints.append({
        'id': f'ladder_frac_{frac}',
        'family': 'ladder',
        'kind': 'trained_checkpoint',
        'frac': frac,
        'dev_bleu4_official': dev_bleu,
        'dev_wer_official': dev_wer,
        'phx_public_PURE_GT_gap': gap,
        'has_gap': True, 'has_dev': True,
        'enters_dose_response': True,
        'enters_gate_eval': True,
    })

# Config-faithful
for seed, dev_bleu, dev_wer, gap, pt, sh, ep in cfaithful_data:
    checkpoints.append({
        'id': f'config_faithful_seed_{seed}',
        'family': 'config_faithful',
        'kind': 'trained_checkpoint',
        'seed': seed,
        'dev_bleu4_official': dev_bleu,
        'dev_wer_official': dev_wer,
        'phx_public_PURE_GT_gap': gap,
        'pass_through_ratio': pt,
        'seen_holdout_bleu': sh,
        'epoch_selected': ep,
        'has_gap': True, 'has_dev': True,
        'enters_dose_response': True,
        'enters_gate_eval': True,
    })

# Step-faithful
for seed, dev_bleu, gap, pt, sh, stopped in step_faithful_data:
    checkpoints.append({
        'id': f'step_faithful_seed_{seed}',
        'family': 'step_faithful',
        'kind': 'trained_checkpoint',
        'seed': seed,
        'dev_bleu4_official': dev_bleu,
        'phx_public_PURE_GT_gap': gap,
        'pass_through_ratio': pt,
        'seen_holdout_bleu': sh,
        'stopped_step': stopped,
        'has_gap': True, 'has_dev': True,
        'enters_dose_response': True,
        'enters_gate_eval': True,
    })

# Confirmation
for seed, dev_bleu, gap, pt, sh in confirmation_data:
    checkpoints.append({
        'id': f'confirmation_seed_{seed}',
        'family': 'confirmation',
        'kind': 'trained_checkpoint',
        'seed': seed,
        'dev_bleu4_official': dev_bleu,
        'phx_public_PURE_GT_gap': gap,
        'pass_through_ratio': pt,
        'seen_holdout_bleu': sh,
        'has_gap': True, 'has_dev': True,
        'enters_dose_response': True,
        'enters_gate_eval': True,
    })

# Long-schedule (2 seeds, one reaches 10.02)
for i, seed in enumerate([101, 202]):
    checkpoints.append({
        'id': f'long_schedule_seed_{seed}',
        'family': 'long_schedule',
        'kind': 'trained_checkpoint',
        'seed': seed,
        'dev_bleu4_official': 10.02 if i == 0 else None,
        'phx_public_PURE_GT_gap': -0.45 if i == 0 else None,
        'has_gap': i == 0, 'has_dev': True,
        'enters_dose_response': i == 0,  # only the 10.02 point enters
        'enters_gate_eval': True,
    })

# Rescue lr (8) — family-level only, per-seed not in main paper
for i in range(8):
    checkpoints.append({
        'id': f'rescue_lr_idx_{i}',
        'family': 'rescue_lr',
        'kind': 'trained_checkpoint',
        'has_gap': False, 'has_dev': True,
        'enters_dose_response': False,
        'enters_gate_eval': True,
        'note': f'lr rescue; per-seed values in artifact training logs',
    })

# Rescue expanded (12 total: 5 variant families × 2 seeds + 2 from prior)
# wd0 s202 is the only one with PHX-public gap
rescue_variants = [
    ('batch_128', 2), ('batch_512', 2),
    ('dropout_0.2', 2),
    ('weight_decay_0', 2),
    ('label_smoothing_0.1', 2),
    ('long_600ep_lr5e-4', 2),
]
for variant, n in rescue_variants:
    for i in range(n):
        is_wd0_s202 = (variant == 'weight_decay_0' and i == 1)
        checkpoints.append({
            'id': f'rescue2_{variant}_seed_{i}',
            'family': 'rescue2',
            'kind': 'trained_checkpoint',
            'variant': variant,
            'has_gap': is_wd0_s202,
            'phx_public_PURE_GT_gap': +0.25 if is_wd0_s202 else None,
            'phx_public_GT': 10.73 if is_wd0_s202 else None,
            'phx_public_PURE': 10.98 if is_wd0_s202 else None,
            'dev_bleu4_official': 9.92 if is_wd0_s202 else None,
            'has_dev': True,
            'enters_dose_response': is_wd0_s202,
            'enters_gate_eval': True,
            'note': 'best rescue; wd0 s202' if is_wd0_s202 else variant,
        })

# Big-arch (4)
for i in range(4):
    checkpoints.append({
        'id': f'big_arch_idx_{i}',
        'family': 'big_arch',
        'kind': 'trained_checkpoint',
        'has_gap': False, 'has_dev': True,
        'enters_dose_response': False,
        'enters_gate_eval': True,
    })

# BT-retrained holdout (1)
checkpoints.append(bt_retrained)

# Crossfit (2)
for i in range(2):
    checkpoints.append({
        'id': f'crossfit_{chr(65+i)}',  # A, B
        'family': 'crossfit',
        'kind': 'trained_checkpoint',
        'has_gap': False, 'has_dev': False,
        'enters_dose_response': False,
        'enters_gate_eval': False,
    })

# Distillation (13 of 15)
for alpha_info in distill_alpha_summary:
    alpha = alpha_info['alpha']
    for seed in alpha_info['seeds_completed']:
        checkpoints.append({
            'id': f'distill_alpha_{alpha}_seed_{seed}',
            'family': 'distillation',
            'kind': 'trained_checkpoint',
            'alpha': alpha,
            'seed': seed,
            'has_gap': True, 'has_dev': True,
            'enters_dose_response': True,
            'enters_gate_eval': False,  # distillation uses different objective, excluded from gate
            'note': f'per-alpha gap range: {alpha_info["gap_range"]}, dev max: {alpha_info["dev_bleu4_max"]}',
        })
    # Add OOM placeholders for the 2 missing
    for seed in alpha_info.get('seeds_oom', []):
        checkpoints.append({
            'id': f'distill_alpha_{alpha}_seed_{seed}_OOM',
            'family': 'distillation',
            'kind': 'failed_checkpoint',
            'alpha': alpha,
            'seed': seed,
            'has_gap': False, 'has_dev': False,
            'enters_dose_response': False,
            'enters_gate_eval': False,
            'note': 'CUDA OOM on first batch; pending rerun with grad_accum=4',
        })

# ===================== Compute summary counts =====================
total_entries = len(checkpoints)
released_entries = sum(1 for c in checkpoints if c.get('family') == 'released')
trained_entries = total_entries - released_entries
planned_entries = trained_entries + 2  # +2 distillation OOM (already counted as failed)
failed_entries = sum(1 for c in checkpoints if c.get('kind') == 'failed_checkpoint')
completed_trained = trained_entries - failed_entries

has_gap = sum(1 for c in checkpoints if c.get('has_gap') and c.get('family') != 'released')
has_dev = sum(1 for c in checkpoints if c.get('has_dev') and c.get('family') != 'released')
in_dose = sum(1 for c in checkpoints if c.get('enters_dose_response') and c.get('family') != 'released')
in_gate = sum(1 for c in checkpoints if c.get('enters_gate_eval') and c.get('family') != 'released')
in_pi = sum(1 for c in checkpoints if c.get('enters_prediction_interval') and c.get('family') != 'released')

# Family counts
fam_counts = Counter(c.get('family') for c in checkpoints)

print('=' * 60)
print('UNIFIED CHECKPOINT REGISTRY v2')
print('=' * 60)
print(f'Total entries: {total_entries}')
print(f'  Released: {released_entries}')
print(f'  Trained (completed): {completed_trained}')
print(f'  Trained (failed/OOM): {failed_entries}')
print(f'  Planned total: {planned_entries}')
print()
print(f'Counts (excluding released):')
print(f'  has_gap:        {has_gap}')
print(f'  has_dev:        {has_dev}')
print(f'  enters_dose_response: {in_dose}')
print(f'  enters_gate_eval:     {in_gate}')
print(f'  enters_prediction_interval: {in_pi}')
print()
print('Per family:')
for fam, n in sorted(fam_counts.items()):
    print(f'  {fam}: {n}')

print()
print('Paper-claim verification:')
print(f'  Paper says "37 in dose-response": actual = {in_dose}  ', '✓' if in_dose == 37 else '✗')
print(f'  Paper says "50 gate-eligible (non-distill)": gate = {in_gate}, distill in gate = {sum(1 for c in checkpoints if c.get("family")=="distillation" and c.get("enters_gate_eval"))}')
print(f'  Paper says "13 distillation completed": actual completed = {sum(1 for c in checkpoints if c.get("family")=="distillation" and c.get("kind")=="trained_checkpoint")}')
print(f'  Paper says "2 distillation OOM": actual OOM = {sum(1 for c in checkpoints if c.get("family")=="distillation" and c.get("kind")=="failed_checkpoint")}')
print(f'  Paper says "63 with dev metrics": actual = {has_dev}')
print(f'  Paper says "14 prediction interval": actual = {in_pi}')

# Write
out = {
    'schema': 'unified-checkpoint-registry-v2',
    'generated_by': 'build_unified_checkpoint_registry.py',
    'summary': {
        'total_entries': total_entries,
        'released': released_entries,
        'trained_completed': completed_trained,
        'trained_failed': failed_entries,
        'planned_total': planned_entries,
        'has_gap': has_gap,
        'has_dev': has_dev,
        'enters_dose_response': in_dose,
        'enters_gate_eval': in_gate,
        'enters_prediction_interval': in_pi,
        'family_counts': dict(fam_counts),
    },
    'released': released,
    'checkpoints': checkpoints,
}
outpath = ARTI / 'results' / 'unified_checkpoint_registry.json'
outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f'\nWritten: {outpath}')
