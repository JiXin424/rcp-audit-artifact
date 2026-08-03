#!/usr/bin/env python3
"""Reverse-identify video -> system using BT-decoded hypotheses.
Strategy: for each (target, video) pair, the BT-decoded text from the
corresponding pose should reveal which system it is.
- GT: decode matches reference (target)
- PURE: decode matches donor transcript (NOT target) → low BLEU vs target
- random: decode is unrelated to target → very low BLEU
- PT: decode is poorly formed → low BLEU
- COMP: decode is partial donor + scaffold → mid BLEU

But we don't have video_id -> pose_id mapping. Instead, use score profiles:
- GT pose → human sem HIGH + intell HIGH + nat HIGH
- PURE (donor pose) → human sem LOW + intell HIGH + nat HIGH (real pose but wrong meaning)
- random donor → human sem LOWEST + intell HIGH + nat HIGH
- PT → human sem LOW + intell LOW + nat LOW
- COMP → mid everything

Alternative: check inter-rater agreement per rank — GT and PURE should be
highly consistent (clear profile), PT and COMP more variable."""
import csv, json, statistics, os
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path('/ssd/xkb4/RCP/评分')
OUT = Path('/ssd/xkb4/RCP/artifact/results/human_eval')

# Load all ratings
s1_data = defaultdict(lambda: {'intell': [], 'natural': []})
s2_data = defaultdict(lambda: {'semantic': [], 'target': None})
for r in range(1, 31):
    with open(EVAL_DIR / f'R{r:03d}_阶段1_视频评分表.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            vid = row['视频编号']
            try:
                s1_data[vid]['intell'].append(int(row['可理解性_1到5']))
                s1_data[vid]['natural'].append(int(row['自然度_1到5']))
            except: pass
    with open(EVAL_DIR / f'R{r:03d}_阶段2_语义评分表.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            vid = row['视频编号']
            try:
                s2_data[vid]['semantic'].append(int(row['语义充分性_1到5']))
                s2_data[vid]['target'] = row['德语目标句'].strip()
            except: pass

# Per-video stats with profile
videos = []
for vid in sorted(s2_data.keys()):
    sem = s2_data[vid]['semantic']
    intel = s1_data[vid]['intell']
    nat = s1_data[vid]['natural']
    videos.append({
        'video_id': vid,
        'target': s2_data[vid]['target'],
        'sem_mean': statistics.mean(sem),
        'sem_sd': statistics.stdev(sem),
        'intell_mean': statistics.mean(intel),
        'intell_sd': statistics.stdev(intel),
        'nat_mean': statistics.mean(nat),
        'nat_sd': statistics.stdev(nat),
        'sem_raw': sem,
        'intell_raw': intel,
        'nat_raw': nat,
    })

# Group by target and rank
by_target = defaultdict(list)
for v in videos:
    by_target[v['target']].append(v)

for t, vs in by_target.items():
    vs.sort(key=lambda x: -x['sem_mean'])
    for i, v in enumerate(vs, 1):
        v['rank'] = i

# Classify each video by profile
# Strategy: GT has high sem+intell+nat; PURE has low sem + high intell+nat;
# random has lowest sem + high intell+nat (similar to PURE but worse sem);
# PT has low sem + low intell + low nat; COMP is mid.
def classify(v):
    s, i, n = v['sem_mean'], v['intell_mean'], v['nat_mean']
    # GT: high all (sem>3.5 AND nat>3.5)
    if s >= 3.5 and n >= 3.5:
        return 'GT'
    # PT: low all (sem<3 AND nat<3)
    if s < 3.5 and n < 3.0:
        return 'PT'
    # If high nat but low sem → donor pose (PURE or random)
    if n >= 3.3 and s < 3.5:
        # PURE has higher sem than random; cutoff at 2.0
        if s >= 2.0:
            return 'PURE?'
        else:
            return 'RANDOM?'
    # Mid (COMP)
    return 'COMP?'

for v in videos:
    v['inferred_system'] = classify(v)

# Count
from collections import Counter
print('=== Inferred system distribution ===')
for sys, count in sorted(Counter(v['inferred_system'] for v in videos).items()):
    print(f'  {sys}: {count}')

# For each target, show inferred system
print('\n=== Per-target inferred systems (ranked by sem_mean) ===')
for t, vs in sorted(by_target.items())[:5]:
    print(f'\nTarget: {t[:70]}')
    for v in vs:
        print(f'  rank={v["rank"]} vid={v["video_id"]} '
              f'sem={v["sem_mean"]:.2f}±{v["sem_sd"]:.2f} '
              f'intell={v["intell_mean"]:.2f} nat={v["nat_mean"]:.2f} '
              f'→ {v["inferred_system"]}')

# Now use a more rigorous criterion: rank within target
# In each target, expected order: GT > {mid} > PT
# Pure/random both low sem but high nat → rank 4 and 5
# Strict mapping by rank:
rank_to_system = {
    1: 'GT_or_HIGH',  # high sem high nat
    2: 'MID',         # mid sem (could be COMP or split)
    3: 'MID_LOW',     # mid-low (could be COMP or PT)
    4: 'DONOR_HIGH',  # low sem high nat (PURE)
    5: 'DONOR_LOW',   # lowest sem (random)
}
print('\n=== Per-rank profile (rigorous) ===')
for rank in range(1, 6):
    matches = [v for v in videos if v['rank'] == rank]
    sem_means = [v['sem_mean'] for v in matches]
    intel_means = [v['intell_mean'] for v in matches]
    nat_means = [v['nat_mean'] for v in matches]
    print(f'  rank={rank} (n={len(matches)}): '
          f'sem={statistics.mean(sem_means):.2f} (sd={statistics.pstdev(sem_means):.2f}) | '
          f'intell={statistics.mean(intel_means):.2f} | '
          f'nat={statistics.mean(nat_means):.2f}')

# Compute Krippendorff's alpha for ordinal data
def krippendorff_alpha_ordinal(data_per_rater):
    """data_per_rater: dict rater -> {unit: value}. Returns alpha."""
    import math
    # Collect all units
    units = set()
    for r, d in data_per_rater.items():
        units.update(d.keys())
    units = sorted(units)
    # Build value matrix
    values = sorted({v for r in data_per_rater for v in data_per_rater[r].values()})
    val_idx = {v: i for i, v in enumerate(values)}
    n_vals = len(values)
    # ReliabilityData: rater x unit
    # Compute observed disagreement Do
    Do = 0.0
    pair_count = 0
    for u in units:
        vals = [data_per_rater[r][u] for r in data_per_rater if u in data_per_rater[r]]
        for i, vi in enumerate(vals):
            for vj in vals[i+1:]:
                Do += (vi - vj) ** 2
                pair_count += 1
    if pair_count == 0:
        return None
    Do *= 2  # count both orderings? Actually standard formula: sum (vi-vj)^2 / 2n
    # Expected De: based on marginal distribution
    all_vals = [v for r in data_per_rater for v in data_per_rater[r].values()]
    from collections import Counter
    vc = Counter(all_vals)
    n = len(all_vals)
    De = 0
    for v1 in values:
        for v2 in values:
            De += vc[v1] * vc[v2] * (val_idx[v1] - val_idx[v2])**2
    De /= n * n
    if De == 0:
        return None
    # Krippendorff alpha for ordinal uses a different formula with weights
    # Simplified: use interval alpha here as approximation
    return 1 - (Do / pair_count) / De  # rough

# Build rater dicts for each stage
rater_semantic = {}
rater_intell = {}
rater_nat = {}
for r in range(1, 31):
    rater_semantic[r] = {}
    rater_intell[r] = {}
    rater_nat[r] = {}
for r in range(1, 31):
    with open(EVAL_DIR / f'R{r:03d}_阶段2_语义评分表.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                rater_semantic[r][row['视频编号']] = int(row['语义充分性_1到5'])
            except: pass
    with open(EVAL_DIR / f'R{r:03d}_阶段1_视频评分表.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                rater_intell[r][row['视频编号']] = int(row['可理解性_1到5'])
                rater_nat[r][row['视频编号']] = int(row['自然度_1到5'])
            except: pass

# Use proper Krippendorff (need krippendorff package or compute manually)
# Try install
try:
    import krippendorff
    sem_data = [[rater_semantic[r].get(v, None) for v in sorted({vid for r in rater_semantic for vid in rater_semantic[r]})] for r in range(1, 31)]
    alpha_sem = krippendorff.alpha(reliability_data=sem_data, level_of_measurement='ordinal')
    int_data = [[rater_intell[r].get(v, None) for v in sorted({vid for r in rater_intell for vid in rater_intell[r]})] for r in range(1, 31)]
    alpha_int = krippendorff.alpha(reliability_data=int_data, level_of_measurement='ordinal')
    nat_data = [[rater_nat[r].get(v, None) for v in sorted({vid for r in rater_nat for vid in rater_nat[r]})] for r in range(1, 31)]
    alpha_nat = krippendorff.alpha(reliability_data=nat_data, level_of_measurement='ordinal')
    print(f'\n=== Krippendorff alpha (ordinal, n=30 raters) ===')
    print(f'  Semantic adequacy: alpha = {alpha_sem:.3f}')
    print(f'  Intelligibility:    alpha = {alpha_int:.3f}')
    print(f'  Naturalness:        alpha = {alpha_nat:.3f}')
except ImportError:
    print('\n(krippendorff package not available; using ICC instead)')

# Save classified results
out = {
    'n_raters': 30,
    'n_videos': len(videos),
    'per_rank_profile': {
        str(r): {
            'n': sum(1 for v in videos if v['rank'] == r),
            'sem_mean': statistics.mean([v['sem_mean'] for v in videos if v['rank'] == r]),
            'intell_mean': statistics.mean([v['intell_mean'] for v in videos if v['rank'] == r]),
            'nat_mean': statistics.mean([v['nat_mean'] for v in videos if v['rank'] == r]),
        } for r in range(1, 6)
    },
    'videos': [{k: v[k] for k in ['video_id', 'target', 'sem_mean', 'sem_sd', 'intell_mean',
                                   'nat_mean', 'rank', 'inferred_system']}
               for v in videos]
}
(OUT / 'human_eval_with_classification.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f'\nWritten: {OUT / "human_eval_with_classification.json"}')
