#!/usr/bin/env python3
"""Analyze human evaluation data from /ssd/xkb4/RCP/评分/.
30 raters x 100 videos (20 target sentences x 5 systems).
Reverse-identify video -> system by semantic-score ranking."""
from __future__ import annotations
import csv, json, os, statistics
from collections import defaultdict, Counter
from pathlib import Path

EVAL_DIR = Path('/ssd/xkb4/RCP/评分')
OUT = Path('/ssd/xkb4/RCP/artifact/results/human_eval')
OUT.mkdir(parents=True, exist_ok=True)

# Stage 1: video-level (intelligibility, naturalness)
s1_by_video = defaultdict(lambda: {'intell': [], 'natural': []})
s1_by_rater_video = {}  # (rater, video) -> (intell, natural)
s1_tech_faults = defaultdict(int)
for r in range(1, 31):
    fn = EVAL_DIR / f'R{r:03d}_阶段1_视频评分表.csv'
    with open(fn, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            vid = row['视频编号']
            try:
                intell = int(row['可理解性_1到5'])
                natural = int(row['自然度_1到5'])
            except:
                continue
            s1_by_video[vid]['intell'].append(intell)
            s1_by_video[vid]['natural'].append(natural)
            s1_by_rater_video[(r, vid)] = (intell, natural)
            if row.get('技术故障_是或否', '').strip() in ('是', 'yes', 'Yes', 'YES'):
                s1_tech_faults[vid] += 1

# Stage 2: target meaning adequacy (1-5)
s2_by_video = defaultdict(lambda: {'semantic': [], 'target': None})
s2_by_rater_video = {}
for r in range(1, 31):
    fn = EVAL_DIR / f'R{r:03d}_阶段2_语义评分表.csv'
    with open(fn, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            vid = row['视频编号']
            try:
                sem = int(row['语义充分性_1到5'])
            except:
                continue
            s2_by_video[vid]['semantic'].append(sem)
            s2_by_video[vid]['target'] = row['德语目标句'].strip()
            s2_by_rater_video[(r, vid)] = sem

# Per-video aggregation
videos = sorted(s2_by_video.keys())
print(f'Total videos: {len(videos)}')
print(f'Raters per video (stage 2): min={min(len(v["semantic"]) for v in s2_by_video.values())}, '
      f'max={max(len(v["semantic"]) for v in s2_by_video.values())}')

per_video = []
for vid in videos:
    sem = s2_by_video[vid]['semantic']
    intel = s1_by_video[vid]['intell']
    nat = s1_by_video[vid]['natural']
    per_video.append({
        'video_id': vid,
        'target': s2_by_video[vid]['target'],
        'n_raters_semantic': len(sem),
        'semantic_mean': statistics.mean(sem),
        'semantic_median': statistics.median(sem),
        'semantic_sd': statistics.stdev(sem) if len(sem) > 1 else 0,
        'intell_mean': statistics.mean(intel) if intel else None,
        'intell_sd': statistics.stdev(intel) if len(intel) > 1 else 0,
        'natural_mean': statistics.mean(nat) if nat else None,
        'natural_sd': statistics.stdev(nat) if len(nat) > 1 else 0,
        'tech_faults': s1_tech_faults.get(vid, 0),
    })

# Group by target -> 5 videos each
by_target = defaultdict(list)
for v in per_video:
    by_target[v['target']].append(v)

# Rank within each target (5 videos per target)
# Higher semantic_mean = better
for target, vids in by_target.items():
    vids.sort(key=lambda x: -x['semantic_mean'])
    for rank, v in enumerate(vids, 1):
        v['rank_within_target'] = rank

# For each rank position (1..5), compute aggregate stats
print('\n=== Per-rank aggregate (semantic_mean) ===')
for rank in range(1, 6):
    matches = [v for v in per_video if v['rank_within_target'] == rank]
    sem_means = [v['semantic_mean'] for v in matches]
    intel_means = [v['intell_mean'] for v in matches if v['intell_mean']]
    nat_means = [v['natural_mean'] for v in matches if v['natural_mean']]
    print(f'  Rank {rank} (n={len(matches)}): '
          f'sem={statistics.mean(sem_means):.2f}±{statistics.stdev(sem_means):.2f}, '
          f'intell={statistics.mean(intel_means):.2f}, '
          f'nat={statistics.mean(nat_means):.2f}')

# Save full data
out = {
    'n_raters': 30,
    'n_videos': len(videos),
    'n_targets': len(by_target),
    'per_video': per_video,
    'by_target': {t: [{'video_id': v['video_id'], 'sem_mean': v['semantic_mean'],
                       'rank': v['rank_within_target']} for v in vids]
                  for t, vids in by_target.items()}
}
(OUT / 'human_eval_aggregate.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f'\nWritten: {OUT / "human_eval_aggregate.json"}')

# Show per-target rank patterns
print('\n=== Per-target top/bottom (semantic_mean) ===')
for target, vids in sorted(by_target.items())[:5]:
    print(f'\nTarget: {target[:70]}')
    for v in vids:
        print(f'  rank={v["rank_within_target"]} vid={v["video_id"]} sem={v["semantic_mean"]:.2f} intell={v["intell_mean"]:.2f} nat={v["natural_mean"]:.2f}')
