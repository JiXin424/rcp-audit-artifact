#!/usr/bin/env python3
"""Try hash matching to identify video -> system."""
import hashlib, json, csv
from pathlib import Path

EVAL_DIR = Path('/ssd/xkb4/RCP/评分')
ARTI = Path('/ssd/xkb4/RCP/artifact')

# Load target -> item_id mapping
gt = json.load(open(ARTI / 'data/cells/cp0_GT-v1.json'))
items = {it['id']: it for it in gt['metrics']['items']}
target_to_item = {}
for iid, it in items.items():
    target_to_item.setdefault(it['reference'].strip(), []).append(iid)

# Load video_id -> target
vid_to_target = {}
for r in range(1, 31):
    with open(EVAL_DIR / f'R{r:03d}_阶段2_语义评分表.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            vid_to_target[row['视频编号']] = row['德语目标句'].strip()

video_ids = sorted(vid_to_target.keys())

# Test on first 5 videos
test_vids = video_ids[:5]
print(f'Sample video IDs: {test_vids}')
print(f'Length: {len(test_vids[0])}')

# Check UUID version digit (char 12)
for v in test_vids:
    print(f'  {v} version_char={v[12]} (4=uuid4, 5=uuid5/hash)')

# Try sha256 of various combinations
test_vid = test_vids[0]
test_target = vid_to_target[test_vid]
test_items = target_to_item.get(test_target, [])
print(f'\nTest: vid={test_vid}, target={test_target[:60]}')
print(f'Item IDs: {test_items}')

systems = ['GT', 'PT', 'PURE', 'COMP', 'RANDOM', 'TN-PURE', 'TN-PTCOMP',
           'GT-v1', 'PT-v1', 'TN-PURE-v1', 'TN-PTCOMP-v1', 'RAND640',
           '0', '1', '2', '3', '4', '5']

found = []
for sys in systems:
    for item_id in test_items:
        for fmt in [f'{sys}_{item_id}', f'{item_id}_{sys}',
                    f'{sys}-{item_id}', f'{item_id}-{sys}',
                    f'{sys}{item_id}', f'{item_id}{sys}',
                    f'{sys}:{item_id}', f'{item_id}:{sys}']:
            for algo_name in ['sha256', 'sha1', 'md5', 'sha512']:
                algo = getattr(hashlib, algo_name)
                h = algo(fmt.encode()).hexdigest().upper()
                if h.startswith(test_vid):
                    print(f'  MATCH: sys={sys}, fmt={fmt!r}, algo={algo_name}')
                    print(f'    hash={h[:32]}...')
                    found.append((sys, fmt, algo_name))
                # Try lowercase
                h_low = algo(fmt.encode()).hexdigest()
                if h_low.startswith(test_vid.lower()):
                    print(f'  MATCH (lower): sys={sys}, fmt={fmt!r}, algo={algo_name}')

# Try uuid5 (namespace + name)
import uuid
namespaces = [uuid.NAMESPACE_DNS, uuid.NAMESPACE_URL, uuid.NAMESPACE_OID, uuid.NAMESPACE_X500]
for ns in namespaces:
    for sys in systems:
        for item_id in test_items:
            for fmt in [f'{sys}_{item_id}', f'{item_id}_{sys}', f'{sys}']:
                u = uuid.uuid5(ns, fmt)
                if str(u).replace('-', '').upper().startswith(test_vid):
                    print(f'  UUID5 MATCH: ns={ns}, fmt={fmt!r}')
                u4 = uuid.uuid4()  # random
                # can't predict uuid4

print(f'\nTotal hash matches: {len(found)}')

# Try: maybe it's the SHA-256 of the pose tensor file, indexed by sequence
# Or maybe it's from a separate "videos" manifest that's lost
# Final fallback: rank-based system assignment using profile
print('\n--- Falling back to rank-based system identification ---')
print('Rank 1 (sem 4.48, intell 3.82, nat 3.90) → likely GT (real human pose, semantically correct)')
print('Rank 5 (sem 1.83, intell 3.59, nat 3.71) → likely random donor (real pose, wrong meaning)')
print('Rank 2-4: mixed profile; need additional analysis')
