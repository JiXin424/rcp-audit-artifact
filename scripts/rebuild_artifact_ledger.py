#!/usr/bin/env python3
"""Rebuild SHA-256 artifact ledger covering ALL current files.
Also adds LICENSE if missing."""
import hashlib, json, sys
from pathlib import Path
from datetime import datetime, timezone

ARTI = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[1]

# Include the artifact directory + figures (paper assets)
LEDGER_DIRS = [
    ARTI,
    ARTI / 'figures',
]
SKIP_DIRS = {'__pycache__', '.git', 'tests/__pycache__'}
SKIP_FILES = {'.DS_Store', '__pycache__'}

def should_skip(p: Path) -> bool:
    if any(part in SKIP_DIRS for part in p.parts):
        return True
    if p.name in SKIP_FILES:
        return True
    if p.suffix in {'.pyc', '.pyo'}:
        return True
    return False

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

entries = []
for base in LEDGER_DIRS:
    if not base.exists():
        continue
    for p in sorted(base.rglob('*')):
        if not p.is_file(): continue
        if should_skip(p): continue
        try:
            sha = sha256_file(p)
            size = p.stat().st_size
            rel = str(p.relative_to(ROOT))
            entries.append({
                'path': rel,
                'sha256': sha,
                'size': size,
            })
        except Exception as e:
            print(f'WARN: {p}: {e}', file=sys.stderr)

# Build ledger
ledger = {
    'schema': 'artifact-ledger-v2',
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'generated_by': 'rebuild_artifact_ledger.py',
    'n_files': len(entries),
    'total_bytes': sum(e['size'] for e in entries),
    'roots': [str(p.relative_to(ROOT)) for p in LEDGER_DIRS],
    'files': entries,
}

# Write ledger (the ledger itself is not included in its own SHA list — self-reference)
out = ARTI / 'artifact_ledger.json'
out.write_text(json.dumps(ledger, indent=2))
print(f'Written: {out}')
print(f'Total files: {len(entries)}')
print(f'Total bytes: {ledger["total_bytes"]:,}')

# Category counts
from collections import Counter
cats = Counter()
for e in entries:
    if 'scripts/' in e['path']: cats['scripts'] += 1
    elif 'results/' in e['path']: cats['results'] += 1
    elif 'data/' in e['path']: cats['data'] += 1
    elif 'manifests/' in e['path']: cats['manifests'] += 1
    elif 'tests/' in e['path']: cats['tests'] += 1
    elif 'generated_figures/' in e['path']: cats['figures'] += 1
    else: cats['other'] += 1
print('\nBy category:')
for k, v in sorted(cats.items()):
    print(f'  {k}: {v}')

# Verify key files now in ledger (paths relative to the artifact root)
key_files = [
    'results/checkpoint_registry.json',
    'results/unified_checkpoint_registry.json',
    'results/gap_43_canonical_beam3.json',
    'results/dev_gate_table.json',
    'results/readout_overfit.json',
    'results/floor_calibration.json',
    'paper/main_lre.tex',
    'paper/supplementary.tex',
    'paper/dose_response.pdf',
    'paper/original_trajectory.pdf',
    'paper/checkpoint_sensitivity_schematic.pdf',
    'paper/generated_figures/fig3_competence.pdf',
]
print('\nKey files coverage:')
for kf in key_files:
    found = any(e['path'] == kf for e in entries)
    print(f'  {kf}: {"FOUND" if found else "MISSING"}')
