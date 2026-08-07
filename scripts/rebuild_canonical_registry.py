#!/usr/bin/env python3
"""Rebuild canonical_checkpoint_registry.json with checkpoint SHA-256 as primary key.

Single source of truth. Run this after adding/removing any checkpoint file.
All paper counts derive from the summary block of this file.

Usage:
    python scripts/rebuild_canonical_registry.py
"""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints"
GAP_FILE = ROOT / "results/gap_43_canonical_beam3.json"
OUT = ROOT / "results/canonical_checkpoint_registry.json"

# --- Load gap data ---
gap_data = json.loads(GAP_FILE.read_text()) if GAP_FILE.exists() else {}

# --- Scan all checkpoints ---
# sha -> {family, seeds, paths, ...}
binaries = {}  # sha256 -> info dict
train_runs = []  # list of all training runs

for ckpt_path in sorted(CKPT_ROOT.rglob("best.ckpt")):
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    rel = str(ckpt_path.relative_to(CKPT_ROOT))
    parts = rel.split("/")
    family = parts[0]

    if sha not in binaries:
        binaries[sha] = {
            "checkpoint_sha256": sha,
            "families": set(),
            "training_runs": [],
            "paths": [],
        }
    binaries[sha]["families"].add(family)
    binaries[sha]["training_runs"].append(rel)
    binaries[sha]["paths"].append(rel)

# --- Map gap entries to checkpoints ---
# Build lookup: gap key -> matching SHA
# gap keys: reco_101, cf_202, distill_a0.5_303, sf_505, ladder_0125, etc.
gap_to_sha = {}
for sha, info in binaries.items():
    for run in info["training_runs"]:
        parts = run.split("/")
        family = parts[0]
        sub = parts[1] if len(parts) > 1 else ""
        # Generate possible gap names
        candidates = []
        if family == "reconstructions":
            seed = sub.replace("seed_", "")
            candidates.append(f"reco_{seed}")
        elif family == "distillation":
            # alpha_X.Y_seed_NNN -> distill_aX.Y_NNN
            m = re.match(r"alpha_([\d.]+)_seed_(\d+)", sub)
            if m:
                alpha = m.group(1).replace("0.", "0.")  # keep as-is
                candidates.append(f"distill_a{alpha}_{m.group(2)}")
        elif family == "config_faithful":
            seed = sub.replace("seed_", "")
            candidates.append(f"cf_{seed}")
            candidates.append(f"cfaith_{seed}")
        elif family == "confirmation":
            seed = sub.replace("seed_", "")
            candidates.append(f"conf_{seed}")
        elif family == "ladder":
            frac = sub.replace("frac_", "")
            candidates.append(f"ladder_{frac}")
        elif family == "long_schedule":
            seed = sub.replace("seed_", "")
            candidates.append(f"ls_{seed}")
            candidates.append(f"long_sched_{'best' if seed == '202' else 'other'}")
        elif family == "rescue":
            candidates.append("rescue_wd0")
        elif family == "step_faithful":
            seed = sub.replace("seed_", "")
            candidates.append(f"sf_{seed}")
            candidates.append(f"stepfaith_{seed}")
        elif family == "released":
            candidates.append("released")
        elif family == "finetune_released":
            # e.g. lr5e-06_seed42
            candidates.append(f"finetune_{sub}")

        for c in candidates:
            if c in gap_data:
                gap_to_sha[c] = sha

# --- Build checkpoint entries ---
# One entry per unique binary
checkpoints = []
for sha, info in sorted(binaries.items()):
    families = sorted(info["families"])
    runs = sorted(info["training_runs"])

    # Determine primary family and gap key
    primary_family = families[0]
    gap_key = None
    gap_value = None
    gt_bleu = None
    pure_bleu = None

    # Find gap key for this binary
    for run in runs:
        parts = run.split("/")
        family = parts[0]
        sub = parts[1] if len(parts) > 1 else ""

        keys_to_try = []
        if family == "reconstructions":
            keys_to_try.append(f"reco_{sub.replace('seed_', '')}")
        elif family == "distillation":
            m = re.match(r"alpha_([\d.]+)_seed_(\d+)", sub)
            if m:
                keys_to_try.append(f"distill_a{m.group(1)}_{m.group(2)}")
        elif family == "config_faithful":
            keys_to_try.append(f"cf_{sub.replace('seed_', '')}")
            keys_to_try.append(f"cfaith_{sub.replace('seed_', '')}")
        elif family == "confirmation":
            keys_to_try.append(f"conf_{sub.replace('seed_', '')}")
        elif family == "ladder":
            keys_to_try.append(f"ladder_{sub.replace('frac_', '')}")
        elif family == "long_schedule":
            keys_to_try.append(f"ls_{sub.replace('seed_', '')}")
        elif family == "rescue":
            keys_to_try.append("rescue_wd0")
        elif family == "step_faithful":
            keys_to_try.append(f"sf_{sub.replace('seed_', '')}")
        elif family == "released":
            keys_to_try.append("released")
        elif family == "finetune_released":
            keys_to_try.append(f"finetune_{sub}")

        for k in keys_to_try:
            if k in gap_data:
                gap_key = k
                g = gap_data[k]
                if isinstance(g, dict):
                    gap_value = g.get("gap")
                    gt_bleu = g.get("gt_bleu")
                    pure_bleu = g.get("pure_bleu")
                break
        if gap_key:
            break

    # Check if gap entry exists from filesystem mapping
    if not gap_key and sha in gap_to_sha:
        gap_key = gap_to_sha[sha]
        g = gap_data.get(gap_key, {})
        if isinstance(g, dict):
            gap_value = g.get("gap")
            gt_bleu = g.get("gt_bleu")
            pure_bleu = g.get("pure_bleu")

    # Degenerate: gap ≈ 0 (distillation α=1.0) OR BLEU ≈ 0 (failed training)
    gap_near_zero = gap_value is not None and abs(gap_value) < 1e-6
    bleu_near_zero = (gt_bleu is not None and gt_bleu < 0.01) or \
                     (pure_bleu is not None and pure_bleu < 0.01)
    is_degenerate = gap_near_zero or bleu_near_zero

    entry = {
        "checkpoint_sha256": sha,
        "families": families,
        "training_runs": runs,
        "primary_family": primary_family,
        "gap_key": gap_key,
        "secondary_gap_keys": [],
        "has_gap": gap_key is not None,
        "gap": gap_value,
        "gt_bleu": gt_bleu,
        "pure_bleu": pure_bleu,
        "is_degenerate": is_degenerate,
        "is_released": "released" in families,
        "training_run_count": len(runs),
        "unique_binary": True,
    }
    checkpoints.append(entry)

# Map secondary gap keys: gap entries that share a binary with another entry
sha_to_entry = {c["checkpoint_sha256"]: c for c in checkpoints if c["checkpoint_sha256"]}
for gap_key, sha in gap_to_sha.items():
    entry = sha_to_entry.get(sha)
    if entry and entry["gap_key"] != gap_key:
        entry["secondary_gap_keys"].append(gap_key)

# Also add gap entries that have no corresponding binary
for gap_key in sorted(gap_data.keys()):
    if gap_key in ("_meta",):
        continue
    # Check if already mapped as primary or secondary
    already = any(c["gap_key"] == gap_key or gap_key in c.get("secondary_gap_keys", []) for c in checkpoints)
    if not already:
        g = gap_data[gap_key]
        checkpoints.append({
            "checkpoint_sha256": None,
            "families": [gap_key],
            "training_runs": [],
            "primary_family": gap_key,
            "gap_key": gap_key,
            "has_gap": True,
            "gap": g.get("gap") if isinstance(g, dict) else None,
            "gt_bleu": g.get("gt_bleu") if isinstance(g, dict) else None,
            "pure_bleu": g.get("pure_bleu") if isinstance(g, dict) else None,
            "is_degenerate": abs(g.get("gap", 0)) < 1e-6 if isinstance(g, dict) else False,
            "is_released": gap_key == "released",
            "training_run_count": 0,
            "unique_binary": False,
            "note": "gap entry without retained checkpoint binary",
        })

# --- Compute summary ---
total_training_runs = sum(c["training_run_count"] for c in checkpoints)
unique_binaries = sum(1 for c in checkpoints if c["unique_binary"])
has_gap_count = sum(1 for c in checkpoints if c["has_gap"])
non_degenerate = sum(1 for c in checkpoints if c["has_gap"] and not c["is_degenerate"])
degenerate = sum(1 for c in checkpoints if c["has_gap"] and c["is_degenerate"])
released_count = sum(1 for c in checkpoints if c["is_released"])
non_released_with_gap = has_gap_count - released_count
non_released_non_degenerate = non_degenerate - (1 if any(c["is_released"] and c["has_gap"] and not c["is_degenerate"] for c in checkpoints) else 0)

# sha collisions (training runs sharing same binary)
sha_collisions = [c for c in checkpoints if c["training_run_count"] > 1]

output = {
    "schema": "canonical-checkpoint-registry-v4",
    "generated_by": "scripts/rebuild_canonical_registry.py",
    "note": "checkpoint_sha256 is the primary key. Every unique binary has one entry. Training runs that share a binary (e.g., cf_202==ls_202) are listed under training_runs.",
    "summary": {
        "total_training_runs": total_training_runs,
        "unique_checkpoint_binaries": unique_binaries,
        "total_with_gap": has_gap_count,
        "non_released_with_gap": non_released_with_gap,
        "non_degenerate": non_degenerate,
        "non_released_non_degenerate": non_released_non_degenerate,
        "degenerate": degenerate,
        "released": released_count,
        "sha_collisions": len(sha_collisions),
    },
    "sha_collisions": [
        {
            "checkpoint_sha256": c["checkpoint_sha256"],
            "training_runs": c["training_runs"],
            "families": c["families"],
            "note": "These training runs produced the same binary; counted once as a unique checkpoint.",
        }
        for c in sha_collisions
    ],
    "degenerate_entries": [
        {
            "gap_key": c["gap_key"],
            "families": c["families"],
            "gap": c["gap"],
            "note": "BLEU-4 gap ≈ 0; excluded from non-degenerate count.",
        }
        for c in checkpoints if c["is_degenerate"]
    ],
    "entries_without_binary": [
        {
            "gap_key": c["gap_key"],
            "gap": c["gap"],
            "note": c.get("note", ""),
        }
        for c in checkpoints if not c["unique_binary"]
    ],
    "checkpoints": checkpoints,
}

OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"Canonical registry written to {OUT}")
print(f"  Total training runs: {total_training_runs}")
print(f"  Unique binaries: {unique_binaries}")
print(f"  With gap: {has_gap_count} ({non_released_with_gap} non-released)")
print(f"  Non-degenerate: {non_degenerate} ({non_released_non_degenerate} non-released)")
print(f"  Degenerate: {degenerate}")
print(f"  SHA collisions: {len(sha_collisions)}")
for c in sha_collisions:
    print(f"    {c['checkpoint_sha256'][:16]}... : {', '.join(c['families'])}")
