#!/usr/bin/env python3
"""Build a single canonical accounting table from the disk-scanned registry.

This is the SINGLE SOURCE OF TRUTH for all run/checkpoint counts in the paper.
Every number in the abstract, tables, and text must match this output.

Reads: results/canonical_checkpoint_registry.json (schema v4, disk scan)
Writes: results/accounting_table.json

Usage:
    python3 scripts/build_accounting_table.py
"""
import json
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "results" / "canonical_checkpoint_registry.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "accounting_table.json"

# Family -> tier mapping (must match registry v5)
FAMILY_TO_TIER = OrderedDict([
    ("faithful",                "faithful"),
    ("config_faithful",         "near_faithful"),
    ("step_faithful",           "near_faithful"),
    ("reconstructions_primary", "secondary"),
    ("reconstructions_extension", "secondary"),
    ("confirmation",            "diagnostic"),
    ("joint_loss_greedy",       "diagnostic"),
    ("joint_loss_beam3",        "diagnostic"),
    ("joint_loss_greedy_supp",  "diagnostic"),
    ("rescue2",                 "diagnostic"),
    ("rescue_lr",               "diagnostic"),
    ("distillation",            "diagnostic"),
    ("ladder",                  "diagnostic"),
    ("long_schedule",           "diagnostic"),
    ("big_arch",                "diagnostic"),
    ("crossfit",                "holdout"),
    ("bt_retrained_holdout",    "holdout"),
])

TIER_DISPLAY = OrderedDict([
    ("faithful",       "Faithful (config-faithful best-effort, seed-42-first)"),
    ("near_faithful", "Near-faithful (partially-corrected recipe)"),
    ("secondary",     "Secondary (legacy-implementation replication)"),
    ("diagnostic",    "Diagnostic (post-hoc sensitivity)"),
    ("holdout",       "Holdout controls"),
])

# Display order within each tier
FAMILY_DISPLAY = {
    "faithful": [
        ("faithful",                "Faithful (train_faithful.py, seeds 42-49)"),
    ],
    "near_faithful": [
        ("config_faithful",         "Validation-freq-misread"),
        ("step_faithful",           "Step-corrected (re-trained)"),
    ],
    "secondary": [
        ("reconstructions_primary", "Paper-derived (primary seeds)"),
        ("reconstructions_extension", "Paper-derived (extension seeds)"),
    ],
    "diagnostic": [
        ("joint_loss_greedy",       "Joint-loss (greedy selection)"),
        ("joint_loss_beam3",        "Joint-loss (beam-3 selection)"),
        ("joint_loss_greedy_supp",  "Joint-loss greedy (supplementary, 1809/1810)"),
        ("confirmation",            "Confirmation"),
        ("rescue2",                 "Rescue (dropout/wd variants)"),
        ("rescue_lr",               "Rescue (lr variants)"),
        ("distillation",            "Distillation"),
        ("ladder",                  "Ladder (data fractions)"),
        ("long_schedule",           "Long-schedule"),
        ("big_arch",                "Large-arch"),
    ],
    "holdout": [
        ("crossfit",                "Cross-fit holdout"),
        ("bt_retrained_holdout",    "BT-retrained holdout"),
    ],
}


def build_accounting():
    if not REGISTRY_PATH.exists():
        print(f"ERROR: registry not found at {REGISTRY_PATH}", file=sys.stderr)
        print("Run scripts/build_checkpoint_registry.py first.", file=sys.stderr)
        sys.exit(2)

    with open(REGISTRY_PATH) as f:
        reg = json.load(f)

    checkpoints = reg["checkpoints"]
    summary = reg["summary"]

    # Build per-family stats
    family_stats = defaultdict(lambda: {
        "runs": 0, "decoded": 0, "non_degenerate": 0,
        "has_dev": 0, "gate_eligible": 0, "negative": 0, "positive": 0, "zero": 0,
        "gap_min": None, "gap_max": None,
    })
    for ckpt in checkpoints:
        fam = ckpt["family"]
        s = family_stats[fam]
        s["runs"] += 1
        if ckpt.get("has_gap"):
            s["decoded"] += 1
            if not ckpt.get("degenerate"):
                s["non_degenerate"] += 1
                g = ckpt.get("gap")
                if g is not None:
                    if g < 0: s["negative"] += 1
                    elif g == 0: s["zero"] += 1
                    else: s["positive"] += 1
                    s["gap_min"] = g if s["gap_min"] is None else min(s["gap_min"], g)
                    s["gap_max"] = g if s["gap_max"] is None else max(s["gap_max"], g)
        if ckpt.get("has_dev"):
            s["has_dev"] += 1

    # Tier summaries
    tier_summary = OrderedDict()
    for tier in TIER_DISPLAY:
        tier_runs = tier_decoded = tier_non_deg = tier_neg = tier_pos = 0
        tier_gap_min = tier_gap_max = None
        for fam, t in FAMILY_TO_TIER.items():
            if t != tier: continue
            s = family_stats.get(fam)
            if s is None: continue
            tier_runs += s["runs"]
            tier_decoded += s["decoded"]
            tier_non_deg += s["non_degenerate"]
            tier_neg += s["negative"]
            tier_pos += s["positive"]
            if s["gap_min"] is not None:
                tier_gap_min = s["gap_min"] if tier_gap_min is None else min(tier_gap_min, s["gap_min"])
            if s["gap_max"] is not None:
                tier_gap_max = s["gap_max"] if tier_gap_max is None else max(tier_gap_max, s["gap_max"])
        tier_summary[tier] = {
            "runs": tier_runs, "decoded": tier_decoded,
            "non_degenerate": tier_non_deg,
            "negative": tier_neg, "positive": tier_pos,
            "gap_range": [round(tier_gap_min, 4) if tier_gap_min is not None else None,
                          round(tier_gap_max, 4) if tier_gap_max is not None else None],
        }

    # Build output
    output = OrderedDict()
    output["schema"] = "accounting-table-v2"
    output["source"] = "canonical_checkpoint_registry.json (schema v4, disk scan)"
    output["note"] = ("Single source of truth for all paper counts. Generated by "
                      "scripts/build_accounting_table.py from the disk-scanned registry. "
                      "Run scripts/build_checkpoint_registry.py first if checkpoints changed.")

    output["headline_counts"] = {
        "total_trained_runs": summary["total_trained_runs"],
        "total_decoded_gap_panel": summary["decoded_gap_panel"],
        "total_unique_binaries": summary["unique_binaries"],
        "total_non_degenerate": summary["non_degenerate"],
        "sha256_collisions": summary["sha256_collisions"],
        "released_evaluator": 1,
        "released_weight_perturbations": 11,  # finetune_released/ — not from-scratch training
        "total_checkpoint_artifacts": summary["total_trained_runs"] + 1,  # +released
    }
    # Note: released-weight fine-tunes live in checkpoints/finetune_released/ and are
    # NOT counted as from-scratch training. The scanner skips this directory.

    output["gap_panel_summary"] = {
        "decoded": summary["decoded_gap_panel"],
        "degenerate": summary["degenerate"],
        "non_degenerate": summary["non_degenerate"],
        "negative": summary["negative_gap_count"],
        "zero": summary["zero_gap_count"],
        "positive": summary["positive_gap_count"],
        "gap_range": summary["gap_range"],
    }

    output["tier_summary"] = OrderedDict()
    for tier, label in TIER_DISPLAY.items():
        ts = tier_summary[tier]
        output["tier_summary"][label] = {
            "runs": ts["runs"],
            "decoded": ts["decoded"],
            "unique_binaries": ts["decoded"],  # SHA collisions accounted at headline level
            "non_degenerate": ts["non_degenerate"],
            "negative": ts["negative"],
            "positive": ts["positive"],
            "gap_range": ts["gap_range"],
        }

    output["family_detail"] = OrderedDict()
    for tier, label in TIER_DISPLAY.items():
        for fam_key, fam_display in FAMILY_DISPLAY[tier]:
            if fam_key in family_stats:
                s = family_stats[fam_key]
                output["family_detail"][fam_display] = {
                    "tier": label.split(" (")[0],
                    "family_key": fam_key,
                    "runs": s["runs"],
                    "decoded": s["decoded"],
                    "non_degenerate": s["non_degenerate"],
                    "has_dev": s["has_dev"],
                    "negative": s["negative"],
                    "positive": s["positive"],
                    "gap_range": [round(s["gap_min"], 4) if s["gap_min"] is not None else None,
                                  round(s["gap_max"], 4) if s["gap_max"] is not None else None],
                }

    # Internal verification (no hardcoded total — derived from disk scan)
    errors = []
    # Family sums == tier sums == headline
    fam_runs_total = sum(s["runs"] for s in family_stats.values())
    tier_runs_total = sum(t["runs"] for t in tier_summary.values())
    if fam_runs_total != summary["total_trained_runs"]:
        errors.append(f"family runs sum={fam_runs_total} != headline trained={summary['total_trained_runs']}")
    if tier_runs_total != summary["total_trained_runs"]:
        errors.append(f"tier runs sum={tier_runs_total} != headline trained={summary['total_trained_runs']}")
    # Decoded <= trained
    if summary["decoded_gap_panel"] > summary["total_trained_runs"]:
        errors.append(f"decoded={summary['decoded_gap_panel']} > trained={summary['total_trained_runs']}")
    # Non-degenerate <= decoded
    if summary["non_degenerate"] > summary["decoded_gap_panel"]:
        errors.append(f"non_degenerate={summary['non_degenerate']} > decoded={summary['decoded_gap_panel']}")
    # Unique <= decoded
    if summary["unique_binaries"] > summary["decoded_gap_panel"]:
        errors.append(f"unique={summary['unique_binaries']} > decoded={summary['decoded_gap_panel']}")
    # neg + zero + pos == non_degenerate
    gap_sum = summary["negative_gap_count"] + summary["zero_gap_count"] + summary["positive_gap_count"]
    if gap_sum != summary["non_degenerate"]:
        errors.append(f"neg+zero+pos={gap_sum} != non_degenerate={summary['non_degenerate']}")

    output["_verification"] = {
        "all_checks_passed": len(errors) == 0,
        "errors": errors,
    }

    # Print summary
    print("=" * 80)
    print("CANONICAL ACCOUNTING TABLE v2 (from disk-scanned registry v4)")
    print("=" * 80)
    print(f"\n{'Tier / Family':<50} {'Runs':>5} {'Dec':>5} {'NonD':>5} {'Neg':>5} {'Pos':>5} {'Gap range':>18}")
    print("-" * 95)
    for tier, label in TIER_DISPLAY.items():
        ts = tier_summary[tier]
        gap_str = f"[{ts['gap_range'][0]:+.2f}, {ts['gap_range'][1]:+.2f}]" if ts["gap_range"][0] is not None else "---"
        print(f"  {label:<48} {ts['runs']:>5} {ts['decoded']:>5} {ts['non_degenerate']:>5} {ts['negative']:>5} {ts['positive']:>5} {gap_str:>18}")
        for fam_key, fam_display in FAMILY_DISPLAY[tier]:
            if fam_key in family_stats:
                s = family_stats[fam_key]
                gap_str = f"[{s['gap_min']:+.2f}, {s['gap_max']:+.2f}]" if s["gap_min"] is not None else "---"
                print(f"    {fam_display:<46} {s['runs']:>5} {s['decoded']:>5} {s['non_degenerate']:>5} {s['negative']:>5} {s['positive']:>5} {gap_str:>18}")
    print("-" * 95)
    hc = output["headline_counts"]
    gs = output["gap_panel_summary"]
    gap_range = gs["gap_range"]
    gap_str = f"[{gap_range[0]:+.2f}, {gap_range[1]:+.2f}]"
    print(f"  {'TOTAL':<48} {hc['total_trained_runs']:>5} {hc['total_decoded_gap_panel']:>5} {hc['total_non_degenerate']:>5} {gs['negative']:>5} {gs['positive']:>5} {gap_str:>18}")
    print(f"\n  Unique binaries (SHA-256): {hc['total_unique_binaries']}  ({hc['sha256_collisions']} collision(s))")
    print(f"  Degenerate:                {gs['degenerate']}")
    print(f"  Released-weight perturbations (not from-scratch): {hc['released_weight_perturbations']}")

    if errors:
        print(f"\n⚠ VERIFICATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"\n✓ All verification checks passed.")

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_accounting()
