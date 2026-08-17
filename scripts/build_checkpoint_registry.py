#!/usr/bin/env python3
"""Build canonical checkpoint registry v4 by scanning disk.

Single source of truth for every per-run fact in the paper. All counts in
main_lre.tex, supplementary.tex, README.md, claim_manifest.json, and
accounting_table.json must be derived from this registry.

Discovery rule: walk checkpoints/*/*/best.ckpt + training_log.json. Skip
  - released/             (audit target, not a training run)
  - finetune_released/    (weight perturbations, not from-scratch)
  - step_faithful_legacy_nll/  (the previously-mislabelled runs; archived)
  - *_OOM/, *_failed/     (incomplete training)

For each remaining run, extract from training_log.json:
  seed, protocol, selection, validation_freq_steps (None=epoch),
  rec_weight, trans_weight, has_dev, dev metrics

Then:
  - Compute SHA-256 of best.ckpt
  - Look up corresponding gap_43_canonical_beam3_items/<prefix>_{gt,pure}.json
  - If gap files exist, compute corpus sacreBLEU + gap (PURE - REC)
  - Mark degenerate if max(GT_bleu, PURE_bleu) < 0.5 or all-empty hypotheses

Family derivation:
  checkpoints/reconstructions/seed_X        -> reconstructions_primary (X<=606) | reconstructions_extension (X>=707)
  checkpoints/config_faithful/seed_X        -> config_faithful
  checkpoints/step_faithful/seed_X          -> step_faithful  (true step-corrected, post re-train)
  checkpoints/confirmation/seed_X           -> confirmation
  checkpoints/long_schedule/seed_X          -> long_schedule
  checkpoints/rescue/seed_X_wd0|...         -> rescue2  (dropout/wd variants)
  checkpoints/rescue/seed_X_lrY             -> rescue_lr (lr variants)
  checkpoints/distillation/alpha_X_seed_Y   -> distillation
  checkpoints/ladder/frac_X                 -> ladder
  checkpoints/big_arch/X                    -> big_arch
  checkpoints/crossfit/X                    -> crossfit
  checkpoints/bt_retrained_holdout/X        -> bt_retrained_holdout
  checkpoints/reconstructions_v2/seed_18XX  -> joint_loss_greedy (1801-1808 main; 1809/1810 supplementary)
  checkpoints/reconstructions_v3/seed_19XX  -> joint_loss_beam3
  checkpoints/faithful/seed_4X              -> faithful (seed-42-first, batch-norm, joint, beam-3)

Usage:
    python3 scripts/build_checkpoint_registry.py
"""
from __future__ import annotations
import hashlib, json, os, sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_ROOT = ROOT / "checkpoints"
GAP_ITEMS_DIR = ROOT / "results" / "gap_43_canonical_beam3_items"
OUT_PATH = ROOT / "results" / "canonical_checkpoint_registry.json"

# Directories to skip (not from-scratch reconstruction training)
SKIP_DIRS = {"released", "finetune_released", "step_faithful_legacy_nll", "retrain_logs"}

# Family-tier mapping (must match paper stratification)
FAMILY_TO_TIER = {
    "faithful":                "faithful",  # seed-42-first, batch-norm, joint loss, beam-3 select
    "config_faithful":         "near_faithful",
    "step_faithful":           "near_faithful",
    "reconstructions_primary": "secondary",
    "reconstructions_extension": "secondary",
    "confirmation":            "diagnostic",
    "joint_loss_greedy":       "diagnostic",
    "joint_loss_beam3":        "diagnostic",
    "joint_loss_greedy_supp":  "diagnostic",  # 1809/1810 stray v2 seeds
    "rescue2":                 "diagnostic",
    "rescue_lr":               "diagnostic",
    "distillation":            "diagnostic",
    "ladder":                  "diagnostic",
    "long_schedule":           "diagnostic",
    "big_arch":                "diagnostic",
    "crossfit":                "holdout",
    "bt_retrained_holdout":    "holdout",
}

# Prefix in gap_43_canonical_beam3_items/ derived from family + seed
# (must match the decoding harness naming)
def gap_panel_prefix(family: str, seed, dir_name: str) -> str | None:
    """Return the filename prefix used in gap_43_canonical_beam3_items/, or None if not decoded."""
    if family == "config_faithful":
        return f"cf_{seed}"
    if family == "faithful":
        return f"faithful_{seed}"
    if family == "step_faithful":
        return f"sf_{seed}"
    if family in ("reconstructions_primary", "reconstructions_extension"):
        return f"reco_{seed}"
    if family == "confirmation":
        return f"conf_{seed}"
    if family == "long_schedule":
        return f"ls_{seed}"
    if family == "distillation":
        # distill_a0.0_101 etc. — caller passes (alpha, seed); handled separately
        return None
    if family == "ladder":
        # ladder_0125, ladder_025, etc. — handled separately
        return None
    if family == "rescue2":
        # rescue_wd0 etc. — handled separately
        return None
    if family in ("joint_loss_greedy", "joint_loss_beam3", "joint_loss_greedy_supp"):
        return f"sf_{seed}"
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_corpus_bleu(gt_items: list, pure_items: list) -> tuple[float, float, float]:
    """Return (gt_bleu, pure_bleu, gap)."""
    try:
        import logging
        logging.getLogger("sacrebleu").setLevel(logging.ERROR)
        from sacrebleu import corpus_bleu
        gt_hyps = [x.get("hypothesis", "") for x in gt_items]
        gt_refs = [x.get("reference", "") for x in gt_items]
        pure_hyps = [x.get("hypothesis", "") for x in pure_items]
        pure_refs = [x.get("reference", "") for x in pure_items]
        gt_bleu = corpus_bleu(gt_hyps, [gt_refs], force=True).score
        pure_bleu = corpus_bleu(pure_hyps, [pure_refs], force=True).score
        return gt_bleu, pure_bleu, pure_bleu - gt_bleu
    except Exception as e:
        print(f"  WARN: BLEU compute failed: {e}", file=sys.stderr)
        return None, None, None


def is_degenerate(gt_items: list, pure_items: list, gt_bleu, pure_bleu) -> tuple[bool, str | None]:
    """Check if decoded output is degenerate (BLEU≈0 or all-empty)."""
    if gt_bleu is None or pure_bleu is None:
        return False, None
    if max(gt_bleu, pure_bleu) < 0.5:
        return True, f"BLEU near zero (GT={gt_bleu:.2f}, PURE={pure_bleu:.2f})"
    # Check all-empty hypotheses
    gt_empty = all(not x.get("hypothesis", "").strip() for x in gt_items)
    pure_empty = all(not x.get("hypothesis", "").strip() for x in pure_items)
    if gt_empty and pure_empty:
        return True, "all-empty hypotheses in both GT and PURE"
    return False, None


def derive_family(dir_name: str, seed_dir: str, log: dict) -> tuple[str, str]:
    """Derive (family, run_id) from directory structure + training log."""
    seed = log.get("seed", seed_dir)
    protocol = log.get("protocol", "")
    if dir_name == "reconstructions":
        if isinstance(seed, int) and seed <= 606:
            return "reconstructions_primary", f"reco_{seed}"
        return "reconstructions_extension", f"reco_{seed}"
    if dir_name == "config_faithful":
        return "config_faithful", f"cf_{seed}"
    if dir_name == "faithful":
        return "faithful", f"faithful_{seed}"
    if dir_name == "step_faithful":
        return "step_faithful", f"sf_{seed}"
    if dir_name == "confirmation":
        return "confirmation", f"conf_{seed}"
    if dir_name == "long_schedule":
        return "long_schedule", f"ls_{seed}"
    if dir_name == "distillation":
        # alpha_X_seed_Y format
        return "distillation", f"distill_a{log.get('alpha', '?')}_s{seed}"
    if dir_name == "ladder":
        # frac_0125, frac_025 etc
        frac = seed_dir.replace("frac_", "").replace("frac", "")
        return "ladder", f"ladder_{frac}"
    if dir_name == "big_arch":
        return "big_arch", f"big_arch_{seed_dir}"
    if dir_name == "crossfit":
        return "crossfit", f"crossfit_{seed_dir}"
    if dir_name == "bt_retrained_holdout":
        return "bt_retrained_holdout", f"bt_retrained_{seed_dir}"
    if dir_name == "rescue":
        # Use variant info from log; default to rescue2
        variant = log.get("variant", seed_dir)
        if "lr" in str(variant).lower() or "lr" in str(seed_dir).lower():
            return "rescue_lr", f"rescue_lr_{seed_dir}"
        if "wd0" in str(variant) or "wd0" in str(seed_dir):
            return "rescue2", "rescue_wd0"
        return "rescue2", f"rescue2_{seed_dir}"
    if dir_name == "reconstructions_v2":
        # 1801-1808 main; 1809/1810 supplementary (stray)
        if isinstance(seed, int) and seed <= 1808:
            return "joint_loss_greedy", f"sf_{seed}"
        return "joint_loss_greedy_supp", f"sf_{seed}"
    if dir_name == "reconstructions_v3":
        return "joint_loss_beam3", f"sf_{seed}"
    # Fallback
    return f"unknown_{dir_name}", f"unknown_{dir_name}_{seed_dir}"


def find_gap_files(prefix: str, family: str, log: dict, seed_dir: str) -> tuple[Path, Path] | None:
    """Locate the (gt, pure) per-item JSON pair for this run."""
    if prefix is None:
        return None
    gt = GAP_ITEMS_DIR / f"{prefix}_gt.json"
    pure = GAP_ITEMS_DIR / f"{prefix}_pure.json"
    if gt.exists() and pure.exists():
        return gt, pure
    return None


def main():
    if not CKPT_ROOT.exists():
        print(f"ERROR: {CKPT_ROOT} does not exist", file=sys.stderr)
        sys.exit(2)

    checkpoints = []
    sha_cache = {}

    # Walk checkpoints/<family_dir>/<seed_dir>/best.ckpt
    for family_dir in sorted(CKPT_ROOT.iterdir()):
        if not family_dir.is_dir():
            continue
        if family_dir.name in SKIP_DIRS:
            continue
        for seed_dir in sorted(family_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            if seed_dir.name.startswith("_") or seed_dir.name.endswith("_OOM") or seed_dir.name.endswith("_failed"):
                continue
            best_ckpt = seed_dir / "best.ckpt"
            log_path = seed_dir / "training_log.json"
            has_ckpt = best_ckpt.exists()
            has_log = log_path.exists()
            # Require at least one of the two as evidence of training
            if not has_ckpt and not has_log:
                # Empty directory (e.g. big_arch/, crossfit/, bt_retrained_holdout/ that were
                # never actually trained in this repo) — do not count.
                continue
            if not has_log:
                # best.ckpt without training_log.json — count but flag
                print(f"  NOTE: {seed_dir} has best.ckpt but no training_log.json — counting with missing_log flag", file=sys.stderr)
                log = {"seed": seed_dir.name, "protocol": "unknown (no training_log)",
                       "selection": None, "validation_freq_steps": None,
                       "rec_weight": None, "trans_weight": None,
                       "validations": [], "best": {}}
            else:
                try:
                    with open(log_path) as f:
                        log = json.load(f)
                except Exception as e:
                    print(f"  WARN: cannot parse {log_path}: {e}", file=sys.stderr)
                    continue

            family, run_id = derive_family(family_dir.name, seed_dir.name, log)
            seed = log.get("seed")
            protocol = log.get("protocol")
            selection = log.get("selection")
            val_freq_steps = log.get("validation_freq_steps")  # None means epoch-level
            rec_weight = log.get("rec_weight", log.get("recognition_loss_weight"))
            trans_weight = log.get("trans_weight", log.get("translation_loss_weight"))

            # Determine validation schedule label
            if val_freq_steps is None:
                validation = "epoch"
            else:
                validation = f"step-{val_freq_steps}"

            # Determine loss label
            if rec_weight is not None and rec_weight == 0:
                loss = "translation-only"
            elif rec_weight is not None and rec_weight > 0:
                loss = "joint-CTC+translation"
            else:
                loss = log.get("loss", "translation-only")

            # Has dev: training_log has validations list OR best entry
            validations = log.get("validations") or log.get("epochs_log") or []
            has_dev = len(validations) > 0
            best = log.get("best", {})
            best_dev_metric = best.get("dev_metric")
            best_step = best.get("step")
            missing_log = not has_log

            # Look up gap panel entries
            prefix = gap_panel_prefix(family, seed, seed_dir.name)
            # Special-case distillation, ladder, rescue naming
            if family == "distillation":
                # dir like alpha_0.0_seed_101
                parts = seed_dir.name.split("_")
                if len(parts) >= 4:
                    a = parts[1]; s = parts[3]
                    prefix = f"distill_a{a}_{s}"
            elif family == "ladder":
                # dir like frac_05 or frac_0125
                fname = seed_dir.name.replace("frac_", "")
                prefix = f"ladder_{fname}"
            elif family == "rescue2" and "wd0" in str(log.get("variant", "")):
                prefix = "rescue_wd0"

            gt_path, pure_path = None, None
            gap_value = None; gt_bleu = None; pure_bleu = None
            has_gap = False; degenerate = False; degenerate_reason = None
            n_items = None

            if prefix is not None:
                gap_pair = find_gap_files(prefix, family, log, seed_dir.name)
                if gap_pair is not None:
                    gt_path, pure_path = gap_pair
                    try:
                        with open(gt_path) as f: gt_items = json.load(f)
                        with open(pure_path) as f: pure_items = json.load(f)
                        n_items = len(gt_items)
                        gt_bleu, pure_bleu, gap_value = compute_corpus_bleu(gt_items, pure_items)
                        has_gap = True
                        degenerate, degenerate_reason = is_degenerate(gt_items, pure_items, gt_bleu, pure_bleu)
                    except Exception as e:
                        print(f"  WARN: cannot compute gap for {prefix}: {e}", file=sys.stderr)

            # SHA-256 of best.ckpt (cache by path + mtime + size)
            sha256 = None
            if best_ckpt.exists():
                key = (str(best_ckpt), best_ckpt.stat().st_mtime, best_ckpt.stat().st_size)
                if key in sha_cache:
                    sha256 = sha_cache[key]
                else:
                    try:
                        sha256 = sha256_file(best_ckpt)
                        sha_cache[key] = sha256
                    except Exception as e:
                        print(f"  WARN: SHA-256 failed for {best_ckpt}: {e}", file=sys.stderr)

            entry = OrderedDict([
                ("run_id", run_id),
                ("family", family),
                ("tier", FAMILY_TO_TIER.get(family, "unknown")),
                ("seed", seed),
                ("protocol", protocol),
                ("loss", loss),
                ("validation", validation),
                ("selection", selection),
                ("rec_weight", rec_weight),
                ("trans_weight", trans_weight),
                ("best_step", best_step),
                ("best_dev_metric_training", best_dev_metric),
                ("has_dev", has_dev),
                ("has_gap", has_gap),
                ("gap", round(gap_value, 4) if gap_value is not None else None),
                ("gt_bleu", round(gt_bleu, 2) if gt_bleu is not None else None),
                ("pure_bleu", round(pure_bleu, 2) if pure_bleu is not None else None),
                ("n_items", n_items),
                ("degenerate", degenerate),
                ("degenerate_reason", degenerate_reason),
                ("sha256", sha256),
                ("missing_log", missing_log),
                ("checkpoint_path", str(best_ckpt.relative_to(ROOT)) if best_ckpt.exists() else None),
                ("training_log_path", str(log_path.relative_to(ROOT)) if has_log else None),
                ("training_dir", str(seed_dir.relative_to(ROOT))),
            ])
            checkpoints.append(entry)

    # Compute summary
    n_total = len(checkpoints)
    n_has_dev = sum(1 for c in checkpoints if c["has_dev"])
    n_has_gap = sum(1 for c in checkpoints if c["has_gap"])
    n_degenerate = sum(1 for c in checkpoints if c["has_gap"] and c["degenerate"])
    n_non_degenerate = n_has_gap - n_degenerate
    # Unique binaries: distinct SHA-256 among gap-decoded runs
    gap_shas = [c["sha256"] for c in checkpoints if c["has_gap"] and c["sha256"] is not None]
    n_unique_binaries = len(set(gap_shas))
    sha_collisions = n_has_gap - n_unique_binaries  # number of duplicates

    # Negative/positive gap counts among non-degenerate
    non_deg_gaps = [c["gap"] for c in checkpoints if c["has_gap"] and not c["degenerate"] and c["gap"] is not None]
    n_negative = sum(1 for g in non_deg_gaps if g < 0)
    n_zero = sum(1 for g in non_deg_gaps if g == 0)
    n_positive = sum(1 for g in non_deg_gaps if g > 0)
    if non_deg_gaps:
        gap_min = min(non_deg_gaps); gap_max = max(non_deg_gaps)
    else:
        gap_max = gap_min = None

    # Family counts
    family_counts = defaultdict(int)
    for c in checkpoints:
        family_counts[c["family"]] += 1

    # Tier counts
    tier_counts = defaultdict(int)
    for c in checkpoints:
        tier_counts[c["tier"]] += 1

    summary = OrderedDict([
        ("schema", "canonical-checkpoint-registry-v4"),
        ("generated_by", "scripts/build_checkpoint_registry.py (disk scan)"),
        ("total_trained_runs", n_total),
        ("has_dev_metrics", n_has_dev),
        ("decoded_gap_panel", n_has_gap),
        ("unique_binaries", n_unique_binaries),
        ("sha256_collisions", sha_collisions),
        ("degenerate", n_degenerate),
        ("non_degenerate", n_non_degenerate),
        ("negative_gap_count", n_negative),
        ("zero_gap_count", n_zero),
        ("positive_gap_count", n_positive),
        ("gap_range", [round(gap_min, 4) if gap_min is not None else None,
                       round(gap_max, 4) if gap_max is not None else None]),
        ("tier_counts", dict(tier_counts)),
        ("family_counts", dict(family_counts)),
        ("scan_root", str(CKPT_ROOT.relative_to(ROOT))),
        ("scan_excluded", sorted(SKIP_DIRS)),
    ])

    output = OrderedDict([
        ("schema", "canonical-checkpoint-registry-v4"),
        ("generated_by", "scripts/build_checkpoint_registry.py (disk scan)"),
        ("generated_note", "Single source of truth for every per-run fact in the paper. All counts in main_lre.tex, supplementary.tex, README.md, claim_manifest.json, and accounting_table.json must be derived from this registry. Disk scan walks checkpoints/*/*/best.ckpt + training_log.json and joins with results/gap_43_canonical_beam3_items/<prefix>_{gt,pure}.json."),
        ("summary", summary),
        ("checkpoints", checkpoints),
    ])

    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # Print summary
    print("=" * 70)
    print("CANONICAL CHECKPOINT REGISTRY v4 (disk scan)")
    print("=" * 70)
    print(f"Total trained runs:         {n_total}")
    print(f"  with dev metrics:         {n_has_dev}")
    print(f"  decoded on gap panel:     {n_has_gap}")
    print(f"  unique binaries (SHA-256):{n_unique_binaries}  ({sha_collisions} collision(s))")
    print(f"  degenerate:               {n_degenerate}")
    print(f"  non-degenerate:           {n_non_degenerate}")
    print(f"  negative gap (non-deg):   {n_negative}")
    print(f"  zero gap (non-deg):       {n_zero}")
    print(f"  positive gap (non-deg):   {n_positive}")
    if gap_min is not None:
        print(f"  non-deg gap range:        [{gap_min:.4f}, {gap_max:.4f}]")
    print()
    print("By tier:")
    for tier, n in sorted(tier_counts.items()):
        print(f"  {tier}: {n}")
    print()
    print("By family:")
    for fam, n in sorted(family_counts.items()):
        print(f"  {fam}: {n}")
    print()
    print(f"Written: {OUT_PATH}")


if __name__ == "__main__":
    main()
