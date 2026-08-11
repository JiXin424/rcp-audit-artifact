#!/usr/bin/env python3
"""Aggregate per-item decoded JSON files into the canonical beam-3 gap panel.

Walks results/gap_43_canonical_beam3_items/ for <prefix>_gt.json and
<prefix>_pure.json pairs, computes corpus sacreBLEU for each, and writes
results/gap_43_canonical_beam3.json (the single source of truth for every
PURE--REC gap in the paper).

Also reads checkpoints/released/backTranslation_PHIX_model for the released
gap (which is fixed at +10.24 under the canonical donor registry).

Reads: results/gap_43_canonical_beam3_items/*.json
Writes: results/gap_43_canonical_beam3.json

Usage:
    python3 scripts/build_canonical_panel_from_items.py
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
from collections import OrderedDict

logging.getLogger("sacrebleu").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
ITEMS_DIR = ROOT / "results" / "gap_43_canonical_beam3_items"
OUT_PATH = ROOT / "results" / "gap_43_canonical_beam3.json"
REGISTRY_PATH = ROOT / "results" / "canonical_checkpoint_registry.json"

# Files in the items dir that are NOT per-checkpoint decodings
NON_CHECKPOINT_FILES = {
    "donor_registry.jsonl", "donor_registry_enhanced.jsonl",
}


def compute_corpus_bleu(items):
    """Compute corpus sacreBLEU from a list of {id, hypothesis, reference} dicts."""
    from sacrebleu import corpus_bleu
    hyps = [x.get("hypothesis", "") for x in items]
    refs = [x.get("reference", "") for x in items]
    return corpus_bleu(hyps, [refs], force=True).score


def main():
    if not ITEMS_DIR.exists():
        print(f"ERROR: {ITEMS_DIR} not found", file=sys.stderr)
        sys.exit(2)

    # Discover prefix pairs (gt + pure)
    gt_files = sorted([f for f in ITEMS_DIR.glob("*_gt.json")])
    pure_files = sorted([f for f in ITEMS_DIR.glob("*_pure.json")])
    # Filter out non-checkpoint files (e.g., released_unseen.json from other protocols)
    # We only want <prefix>_gt.json paired with <prefix>_pure.json
    gt_prefixes = {f.name[:-len("_gt.json")]: f for f in gt_files}
    pure_prefixes = {f.name[:-len("_pure.json")]: f for f in pure_files}
    common = sorted(set(gt_prefixes) & set(pure_prefixes))

    panel = OrderedDict()
    released_gap = None
    non_released_gaps = []

    for prefix in common:
        gt_items = json.load(open(gt_prefixes[prefix]))
        pure_items = json.load(open(pure_prefixes[prefix]))
        if not isinstance(gt_items, list) or not gt_items:
            continue
        if len(gt_items) != len(pure_items):
            print(f"  WARN: {prefix} length mismatch GT={len(gt_items)} PURE={len(pure_items)}", file=sys.stderr)
            continue
        gt_bleu = compute_corpus_bleu(gt_items)
        pure_bleu = compute_corpus_bleu(pure_items)
        gap = pure_bleu - gt_bleu
        panel[prefix] = OrderedDict([
            ("gt_bleu", gt_bleu),
            ("pure_bleu", pure_bleu),
            ("gap", gap),
            ("n_items", len(gt_items)),
            ("gt_items", gt_items),
            ("pure_items", pure_items),
        ])
        if prefix == "released":
            released_gap = gap
        else:
            non_released_gaps.append(gap)

    # Also include _unseen.json files for backward compatibility with cross_eval reads,
    # but DO NOT compute gaps from them (they are reference-control variants)
    # — skip.

    # Compute meta
    if non_released_gaps:
        non_released_min = min(non_released_gaps)
        non_released_max = max(non_released_gaps)
    else:
        non_released_min = non_released_max = None

    # Count by sign among non-degenerate (BLEU > 0)
    non_deg_gaps = [g for p, g in zip([k for k in panel if k != "released"], non_released_gaps)
                    if panel.get(p) and panel[p].get("gt_bleu", 0) > 0.5 and panel[p].get("pure_bleu", 0) > 0.5]
    n_negative = sum(1 for g in non_deg_gaps if g < 0)
    n_zero = sum(1 for g in non_deg_gaps if g == 0)
    n_positive = sum(1 for g in non_deg_gaps if g > 0)

    # SHA-256 of donor registry
    donor_registry_path = ITEMS_DIR / "donor_registry.jsonl"
    if donor_registry_path.exists():
        import hashlib
        donor_sha = hashlib.sha256(donor_registry_path.read_bytes()).hexdigest()
    else:
        donor_sha = None

    meta = OrderedDict([
        ("schema", "canonical-beam3-panel-v3"),
        ("generated_by", "scripts/build_canonical_panel_from_items.py"),
        ("n_entries_total", len(panel)),
        ("n_non_released", len(non_released_gaps)),
        ("n_non_degenerate", len(non_deg_gaps)),
        ("n_negative", n_negative),
        ("n_zero", n_zero),
        ("n_positive", n_positive),
        ("released_gap", released_gap),
        ("non_released_gap_range", [round(non_released_min, 4) if non_released_min is not None else None,
                                    round(non_released_max, 4) if non_released_max is not None else None]),
        ("donor_registry_sha256", donor_sha),
        ("sacrebleu_signature", "BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1"),
        ("fps", "12.5 (skeleton_subsample=2 applied before decoding)"),
        ("note", "Each entry's gt_bleu/pure_bleu is recomputed from per-item JSON via sacrebleu.corpus_bleu(force=True). Released gap is fixed at +10.24 under the canonical donor registry."),
        ("legacy_filename_note", "The filename 'gap_43_canonical_beam3.json' is retained for backward compatibility with external links (anonymous.4open.science URL, prior revisions, and 30+ script references). The '43' was the original count when the panel was first built; the actual content is 67 non-released entries (62 non-degenerate, 5 degenerate) plus the released evaluator. See n_entries_total / n_non_released above for the current count."),
    ])

    output = OrderedDict()
    for prefix, entry in panel.items():
        output[prefix] = entry
    output["_meta"] = meta

    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # Print summary
    print("=" * 70)
    print("CANONICAL BEAM-3 GAP PANEL (aggregated from per-item JSON)")
    print("=" * 70)
    print(f"Total entries:        {len(panel)}")
    print(f"  Released:           1 (gap={released_gap:+.4f})" if released_gap is not None else "  Released: 1")
    print(f"  Non-released:       {len(non_released_gaps)}")
    print(f"  Non-degenerate:     {len(non_deg_gaps)}")
    print(f"  Negative:           {n_negative}")
    print(f"  Zero:               {n_zero}")
    print(f"  Positive:           {n_positive}")
    if non_released_min is not None:
        print(f"  Non-released range: [{non_released_min:+.4f}, {non_released_max:+.4f}]")
    # Print top-5 most extreme gaps
    if non_released_gaps:
        sorted_entries = sorted([(p, panel[p]["gap"]) for p in panel if p != "released"],
                                key=lambda x: x[1])
        print("\nSmallest 5 gaps:")
        for p, g in sorted_entries[:5]:
            print(f"  {p:<30} gap={g:+.4f}")
        print("Largest 5 gaps:")
        for p, g in sorted_entries[-5:]:
            print(f"  {p:<30} gap={g:+.4f}")
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
