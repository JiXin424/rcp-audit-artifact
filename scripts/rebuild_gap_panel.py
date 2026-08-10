#!/usr/bin/env python3
"""Rebuild results/gap_43_canonical_beam3.json from per-item JSON files.

After adding new checkpoints (e.g. step-corrected seeds 1701-1708), the
per-item files exist in results/gap_43_canonical_beam3_items/ but the summary
JSON may be stale. This script recomputes corpus BLEU and the PURE-REC gap for
every checkpoint that has both <prefix>_gt.json and <prefix>_pure.json, and
writes the unified panel. No re-decoding.

Output: results/gap_43_canonical_beam3.json
"""
import json
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results" / "gap_43_canonical_beam3_items"
OUT = ROOT / "results" / "gap_43_canonical_beam3.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def main():
    gt_files = sorted(ITEMS.glob("*_gt.json"))
    panel = {}
    for gtf in gt_files:
        prefix = gtf.name[:-len("_gt.json")]
        puf = ITEMS / f"{prefix}_pure.json"
        if not puf.exists():
            continue
        gt = json.load(open(gtf))
        pure = json.load(open(puf))
        refs = [it["reference"] for it in gt]
        gt_bleu = BLEU.corpus_score([it["hypothesis"] for it in gt],
                                    [refs]).score
        pure_bleu = BLEU.corpus_score([it["hypothesis"] for it in pure],
                                      [refs]).score
        panel[prefix] = {
            "gt_bleu": gt_bleu,
            "pure_bleu": pure_bleu,
            "gap": pure_bleu - gt_bleu,
            "n_items": len(gt),
            "gt_items": gt,
            "pure_items": pure,
        }

    # Preserve _meta from the previous file if present
    if OUT.exists():
        try:
            old = json.load(open(OUT))
            if isinstance(old, dict) and "_meta" in old:
                panel["_meta"] = old["_meta"]
        except Exception:
            pass

    json.dump(panel, open(OUT, "w"), ensure_ascii=False)
    n = len(panel) - (1 if "_meta" in panel else 0)
    print(f"Rebuilt {OUT.name}: {n} checkpoint entries")

    # Show step-corrected + released for verification
    print("\n=== step-corrected + released gaps ===")
    for k in sorted(panel):
        if k == "_meta":
            continue
        if k == "released" or k.startswith("sf_"):
            print(f"  {k:12s}: REC={panel[k]['gt_bleu']:.2f} "
                  f"PURE={panel[k]['pure_bleu']:.2f} gap={panel[k]['gap']:+.2f}")

    # Verify sf_1703 (the +0.24 case) is present
    if "sf_1703" in panel:
        print(f"\n✓ sf_1703 present: gap={panel['sf_1703']['gap']:+.2f}")
    else:
        print("\n✗ sf_1703 MISSING")


if __name__ == "__main__":
    main()
