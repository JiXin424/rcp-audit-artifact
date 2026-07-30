#!/usr/bin/env python3
"""Reviewer-round diagnostics from frozen canonical evaluator cells."""

import argparse
import csv
import json
from pathlib import Path

import sacrebleu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.cells.glob("cp*_*.json")):
        cell = json.loads(path.read_text())
        items = cell["metrics"]["items"]
        hyps = [item["hypothesis"] for item in items]
        refs = [item["reference"] for item in items]
        score = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a", smooth_method="exp", force=True)
        hyp_tokens = [len(h.split()) for h in hyps]
        ref_tokens = [len(r.split()) for r in refs]
        rows.append({
            "checkpoint_index": cell["checkpoint_index"],
            "family": cell["family"],
            "seed": cell["seed"],
            "system_id": cell["system_id"],
            "fractional_bleu4": score.score / 100.0,
            "bleu_0_100": score.score,
            "empty_count": sum(not h.strip() for h in hyps),
            "empty_rate": sum(not h.strip() for h in hyps) / len(hyps),
            "hyp_tokens": sum(hyp_tokens),
            "ref_tokens": sum(ref_tokens),
            "hyp_ref_token_ratio": sum(hyp_tokens) / sum(ref_tokens),
            "brevity_penalty": score.bp,
            "precision_1": score.precisions[0],
            "precision_2": score.precisions[1],
            "precision_3": score.precisions[2],
            "precision_4": score.precisions[3],
            "signature": cell["sacrebleu_signature"],
        })

    fieldnames = list(rows[0])
    with (args.output / "bleu_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "bleu_decomposition.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"cells": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
