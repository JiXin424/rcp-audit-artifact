#!/usr/bin/env python3
"""E4 follow-up: recompute chrF/WER/ROUGE-L under the OFFICIAL protocol for all
7 primary evaluators x 4 systems (28 cells), so the paper's metric_sensitivity and
cross_metric_gap tables can be updated to official-protocol values."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
OUT = ROOT / "revision_20260729_round5/results/e4_official_protocol_all_cells.json"
EVAL_ROOT = Path(__file__).resolve().parents[1] / "src/models"
sys.path.insert(0, str(EVAL_ROOT))
from metrics import wer as official_wer, chrf as official_chrf, rouge as official_rouge  # noqa: E402

SYSTEMS = ["GT-v1", "PT-v1", "TN-PTCOMP-v1", "TN-PURE-v1"]
CPS = {f"cp{i}": n for i, n in enumerate(["original", "seed_101", "seed_202", "seed_303", "seed_404", "seed_505", "seed_606"])}

out = {}
for cp, eval_name in CPS.items():
    out[eval_name] = {}
    for s in SYSTEMS:
        d = json.load(open(CELLS / f"{cp}_{s}.json"))
        items = d["metrics"]["items"]
        refs = [it["reference"] for it in items]
        hyps = [it["hypothesis"] for it in items]
        out[eval_name][s] = {
            "chrf_official": official_chrf(hyps, refs),
            "wer_official": official_wer(hyps, refs),
            "rouge_official": official_rouge(hyps, refs),
        }
    gap = {m: round(out[eval_name]["TN-PURE-v1"][m] - out[eval_name]["GT-v1"][m], 4)
           for m in ["chrf_official", "wer_official", "rouge_official"]}
    print(eval_name, {s: {k: round(v, 3) for k, v in out[eval_name][s].items()} for s in ["GT-v1", "TN-PURE-v1"]}, "gap:", gap)

OUT.write_text(json.dumps(out, indent=2))
print("wrote", OUT)
