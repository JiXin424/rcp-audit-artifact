#!/usr/bin/env python3
"""E6b: recompute GT and TN-PURE text scores against the public human sign-to-text
back-translations (Czehmann et al. 2026, sacre-bird-phoenix), addressing reviewer Major #4.

Comparisons (corpus BLEU-4 sacrebleu 2.5.1 tok:13a and official-protocol chrF):
  GT decode    vs original reference      (paper baseline 12.78)
  PURE decode  vs original reference      (paper baseline 23.79)
  GT decode    vs human back-translation
  PURE decode  vs human back-translation
  original reference vs human back-translation  (reference validity sanity check)
  donor transcript  vs human back-translation   (retrieval upper bound without evaluator)
Reported on all mappable items and on the high-confidence (confidence=1.0) subset.
"""
import json, sys
from pathlib import Path

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
CSV_FULL = ROOT / "revision_20260729_round5/data_sacrebird/test_full_annotations_sacrebirdphoenix.csv"
CSV_HC = ROOT / "revision_20260729_round5/data_sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
REG = ROOT / "revision_20260728_canonical_rebuild/registry/query_donor_registry.jsonl"
OUT = ROOT / "revision_20260729_round5/results/e6b_human_reference_sensitivity.json"

import sacrebleu  # noqa: E402
sys.path.insert(0, "/ssd/xkb4/SignDiff/SLRTP2025_eval")
from metrics import chrf as official_chrf  # noqa: E402

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp", effective_order=False, force=True)


def load_csv(path):
    out = {}
    for i, line in enumerate(path.read_text().splitlines()):
        if i == 0:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        out[parts[0]] = parts[1]
    return out


def score(hyps, refs):
    bleu = BLEU.corpus_score(hyps, [refs]).score
    chrf = official_chrf(hyps, refs)
    return {"bleu4": round(bleu, 3), "chrf": round(chrf, 3)}


def main():
    human_full = load_csv(CSV_FULL)
    human_hc = load_csv(CSV_HC)
    gt_items = json.load(open(CELLS / "cp0_GT-v1.json"))["metrics"]["items"]
    pure_items = json.load(open(CELLS / "cp0_TN-PURE-v1.json"))["metrics"]["items"]
    donor_text = {json.loads(l)["query_id"]: json.loads(l)["donor_text"] for l in open(REG)}

    ids = [it["id"] for it in gt_items]
    hyps_gt = {it["id"]: it["hypothesis"] for it in gt_items}
    hyps_pure = {it["id"]: it["hypothesis"] for it in pure_items}
    refs_orig = {it["id"]: it["reference"] for it in gt_items}

    def run(id_set, ref_map, label):
        id_list = sorted(id_set)
        rows = {
            "GT_decode": score([hyps_gt[i] for i in id_list], [ref_map[i] for i in id_list]),
            "PURE_decode": score([hyps_pure[i] for i in id_list], [ref_map[i] for i in id_list]),
            "original_reference_as_hyp": score([refs_orig[i] for i in id_list], [ref_map[i] for i in id_list]),
            "donor_transcript_as_hyp": score([donor_text[i] for i in id_list], [ref_map[i] for i in id_list]),
        }
        return {"label": label, "n": len(id_list), "scores": rows,
                "PURE_minus_GT_bleu": round(rows["PURE_decode"]["bleu4"] - rows["GT_decode"]["bleu4"], 3),
                "PURE_minus_GT_chrf": round(rows["PURE_decode"]["chrf"] - rows["GT_decode"]["chrf"], 3)}

    out = {"note": "human back-translations: Czehmann et al. 2026 (CC BY-NC-SA 4.0), joined by PHOENIX test ID",
           "against_original_references": run(set(ids) & set(refs_orig), refs_orig, "original speech-transcribed references (baseline)"),
           "against_human_full": run(set(ids) & set(human_full), human_full, "human back-translations, all confidences"),
           "against_human_highconf": run(set(ids) & set(human_hc), human_hc, "human back-translations, confidence=1.0 subset")}
    # empty back-translations?
    empties = [i for i in ids if i in human_full and not human_full[i].strip()]
    out["n_empty_human_bt"] = len(empties)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
