#!/usr/bin/env python3
"""E-H' CPU items: pass-through bootstrap CI + chrF corroboration + gate threshold sensitivity."""
import json, sys
from pathlib import Path
import numpy as np
import sacrebleu

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
REG = ROOT / "revision_20260728_canonical_rebuild/registry/query_donor_registry.jsonl"
R5 = ROOT / "revision_20260729_round5"
OUT = R5 / "results/e12h_minor.json"
B = 10_000
SBLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp", effective_order=False, force=True)
CHRF = sacrebleu.metrics.CHRF(word_order=0, char_order=6, beta=2)

donor_text = {json.loads(l)["query_id"]: json.loads(l)["donor_text"] for l in open(REG)}
items = json.load(open(CELLS / "cp0_TN-PURE-v1.json"))["metrics"]["items"]
gt_items = json.load(open(CELLS / "cp0_GT-v1.json"))["metrics"]["items"]
n = len(items)
rng = np.random.default_rng(42)

ratios, ratios_chrf = [], []
for b in range(B):
    idx = rng.integers(0, n, n)
    ad, aq, cd, cq = [], [], [], []
    for i in idx:
        it = items[i]
        dt = donor_text[it["id"]]
        ad.append(SBLEU.sentence_score(it["hypothesis"], [dt]).score / 100)
        aq.append(SBLEU.sentence_score(it["hypothesis"], [it["reference"]]).score / 100)
        cd.append(CHRF.sentence_score(it["hypothesis"], [dt]).score)
        cq.append(CHRF.sentence_score(it["hypothesis"], [it["reference"]]).score)
    ratios.append(np.mean(ad) / np.mean(aq))
    ratios_chrf.append(np.mean(cd) / np.mean(cq))

ad = [SBLEU.sentence_score(it["hypothesis"], [donor_text[it["id"]]]).score / 100 for it in items]
aq = [SBLEU.sentence_score(it["hypothesis"], [it["reference"]]).score / 100 for it in items]
cd = [CHRF.sentence_score(it["hypothesis"], [donor_text[it["id"]]]).score for it in items]
cq = [CHRF.sentence_score(it["hypothesis"], [it["reference"]]).score for it in items]

# GT pass-through baseline for contrast (GT decode vs its own reference = 1.0 by definition for chrF/BLEU vs reference)
out = {
    "original_pass_through_bleu_ratio": {"point": float(np.mean(ad) / np.mean(aq)),
                                          "ci": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                                          "mean_donor": float(np.mean(ad)), "mean_query": float(np.mean(aq))},
    "original_pass_through_chrf_ratio": {"point": float(np.mean(cd) / np.mean(cq)),
                                          "ci": [float(np.percentile(ratios_chrf, 2.5)), float(np.percentile(ratios_chrf, 97.5))],
                                          "mean_donor": float(np.mean(cd)), "mean_query": float(np.mean(cq))},
}
print(json.dumps(out, indent=1))

# gate threshold sensitivity
ORIG = {"bleu": 13.378651856913777, "wer": 77.48698273499589}
devs = {}
for f, src in [(ROOT / "revision_20260728_round4/results/r5_extension_seeds_dev.json", None)]:
    d = json.load(open(f))
    rows = d if isinstance(d, list) else d.get("rows", [])
    for r in rows:
        devs[f"seed_{r['seed']}"] = {"bleu": (r.get("dev_bleu4") or r.get("dev_bleu")) * 100, "wer": (r.get("dev_wer")) * 100}
prim = {101: (8.57, 85.66), 202: (8.81, 83.01), 303: (9.67, 82.54), 404: (7.19, 85.15), 505: (9.05, 81.98), 606: (8.09, 81.82)}
for k, v in prim.items():
    devs[f"seed_{k}"] = {"bleu": v[0], "wer": v[1]}
resc = json.load(open(R5 / "results/e8_rescue2_dev_eval.json"))
for k, v in resc.items():
    devs[k.replace("rescue2_", "")] = {"bleu": v["dev_bleu4"] * 100, "wer": v["dev_wer"] * 100}
lad = json.load(open(R5 / "results/e11b_ladder_gaps.json"))
for k, v in lad.items():
    devs["ladder_" + k] = {"bleu": v["dev_bleu"] * 100, "wer": v["dev_wer"] * 100}

sens = {}
for tb in [0.5, 1.0, 1.5, 2.0]:
    for tw in [1.5, 3.0, 4.5]:
        n_pass = sum(1 for v in devs.values()
                     if abs(v["bleu"] - ORIG["bleu"]) <= tb and abs(v["wer"] - ORIG["wer"]) <= tw)
        sens[f"bleu±{tb}_wer±{tw}"] = {"n_pass": n_pass, "n_total": len(devs)}
out["gate_sensitivity"] = sens
print(json.dumps(sens, indent=1))
OUT.write_text(json.dumps(out, indent=1))
