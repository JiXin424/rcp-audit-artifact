#!/usr/bin/env python3
"""E8 Pareto analysis: multi-objective checkpoint selection on dense trajectories.

Uses p0_dev_bleu_along_trajectory.json (per-epoch dev BLEU-4/WER/NLL for 6 primary
seeds). For each seed: Pareto frontier over (dev NLL down, dev BLEU up, dev WER down);
best achievable dev BLEU under any selection objective; distance to competence gate
(|dev BLEU - 0.1338| <= 0.01 and |dev WER - 0.7749| <= 0.03).
"""
import json
from pathlib import Path

TRAJ = Path("/ssd/xkb4/RCP/revision_20260728_round4/results/p0_dev_bleu_along_trajectory.json")
OUT = Path("/ssd/xkb4/RCP/revision_20260729_round5/results/e8_pareto_selection.json")
ORIG = {"dev_bleu4": 0.1338, "dev_wer": 0.7749}
GATE = {"bleu": 0.01, "wer": 0.03}

rows = json.load(open(TRAJ))["rows"]
by_seed = {}
for r in rows:
    if r["seed"] is None:
        continue
    by_seed.setdefault(r["seed"], []).append(r)

report = {"gate": GATE, "original": ORIG, "seeds": {}}
all_epochs = []
for seed, rs in sorted(by_seed.items()):
    rs = sorted(rs, key=lambda r: r["epoch"])
    for r in rs:
        all_epochs.append(r)
    # Pareto frontier: minimize nll, maximize bleu, minimize wer
    pareto = []
    for r in rs:
        dominated = False
        for q in rs:
            if q is r:
                continue
            if (q["dev_nll"] <= r["dev_nll"] and q["dev_bleu4"] >= r["dev_bleu4"] and q["dev_wer"] <= r["dev_wer"]
                    and (q["dev_nll"] < r["dev_nll"] or q["dev_bleu4"] > r["dev_bleu4"] or q["dev_wer"] < r["dev_wer"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r["epoch"])
    best_bleu = max(rs, key=lambda r: r["dev_bleu4"])
    best_wer = min(rs, key=lambda r: r["dev_wer"])
    # selection by dev BLEU (decoded) instead of NLL: would pick best_bleu epoch
    report["seeds"][str(seed)] = {
        "n_saved_epochs": len(rs),
        "pareto_epochs": sorted(pareto),
        "nll_selected": next(({"epoch": r["epoch"], "dev_bleu4": r["dev_bleu4"], "dev_wer": r["dev_wer"], "dev_nll": r["dev_nll"]} for r in rs if r["is_best"]), None),
        "bleu_selected": {"epoch": best_bleu["epoch"], "dev_bleu4": best_bleu["dev_bleu4"], "dev_wer": best_bleu["dev_wer"], "dev_nll": best_bleu["dev_nll"]},
        "wer_selected": {"epoch": best_wer["epoch"], "dev_bleu4": best_wer["dev_bleu4"], "dev_wer": best_wer["dev_wer"], "dev_nll": best_wer["dev_nll"]},
        "max_dev_bleu4": best_bleu["dev_bleu4"],
        "min_dev_wer": best_wer["dev_wer"],
        "gate_reachable_bleu": best_bleu["dev_bleu4"] >= ORIG["dev_bleu4"] - GATE["bleu"],
        "gate_reachable_wer": best_wer["dev_wer"] <= ORIG["dev_wer"] + GATE["wer"],
    }

global_best_bleu = max(all_epochs, key=lambda r: r["dev_bleu4"])
global_best_wer = min(all_epochs, key=lambda r: r["dev_wer"])
report["global"] = {
    "max_dev_bleu4_any_epoch": {"seed": global_best_bleu["seed"], "epoch": global_best_bleu["epoch"], "value": global_best_bleu["dev_bleu4"]},
    "min_dev_wer_any_epoch": {"seed": global_best_wer["seed"], "epoch": global_best_wer["epoch"], "value": global_best_wer["dev_wer"]},
    "gate_reachable_by_any_selection": bool(
        global_best_bleu["dev_bleu4"] >= ORIG["dev_bleu4"] - GATE["bleu"]
        and global_best_wer["dev_wer"] <= ORIG["dev_wer"] + GATE["wer"]),
}
OUT.write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
