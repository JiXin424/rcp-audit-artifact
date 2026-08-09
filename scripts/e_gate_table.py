#!/usr/bin/env python3
"""Unified full-pool dev-set table under the uniform decoding protocol (C1 fix).

Merges the 30 full-readout decodes (train/dev/test, beam-3, alpha=-1, [::2],
sacreBLEU 13a/exp/effective_order=False) with the 23 dev-only decodes
(e_dev_uniform.py, identical protocol + official normalized WER), computing
official jiwer-normalized WER for the full-readout checkpoints offline from
their committed per-item hypotheses. Re-evaluates the competence gate
(|dev BLEU - 13.38| <= 1.0 and dev WER <= 86.37) under this single protocol.

Output: results/dev_gate_table.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path_ready = False


def corpus_wer(hyps, refs):
    """Official SLRTP2025 normalized WER (jiwer 3.1.0), imported lazily."""
    global sys_path_ready
    if not sys_path_ready:
        import sys
        sys.path.insert(0, str(ROOT))
        sys_path_ready = True
    from src.evaluation.bleu import corpus_wer as _cw
    return _cw(hyps, refs)["wer"]


# full_readout id -> paper id
FULL_ID_MAP = {
    f"seed_{s}": f"reco_{s}"
    for s in [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102,
              1203, 1304, 1405]
}
FULL_ID_MAP.update({
    f"alpha_{a}_seed_{s}": f"distill_a{a}_{s}"
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]
    for s in [101, 202, 303]
})
FULL_ID_MAP["seed_202_wd0"] = "rescue_wd0"
FULL_ID_MAP["backTranslation_PHIX_model"] = "released"

# dev_uniform dir name -> paper id (checkpoint source dir recorded in the file)
DEV_UNIFORM_MAP = {
    "seed_101": "cf_101",
    "seed_202": "cf_202",       # long-schedule seed_202 is byte-identical
    "seed_303": "cf_303",
    "seed_404": "cf_404",
    "seed_505": "sf_505",
    "seed_606": "sf_606",
    "seed_1506": "conf_1506",
    "seed_1607": "conf_1607",
    "frac_0125": "ladder_0125",
    "frac_025": "ladder_025",
    "frac_05": "ladder_05",
    "frac_075": "ladder_075",
    "backTranslation_PHIX_model": "released",
}
DEV_UNIFORM_MAP.update({
    f"lr{lr:.0e}_seed{sd}": f"ft_lr{lr:.0e}_seed{sd}"
    for lr in [1e-5, 3e-5, 5e-6]
    for sd in [42, 123, 789]
})

FAMILY = {
    "released": "released",
    "rescue_wd0": "rescue",
    **{f"reco_{s}": "reconstructions" for s in
       [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203,
        1304, 1405]},
    **{f"distill_a{a}_{s}": "distillation" for a in [0.0, 0.25, 0.5, 0.75, 1.0]
       for s in [101, 202, 303]},
    **{f"cf_{s}": "config_faithful" for s in [101, 202, 303, 404]},
    "sf_505": "step_faithful",
    "sf_606": "step_faithful",
    "conf_1506": "confirmation",
    "conf_1607": "confirmation",
    "ls_202": "long_schedule",
    "ladder_0125": "ladder",
    "ladder_025": "ladder",
    "ladder_05": "ladder",
    "ladder_075": "ladder",
    **{f"ft_lr{lr:.0e}_seed{sd}": "finetune" for lr in [1e-5, 3e-5, 5e-6]
       for sd in [42, 123, 789]},
}

RELEASED_DEV_BLEU = 13.38
RELEASED_DEV_WER = 83.37
D_BLEU = 1.0
D_WER = 3.0


def main():
    rows = {}

    # ---- full_readout (30 checkpoints) ----
    for f in sorted((ROOT / "results/full_readout").glob("*.json")):
        d = json.load(open(f))
        pid = FULL_ID_MAP.get(f.stem)
        if pid is None:
            continue
        dev = d["splits"]["dev"]
        hyps = [it["hyp"] for it in dev["per_item"]]
        refs = [it["ref"] for it in dev["per_item"]]
        wer = corpus_wer(hyps, refs)
        rows[pid] = {
            "id": pid, "family": FAMILY[pid],
            "source": f"full_readout/{f.name}",
            "dev_bleu": dev["bleu"], "dev_em": dev["em"],
            "dev_wer": float(wer),
            "train_bleu": d["splits"]["train"]["bleu"],
            "test_bleu": d["splits"]["test"]["bleu"],
        }

    # ---- dev_uniform (23 decodes; cf_202 == ls_202 same binary) ----
    for f in sorted((ROOT / "results/dev_uniform").glob("*.json")):
        d = json.load(open(f))
        pid = DEV_UNIFORM_MAP.get(f.stem)
        if pid is None:
            continue
        row = {
            "id": pid, "family": FAMILY[pid],
            "source": f"dev_uniform/{f.name}",
            "dev_bleu": d["bleu"], "dev_em": d["em"],
            "dev_wer": d["wer"],
        }
        if pid in rows:
            # overlap (released): keep the full_readout row, record dev_uniform
            # agreement as a cross-path consistency check
            rows[pid]["dev_uniform_agreement"] = {
                "bleu": d["bleu"], "em": d["em"], "wer": d["wer"],
            }
        else:
            rows[pid] = row
    # long-schedule seed-202 shares the cf_202 binary
    rows["ls_202"] = {
        "id": "ls_202", "family": "long_schedule",
        "source": "dev_uniform/seed_202.json (byte-identical to cf_202; "
                  "SHA-256 911de1fe...830c7)",
        "dev_bleu": rows["cf_202"]["dev_bleu"],
        "dev_em": rows["cf_202"]["dev_em"],
        "dev_wer": rows["cf_202"]["dev_wer"],
    }

    # ---- gate ----
    for r in rows.values():
        r["gate_bleu"] = abs(r["dev_bleu"] - RELEASED_DEV_BLEU) <= D_BLEU
        r["gate_wer"] = r["dev_wer"] - RELEASED_DEV_WER <= D_WER
        r["gate_pass"] = r["gate_bleu"] and r["gate_wer"]
        r["bleu_shortfall"] = RELEASED_DEV_BLEU - r["dev_bleu"]
        r["wer_excess"] = r["dev_wer"] - (RELEASED_DEV_WER + D_WER)

    # ---- summaries ----
    non_released = [r for r in rows.values() if r["id"] != "released"]
    # recipe-constructed = the 52 non-distillation runs with dev metrics
    # (registry: 14 reco + 8 rescue-lr + 12 rescue-expanded + 4 ladder +
    # 4 config-faithful + 2 step-faithful + 4 large-arch + 2 confirmation +
    # 2 long-schedule); excludes the 9 released-weight fine-tunes and the
    # 15 distillation students (teacher-distilled from released weights).
    # 28 of the 52 have released weights and are re-decoded here under the
    # uniform protocol; the remaining 24 (rescue-lr 8, rescue-expanded 11,
    # large-arch 4, long-schedule second seed 1) are withheld from the
    # artifact and retain their training-log dev metrics (max dev BLEU-4
    # 10.02, long-schedule best seed; none passes the gate).
    recipe = [r for r in non_released if r["family"] != "finetune"
              and r["family"] != "distillation"]
    dev_bleus = np.array([r["dev_bleu"] for r in non_released])

    gate = {
        "recipe_constructed_n_registry": 52,
        "recipe_constructed_n_decodable_uniform": len(recipe),
        "recipe_constructed_n_pass_gate": sum(r["gate_pass"] for r in recipe),
        "recipe_constructed_n_pass_bleu_half": sum(
            r["gate_bleu"] for r in recipe),
        "recipe_constructed_n_pass_wer_half": sum(
            r["gate_wer"] for r in recipe),
        "recipe_constructed_bleu_shortfall_range": [
            round(min(r["bleu_shortfall"] for r in recipe), 2),
            round(max(r["bleu_shortfall"] for r in recipe), 2)],
        "recipe_constructed_wer_excess_range": [
            round(min(r["wer_excess"] for r in recipe), 2),
            round(max(r["wer_excess"] for r in recipe), 2)],
    }
    # full-gate grid over the 52 recipe-constructed checkpoints
    grid = {}
    for dB in [0.5, 1.0, 1.5, 2.0]:
        for dW in [1.5, 3.0, 4.5]:
            n = sum(
                1 for r in recipe
                if abs(r["dev_bleu"] - RELEASED_DEV_BLEU) <= dB
                and r["dev_wer"] - RELEASED_DEV_WER <= dW
            )
            grid[f"{dB}/{dW}"] = n
    gate["grid_sensitivity_n_pass_recipe_constructed"] = grid

    # finetune gate status (released-weight perturbations, SI Sup. F)
    finetunes = [r for r in non_released if r["family"] == "finetune"]
    gate["finetune_gate"] = {
        "n": len(finetunes),
        "n_pass_gate": sum(r["gate_pass"] for r in finetunes),
        "dev_bleu_range": [round(min(r["dev_bleu"] for r in finetunes), 2),
                           round(max(r["dev_bleu"] for r in finetunes), 2)],
        "dev_wer_range": [round(min(r["dev_wer"] for r in finetunes), 2),
                          round(max(r["dev_wer"] for r in finetunes), 2)],
    }
    gate["withheld_24_training_log_dev_max_bleu"] = 10.02
    gate["withheld_24_note"] = ("24 registered recipe-constructed runs lack "
                                "released weights (artifact withholds them): "
                                "rescue-lr 8, rescue-expanded 11, large-arch "
                                "4, long-schedule second seed 1. Their dev "
                                "metrics come from training logs under the "
                                "original selection protocol (max dev BLEU-4 "
                                "10.02); none passes the gate.")

    families = {}
    for r in non_released:
        fam = r["family"]
        fam_rows = families.setdefault(fam, [])
        fam_rows.append(r)
    fam_summary = {}
    for fam, fam_rows in families.items():
        fam_summary[fam] = {
            "n": len(fam_rows),
            "dev_bleu_range": [round(min(x["dev_bleu"] for x in fam_rows), 2),
                               round(max(x["dev_bleu"] for x in fam_rows), 2)],
            "dev_wer_range": [round(min(x["dev_wer"] for x in fam_rows), 2),
                              round(max(x["dev_wer"] for x in fam_rows), 2)],
        }

    out = {
        "protocol": "full-pool dev (515 items), beam=3, alpha=-1, [::2] "
                    "subsample, sacrebleu 13a/exp/effective_order=False, "
                    "official jiwer 3.1.0 normalized WER",
        "released_dev": {"bleu": RELEASED_DEV_BLEU, "wer": RELEASED_DEV_WER,
                         "em": rows["released"]["dev_em"]},
        "gate_definition": f"|dev BLEU - {RELEASED_DEV_BLEU}| <= {D_BLEU} "
                           f"and dev WER <= {RELEASED_DEV_WER + D_WER}",
        "n_checkpoints": len(rows),
        "n_pass_gate": sum(1 for r in non_released if r["gate_pass"]),
        "n_pass_bleu_half": sum(1 for r in non_released if r["gate_bleu"]),
        "n_pass_wer_half": sum(1 for r in non_released if r["gate_wer"]),
        "non_released_dev_bleu_min": float(dev_bleus.min()),
        "non_released_dev_bleu_max": float(dev_bleus.max()),
        "bleu_shortfall_range": [round(min(r["bleu_shortfall"] for r in
                                           non_released), 2),
                                 round(max(r["bleu_shortfall"] for r in
                                           non_released), 2)],
        "family_summary": fam_summary,
        "gate": gate,
        "checkpoints": {r["id"]: r for r in sorted(rows.values(),
                                                   key=lambda x: x["id"])},
    }

    dest = ROOT / "results/dev_gate_table.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "checkpoints"},
                     indent=1))


if __name__ == "__main__":
    main()
