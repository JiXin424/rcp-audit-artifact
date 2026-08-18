#!/usr/bin/env python3
"""Readout-gap Spearman correlation over the CURRENT uniform-readout set (round 33).

Replaces the stale 31-point values (rho=-0.082 readout~gap) that predate the
eight faithful full-pool readouts. The current set is 38 constructible
checkpoints with uniform full-pool readouts (results/full_readout_summary.json)
plus the released evaluator (78.8), i.e. the "39 checkpoints" of main text
Sec. 4.2; rescue_wd0 has no canonical gap, so gap correlations use 37
constructible + released = 38 points.

Outputs Spearman correlations under four counting conventions, plus
readout~dev and gap~dev using dev_gate_table.json, and writes
results/readout_gap_correlation.json.

No GPU needed.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
READOUT = ROOT / "results/full_readout_summary.json"
REGISTRY = ROOT / "results/canonical_checkpoint_registry.json"
DEVGATE = ROOT / "results/dev_gate_table.json"
OUT = ROOT / "results/readout_gap_correlation.json"

RELEASED_READOUT = 78.81879546693305   # backTranslation_PHIX_model train BLEU
RELEASED_GAP = 10.24331582979233       # canonical panel released gap
RELEASED_DEV = 13.38


def map_key(k: str) -> str:
    """full_readout_summary key -> canonical registry run_id."""
    if k.startswith("alpha_"):
        a, s = k.split("_seed_")
        return f"distill_a{a[6:]}_s{s}"
    if k == "seed_202_wd0":
        return "rescue_wd0"
    if k.startswith("seed_"):
        n = k.split("_")[1]
        return f"faithful_{n}" if int(n) < 100 else f"reco_{n}"
    return k


def main():
    ro = json.load(open(READOUT))
    reg = json.load(open(REGISTRY))
    cps = {c["run_id"]: c for c in reg["checkpoints"]}

    # dev BLEU join: dev_gate_table (incl. faithful), dev_uniform (legacy/
    # sf/ladder), new_distill_dev_bleu4 (distills) -- first hit wins.
    dev_by_key = {}
    devg = json.load(open(DEVGATE))["checkpoints"]
    for rid, e in devg.items():
        dev_by_key[rid] = e["dev_bleu"]
    for f in sorted((ROOT / "results/dev_uniform").glob("*.json")):
        d = json.load(open(f))
        k = f.stem
        if k == "backTranslation_PHIX_model":
            k = "released"
        if k not in dev_by_key:
            dev_by_key[k] = d.get("bleu")
    try:
        nd = json.load(open(ROOT / "results/new_distill_dev_bleu4.json"))
        for k, e in nd.items():
            if k not in dev_by_key:
                dev_by_key[k] = e["dev_bleu4"]
    except FileNotFoundError:
        pass

    def dev_for(rid, raw_key):
        for cand in (rid, raw_key):
            if cand in dev_by_key and dev_by_key[cand] is not None:
                return dev_by_key[cand]
        return None

    rows = []
    for k, v in ro.items():
        rid = map_key(k)
        c = cps[rid]
        rows.append({"run_id": rid, "readout": v["train"]["bleu"],
                     "gap": c["gap"], "dev_bleu": dev_for(rid, k),
                     "degenerate": c["degenerate"]})
    n_readout = len(rows)
    rows.append({"run_id": "released", "readout": RELEASED_READOUT,
                 "gap": RELEASED_GAP, "dev_bleu": RELEASED_DEV,
                 "degenerate": False})

    def sp(sub, a, b):
        sub = [r for r in sub if r[a] is not None and r[b] is not None]
        if len(sub) < 3:
            return None
        rho, p = spearmanr([r[a] for r in sub], [r[b] for r in sub])
        return {"rho": round(float(rho), 4), "p": float(p), "n": len(sub)}

    cons = all_rows = rows
    excl_rel = [r for r in rows if r["run_id"] != "released"]
    nd_all = [r for r in rows if not r["degenerate"]]
    nd_excl = [r for r in excl_rel if not r["degenerate"]]

    out = {
        "schema": "readout-gap-correlation-v1",
        "generated_by": "scripts/e_readout_gap_correlation.py",
        "note": ("Current-set Spearman correlations for SI tab:readout_assoc. "
                 "Supersedes the 31-point table (rho=-0.082 readout~gap) which "
                 "predates the eight faithful full-pool readouts. rescue_wd0 "
                 "has a uniform readout but no canonical gap (not in the 74-run "
                 "decoded gap panel), so it enters readout~dev only."),
        "n_readout_entries_constructible": n_readout,
        "conventions": {
            "all_with_gap_incl_released": "released + constructible with canonical gap",
            "excl_released": "constructible only",
            "non_degenerate_incl_released": "adds released, drops 3 empty-output distill students",
            "non_degenerate_excl_released": "constructible non-degenerate only",
        },
        "correlations": {
            "readout_vs_gap": {
                "all_incl_released": sp(all_rows, "readout", "gap"),
                "excl_released": sp(excl_rel, "readout", "gap"),
                "non_degenerate_incl_released": sp(nd_all, "readout", "gap"),
                "non_degenerate_excl_released": sp(nd_excl, "readout", "gap"),
            },
            "readout_vs_dev_bleu": {
                "all_incl_released": sp(all_rows, "readout", "dev_bleu"),
                "excl_released": sp(excl_rel, "readout", "dev_bleu"),
                "non_degenerate_incl_released": sp(nd_all, "readout", "dev_bleu"),
                "non_degenerate_excl_released": sp(nd_excl, "readout", "dev_bleu"),
            },
            "gap_vs_dev_bleu": {
                "all_incl_released": sp(all_rows, "gap", "dev_bleu"),
                "excl_released": sp(excl_rel, "gap", "dev_bleu"),
                "non_degenerate_incl_released": sp(nd_all, "gap", "dev_bleu"),
                "non_degenerate_excl_released": sp(nd_excl, "gap", "dev_bleu"),
            },
        },
        "interpretation_note": (
            "Within the constructible family the readout~gap association is "
            "NEGATIVE (higher-readout checkpoints have more negative gaps), "
            "and the released evaluator is the single positive-gap point far "
            "outside the family trend on both axes. The data therefore support "
            "co-occurrence in one checkpoint plus an uncovered readout region, "
            "not 'the probe response tracks the signature'."
        ),
        "rows": rows,
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)

    print(f"Constructible readout entries: {n_readout} (+released)")
    for pair, d in out["correlations"].items():
        print(f"\n{pair}:")
        for conv, v in d.items():
            if v:
                print(f"  {conv:36s} rho={v['rho']:+.3f}  p={v['p']:.3g}  n={v['n']}")
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
