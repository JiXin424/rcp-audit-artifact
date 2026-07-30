#!/usr/bin/env python3
"""Round 5 E1 analysis: full 2x2 factorial report at the corrected 12.5 fps protocol.

Per evaluator: four cell means (corpus BLEU from additive sufficient statistics;
token-weighted NLL), seen/unseen main effect, similarity main effect, interaction.
Uncertainty: paired item bootstrap (10,000 resamples, seed 42) per evaluator for the
original evaluator; seed-level mean/t-CI across the six reconstructions.
Holm correction across the three contrasts per metric.
Also recomputes the legacy 25 fps interaction from task7 shards for the protocol-correction note.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path("/ssd/xkb4/RCP/revision_20260729_round5")
CELL_DIR = ROOT / "results/e1_factorial_cells_12p5fps"
TASK7 = Path("/ssd/xkb4/RCP/revision_20260728_major/results/task7_canonical_v2")
OUT = ROOT / "results/e1_factorial_full_report.json"
CELLS = ["seen_high", "seen_low", "unseen_high", "unseen_low"]
B = 10_000
SEED = 42


def item_stats(cell):
    """Additive per-item sufficient statistics."""
    bleu_counts, bleu_totals = [], []
    sys_len, ref_len = [], []
    nll_sum, tok = [], []
    for it in cell["items"]:
        sb = it["segment_bleu"]
        bleu_counts.append(sb["counts"]); bleu_totals.append(sb["totals"])
        sys_len.append(sb["system_length"]); ref_len.append(sb["reference_length"])
        nll_sum.append(it["nll_sum"]); tok.append(it["token_count"])
    return {"c": np.array(bleu_counts), "t": np.array(bleu_totals),
            "sys": np.array(sys_len, dtype=float), "ref": np.array(ref_len, dtype=float),
            "nll": np.array(nll_sum, dtype=float), "tok": np.array(tok, dtype=float)}


def corpus_bleu(idx, st):
    c = st["c"][idx].sum(0); t = st["t"][idx].sum(0)
    p = c / t
    if (p <= 0).any():
        return 0.0
    sysl = st["sys"][idx].sum(); refl = st["ref"][idx].sum()
    bp = 1.0 if sysl > refl else math.exp(1 - refl / max(sysl, 1e-9))
    return float(bp * math.exp(np.log(p).mean()) * 100)


def tok_nll(idx, st):
    return float(st["nll"][idx].sum() / max(st["tok"][idx].sum(), 1))


def contrasts(vals):
    sh, sl, uh, ul = vals
    return {"seen_main": (sh + sl) / 2 - (uh + ul) / 2,
            "sim_main": (sh + uh) / 2 - (sl + ul) / 2,
            "interaction": (sh - sl) - (uh - ul)}


def bootstrap_eval(cells_st, metric):
    n = len(cells_st["seen_high"]["sys"])
    rng = np.random.default_rng(SEED)
    fn = corpus_bleu if metric == "bleu" else tok_nll
    point = [fn(np.arange(n), cells_st[c]) for c in CELLS]
    point_c = contrasts(point)
    boots = {k: [] for k in point_c}
    cells_b = {k: [] for k in CELLS}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals = [fn(idx, cells_st[c]) for c in CELLS]
        cc = contrasts(vals)
        for k in boots:
            boots[k].append(cc[k])
        for i, cn in enumerate(CELLS):
            cells_b[cn].append(vals[i])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    pvals = {}
    for k, v in boots.items():
        v = np.asarray(v)
        # two-sided bootstrap p with (b+1)/(B+1) correction
        b_side = min(int((v <= 0).sum()), int((v >= 0).sum()))
        pvals[k] = min(1.0, (2 * b_side + 1) / (B + 1))
    cell_ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in cells_b.items()}
    return point, point_c, ci, pvals, cell_ci


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    m = len(order)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def main():
    report = {"protocol": "12.5 fps corrected", "n_items": 605, "B": B, "evaluators": {}}
    seed_contrasts = {"bleu": {"seen_main": [], "sim_main": [], "interaction": []},
                      "nll": {"seen_main": [], "sim_main": [], "interaction": []}}
    for path in sorted(CELL_DIR.glob("cells_*.json")):
        name = json.load(open(path))["evaluator"]
        d = json.load(open(path))
        st = {c: item_stats(d["conditions"][c]) for c in CELLS}
        for metric in ["bleu", "nll"]:
            point, pc, ci, pvals, cell_ci = bootstrap_eval(st, metric)
            report["evaluators"].setdefault(name, {})[metric] = {
                "cell_means": dict(zip(CELLS, point)), "cell_ci": cell_ci,
                "contrasts": pc, "contrast_ci": ci, "contrast_p": pvals,
                "contrast_p_holm": holm(pvals)}
            if name.startswith("seed_"):
                for k in seed_contrasts[metric]:
                    seed_contrasts[metric][k].append(pc[k])
        print(f"{name}: done")
    # seed-level summary
    seed_summary = {}
    from scipy import stats as sst
    for metric in seed_contrasts:
        seed_summary[metric] = {}
        for k, v in seed_contrasts[metric].items():
            v = np.asarray(v)
            t = sst.t.ppf(0.975, len(v) - 1)
            seed_summary[metric][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                       "ci": [float(v.mean() - t * v.std(ddof=1) / np.sqrt(len(v))),
                                              float(v.mean() + t * v.std(ddof=1) / np.sqrt(len(v)))],
                                       "n_seeds": len(v)}
    report["seed_level_summary"] = seed_summary

    # Legacy 25 fps interactions from task7 shards (for the correction note)
    legacy = {}
    for shard in sorted(TASK7.glob("score_seed_*.json")):
        if shard.name.endswith(".provenance.json"):
            continue
        x = json.load(open(shard))
        by = {c["condition"]: c["metrics"] for c in x["conditions"]}
        bleu_v = [by[c]["decoded_bleu"] for c in CELLS]
        nll_v = [by[c]["teacher_forced_nll_per_token"] for c in CELLS]
        legacy[str(x["seed"])] = {"bleu_interaction": contrasts(bleu_v)["interaction"],
                                  "nll_interaction": contrasts(nll_v)["interaction"],
                                  "bleu_seen_main": contrasts(bleu_v)["seen_main"],
                                  "nll_seen_main": contrasts(nll_v)["seen_main"]}
    report["legacy_25fps_interactions"] = legacy
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({"seed_level_summary": seed_summary,
                      "original_bleu": report["evaluators"]["original"]["bleu"]["contrasts"],
                      "original_nll": report["evaluators"]["original"]["nll"]["contrasts"]}, indent=1))


if __name__ == "__main__":
    main()
