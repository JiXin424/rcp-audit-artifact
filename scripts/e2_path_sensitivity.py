#!/usr/bin/env python3
"""E2 analysis: alternative telescoping paths + Shapley-style averaging (original evaluator).

Cells on the 622-query common support (corpus BLEU from sufficient statistics):
  S1 SEEN-PURE-v1 (7060, max), S2 SEEN-PURE-RAND640-v1 (640, max),
  S3 SEEN-PURE-MATCHED-v1 (7060, match), S4 SEEN-RAND640-MATCHED-v1 (640, match),
  U  UNSEEN-PURE-v1 (test640, max).
Path 1 (size->selection->origin):  S1 -> S2 -> S4 -> U
Path 2 (selection->size->origin):  S1 -> S3 -> S4 -> U
Canonical paper path (for contrast): S1 -> S2 -> S3 -> U  (C conflates size+selection)
Shapley averages: size = mean(S1-S2, S3-S4); selection = mean(S2-S4, S1-S3); origin = S4-U.
10,000-resample query-index bootstrap CIs for every contrast.
"""
import json, math, re, sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path("/ssd/xkb4/RCP")
EVAL_JSON = ROOT / "revision_20260728_round4/results/r5_common_support_eval.json"
S4_JSON = ROOT / "revision_20260729_round5/results/decoded/SEEN-RAND640-MATCHED-v1__original.json"
OUT = ROOT / "revision_20260729_round5/results/e2_path_sensitivity.json"
B = 10_000


def tokenize13a(text):
    text = text.replace("<skipped>", "").replace("-\n", "")
    text = " ".join(re.findall(r"\S+", text))
    text = re.sub(r"([\{-\~\[-\^\`\{-\}])", r" \1 ", text)
    text = re.sub(r"([\.\!\;\:\?\,])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower().split(" ") if text.strip() else []


def suff(hyp, ref):
    c = np.zeros(4); t = np.zeros(4)
    for n in range(1, 5):
        rng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hng = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        c[n - 1] = sum(min(v, rng.get(g, 0)) for g, v in hng.items())
        t[n - 1] = max(len(hyp) - n + 1, 0)
    return c, t, len(hyp), len(ref)


def bleu(C, T, SL, RL):
    c = C.sum(0); t = T.sum(0); sl = SL.sum(); rl = RL.sum()
    if sl == 0:
        return 0.0
    bp = 1.0 if sl >= rl else math.exp(1 - rl / sl)
    p = [(0.5 / t[n] if t[n] and c[n] == 0 else (c[n] / t[n] if t[n] else 0.0)) for n in range(4)]
    if min(p) <= 0:
        return 0.0
    return float(bp * math.exp(sum(math.log(x) for x in p) / 4) * 100)


def main():
    d = json.loads(EVAL_JSON.read_text())
    common_ids = d["common_support_ids"]
    id_set = set(common_ids)
    by_es = {(r["evaluator"], r["system"]): r["items"] for r in d["rows"]}
    s4 = json.load(open(S4_JSON))
    s4_items = [it for it in s4["items"] if it["id"] in id_set]
    s4_items.sort(key=lambda x: common_ids.index(x["id"]))

    systems = {"S1": by_es[("original", "SEEN-PURE-v1")],
               "S2": by_es[("original", "SEEN-PURE-RAND640-v1")],
               "S3": by_es[("original", "SEEN-PURE-MATCHED-v1")],
               "S4": s4_items,
               "U": by_es[("original", "UNSEEN-PURE-v1")]}
    for name, items in systems.items():
        assert [it["id"] for it in items] == common_ids, name

    stats = {}
    for name, items in systems.items():
        n = len(items)
        C = np.zeros((n, 4)); T = np.zeros((n, 4)); SL = np.zeros(n); RL = np.zeros(n)
        for i, it in enumerate(items):
            c, t, sl, rl = suff(tokenize13a(it["hypothesis"]), tokenize13a(it["reference"]))
            C[i], T[i], SL[i], RL[i] = c, t, sl, rl
        stats[name] = (C, T, SL, RL)

    n = len(common_ids)
    rng = np.random.default_rng(42)
    contrasts = {
        "path1_size(S1-S2)": ("S1", "S2"),
        "path1_selection(S2-S4)": ("S2", "S4"),
        "path1_origin(S4-U)": ("S4", "U"),
        "path2_selection(S1-S3)": ("S1", "S3"),
        "path2_size(S3-S4)": ("S3", "S4"),
        "path2_origin(S4-U)": ("S4", "U"),
        "canonical_C(S2-S3)": ("S2", "S3"),
        "canonical_D(S3-U)": ("S3", "U"),
        "shapley_size": "avg",
        "shapley_selection": "avg2",
        "shapley_origin": ("S4", "U"),
        "total_A(S1-U)": ("S1", "U"),
    }
    point = {}
    boots = {k: np.zeros(B) for k in contrasts}

    def sys_bleu(name, idx):
        C, T, SL, RL = stats[name]
        return bleu(C[idx], T[idx], SL[idx], RL[idx])

    def compute(idx):
        v = {k: sys_bleu(k, idx) for k in stats}
        out = {}
        for k, spec in contrasts.items():
            if spec == "avg":
                out[k] = ((v["S1"] - v["S2"]) + (v["S3"] - v["S4"])) / 2
            elif spec == "avg2":
                out[k] = ((v["S2"] - v["S4"]) + (v["S1"] - v["S3"])) / 2
            else:
                out[k] = v[spec[0]] - v[spec[1]]
        out["_cells"] = v
        return out

    pt = compute(np.arange(n))
    point = {k: pt[k] for k in contrasts}
    cells = pt["_cells"]
    for b in range(B):
        idx = rng.integers(0, n, n)
        r = compute(idx)
        for k in contrasts:
            boots[k][b] = r[k]
    out = {"n": n, "cells_bleu": cells, "contrasts": {}}
    for k in contrasts:
        out["contrasts"][k] = {"value": float(point[k]),
                               "ci95": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]}
    # S4 match-quality audit
    reg = [json.loads(l) for l in open(ROOT / "revision_20260729_round5/results/e2_seen_rand640_matched_registry.jsonl")]
    dj = [r["jaccard_abs_diff"] for r in reg]
    out["s4_match_quality"] = {"n": len(reg), "mean_abs_dj": float(np.mean(dj)),
                               "median_abs_dj": float(np.median(dj)), "max_abs_dj": float(np.max(dj)),
                               "frac_dj_gt_0p1": float(np.mean([x > 0.1 for x in dj]))}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
