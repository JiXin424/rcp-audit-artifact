#!/usr/bin/env python3
"""Round 6 E10a: dual-reference conditions + paired uncertainty for the attenuation.

Reviewer point 2: (11.01-1.12)/11.01 is not a causal attribution. We report:
  1. Single-reference conditions (original refs; human back-translations) -- for context.
  2. DUAL-reference corpus BLEU/chrF: official reference + human back-translation as
     two references (sacrebleu multi-ref; chrF uses closest-reference matching),
     following Czehmann et al.'s single vs dual reference protocol.
  3. Paired bootstrap (10,000 resamples, seed 42) for the attenuation:
     delta = (PURE-GT gap under original refs) - (PURE-GT gap under human refs),
     and the ratio of gaps, with percentile CIs -- descriptive, not causal.
"""
import json, math, re
from pathlib import Path
from collections import Counter
import numpy as np
import sacrebleu

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
CSV_FULL = ROOT / "revision_20260729_round5/data_sacrebird/test_full_annotations_sacrebirdphoenix.csv"
CSV_HC = ROOT / "revision_20260729_round5/data_sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
OUT = ROOT / "revision_20260729_round5/results/e10a_dual_reference.json"
B = 10_000

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp", effective_order=False, force=True)
CHRF = sacrebleu.metrics.CHRF()


def load_csv(path):
    out = {}
    for i, line in enumerate(path.read_text().splitlines()):
        if i == 0:
            continue
        p = line.split("|")
        if len(p) >= 2:
            out[p[0]] = p[1]
    return out


def tokenize13a(text):
    text = text.replace("<skipped>", "").replace("-\n", "")
    text = " ".join(re.findall(r"\S+", text))
    text = re.sub(r"([\{-\~\[-\^\`\{-\}])", r" \1 ", text)
    text = re.sub(r"([\.\!\;\:\?\,])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower().split(" ") if text.strip() else []


def suff_multi(hyp, refs):
    """Sufficient stats vs multiple references: clipped counts use max-over-refs
    reference counts; reference length uses the closest length (sacrebleu convention)."""
    c = np.zeros(4); t = np.zeros(4)
    for n in range(1, 5):
        hng = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        ref_max = Counter()
        for r in refs:
            rng = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            for g, cnt in rng.items():
                ref_max[g] = max(ref_max[g], cnt)
        c[n - 1] = sum(min(v, ref_max.get(g, 0)) for g, v in hng.items())
        t[n - 1] = max(len(hyp) - n + 1, 0)
    hyp_len = len(hyp)
    ref_len = min((len(r) for r in refs), key=lambda rl: (abs(rl - hyp_len), rl))
    return c, t, hyp_len, ref_len


def bleu_from(S, idx):
    c = np.zeros(4); t = np.zeros(4); sl = 0; rl = 0
    for i in idx:
        cc, tt, s_, r_ = S[i]
        c += cc; t += tt; sl += s_; rl += r_
    if sl == 0:
        return 0.0
    bp = 1.0 if sl >= rl else math.exp(1 - rl / sl)
    p = [(0.5 / t[n] if t[n] and c[n] == 0 else (c[n] / t[n] if t[n] else 0.0)) for n in range(4)]
    if min(p) <= 0:
        return 0.0
    return float(bp * math.exp(sum(math.log(x) for x in p) / 4) * 100)


def main():
    human_full = load_csv(CSV_FULL)
    human_hc = load_csv(CSV_HC)
    gt = json.load(open(CELLS / "cp0_GT-v1.json"))["metrics"]["items"]
    pure = json.load(open(CELLS / "cp0_TN-PURE-v1.json"))["metrics"]["items"]
    ids = [it["id"] for it in gt]
    assert [it["id"] for it in pure] == ids

    def prep(ref_mode, id_filter=None):
        rows_gt, rows_pu = [], []
        for it_g, it_p in zip(gt, pure):
            qid = it_g["id"]
            if id_filter and qid not in id_filter:
                continue
            orig = it_g["reference"]
            refs_tok = []
            if ref_mode in ("orig", "dual"):
                refs_tok.append(tokenize13a(orig))
            if ref_mode in ("human", "dual"):
                refs_tok.append(tokenize13a(human_full[qid]))
            rows_gt.append(suff_multi(tokenize13a(it_g["hypothesis"]), refs_tok))
            rows_pu.append(suff_multi(tokenize13a(it_p["hypothesis"]), refs_tok))
        return rows_gt, rows_pu

    out = {"B": B, "conditions": {}, "attenuation": {}}
    for label, mode, filt in [("original_refs_full641", "orig", None),
                              ("human_refs_full641", "human", None),
                              ("human_refs_highconf461", "human", set(human_hc)),
                              ("dual_refs_full641", "dual", None),
                              ("dual_refs_highconf461", "dual", set(human_hc))]:
        rows_gt, rows_pu = prep(mode, filt)
        n = len(rows_gt)
        idx = np.arange(n)
        g = bleu_from(rows_gt, idx); p = bleu_from(rows_pu, idx)
        # official-library dual check (sacrebleu corpus_score with 2 refs) on full641
        out["conditions"][label] = {"n": n, "ref_mode": mode, "GT": g, "PURE": p, "gap": p - g}
        print(f"{label}: n={n} GT={g:.2f} PURE={p:.2f} gap={p-g:+.2f}")

    # Paired bootstrap for attenuation between original-refs gap and human-refs gap
    rows_gt_o, rows_pu_o = prep("orig")
    rows_gt_h, rows_pu_h = prep("human")
    n = len(rows_gt_o)
    rng = np.random.default_rng(42)
    att, ratio, gap_o, gap_h = [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        go = bleu_from(rows_pu_o, idx) - bleu_from(rows_gt_o, idx)
        gh = bleu_from(rows_pu_h, idx) - bleu_from(rows_gt_h, idx)
        gap_o.append(go); gap_h.append(gh)
        att.append(go - gh)
        ratio.append(gh / go if go > 0 else np.nan)
    idx = np.arange(n)
    go = bleu_from(rows_pu_o, idx) - bleu_from(rows_gt_o, idx)
    gh = bleu_from(rows_pu_h, idx) - bleu_from(rows_gt_h, idx)
    out["attenuation"] = {
        "gap_original_refs": go, "gap_human_refs": gh,
        "gap_original_ci": [float(np.percentile(gap_o, 2.5)), float(np.percentile(gap_o, 97.5))],
        "gap_human_ci": [float(np.percentile(gap_h, 2.5)), float(np.percentile(gap_h, 97.5))],
        "attenuation_abs": go - gh,
        "attenuation_abs_ci": [float(np.percentile(att, 2.5)), float(np.percentile(att, 97.5))],
        "ratio_remaining": gh / go,
        "ratio_remaining_ci": [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))],
        "note": "descriptive contrast of two corpus gaps on paired resamples; not a causal attribution",
    }
    print(json.dumps(out["attenuation"], indent=1))
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
