#!/usr/bin/env python3
"""Paired per-item reference-frame visualization [reviewer R2-6].

On the matched 461-item confidence=1 subset, plots per-item sentence-level
sacreBLEU under original vs human references for REC (recorded test poses) and
PURE (TN-PURE-v1 replay), plus the per-item paired gap under the two reference
frames. Uses the canonical released-evaluator hypotheses
(results/gap_43_canonical_beam3_items/released_{gt,pure}.json).

Output: generated_figures/ref_frame_paired.pdf and
        results/ref_frame_paired_items.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "results/gap_43_canonical_beam3_items"
CSV_HC = ROOT / "data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv"
OUT_FIG = ROOT / "generated_figures/ref_frame_paired.pdf"
OUT_JSON = ROOT / "results/ref_frame_paired_items.json"

BLEU = sacrebleu.metrics.BLEU(tokenize="13a", smooth_method="exp",
                              effective_order=False, force=True)


def load_hc(path):
    out = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("|")
        if len(parts) < 3:
            continue
        try:
            conf = float(parts[2])
        except ValueError:
            continue
        out[parts[0]] = (parts[1], conf)
    return out


def sent_bleu(hyp, ref):
    return BLEU.sentence_score(hyp, [ref]).score


def main():
    gt = {x["id"]: x for x in json.load(open(ITEMS / "released_gt.json"))}
    pure = {x["id"]: x for x in json.load(open(ITEMS / "released_pure.json"))}
    hc = load_hc(CSV_HC)
    ids = sorted(set(gt) & set(pure) & {k for k, (_, c) in hc.items() if c == 1.0})
    assert len(ids) == 461, len(ids)

    rows = []
    for i in ids:
        ref_o = gt[i]["reference"]
        ref_h = hc[i][0]
        rows.append({
            "id": i,
            "rec_orig": sent_bleu(gt[i]["hypothesis"], ref_o),
            "rec_human": sent_bleu(gt[i]["hypothesis"], ref_h),
            "pure_orig": sent_bleu(pure[i]["hypothesis"], ref_o),
            "pure_human": sent_bleu(pure[i]["hypothesis"], ref_h),
        })
    for r in rows:
        r["gap_orig"] = r["pure_orig"] - r["rec_orig"]
        r["gap_human"] = r["pure_human"] - r["rec_human"]

    OUT_JSON.write_text(json.dumps({
        "schema": "ref-frame-paired-v1",
        "n": len(rows),
        "sentence_bleu": "sacrebleu 2.5.1, tok=13a, smooth=exp, "
                         "effective_order=False, sentence-level",
        "items": rows}, ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    ax = axes[0]
    ax.scatter([r["rec_orig"] for r in rows], [r["rec_human"] for r in rows],
               s=8, alpha=0.45, label="REC (recorded test poses)", color="#b2182b")
    ax.scatter([r["pure_orig"] for r in rows], [r["pure_human"] for r in rows],
               s=8, alpha=0.45, label="PURE (replay probe)", color="#2166ac")
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlabel("sentence BLEU vs original reference")
    ax.set_ylabel("sentence BLEU vs human reference")
    ax.set_title("(a) Per-item scores, matched 461 items")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    ax = axes[1]
    ax.scatter([r["gap_orig"] for r in rows], [r["gap_human"] for r in rows],
               s=8, alpha=0.45, color="#4d9221")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("per-item gap (PURE $-$ REC) vs original reference")
    ax.set_ylabel("per-item gap (PURE $-$ REC) vs human reference")
    ax.set_title("(b) Per-item paired gap attenuation")
    mo = sum(r["gap_orig"] for r in rows) / len(rows)
    mh = sum(r["gap_human"] for r in rows) / len(rows)
    ax.scatter([mo], [mh], marker="*", s=220, color="black", zorder=5,
               label=f"mean ({mo:.1f}, {mh:.1f})")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"saved {OUT_FIG}")
    print(f"mean gap orig {mo:.2f} -> human {mh:.2f}")


if __name__ == "__main__":
    main()
