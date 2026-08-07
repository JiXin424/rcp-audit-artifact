#!/usr/bin/env python3
"""Summarize + plot experiment A1 (released-checkpoint weight perturbation)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "results/released_perturbation.json"
OUT_FIG = ROOT / "generated_figures/released_perturbation.pdf"
OUT_JSON = ROOT / "results/released_perturbation_summary.json"


def main():
    d = json.load(open(IN))
    recs = d["records"]
    # group by sigma
    by_sigma = {}
    for r in recs:
        s = r["sigma"]
        by_sigma.setdefault(s, []).append(r)

    rows = []
    for s in sorted(by_sigma):
        rs = by_sigma[s]
        gaps = [r["gap"] for r in rs]
        devs = [r["dev_bleu"] for r in rs]
        gts = [r["gt_bleu"] for r in rs]
        purs = [r["pure_bleu"] for r in rs]
        row = {
            "sigma": s, "n": len(rs),
            "gap_mean": float(np.mean(gaps)),
            "gap_min": float(min(gaps)), "gap_max": float(max(gaps)),
            "dev_mean": float(np.mean(devs)),
            "gt_mean": float(np.mean(gts)),
            "pure_mean": float(np.mean(purs)),
        }
        # train readout only at sigma==0 or seed 101 entries
        trs = [r for r in rs if "train_bleu" in r]
        if trs:
            row["train_bleu"] = float(np.mean([r["train_bleu"] for r in trs]))
            row["train_em"] = float(np.mean([r["train_em"] for r in trs]))
        rows.append(row)

    OUT_JSON.write_text(json.dumps({"schema": "perturbation-summary-v1",
                                    "rows": rows}, indent=1))
    for r in rows:
        print(f"sigma={r['sigma']:.0e}  gap={r['gap_mean']:+.2f} "
              f"[{r['gap_min']:+.2f},{r['gap_max']:+.2f}]  "
              f"dev={r['dev_mean']:.2f}  gt={r['gt_mean']:.2f}  "
              f"pure={r['pure_mean']:.2f}"
              + (f"  train={r.get('train_bleu',float('nan')):.2f}"
                 f"/{r.get('train_em',float('nan'))*100:.1f}%"
                 if "train_bleu" in r else ""))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    sig = [r["sigma"] for r in rows]
    gap_m = [r["gap_mean"] for r in rows]
    gap_lo = [r["gap_mean"] - r["gap_min"] for r in rows]
    gap_hi = [r["gap_max"] - r["gap_mean"] for r in rows]
    dev_m = [r["dev_mean"] for r in rows]
    gt_m = [r["gt_mean"] for r in rows]
    pure_m = [r["pure_mean"] for r in rows]
    tr_m = [r.get("train_bleu", np.nan) for r in rows]

    ax1.errorbar(sig, gap_m, yerr=[gap_lo, gap_hi], marker="o", lw=2,
                 capsize=4, color="#b2182b", label="PURE$-$REC gap")
    ax1.axhline(0, color="k", lw=0.6, ls="--")
    ax1.set_xscale("symlog", linthresh=1e-4)
    ax1.set_xlabel("weight noise scale $\\sigma$")
    ax1.set_ylabel("PURE $-$ REC gap (BLEU)")
    ax1.set_title("(a) Reversal vs weight perturbation")
    ax1.legend(loc="lower left", frameon=False)
    ax1.grid(True, alpha=0.3)

    ax2.plot(sig, dev_m, marker="s", lw=2, color="#2166ac", label="dev BLEU-4")
    ax2.plot(sig, gt_m, marker="o", lw=1.5, color="#4daf4a", label="REC BLEU-4")
    ax2.plot(sig, pure_m, marker="^", lw=1.5, color="#ff7f00", label="PURE BLEU-4")
    ax2.plot(sig, tr_m, marker="d", lw=2, color="#984ea3",
             label="train-pool readout BLEU")
    ax2.set_xscale("symlog", linthresh=1e-4)
    ax2.set_xlabel("weight noise scale $\\sigma$")
    ax2.set_ylabel("BLEU-4")
    ax2.set_title("(b) Competence and readout vs $\\sigma$")
    ax2.legend(loc="lower left", frameon=False)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"\nsaved -> {OUT_FIG}")


if __name__ == "__main__":
    main()
