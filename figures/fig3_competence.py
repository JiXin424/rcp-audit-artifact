#!/usr/bin/env python3
"""Fig. 3: the released evaluator lies outside the reconstructible family.

Panel (a): training trajectories on the dev set (optimizer step axis).
  Released evaluator: full 202-point trajectory from its archived validations.txt
  (best at step 1,820, dev BLEU-4 13.38). Reconstructions: 14 seeds, decoded at
  epochs 25/50 (steps 700/1400) plus best.ckpt (step 2,828), shown as a
  median band with per-seed best points (results/full_readout_summary.json).

Panel (b): canonical gap vs decoded dev BLEU-4 by family.
  gaps from results/gap_43_canonical_beam3.json; dev BLEU-4 from
  full_readout_summary.json / epoch_ckpt_dev_bleu_all.json / epoch_decouple.json
  (uniform full-pool beam-3 protocol);
  released dev = 13.38 from validations.txt. Gray band 6.5-9.8; the
  (9.8, 13.38) interval is unobserved for any recipe-constructed checkpoint.

Panel (c): training-pool readout (full-pool free decode, beam=3).
  x = decoded dev BLEU-4, y = full-training-pool free-decode BLEU;
  released point from leakage_sanity.json (78.8 BLEU, 70.7% EM).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "figures"))
from fig_style import (setup, panel_label, C_RELEASED, C_RECO, C_TNPURE,
                       C_DEGEN, M_RELEASED, M_RECO)
from fig_style import FAM_MARKERS, FAM_COLORS

VALID = ROOT / "checkpoints/released/backTranslation_PHIX_model/validations.txt"
GAP = ROOT / "results/gap_43_canonical_beam3.json"
READOUT = ROOT / "results/full_readout_summary.json"
ECKPD = ROOT / "results/epoch_ckpt_dev_bleu_all.json"
EPOCH_DEC = ROOT / "results/epoch_decouple.json"
LEAK = ROOT / "results/leakage_sanity.json"
OUT = ROOT / "figures/fig3_competence.pdf"

PRIMARY = [101, 202, 303, 404, 505, 606]
EXTENSION = [707, 808, 909, 1001, 1102, 1203, 1304, 1405]
ALL_SEEDS = PRIMARY + EXTENSION
STEPS_PER_EPOCH = 28          # 2,828 total steps / 101 epochs
FINAL_STEP = 2828
RELEASED_BEST = (1820, 13.38)  # verified against validations.txt in main()

# gap-JSON id -> family
FAMILY_OF = {}
for s in range(42, 50):
    FAMILY_OF[f"faithful_{s}"] = "faithful"
for s in ALL_SEEDS:
    FAMILY_OF[f"reco_{s}"] = "reconstructions"
    FAMILY_OF[f"seed_{s}"] = "reconstructions"   # readout-summary id
for s in [101, 202, 303, 404]:
    FAMILY_OF[f"cf_{s}"] = "config_faithful"
# step-faithful / confirmation / long-schedule are config-fidelity diagnostics
for s in [505, 606]:
    FAMILY_OF[f"sf_{s}"] = "config_faithful"
for s in [1506, 1607]:
    FAMILY_OF[f"conf_{s}"] = "config_faithful"
FAMILY_OF["ls_202"] = "config_faithful"
FAMILY_OF["rescue_wd0"] = "rescue"
FAMILY_OF["released"] = "released"
# distillation ladder (per-alpha students) belongs to the distillation family
for f in ["0125", "025", "05", "075"]:
    FAMILY_OF[f"ladder_{f}"] = "distillation"

# id -> merged decode-log key for checkpoints without full-readout records
CKPT_KEY = {
    f"cf_{s}": f"checkpoints/config_faithful/seed_{s}/best.ckpt" for s in [101, 202, 303, 404]
}
CKPT_KEY.update({f"sf_{s}": f"checkpoints/step_faithful/seed_{s}/best.ckpt" for s in [505, 606]})
CKPT_KEY.update({f"conf_{s}": f"checkpoints/confirmation/seed_{s}/best.ckpt" for s in [1506, 1607]})
CKPT_KEY["ls_202"] = "checkpoints/long_schedule/seed_202/best.ckpt"
CKPT_KEY["rescue_wd0"] = "checkpoints/rescue/seed_202_wd0/best.ckpt"
for f, p in [("0125", "frac_0125"), ("025", "frac_025"),
             ("05", "frac_05"), ("075", "frac_075")]:
    CKPT_KEY[f"ladder_{f}"] = f"checkpoints/ladder/{p}/best.ckpt"


def load_merged_ckpts():
    """Read results/epoch_ckpt_dev_bleu_all.json (merged decode log)."""
    return json.load(open(ECKPD))


def parse_validations(path):
    steps, b4 = [], []
    for line in open(path):
        m = re.search(r"Steps: (\d+).*?BLEU-4 ([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            b4.append(float(m.group(2)))
    return np.asarray(steps), np.asarray(b4)


def main():
    setup(scale=1.0)
    gap = json.load(open(GAP))
    ro = json.load(open(READOUT))
    epoch_dec = json.load(open(EPOCH_DEC))
    leak = json.load(open(LEAK))["experiment_a_free_decode"]
    ckpts = load_merged_ckpts()

    # ---------- build per-family point sets (dev BLEU, gap) ----------
    def ro_key_of(id_):
        """full_readout_summary key for a gap-JSON id (reco_* / faithful_* / distill_*)."""
        if id_.startswith("reco_"):
            return "seed_" + id_[5:]
        if id_.startswith("faithful_"):
            return "seed_" + id_[len("faithful_"):]
        m = re.match(r"distill_a([\d.]+)_(\d+)$", id_)
        if m:
            return f"alpha_{m.group(1)}_seed_{m.group(2)}"
        return None

    def dev_of(id_):
        """dev BLEU-4 for a canonical id from the best available source."""
        if id_ == "released":
            return RELEASED_BEST[1]
        rk = ro_key_of(id_)
        if rk in ro:
            return ro[rk]["dev"]["bleu"]
        if id_ in CKPT_KEY and CKPT_KEY[id_] in ckpts:
            return ckpts[CKPT_KEY[id_]]["dev_bleu"]
        return None

    families = {f: {"x": [], "y": [], "ids": []} for f in
                ("faithful", "reconstructions", "config_faithful", "rescue",
                 "distillation", "released", "degenerate")}
    for id_, rec in gap.items():
        if id_ == "_meta":
            continue
        dev = dev_of(id_)
        if dev is None:
            continue
        if dev < 1.0:                      # ladder / alpha-1.0 students
            families["degenerate"]["x"].append(dev)
            families["degenerate"]["y"].append(rec["gap"])
            families["degenerate"]["ids"].append(id_)
            continue
        fam = FAMILY_OF.get(id_)
        if fam is None:
            fam = "distillation" if id_.startswith("distill_") else "reconstructions"
        families[fam]["x"].append(dev)
        families[fam]["y"].append(rec["gap"])
        families[fam]["ids"].append(id_)

    # ---------- Panel (a): training trajectories ----------
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(7.6, 3.1),
                                        width_ratios=[1.25, 1.0, 1.0])
    panel_label(axa, "(a)", x=-0.30)
    panel_label(axb, "(b)", x=-0.26)
    panel_label(axc, "(c)", x=-0.26)

    steps, b4 = parse_validations(VALID)
    assert steps[0] == 14 and steps[-1] == FINAL_STEP and len(steps) == 202
    best = int(np.argmax(b4))
    assert (steps[best], round(b4[best], 2)) == RELEASED_BEST, (steps[best], b4[best])

    axa.plot(steps, b4, color=C_RELEASED, lw=1.5, zorder=3, label="Released evaluator")
    axa.plot(*RELEASED_BEST, M_RELEASED, color=C_RELEASED, ms=9, zorder=4)
    axa.annotate("best: step 1,820\ndev BLEU-4 13.38", xy=RELEASED_BEST,
                 xytext=(RELEASED_BEST[0] + 320, RELEASED_BEST[1] + 2.7),
                 fontsize=7.5, va="center", color="#333333",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                           pad=1.2),
                 arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))

    # reconstruction trajectories: epoch 25 (step 700) + final (step 2,828);
    # epoch-50 points (step 1,400) only exist for 4 seeds (303/707/1001/1001)
    xs = [700, FINAL_STEP]
    ys = []
    for s in ALL_SEEDS:
        ys.append([ckpts[f"checkpoints/reconstructions/seed_{s}/epoch_0025.ckpt"]["dev_bleu"],
                   ro[f"seed_{s}"]["dev"]["bleu"]])
    ys = np.asarray(ys)
    med = np.median(ys, axis=0)
    lo = np.min(ys, axis=0)
    hi = np.max(ys, axis=0)
    axa.fill_between(xs, lo, hi, color=C_RECO, alpha=0.18, zorder=1,
                     label="14 reconstructions (median band)")
    axa.plot(xs, med, color=C_RECO, lw=1.2, zorder=2)
    for s in ALL_SEEDS:
        key50 = f"checkpoints/reconstructions/seed_{s}/epoch_0050.ckpt"
        if key50 in ckpts:
            axa.plot(1400, ckpts[key50]["dev_bleu"], "o", color=C_RECO, ms=3,
                     mew=0.4, zorder=3)
    axa.plot([FINAL_STEP] * 14, ys[:, -1], "o", color=C_RECO, ms=4, mew=0.6,
             zorder=3)

    # faithful-family trajectories (train_faithful.py), truncated to 2,828
    # steps for comparability with the released run's endpoint
    f_steps = np.arange(14, FINAL_STEP + 1, 14)
    f_b4 = np.zeros((8, len(f_steps)))
    for i, s in enumerate(range(42, 50)):
        st, b4 = parse_validations(ROOT / f"checkpoints/faithful/seed_{s}/validations.txt")
        m = st <= FINAL_STEP
        f_b4[i, :int(m.sum())] = b4[m]
    axa.fill_between(f_steps, f_b4.min(0), f_b4.max(0),
                     color=FAM_COLORS["faithful"], alpha=0.14, zorder=1,
                     label="8 faithful runs (band)")
    axa.plot(f_steps, np.median(f_b4, 0), color=FAM_COLORS["faithful"],
             lw=1.2, zorder=2)

    axa.set_xlim(0, 3100)
    axa.set_ylim(-0.5, 18.6)
    axa.set_xlabel("optimizer step")
    axa.set_ylabel("decoded dev BLEU-4")
    axa.spines[["top", "right"]].set_visible(False)
    axa.legend(loc="lower left", frameon=False, fontsize=7.5)

    # ---------- Panel (b): gap vs dev BLEU ----------
    axb.axvspan(6.486, 12.615, color="#e8e8e8", zorder=0)
    axb.axhline(0, color="#000000", ls="--", lw=0.8, zorder=1)
    order = ["faithful", "reconstructions", "config_faithful", "rescue",
             "distillation", "released", "degenerate"]
    for fam in order:
        pts = families[fam]
        if not pts["x"]:
            continue
        if fam == "degenerate":
            axb.plot(pts["x"], pts["y"], "o", color=C_DEGEN, mfc="white", ms=5,
                     mew=1.0, zorder=2)
        elif fam == "released":
            axb.plot(pts["x"], pts["y"], M_RELEASED, color=C_RELEASED, ms=13,
                     zorder=5)
            axb.text(pts["x"][0], pts["y"][0] + 1.1, f"+{pts['y'][0]:.2f}",
                     fontsize=7.8, ha="center", color=C_RELEASED)
        else:
            axb.plot(pts["x"], pts["y"], FAM_MARKERS[fam], color=FAM_COLORS[fam],
                     ms=5.5, mew=1.0, ls="", zorder=3)
    axb.text(13.6, -1.4, "residual\nunobserved\ninterval\n(12.62, 13.38)",
             fontsize=7.0, ha="center", va="center", color="#555555",
             style="italic",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                       pad=1.2))
    axb.set_xlim(0, 14.5)
    axb.set_ylim(-4.5, 12.5)
    axb.set_xlabel("decoded dev BLEU-4")
    axb.set_ylabel("gap (PURE \u2212 REC, sacreBLEU)")
    axb.spines[["top", "right"]].set_visible(False)

    # ---------- Panel (c): training-pool readout ----------
    def train_readout_of(id_):
        """Full-pool free-decode train BLEU (beam=3) for a canonical id."""
        rk = ro_key_of(id_)
        if rk in ro:
            return ro[rk]["train"]["bleu"]
        if id_ == "rescue_wd0":          # epoch_decouple rescue record
            for rec in epoch_dec["records"]:
                if rec["label"] == "rescue_wd0_best":
                    return rec["train_bleu"]
        return None

    read_pts = []
    for fam in ("faithful", "reconstructions", "config_faithful", "rescue", "distillation"):
        for x, id_ in zip(families[fam]["x"], families[fam]["ids"]):
            ty = train_readout_of(id_)
            if ty is not None:
                read_pts.append((x, ty, fam))
    for x, ty, fam in read_pts:
        axc.plot(x, ty, FAM_MARKERS[fam], color=FAM_COLORS[fam],
                 ms=5, mew=0.8, zorder=3)
    axc.plot(RELEASED_BEST[1], leak["bleu"], M_RELEASED, color=C_RELEASED,
             ms=14, zorder=5)
    axc.annotate(f"released evaluator\n{leak['bleu']:.1f} BLEU, "
                 f"{leak['em_pct']:.1f}% EM",
                 xy=(RELEASED_BEST[1], leak["bleu"]),
                 xytext=(13.38, 73),
                 fontsize=7.6, va="center", ha="center", color="#333333",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                           pad=1.2),
                 arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))
    axc.set_xlim(0, 14.5)
    axc.set_ylim(0, 85)
    axc.set_xlabel("decoded dev BLEU-4")
    axc.set_ylabel("training-pool free-decode BLEU")
    axc.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT)
    print(f"Wrote: {OUT}")
    for fam in order:
        pts = families[fam]
        if pts["x"]:
            print(f"{fam}: n={len(pts['x'])} dev range "
                  f"[{min(pts['x']):.2f}, {max(pts['x']):.2f}]")


if __name__ == "__main__":
    main()
