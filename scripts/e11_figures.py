#!/usr/bin/env python3
"""Generate dose-response figures: (a) PURE-GT gap vs dev BLEU; (b) pass-through ratio vs dev BLEU."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R5 = Path(__file__).resolve().parents[1] / "results"
FIG = R5 / "figures"
FIG.mkdir(exist_ok=True)

# dev BLEU per checkpoint (fractional)
dev = {"original": 0.1338, "seed_101": 0.0857, "seed_202": 0.0881, "seed_303": 0.0967,
       "seed_404": 0.0719, "seed_505": 0.0905, "seed_606": 0.0809}
# from r5_extension_seeds_dev.json
ext = json.load(open(str(Path(__file__).resolve().parents[1] / "results/evaluator_extension")))
for row in ext if isinstance(ext, list) else ext.get("rows", []):
    dev[f"seed_{row['seed']}"] = row.get("dev_bleu4", row.get("dev_bleu"))
# rescue2
resc = json.load(open(R5 / "results/e8_rescue2_dev_eval.json"))
for k, v in resc.items():
    dev[k.replace("rescue2_", "")] = v["dev_bleu4"]
# ladder + cfaith
lad = json.load(open(R5 / "results/e11b_ladder_gaps.json"))
for k, v in lad.items():
    dev["ladder_" + k] = v["dev_bleu"]

# gaps (fractional -> BLEU points)
gaps = {"original": 11.01, "seed_101": 0.08, "seed_202": -0.98, "seed_303": -0.71,
        "seed_404": -0.36, "seed_505": -0.89, "seed_606": 0.13,
        "seed_707": -0.19, "seed_808": -0.87, "seed_909": -0.47, "seed_1001": -0.27,
        "seed_1102": -1.10, "seed_1203": -0.60, "seed_1304": -0.78, "seed_1405": -0.72,
        "wd0_seed202": 0.25, "bs512_seed101": -0.71}
for k, v in lad.items():
    gaps["ladder_" + k] = v["gap"] * 100

pt = json.load(open(R5 / "results/e11d_pass_through.json"))

def name_key(k):
    return k

# Assemble series
fams = {"14 reconstructions": {"keys": [f"seed_{s}" for s in [101,202,303,404,505,606,707,808,909,1001,1102,1203,1304,1405]], "c": "#1f77b4", "m": "o"},
        "ladder (train subsample)": {"keys": ["ladder_f0.125", "ladder_f0.25", "ladder_f0.5", "ladder_f0.75"], "c": "#2ca02c", "m": "s"},
        "rescue variants": {"keys": [k for k in dev if any(x in k for x in ["bs128", "bs512", "drop", "wd0", "ls0.1", "ep600"])], "c": "#9467bd", "m": "^"},
        "config-faithful": {"keys": ["cfaith101", "cfaith404"], "c": "#8c564b", "m": "D"},
        "released evaluator": {"keys": ["original"], "c": "#d62728", "m": "*"}}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for fam, spec in fams.items():
    xs, ys, ys2 = [], [], []
    for k in spec["keys"]:
        if k not in dev:
            continue
        xs.append(dev[k])
        ys.append(gaps.get(k, np.nan))
        ys2.append(pt.get(k, {}).get("ratio", np.nan))
    if fam == "released evaluator":
        axes[0].scatter(xs, ys, c=spec["c"], marker=spec["m"], s=180, label=fam, zorder=5)
        axes[1].scatter(xs, ys2, c=spec["c"], marker=spec["m"], s=180, label=fam, zorder=5)
    else:
        axes[0].scatter(xs, ys, c=spec["c"], marker=spec["m"], s=40, label=fam, alpha=0.85)
        axes[1].scatter(xs, ys2, c=spec["c"], marker=spec["m"], s=40, label=fam, alpha=0.85)

axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("dev BLEU-4 (fractional)")
axes[0].set_ylabel("PURE $-$ GT gap (BLEU points)")
axes[0].set_title("(a) Gap--competence dose--response is flat")
axes[0].legend(fontsize=7, loc="upper left")
axes[0].set_xlim(0.04, 0.145)

axes[1].set_xlabel("dev BLEU-4 (fractional)")
axes[1].set_ylabel("donor pass-through ratio")
axes[1].set_title("(b) Pass-through ratio vs competence")
axes[1].axhline(3.09, color="#d62728", lw=0.8, ls=":")
axes[1].set_xlim(0.04, 0.145)

fig.tight_layout()
fig.savefig(FIG / "dose_response.pdf")
fig.savefig(FIG / "dose_response.png", dpi=150)
print("wrote", FIG / "dose_response.pdf")
