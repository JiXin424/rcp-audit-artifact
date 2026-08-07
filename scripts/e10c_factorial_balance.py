#!/usr/bin/env python3
"""Round 6 E10c: full design/balance report for the 605-query exposure factorial.

From the frozen selection manifest:
  - high/low LaBSE definition (from tolerances) and realized per-cell similarity stats
  - four-cell covariate balance on the 605 support: similarity, frames (duration),
    word count, signer-match, plus SMDs for seen-vs-unseen within each stratum
  - sample flow: 641 -> 605 (exclusion reasons)
  - donor reuse across the four cells (shared donors between cells/queries)
  - prespecification status (from frozen manifest hashes)
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "revision_20260728_major/results/task7_canonical_v2/selection_manifest.json"
DIAG = ROOT / "revision_20260728_major/results/task7_canonical_v2/selection_diagnostics.json"
OUT = ROOT / "revision_20260729_round5/results/e10c_factorial_design_balance.json"

man = json.load(open(MAN))
diag = json.load(open(DIAG))
rows = [r for r in man["rows"] if r["included"]]
tol = man["tolerances"]

cells = ["seen_high", "seen_low", "unseen_high", "unseen_low"]
covs = ["similarity", "frames", "text_words"]

per_cell = {}
for c in cells:
    vals = {k: np.array([r["cells"][c][k] for r in rows], dtype=float) for k in covs}
    signer = np.array([r["cells"][c]["signer"] == r["target_signer"] for r in rows], dtype=float)
    per_cell[c] = {**{k: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "median": float(np.median(v))} for k, v in vals.items()},
                   "same_signer_rate": float(signer.mean())}


def smd(a, b):
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


balance = {}
for stratum in ["high", "low"]:
    s = per_cell[f"seen_{stratum}"]; u = per_cell[f"unseen_{stratum}"]
    sa = np.array([r["cells"][f"seen_{stratum}"]["similarity"] for r in rows])
    ua = np.array([r["cells"][f"unseen_{stratum}"]["similarity"] for r in rows])
    sf = np.array([r["cells"][f"seen_{stratum}"]["frames"] for r in rows], dtype=float)
    uf = np.array([r["cells"][f"unseen_{stratum}"]["frames"] for r in rows], dtype=float)
    sw = np.array([r["cells"][f"seen_{stratum}"]["text_words"] for r in rows], dtype=float)
    uw = np.array([r["cells"][f"unseen_{stratum}"]["text_words"] for r in rows], dtype=float)
    balance[stratum] = {"similarity_smd": smd(sa, ua), "frames_smd": smd(sf, uf),
                        "words_smd": smd(sw, uw),
                        "seen_same_signer": s["same_signer_rate"], "unseen_same_signer": u["same_signer_rate"]}

# donor reuse across cells
donor_use = Counter()
for r in rows:
    for c in cells:
        donor_use[r["cells"][c]["donor_id"]] += 1
reuse = Counter(donor_use.values())
n_shared = sum(1 for r in rows if len({r["cells"][c]["donor_id"] for c in cells}) < 4)

# diagnostics from manifest (selection-time balance checks)
diags = [r["diagnostics"] for r in rows]
diag_stats = {k: {"max": float(max(d[k] for d in diags)),
                  "mean": float(np.mean([d[k] for d in diags]))}
              for k in diags[0] if isinstance(diags[0][k], (int, float))}

out = {
    "n_included": len(rows), "n_excluded": 641 - len(rows),
    "exclusion_reasons": Counter(r.get("exclusion_reason") for r in man["rows"] if not r["included"]),
    "tolerances": tol,
    "high_low_definition": {
        "high": "the seen/unseen donor pair with maximized LaBSE cosine to the query, subject to |sim(seen_high)-sim(unseen_high)| <= max_cross_similarity_diff",
        "low": "a donor with similarity <= high_similarity - min_high_low_gap, matched across pools within the same cross-pool tolerance",
        "min_high_low_gap": tol["min_high_low_gap"],
        "max_cross_similarity_diff": tol["max_cross_similarity_diff"],
        "duration_caliper": "candidate-target relative frames diff <= 0.35; four-cell max relative range <= 0.30; low-to-high <= 0.20",
        "word_count_caliper": "candidate-target |dwords| <= 8; four-cell max range <= 6; low-to-high <= 4",
        "signer": "same signer required for all four cells (require_same_signer=true)",
        "selection_order": "high pair chosen first (maximize summed similarity, then cross-pool |dsim|, then lexical IDs), then low pair; lexicographic deterministic tie-breaks",
    },
    "per_cell": per_cell,
    "seen_vs_unseen_balance": balance,
    "donor_reuse": {
        "total_cell_slots": len(rows) * 4,
        "unique_donors": len(donor_use),
        "reuse_distribution": {str(k): v for k, v in sorted(reuse.items())},
        "queries_with_cross_cell_donor_collision": n_shared,
        "note": "cross-cell donor identity collision was forbidden by design (validate_design); n_shared counts queries where the same donor id appears in two cells, which should be 0",
    },
    "selection_time_diagnostics": diag_stats,
    "prespecification": {
        "selection_blinded_to_evaluator_outcomes": man.get("selection_blinded_to_evaluator_outcomes"),
        "manifest_canonical_sha256": diag.get("manifest_canonical_sha256"),
        "status": "selection manifest and tolerances frozen before outcome scoring; main effects and interaction are the design's primary contrasts",
    },
}
OUT.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ["n_included", "n_excluded", "seen_vs_unseen_balance", "donor_reuse"]}, indent=1, default=str))
