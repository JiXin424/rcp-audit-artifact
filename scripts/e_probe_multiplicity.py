#!/usr/bin/env python3
"""Probe multiplicity transparency table.

Reports all retrieval strategies and construction variants explored on the
released evaluator before the headline probe was frozen.  The headline probe
(source-text Jaccard, pure donor copy) was selected as the maximal-gap probe;
its CI is therefore post-selection descriptive.

Values are corpus-level sacreBLEU-4 (0--100) on the canonical 641-item PHX-public
set under the released evaluator (REC = 12.78).  Sources:
  - Jaccard / oracle-gloss / random PURE and COMP: Table 1 (isomorphic_conditioning)
  - LaBSE / multi-SBERT PURE: main text Sec 3.1 (conditioning controls)
  - All gaps computed as PURE_score - REC (12.78).

Output: results/probe_multiplicity.json
"""
from __future__ import annotations
import json
from pathlib import Path

REC = 12.78

PROBES = [
    {
        "retriever": "Source-text Jaccard",
        "construction": "Pure donor copy",
        "score": 23.02,
        "selected": True,
        "source": "Table 1; canonical gap panel",
    },
    {
        "retriever": "Source-text Jaccard",
        "construction": "PT-composed",
        "score": 18.86,
        "selected": False,
        "source": "Table 1",
    },
    {
        "retriever": "LaBSE",
        "construction": "Pure donor copy",
        "score": 19.2,
        "selected": False,
        "source": "Sec 3.1 conditioning controls",
    },
    {
        "retriever": "multi-SBERT",
        "construction": "Pure donor copy",
        "score": 16.5,
        "selected": False,
        "source": "Sec 3.1 conditioning controls",
    },
    {
        "retriever": "Oracle gloss (input leakage)",
        "construction": "Pure donor copy",
        "score": 11.4,
        "selected": False,
        "source": "Table 1",
    },
    {
        "retriever": "Oracle gloss (input leakage)",
        "construction": "PT-composed",
        "score": 11.8,
        "selected": False,
        "source": "Table 1",
    },
    {
        "retriever": "Random donor (20 seeds, SD 0.24)",
        "construction": "Pure donor copy",
        "score": 0.90,
        "selected": False,
        "source": "Table 1",
    },
    {
        "retriever": "Random donor (20 seeds, SD 0.19)",
        "construction": "PT-composed",
        "score": 0.77,
        "selected": False,
        "source": "Table 1",
    },
]

for p in PROBES:
    p["gap"] = round(p["score"] - REC, 2)

n_probes = len(PROBES)
max_gap = max(p["gap"] for p in PROBES)
selected = [p for p in PROBES if p["selected"]][0]

output = {
    "description": "All retrieval x construction probes explored on the released evaluator before freezing the headline probe.",
    "rec_reference": REC,
    "n_probes_explored": n_probes,
    "max_gap": max_gap,
    "selected_probe": selected["retriever"] + " / " + selected["construction"],
    "selected_gap": selected["gap"],
    "note": (
        "The headline probe was selected as the maximal-gap probe after exploring "
        f"{n_probes} probe configurations on the same 641-item test set and the same "
        "released evaluator.  Its 95% CI [+8.88, +11.62] is conditional on this "
        "selection and should be read as post-selection descriptive, not confirmatory."
    ),
    "bonferroni_note": (
        f"A Bonferroni correction over {n_probes} probes at alpha=0.05 gives "
        f"alpha={0.05/n_probes:.5f} (z~{1.96 * (n_probes ** 0.5) ** 0.5:.2f} approx); "
        "the corrected CI still excludes zero but is wider.  The more honest "
        "characterization remains descriptive."
    ),
    "probes": PROBES,
}

out_path = Path(__file__).resolve().parents[1] / "results" / "probe_multiplicity.json"
out_path.write_text(json.dumps(output, indent=2) + "\n")
print(f"Wrote {out_path}")
print(f"Selected probe: {selected['retriever']} / {selected['construction']}")
print(f"Selected gap: {selected['gap']}")
print(f"Max gap among all {n_probes} probes: {max_gap}")
