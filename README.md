# Anonymous Review Artifact

## Overview
This artifact contains the code, decoded hypotheses, sufficient statistics,
donor registries, statistical analyses, and machine-readable checkpoint
registry for the audit study of the released SLRTP2025 back-translation
evaluator.

## Quick Start
```bash
# Verify core reproducibility (no GPU needed, ~30 seconds)
make core-audit PYTHON=/path/to/python3
```
Expected output: `checkpoint-GT alignment PASS` and a 60-cell BLEU decomposition
JSON written to `results/bleu_diagnostics/`.

## Structure
```
.
├── Makefile                    # core-audit target
├── README.md                   # this file
├── requirements.lock.txt       # pinned dependency versions
├── artifact_ledger.json        # SHA-256 for every file (203 entries)
├── data/
│   └── cells/                  # 28 decoded-hypothesis cells (7 evaluators × 4 systems)
├── figures/
│   ├── dose_response.pdf       # gap-vs-competence + pass-through dose-response
│   └── original_trajectory.pdf # released evaluator training log trajectory
├── manifests/                  # exclusion manifests, donor registries
├── results/
│   ├── artifact_ledger.json    # machine-readable file digests
│   ├── checkpoint_registry.json # all 51 checkpoints with dev metrics + gate status
│   ├── bleu_diagnostics/       # 60-cell BLEU decomposition
│   └── round5/                 # all analysis result JSONs (E1-E13)
├── scripts/                    # all analysis scripts (38 .py files)
└── tests/                      # checkpoint-GT alignment regression test
```

## Checkpoint Registry
`results/checkpoint_registry.json` is the definitive registry of every BT
evaluator checkpoint (50 trained + 1 released = 51 total), with per-checkpoint
dev BLEU-4, dev WER (official protocol), competence-gate status, and which
analyses each enters.

## Lightweight Reproduction
A text-only package (no pose data needed) can reproduce every text-metric
number and confidence interval from `data/cells/` + `results/` using the
scripts in `scripts/`.

## License
- PHOENIX-2014T: CC BY-NC-SA 3.0 (original RWTH Aachen terms)
- Czehmann et al. (2026) back-translations: CC BY-NC-SA 4.0 (derivative)
- Code: available under the project license (see paper for details)
