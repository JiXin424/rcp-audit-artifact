# Review Artifact — `artifact/` README

This artifact accompanies the manuscript:
**"A Training-Pool Replay Audit of the Released SLRTP2025 Back-Translation Evaluator: Non-Replication Across Independently Trained Checkpoints and a Readout-With-Generalization Signature"**

Repository: [anonymized for double-blind review — the public repository URL will be restored in the camera-ready version]

## 1. Layout

```
artifact/
├── README.md                      # this file
├── Makefile                       # `make core-audit` (operational); `reproduction-dag` needs sibling scripts
├── LICENSE                        # MIT license (source code + derived statistics)
├── requirements.lock.txt          # Python dependency pins
├── artifact_ledger.json           # SHA-256 ledger (306 files; auto-generated from current artifact state)
├── manifests/
│   └── exclusions.jsonl           # IDs absent from released SLRTP2025 pose materialization
├── data/cells/                    # decoded BT hypotheses (60 cells: 15 beam-3 evaluators × 4 systems)
│   ├── cp0_GT-v1.json             # Released evaluator × GT-v1 (641 items)
│   ├── cp0_TN-PURE-v1.json        # Released evaluator × TN-PURE-v1
│   └── ...
├── results/
│   ├── cells_greedy/                     # 30-evaluator greedy-decode cells (cp0-cp29, GT-v1 + TN-PURE-v1)
│   ├── unified_checkpoint_registry.json   # SINGLE SOURCE OF TRUTH (71 entries: 70 trained + 1 released)
│   ├── canonical_checkpoint_registry.json # beam-3 primary registry (matches unified)
│   ├── all_distill_beam3_canonical.json   # 15/15 distillation beam-3 gaps (canonical donor registry)
│   ├── all_distill_beam3.json             # 15/15 distillation beam-3 gaps (simplified retrieval)
│   ├── equivariance.json                  # input-side pose permutation equivariance check
│   ├── input_permutation.json             # input-side degenerate-pose controls
│   ├── leakage_sanity.json                # train-pool free-decode + permutation + stratified EM
│   ├── e6b_matched_subset.json            # matched-subset reference sensitivity
│   ├── family_stratify_cluster.json       # per-family dose-response + cluster bootstrap
│   ├── missing_distill_beam3.json         # 2 re-decoded distillation seeds
│   ├── new_distill_dev_bleu4.json         # dev BLEU-4 for 2 re-decoded seeds
│   ├── human_eval/                        # DGS human evaluation (30 raters × 100 videos; manifest lost)
│   └── ...
├── scripts/                       # 51 analysis scripts
│   ├── e_leakage_sanity.py               # train-pool free-decode + permutation tests
│   ├── e_input_permutation.py            # input-side pose shuffle/zero/noise
│   ├── e_equivariance.py                 # output-equivariance verification
│   ├── e6b_matched_subset.py             # matched confidence=1 subset analysis
│   ├── e_beam3_matched_subset.py         # beam-3 re-decode + matched subset
│   ├── e_family_stratify_cluster.py      # per-family PI + signer/show/date bootstrap
│   ├── decode_missing_distill.py         # decode 2 previously-OOM distillation seeds
│   └── ...
└── tests/                         # pytest regression: checkpoint–GT alignment
```

## 2. Quick reproduction

```bash
# Verify the core 28-cell BLEU decomposition (released evaluator + 6 primary reconstructors)
make core-audit PYTHON=/path/to/python3
```

This regenerates the headline GT 12.78 / PURE 23.79 / gap +11.01 (beam-3) from the artifact cells.

**Note on `reproduction-dag`:** The full DAG references scripts from sibling revision directories (`../revision_...`) that are not included because they depend on intermediate data exceeding the artifact size limit. The saved hypotheses in `data/cells/` and sufficient statistics in `results/` allow full recomputation of all reported numbers via `core-audit`. The DAG scripts will be published with the camera-ready version.

## 3. Data licensing

Source pose tensors and signer-derived videos are **not** redistributed in this artifact. Users reconstructing the complete audit must obtain:

- PHOENIX-2014T (RWTH-PHOENIX-Weather 2014T): CC BY-NC-SA 3.0, distributed by RWTH Aachen
- Human back-translations: Czehmann et al. (2026), CC BY-NC-SA 4.0
- SLRTP2025 pose bundle + released BT checkpoint: provider-specific terms

The MIT license covers source code, manifests, decoded hypothesis JSON files, and derived statistics.

## 4. Checkpoint inventory

The unified registry (`results/unified_checkpoint_registry.json`) is the single source of truth:

| Family | Planned | Trained | Has gap (beam-3) | Has dev |
|---|---|---|---|---|
| Reconstructions (primary) | 6 | 6 | 6 | 6 |
| Reconstructions (extension) | 8 | 8 | 8 | 8 |
| Rescue lr | 8 | 8 | 0 | 8 |
| Rescue expanded | 12 | 12 | 1 | 12 |
| Train-pool ladder | 4 | 4 | 4 | 4 |
| Config-faithful | 4 | 4 | 4 | 4 |
| Step-faithful | 2 | 2 | 2 | 2 |
| Large-arch | 4 | 4 | 0 | 4 |
| Confirmation | 2 | 2 | 2 | 2 |
| Long-schedule | 2 | 2 | 1 | 2 |
| BT-retrained (holdout) | 1 | 1 | 0 | 0 |
| Cross-fit A/B (holdout) | 2 | 2 | 0 | 0 |
| Distillation students | 15 | 15 | 15 | 15 |
| **Total** | **70** | **70** | **43** | **67** |

All 70 planned checkpoints completed training and beam-3 PHX-public evaluation. No OOM failures. The 2 distillation seeds (α=0.5 s303, α=0.75 s303) that previously failed beam-3 evaluation due to GPU contention were re-decoded successfully (`results/missing_distill_beam3.json`, `results/all_distill_beam3_canonical.json`).

## 5. DGS human evaluation data

Per-rater response matrices for the 30-rater × 100-video feasibility study are in `results/human_eval/`. The original video files and UUID-to-system manifest were lost; the study is reported as an aborted feasibility study, not as audit evidence.

## 6. Known discrepancies

- **Canonical GT WER (raw vs official-normalized):** The canonical GT cell stores raw WER 79.26; the paper reports official-normalized WER 85.77 (jiwer 3.1.0 with SLRTP2025 normalization). Both correspond to the same 641 decoded sequences; the difference is the normalization pipeline. BLEU-4 (12.78) is invariant.
- **Donor registry:** The distillation students' beam-3 evaluation uses a rebuilt canonical donor registry (NFKC + Levenshtein + SHA-256). 625/641 items (97.5%) select the same donor as the original registry used for non-distillation checkpoints. See manuscript §Donor-registry consistency note.
