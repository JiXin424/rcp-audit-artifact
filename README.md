# Reviewer Artifact — README

This artifact accompanies the manuscript:

**"Public-Artifact Sufficiency Audit of the SLRTP2025 Back-Translation Evaluator: A Failed Reconstruction and Descriptive Probe Analysis"**

submitted to *Language Resources and Evaluation* (Springer 10579). LRE is
**single-blind**; author information appears in the paper itself. The artifact
is shared anonymously for reviewer verification and will be deposited under a
persistent DOI upon acceptance.

## 0. What this artifact is for

The paper audits whether the released SLRTP2025 back-translation (BT)
evaluator's $+10.24$ sacreBLEU retrieval-vs-recorded-poses reversal on PHX-public
reproduces across checkpoints reconstructed from the publicly documented
training recipe, and characterizes the released evaluator's distinctive
properties. **Every numerical claim in the paper is regenerated from a single
canonical donor registry** (exclusion-fixed; SHA-256
`9170a53026ab3263b451a3632a9e02318745acabf12f0b679921477afbefa301`) by the
targets in `Makefile`. An earlier pre-exclusion materialization (PURE 23.79 /
gap $+11.01$) survives only as a sensitivity analysis in SI~Sup.~N and is NOT
reproduced by any target here.

**Evaluation tiers (canonical counts, auto-generated from registry).** All counts
derive from `results/canonical_checkpoint_registry.json` (schema v4), which uses
**checkpoint SHA-256 as the primary key**. Every unique binary has exactly one
entry; training runs that produce the same binary (e.g., `cf_202` and `ls_202`
share SHA-256 `911de1fe…`) are listed under `training_runs` and counted once.

| Tier | Count | Definition |
|---|---|---|
| Training runs | 48 | Completed `.ckpt` files on disk |
| Unique binaries | 47 | Distinct SHA-256 hashes (1 collision: `cf_202` = `ls_202`) |
| With gap (PHX-public) | 43 | Checkpoints with a PURE--REC gap under the canonical donor registry |
| Non-degenerate | 38 | 43 − 5 (3 × α=1.0 distillation + 2 × ladder fractions with BLEU ≈ 0) |
| Released (audit target) | 1 | Public SLRTP2025 BT evaluator |
| Non-released, non-degenerate | 37 | The set over which the headline gap range $[-2.01,-0.21]$ is reported |

The 5 degenerate entries are: `distill_a1.0_101`, `distill_a1.0_202`,
`distill_a1.0_303` (gap ≈ 0; α=1.0 removes all ground-truth signal), and
`ladder_0125`, `ladder_025` (BLEU ≈ 0; insufficient training data).

`results/gap_43_canonical_beam3_items/` holds per-item beam-3 hypotheses for all
43 checkpoints. The `results/gap_43_canonical_beam3.json` panel now includes
checkpoint SHA-256 for every entry. The single SHA collision (`config_faithful/
seed_202` = `long_schedule/seed_202`) is documented in the registry.

**Legacy files.** `results/unified_checkpoint_registry.json` (schema v2) and
`scripts/build_unified_checkpoint_registry.py` predate the canonical donor
registry and contain pre-exclusion released values (PURE 23.79 / gap +11.01)
and legacy training-time dev BLEU-4 values (including the non-reproducible
α=1.0 dev figure of 10.81). Both are retained for provenance only and carry a
`_deprecated` marker; use `results/gap_43_canonical_beam3.json`,
`results/canonical_checkpoint_registry.json`, and `results/paper_numbers.json`
for all paper numbers.

## 0b. Round-3 revision (C1 protocol unification + reviewer-driven analyses)

Round 3 (2026-08-09) unifies the dev evaluation protocol and adds three
reviewer-driven analyses. **Canonical evaluation protocol:** full pool
(7,060/515/641), beam=3, alpha=−1, `[::2]` subsample, `max_output_len=400`,
sacreBLEU 13a/exp/effective_order=False, official jiwer 3.1.0 normalized WER.
It reproduces the released training log exactly (dev BLEU-4 13.38 / WER
83.37); the earlier 500-sample `e_checkpoint_stats` protocol (which produced
17.95/83.93) is deprecated.

| Analysis | Script | Output | What it does |
|---|---|---|---|
| C1 protocol (Exp 1) | `scripts/e_dev_uniform.py` | `results/dev_uniform/*.json`, `results/dev_gate_table.json` | Uniform-protocol dev BLEU/WER for every checkpoint; competence gate: **0/28 decoded recipe-constructed runs pass** (|dev BLEU−13.38|≤1.0 AND WER≤86.37); 9/9 released-weight fine-tunes pass but are excluded as weight perturbations; 24 withheld runs (training-log max dev 10.02). |
| Readout–overfit (Exp 2) | `scripts/e_readout_overfit.py` | `results/readout_overfit.json` | Spearman across the 31 checkpoints with uniform full-pool readouts: readout tracks decoded competence (ρ=0.916, n=31; ρ=0.873 non-degenerate, n=27) but not the train−dev overfit indicator (ρ=0.040) nor the PURE−REC gap (ρ=−0.082). |
| Floor calibration (Exp 3) | `scripts/e_floor_calibration.py` | `results/floor_calibration.json` | Reference-permutation BLEU floor calibration (reviewer R1-M3). |

Round-3 locked numbers: unobserved competence interval **(9.8, 13.38)**
(decoded family max 9.83 = `cf_101`; training-log max 10.02); released
training-pool readout **78.8 BLEU / 70.7% EM** vs. family max 9.3; 38
non-degenerate decoded runs (37 unique binaries) with strictly negative gaps
$[-2.01, -0.21]$.

## 1. Layout

```
artifact/
├── README.md                       # this file
├── Makefile                        # canonical reproduction targets
├── LICENSE                         # MIT (source code + derived statistics)
├── requirements.lock.txt           # Python dependency pins
├── artifact_ledger.json            # SHA-256 ledger (auto-generated)
├── manifests/
│   └── exclusions.jsonl            # IDs absent from released SLRTP2025 pose materialization
├── data/
│   ├── cells/                      # decoded BT hypotheses (60 cells: 15 beam-3 evaluators × 4 systems)
│   ├── cells_greedy/               # 30-evaluator greedy-decode cells
│   └── sacrebird/                  # Czehmann et al. human back-translation CSVs (CC BY-NC-SA 4.0)
├── results/
│   ├── paper_numbers.json          # MACHINE-READABLE table of every headline number
│   ├── gap_43_canonical_beam3.json # 43-checkpoint PURE-REC gap panel (canonical registry)
│   ├── gap_43_canonical_beam3_items/  # 30×2 beam-3 per-item hypotheses + donor_registry.jsonl
│   ├── canonical_checkpoint_registry.json   # 47 unique binaries, 48 training runs (SHA-256 primary key)
│   ├── canonical_matched_subset.json        # 461-item confidence=1 reference-frame analysis
│   ├── canonical_floor_effect.json          # asymmetric floor-effect numbers
│   ├── full_readout/ + full_readout_summary.json  # uniform full-7060 readout (31 ckpts incl. released)
│   ├── dev_uniform/ + dev_gate_table.json         # C1 uniform dev BLEU/WER + competence gate
│   ├── readout_overfit.json                       # Exp 2 readout–competence/overfit Spearman table
│   ├── floor_calibration.json                     # Exp 3 reference-permutation floor calibration
│   ├── donor_cluster_bootstrap.json         # donor reuse stats + cluster bootstrap CIs
│   ├── ref_frame_paired_items.json          # per-item reference-frame paired data
│   ├── released_perturbation.json           # experiment A1: weight-noise scan at competence parity
│   ├── epoch_decouple.json                  # experiment D: rescue/long-schedule epoch search
│   ├── per_item_gap_decomposition.json      # experiment C: per-item OLS gap decomposition
│   ├── leakage_sanity.json                  # train-pool free-decode + permutation + stratified EM
│   ├── equivariance.json, input_permutation.json   # input-side pose controls
│   ├── family_stratify_cluster.json         # per-family dose-response + signer/show/date bootstrap
│   └── ... (see artifact_ledger.json for the full file list)
├── figures/
│   ├── fig1_audit_design.pdf                 # Fig. 1 audit design (mermaid)
│   ├── fig2_headline.pdf                     # Fig. 2 headline reversal + reconstruction forest
│   ├── fig3_competence.pdf                   # Fig. 3 trajectories / gap vs dev / pool readout
│   ├── fig4_reference_frame.pdf              # Fig. 4 human-reference attenuation
│   ├── fig5_donor_origin.pdf                 # Fig. 5 donor-provenance contrasts
│   ├── released_perturbation.pdf             # A1 weight-noise figure
│   └── per_item_gap_decomposition.pdf        # C coefficient / scatter figure
├── paper/
│   ├── main_lre.tex, main_lre.pdf            # paper source + compiled PDF
│   ├── supplementary.tex, supplementary.pdf  # SI source + PDF
│   └── references.bib
├── scripts/                        # analysis scripts (see Section 4)
└── tests/                          # pytest: checkpoint-GT alignment regression
```

## 2. Quick reproduction (canonical headline)

```bash
make core-audit PYTHON=/path/to/python3
```

This runs (i) the checkpoint-GT alignment regression test and (ii)
`scripts/build_paper_numbers.py`, which reads the canonical per-item
hypotheses in `results/gap_43_canonical_beam3_items/released_{gt,pure}.json`
and regenerates `results/paper_numbers.json` containing every headline number
in the paper. It then asserts the canonical headline matches **GT 12.78 /
PURE 23.02 / gap +10.24 / CI [8.88, 11.62]**. The earlier pre-exclusion
materialization is reported only in SI~Sup.~N.

`make paper` alone rebuilds `paper_numbers.json` (~5 s after the bootstrap
optimization in this revision). Each number in `paper/main_lre.tex` should
match `results/paper_numbers.json`; the build prints a reminder to compare.

## 3. Reviewer round-2 experiments (this revision)

This revision adds three experiments that directly address the reviewer's
competence-confound critique (R2-1) and the mechanism question:

| Target | Script | Output | What it does |
|---|---|---|---|
| `make experiment-A1` | `scripts/e_released_perturbation.py` + `make_figure_perturbation.py` | `results/released_perturbation.json`, `figures/released_perturbation.pdf` | Gaussian weight-noise scan $\sigma\in\{0, 10^{-3}, 3\!\times\!10^{-3}, 10^{-2}, 3\!\times\!10^{-2}\}$ on the released checkpoint. Tests reversal robustness at competence parity. |
| `make experiment-D` | `scripts/e_epoch_decouple.py` | `results/epoch_decouple.json` | Decodes rescue/long-schedule epoch checkpoints (best/ep25/ep50) with full-7060 readout + PHX-public gap. Searches for a readout-gap decoupling point. |
| `make experiment-C` | `scripts/e_per_item_decomposition.py` | `results/per_item_gap_decomposition.json`, `figures/per_item_gap_decomposition.pdf` | OLS regression of per-item PURE-REC gap on lexical/template/signer/length features; $R^2$, standardized coefficients, leave-one-out drops. |
| `make donor-bootstrap` | `scripts/e_donor_cluster_bootstrap.py` | `results/donor_cluster_bootstrap.json` | Donor reuse stats + donor-cluster / two-way / signer / show / date bootstrap CIs on canonical cells (reviewer R2-4). |
| `make full-readout` | `scripts/e_full_readout.py` (8-way launcher in README §4) | `results/full_readout/*.json`, `results/full_readout_summary.json` | Uniform full-7060 train/dev/test readout for 14 reco + 15 distill + 1 rescue (reviewer R2-5). |
| `make ref-frame-figure` | `scripts/make_figure_ref_frame_paired.py` | `figures/ref_frame_paired.pdf` | Paired per-item sentence-BLEU scatter under original vs human references (reviewer R2-6). |
| `make competition` | `scripts/competition_ranking.py` | `results/competition_ranking.json` | Competition-system ranking from public leaderboard data; per-team pose outputs are not publicly archived, so the analysis uses leaderboard scores and two fixed-output cross-evaluator re-decoding studies (noisy-gloss retrieval and USTC-MoE rebuild; see paper §4.4). |

`make round2-all` runs the CPU-only subset (donor-bootstrap, ref-frame-figure,
experiment-C, paper). The GPU experiments (`experiment-A1`, `experiment-D`,
`full-readout`) need one or more GPUs and are run individually with `GPU=N`.

### Full-readout 8-way launcher

`e_full_readout.py` decodes one checkpoint dir per `--ckpt-dir` flag. To run
all 30 checkpoints across 8 GPUs in parallel:

```python
import subprocess, glob, os
ck = sorted(d for d in glob.glob('checkpoints/reconstructions/seed_*') if os.path.isdir(d))
ck += sorted(d for d in glob.glob('checkpoints/distillation/alpha_*') if os.path.isdir(d))
ck += sorted(d for d in glob.glob('checkpoints/rescue/seed_*') if os.path.isdir(d))
for gpu, group in enumerate([ck[i::8] for i in range(8)]):
    cmd = ['python3', 'scripts/e_full_readout.py', '--gpu', str(gpu)]
    for c in group: cmd += ['--ckpt-dir', c]
    subprocess.Popen(cmd, stdout=open(f'full_readout_gpu{gpu}.log', 'w'))
```

## 4. Script inventory (selected)

Canonical headline / registries:
- `build_canonical_panel.py` — deterministic canonical donor registry (SHA-256 above); regenerates `gap_43_canonical_beam3.json` and per-item hypotheses.
- `build_paper_numbers.py` — single-command paper-number rebuild from canonical cells.
- `build_unified_checkpoint_registry.py` — 71-entry checkpoint registry.

Leakage / permutation / readout controls:
- `e_leakage_sanity.py` — train-pool 78.8 BLEU / 70.7% EM readout, permutation, stratified EM, membership control.
- `e_input_permutation.py`, `e_equivariance.py` — input-side pose permutation + output-equivariance verification.
- `e_full_readout.py` — uniform full-7060 readout across the reconstruction family (this revision).

Round-3 protocol analyses:
- `e_dev_uniform.py` — uniform full-pool beam-3 dev BLEU/WER (the canonical C1 protocol).
- `e_readout_overfit.py` — readout–competence/overfit Spearman table (Exp 2).
- `e_gate_table.py` — competence gate table (0/28 pass; 9/9 fine-tunes).
- `e_floor_calibration.py` — reference-permutation BLEU floor calibration (Exp 3, R1-M3).

Bootstrap / decomposition:
- `e_donor_cluster_bootstrap.py` — donor reuse + donor-cluster / two-way / signer / show / date bootstrap CIs.
- `e_family_stratify_cluster.py` — per-family dose-response PI + cluster-aware bootstrap.
- `e_per_item_decomposition.py` — per-item gap OLS decomposition (this revision).
- `e2_path_sensitivity.py`, `e3_balance.py` — donor-pool substitution decomposition.

Reference-frame analysis:
- `e6b_matched_subset.py`, `e_beam3_matched_subset.py` — matched confidence=1 subset analysis.
- `make_figure_ref_frame_paired.py` — paired per-item reference-frame figure (this revision).

Released-checkpoint local perturbation:
- `e_released_perturbation.py` — A1 weight-noise scan (this revision).
- `e_epoch_decouple.py` — D epoch decoupling search (this revision).
- `make_figure_perturbation.py` — A1 figure.

Competition ranking (completed; per-team outputs not archived):
- `competition_ranking.py` — consumes fixed system outputs once placed in `data/competition_systems/`.

## 5. Data licensing

- **PHOENIX-2014T**: CC BY-NC-SA 4.0 (RWTH Aachen) — not redistributed; the
  artifact ships decoded BT hypotheses and derived sufficient statistics only.
- **Czehmann et al. (2026) human back-translations** (`data/sacrebird/*.csv`):
  CC BY-NC-SA 4.0 — included for the matched-subset reference-frame analysis;
  see their release for full terms.
- **SLRTP2025 pose bundle and released BT checkpoint**: challenge terms; the
  checkpoint itself is NOT redistributed here (reviewers obtain it from the
  SLRTP2025 organizers). All decoded hypotheses in `data/cells/` were produced
  from the publicly released checkpoint.
- **CSL-Daily**: provider-specific terms — not redistributed.
- The MIT license in `LICENSE` covers source code, manifests, decoded
  hypothesis JSONs, and derived sufficient-statistic artifacts only.

Pose tensors may retain signer identity; the artifact releases no per-signer
raw poses, only de-identified decoded text and aggregate metrics.

## 6. Regression test

`make regression` (run by `core-audit`) executes
`tests/test_checkpoint_gt_alignment.py`, which asserts that every cross-check
transfer cell uses its evaluator's local recorded-pose baseline as the GT
reference (the audit's central alignment invariant).

## 7. Reproduction DAG (full audit)

The complete reproduction DAG (stages A–N: exposure-symmetric systems,
matched seen/unseen factorials, cluster bootstrap, pool randomization,
cross-fit, scorer equivalence, BERTScore/BLEURT, decoding sensitivity) is
provided in the `scripts/` directory. Each script is self-contained and uses
paths relative to the artifact root. The saved hypotheses in `data/cells/` and
sufficient statistics in `results/` allow full recomputation of every reported
number via `make core-audit` and the per-experiment targets in Section 3.

## 8. Provenance and integrity

- `artifact_ledger.json` records the SHA-256 of every artifact file; rebuild
  with `python3 scripts/rebuild_artifact_ledger.py` (path-fixed in this revision).
- The canonical donor registry's SHA-256 is asserted by `build_paper_numbers.py`.
- All sacreBLEU values use signature `BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1`.
