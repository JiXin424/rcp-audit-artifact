# RCP — Reconstructing the SLRTP2025 Back-Translation Evaluator from Public Artifacts

Paper sources, training scripts, and evaluation harness for the LRE
(*Language Resources and Evaluation*, Springer 10579) submission:

> **Reconstructing the SLRTP2025 Back-Translation Evaluator from Public
> Artifacts: What Reproduces, and What Does Not**

A faithful, config-exact re-implementation (`src/training/train_faithful.py`,
correcting seven defects identified in our earlier reconstruction code)
recovers most of the released evaluator's surface competence (dev BLEU-4
11.59–12.62 vs. 13.38 across seeds 42–49; test decoding up to 13.11 vs. 12.78)
but not its training-pool readout signature (released 78.8 BLEU / 70.7% EM;
faithful family 11.45–12.08 / 2.3–2.5%) nor the response to a constructed
adversarial probe (+10.24 on the released evaluator): all 69 non-degenerate
constructible evaluators give negative PURE–REC gaps (range $[-2.80, -0.21]$;
76 trained / 74 decoded / 73 unique binaries in the disk-scanned registry).

## Two repositories (read this first)

| Repo | Local path | GitHub remote | Role |
|---|---|---|---|
| **RCP (this repo)** | `/ssd/xkb4/RCP/` | `github.com/JiXin424/slrtp2025-replay-audit.git` | Primary working repo: paper sources, experiment scripts, results, figures |
| **rcp-audit-artifact** | `/ssd/xkb4/rcp-audit-artifact/` | `github.com/JiXin424/rcp-audit-artifact.git` | Reviewer artifact bundle (Makefile, requirements.lock.txt, ledger); anonymous URL `anonymous.4open.science/r/rcp-audit-artifact-B314` |

**RCP is the primary repo** — all new work lands here first. The artifact repo
is a manually synced mirror of the review-relevant content; its `Makefile`
regenerates every paper number from the canonical donor registry (SHA-256
`9170a53026ab3263b451a3632a9e02318745acabf12f0b679921477afbefa301`). If the
artifact's `paper/` sources look stale, sync them from this repo (see
`AGENTS.md` §1b). Note: the artifact README still carries the pre-Round-3
title and the old `[10.9, 12.7]` transition interval — the $(10.5, 13.38)$
interval here supersedes it.

## Repository layout

```
src/                  Python package: data, models, training, evaluation, utils
scripts/              46 CLI entry points (40 *.py + 6 *.sh)
configs/              YAML configs (released, reconstruction, distillation, rescue_wd0)
checkpoints/          Trained model weights (.ckpt gitignored; 68 trained + released + 9 finetunes)
data/SLRTP2025/       SLRTP2025 pose records + released BT checkpoint (symlink, not in git)
data/sacrebird/       Czehmann et al. human back-translation CSVs (CC BY-NC-SA 4.0)
results/              Experiment outputs (canonical registries, decoded cells, protocol readouts)
figures/              Figure-generation scripts
generated_figures/    Paper figure PDFs
评分/                  DGS human-evaluation raw response CSVs (30 raters; retained for provenance)
docs/                 Historical plans, specs, design notes
main_lre.tex          Paper main TeX source (LRE, sn-chicago author-year)
supplementary.tex     SI (Sup. A–R)
references.bib        Bibliography
sn-jnl.cls, sn-chicago.bst   Springer Nature LaTeX class / style
```

## Quick start

### Set up

For L1 audit (recompute headline numbers from committed per-item decodes,
verify paper-count consistency), only 4 Python packages are needed:

```bash
cd /ssd/xkb4/rcp-audit-artifact
python3 -m venv .venv
source .venv/bin/activate
make install-audit   # pip install -r requirements-audit.txt
```

For L3 re-decoding and L4 from-scratch retraining, the full
`requirements.lock.txt` (236 packages including training/LLM tooling) is
required; this is **not** needed for L1 audit. CPU-only Linux x86_64 with
Python 3.10–3.12 is sufficient for L1.

The raw SLRTP2025 data is licensed and must be obtained from the challenge
organizers (see Data licensing below); on this host it is symlinked from
`/ssd/xkb4/SignDiff/SLRTP2025_data/`.

### Reproduce the canonical headline

The single source of truth is the canonical donor registry (exclusion-fixed;
SHA-256 `9170a53026ab3263b451a3632a9e02318745acabf12f0b679921477afbefa301`).
One command in the artifact mirror regenerates every paper number:

```bash
cd /ssd/xkb4/rcp-audit-artifact
make core-audit
# → asserts GT 12.78 / PURE 23.02 / gap +10.24 / CI [8.88, 11.62]
# → check-paper passes (all TeX values match canonical numbers)
# → regression passes (transfer-cell local-GT alignment)
```

### Verify paper-count consistency

After `make core-audit`, also run the disk-scanned registry verifier:

```bash
make check-consistency
# → 25+ invariants: registry summary ↔ entry counts; accounting family sums;
#   paper bold total row; gap panel meta ↔ actual range; cross-checks against
#   matched_donor_pool.json, donor_pool_resampling.json, robustness_diagnostics.json.
# All checks must pass (exit code 0).
```

`check_paper_consistency.py` is layout-aware: it finds `paper/main_lre.tex`
in this artifact mirror and `./main_lre.tex` in the RCP working repo
automatically.

### Round-3 experiments (C1 protocol unification + reviewer-driven analyses)

```bash
# Uniform full-pool beam-3 dev BLEU/WER for every checkpoint (C1)
python scripts/e_dev_uniform.py

# Training-pool readout (full-pool free decode, beam-3)
python scripts/e_full_readout.py

# Readout–competence/overfit association table (Exp 2)
python scripts/e_readout_overfit.py

# Competence gate table (0/28 recipe-constructed runs pass)
python scripts/e_gate_table.py

# BLEU floor calibration under reference permutation (R1-M3)
python scripts/e_floor_calibration.py
```

### Train one reconstruction seed

```bash
python scripts/train_reconstruction.py \
    --seed 101 --gpu 0 --epochs 300 \
    --output checkpoints/reconstructions/seed_101
```

### Train the full 14-seed reconstruction set (parallel, 8 GPUs)

```bash
bash scripts/launch_reconstructions.sh
```

### Train step-corrected near-faithful seeds (10 seeds: 505, 606, 1701-1708)

Step-corrected runs use the correct `validation_freq=14` interpretation (14 optimizer steps, not 14 epochs) AND decoded-BLEU checkpoint selection (matching the released recipe). The training script is `src/training/train_matched_v2.py` invoked with `--rec-weight 0` (translation-only loss) and `--selection bleu` (decoded-BLEU selection). One seed takes ~6 min on a single GPU (7060 train items, effective batch 256, early-stop ~epoch 30-50).

```bash
# Single seed
CUDA_VISIBLE_DEVICES=0 python -m src.training.train_matched_v2 \
    --config configs/released.yaml \
    --train-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
    --dev-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
    --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
    --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
    --seed 1701 --gpu 0 --epochs 300 --batch-size 256 --patience 15 \
    --rec-weight 0 --selection bleu \
    --output checkpoints/step_faithful/seed_1701

# 10 seeds in parallel on 8 GPUs (seeds 1701-1708 + 505, 606)
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python -m src.training.train_matched_v2 \
    --config configs/released.yaml \
    --train-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
    --dev-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
    --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
    --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
    --seed $((1701+i)) --gpu 0 --epochs 300 --batch-size 256 --patience 15 \
    --rec-weight 0 --selection bleu \
    --output checkpoints/step_faithful/seed_$((1701+i)) &
done
# Then launch seeds 505 and 606 on freed GPUs
for SEED in 505 606; do
  GPU=$([ $SEED -eq 505 ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$GPU python -m src.training.train_matched_v2 \
    --config configs/released.yaml \
    --train-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
    --dev-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
    --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
    --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
    --seed $SEED --gpu 0 --epochs 300 --batch-size 256 --patience 15 \
    --rec-weight 0 --selection bleu \
    --output checkpoints/step_faithful/seed_$SEED &
done; wait
```

### Decode a step-corrected seed (REC + PURE gap panel + dev BLEU)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/decode_checkpoints_only.py \
    --checkpoints sf_1701:checkpoints/step_faithful/seed_1701 --gpu 0
python scripts/e_dev_uniform.py --gpu 0 --ckpt-dir checkpoints/step_faithful/seed_1701
```

`decode_checkpoints_only.py` writes `results/gap_43_canonical_beam3_items/sf_1701_{gt,pure}.json`. Observed across the 10 step-corrected seeds (Round-10 re-train): PURE--REC gaps in $[-1.37, -0.36]$, **all 10 negative** (per-seed values in SI~Table~S27). The earlier `+0.24` positive gap reported for the legacy `step_faithful` runs was an artifact of those runs using NLL selection + epoch validation (i.e., the same protocol as `config_faithful`); the re-trained runs use the actually-corrected protocol and eliminate the positive gap.

### Compile the paper

```bash
pdflatex -interaction=nonstopmode main_lre.tex
bibtex main_lre
pdflatex -interaction=nonstopmode main_lre.tex
pdflatex -interaction=nonstopmode main_lre.tex
```

(same for `supplementary.tex`)

## Data licensing

The datasets and model weights used in this audit are available under
research-specific terms; they are not "fully public" in the permissive sense.

| Component | Terms | In this artifact? |
|---|---|---|
| PHOENIX-2014T (RWTH Aachen) | Academic research licence; CC licence not displayed on the official landing page. The distributed copy is governed by the version specified in its own licence header. | No (not redistributed). Obtain from RWTH Aachen. |
| SLRTP2025 pose bundle + released BT checkpoint | SLRTP2025 challenge terms. | Yes — `checkpoints/released/` is a verbatim copy under the same challenge terms (research-use only). Other trained weights under `checkpoints/` are outputs of training on this data and inherit the same research-use restriction. |
| Czehmann et al. (2026) human back-translations | CC BY-NC-SA 4.0. | Yes — `data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv` and `test_full_annotations_sacrebirdphoenix.csv` are redistributed under CC BY-NC-SA 4.0; see `data/sacrebird/LICENSE` and `data/sacrebird/NOTICE` for attribution and the share-alike / non-commercial boundary. |
| CSL-Daily | Provider-specific academic terms. | No (not redistributed). |

### Repository licence composition

- Root `/LICENSE` is **MIT** but applies **only to the RCP audit software**
  (scripts, src, Makefile, configs we authored, paper sources, tests). The
  preamble in `/LICENSE` enumerates the scope.
- `/data/sacrebird/LICENSE` is **CC BY-NC-SA 4.0** and governs the two CSVs
  in that directory; it is not overridden by the root MIT Licence.
- `/checkpoints/released/` is governed by **SLRTP2025 challenge terms**; the
  MIT Licence does not extend to it. Other trained checkpoints under
  `/checkpoints/` are outputs of training on PHOENIX-2014T/SLRTP2025 and
  remain under the same research-use restriction.
- Per-directory `LICENSE`/`NOTICE` files document attribution and any
  modifications.

### Signer-identity note

Pose tensors and decoded texts may retain signer identity, biometric
features, and signing style. We release per-item decoded text (low
re-identification risk) and a curated subset of trained checkpoints needed
for the L1/L3 audit; we do not release per-signer raw poses. The Czehmann
back-translations were produced by a Deaf fluent L2 DGS signer; ethical
constraints on reuse (non-commercial, share-alike, attribution) are encoded
in the CC BY-NC-SA 4.0 licence above.

## Provenance and integrity

- Canonical donor registry SHA-256 asserted by `scripts/build_canonical_panel.py`.
- `results/gap_43_canonical_beam3.json` carries per-checkpoint checkpoint SHA-256
  and donor-registry SHA-256; the one collision (`config_faithful/seed_202` ≡
  `long_schedule/seed_202`) is documented in the registry.
- `artifact_ledger.json` (in the artifact mirror) records the SHA-256 of every artifact file.
- Evaluation protocol (C1): full pool (7,060/515/641), beam=3, alpha=−1,
  `[::2]` subsample, max_output_len=400, sacreBLEU 13a/exp/effective_order=False,
  official jiwer 3.1.0 normalized WER — reproduces the released training log
  (dev 13.38 / WER 83.37) exactly.

## History note

A host compromise on 2026-07-14 was cleaned; the released BT checkpoint,
SLRTP2025 raw dataset, decoded BT cells, canonical checkpoint registry, and
paper sources are intact. **Most trained checkpoints were rebuilt from scratch
after the incident**: 14 reconstructions, 4 validation-freq-misread,
10 step-corrected (re-trained in Round-26 with the TRUE step-val + decoded-BLEU
protocol), 16 joint-loss greedy/beam-3, 15 distillation students, 4 ladder,
2 confirmation, 1 long-schedule, 1 rescue (wd0), 9 released-weight fine-tunes.
**Five checkpoint families listed in earlier manual registries (4 large-arch,
8 rescue-lr, 2 cross-fit, 1 BT-retrained holdout, plus 11 of 12 rescue-expanded)
could not be recovered from disk and are omitted from the disk-scanned
registry**; their family-level summary statistics had been retained in the
legacy manual registry but cannot be reproduced from current artifacts. This
is documented in SI~Sup.~D (Table D footnote) and the main-text Limitations
paragraph. The 30-rater DGS evaluation raw response CSVs are preserved in
`评分/`, but the video stimuli and UUID-to-system manifest were lost, so that
feasibility study is not reported in the paper (see Ethics in
`paper/main_lre.tex`). The host is treated as untrusted until rebuilt;
credentials from before 2026-07-14 should be considered compromised.
