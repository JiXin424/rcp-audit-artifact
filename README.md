# RCP — Auditing a Released Back-Translation Evaluator for Sign Language Production

Paper sources, training scripts, and evaluation harness for the LRE
(*Language Resources and Evaluation*, Springer 10579) submission:

> **Auditing a Released Back-Translation Evaluator for Sign Language
> Production: A Failed Reconstruction and Descriptive Probe Analysis
> Audit (Round-9 revision)**

The paper audits whether the SLRTP2025 released back-translation (BT)
evaluator's $+10.24$ sacreBLEU retrieval-vs-recorded-poses (PURE–REC) reversal
on the 641-sequence PHX-public test set reproduces across checkpoints
reconstructed from the publicly documented training recipe. Spoiler: it does
not reproduce — 44 non-degenerate decoded runs (48 unique weight binaries
after SHA-256 deduplication) 43 of 44 have strictly negative PURE–REC gaps (one $+0.24$; range
$[-2.01, -0.21]$), and the released evaluator's competence (dev BLEU-4 13.38
under the uniform beam-3 protocol) and training-pool readout (78.8 BLEU,
70.7% exact match) are outside anything the public recipe constructs (decoded
family decoded max dev BLEU-4 10.5; unobserved competence interval $(9.8, 13.38)$).
The paper is scoped as a public-recipe sufficiency audit, not a causal
checkpoint-identity claim.

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
title and the old `[10.9, 12.7]` transition interval — the $(9.8, 13.38)$
interval here supersedes it.

## Repository layout

```
src/                  Python package: data, models, training, evaluation, utils
scripts/              46 CLI entry points (40 *.py + 6 *.sh)
configs/              YAML configs (released, reconstruction, distillation, rescue_wd0)
checkpoints/          Trained model weights (.ckpt gitignored; 70 trained + released + 9 finetunes)
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

```bash
cd /ssd/xkb4/RCP
python -m venv .venv
source .venv/bin/activate
pip install -r /ssd/xkb4/rcp-audit-artifact/requirements.lock.txt
```

The raw SLRTP2025 data is licensed and must be obtained from the challenge
organizers (see Data licensing below); on this host it is symlinked from
`/ssd/xkb4/SignDiff/SLRTP2025_data/`.

### Reproduce the canonical headline

The single source of truth is the canonical donor registry (exclusion-fixed;
SHA-256 `9170a53026ab3263b451a3632a9e02318745acabf12f0b679921477afbefa301`).
One command in the artifact mirror regenerates every paper number:

```bash
cd /ssd/xkb4/rcp-audit-artifact
make core-audit PYTHON=/path/to/python3
# → asserts GT 12.78 / PURE 23.02 / gap +10.24 / CI [8.88, 11.62]
```

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

Step-corrected runs use the correct `validation_freq=14` interpretation (14 optimizer steps, not 14 epochs) and NLL-based checkpoint selection. One seed takes ~6 min on a single GPU (7060 train items, effective batch 256, early-stop ~epoch 25-45).

```bash
# Single seed
python -m src.training.train_matched \
    --config configs/released.yaml \
    --train-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
    --dev-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
    --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
    --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
    --seed 1701 --gpu 0 --epochs 300 --batch-size 256 --grad-accum 1 --selection nll \
    --output checkpoints/step_faithful/seed_1701

# 8 seeds in parallel on 8 GPUs (seeds 1701-1708)
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python -m src.training.train_matched \
    --config configs/released.yaml \
    --train-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/train.pt \
    --dev-pickle data/SLRTP2025/SLRTP-Sign-Production-Evaluation-Data/data/dev.pt \
    --txt-vocab checkpoints/released/backTranslation_PHIX_model/txt.vocab \
    --gls-vocab checkpoints/released/backTranslation_PHIX_model/gls.vocab \
    --seed $((1701+i)) --gpu 0 --epochs 300 --batch-size 256 --grad-accum 1 --selection nll \
    --output checkpoints/step_faithful/seed_$((1701+i)) &
done; wait
```

### Decode a step-corrected seed (REC + PURE gap panel + dev BLEU)

```bash
python scripts/decode_step_faithful.py --ckpt-dir checkpoints/step_faithful/seed_1701 --gpu 0
python scripts/e_dev_uniform.py --gpu 0 --ckpt-dir checkpoints/step_faithful/seed_1701
```

`decode_step_faithful.py` writes `results/gap_43_canonical_beam3_items/sf_1701_{gt,pure}.json`. Expected across the 10 step-corrected seeds: PURE--REC gaps in $[-1.59, +0.24]$, 9/10 negative (per-seed values in SI~Table~S27).

### Compile the paper

```bash
pdflatex -interaction=nonstopmode main_lre.tex
bibtex main_lre
pdflatex -interaction=nonstopmode main_lre.tex
pdflatex -interaction=nonstopmode main_lre.tex
```

(same for `supplementary.tex`)

## Data licensing

- **PHOENIX-2014T**: RWTH Aachen academic research licence (the distributed copy's specified version; the official landing page does not display a Creative Commons licence) — not redistributed.
- **SLRTP2025 pose bundle and released BT checkpoint**: challenge terms —
  obtain from the [SLRTP2025 organizers](https://github.com/walsharry/SLRTP-Sign-Production-Evaluation).
- **Czehmann et al. (2026) human back-translations**: CC BY-NC-SA 4.0 — not redistributed.
- **CSL-Daily**: provider-specific terms — not redistributed.

Pose tensors may retain signer identity; no per-signer raw poses are released.

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
paper sources are intact. **All trained checkpoints were rebuilt from scratch
after the incident** (14 reconstructions, 15 distillation students, rescue and
diagnostic families, 9 released-weight fine-tunes). The 30-rater DGS
evaluation raw response CSVs are preserved in `评分/`, but the video stimuli
and UUID-to-system manifest were lost, so that feasibility study is not
reported in the paper (see Ethics in `main_lre.tex`). The host is treated as
untrusted until rebuilt; credentials from before 2026-07-14 should be
considered compromised.
