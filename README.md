# RCP — Reconstructing the SLRTP2025 Back-Translation Evaluator

Paper sources, training scripts, and evaluation harness for the LRE
(*Language Resources and Evaluation*, Springer 10579) submission:

> **Reconstructing the SLRTP2025 Back-Translation Evaluator: An Artifact
> Audit and Gradient-Clipping Sensitivity Study**

An audit of whether the released SLRTP2025 back-translation (BT) evaluator
can be reconstructed from its public artifacts (an evaluation-only
repository without training scripts). Headline results:

1. **Seven implementation defects** in our own earlier reconstruction code,
   identified by line-level audit and forensic arithmetic against the
   released training log; correcting them recovers most of the released
   evaluator's surface competence (dev BLEU-4 11.59–12.62 across seeds
   42–49 vs. released 13.38; test-split family mean 12.27 ± 0.46 vs. 12.78).
2. **The eighth defect was a wrong author inference: gradient clipping.**
   All 69 non-degenerate canonical runs shared a clip threshold of 1.0 and
   uniformly failed to reproduce two properties: the training-pool
   memorisation signature (released 78.8 BLEU / 70.7% EM) and the positive
   response to a constructed replay probe (released +10.24). With clipping
   disabled — the default of the candidate upstream framework
   (`neccam/slt`, commit `249d3cd`) when, as in the released config, no
   clip field is specified — the reconstruction reproduces both signatures
   (readout 99.98 / 99.9% EM; probe gap +11.19), confirmed across seeds and
   on a never-explored dev split (+8.07/+8.00/+8.64 vs. released +8.10).
3. The released operating point's **individual coordinates are bracketed**
   by different clip thresholds but not matched jointly; the attribution is
   source-semantic + behavioural, not transcript-confirmed
   (`results/faithful_steps_eval.json`, `results/dev_probe_eval.json`).
4. The probe response **was observed only under tested reference-coupled
   retrieval conditions**; it falls to +0.95 under human back-translations
   and is absent for deployment-realistic systems.

## Repository layout

```
Makefile              make install-audit | core-audit | check-consistency
requirements-audit.txt  L1 audit env (numpy, scipy, sacrebleu, pyyaml)
requirements.lock.txt   full env for L3/L4 (not needed for L1)
claim_manifest.json   claim → file → verify-command map for every headline number
artifact_ledger.json  file inventory with SHA-256s
manifests/            exclusions.jsonl (split reconciliation), donor registries
src/                  Python package: data, models, training, evaluation, utils
  src/training/train_faithful.py   seven-correction training (clip/ctc-norm flags)
scripts/              analysis + training entry points (eval_dev_probe.py,
                      split_reconciliation.py, round34_auto_eval.sh, ...)
results/              per-checkpoint results, decodes, sensitivity grids
checkpoints/          released evaluator bundle + checkpoint metadata
data/                 licence-composed data (sacrebird CSVs; see licensing)
paper/                main_lre.tex + supplementary.tex (submission sources)
docs/                 historical plans and design notes
```

## Quick start (L1 audit — recompute headline numbers)

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-audit    # 4 packages; CPU-only Linux x86_64, Python 3.10–3.12
make core-audit       # rebuilds the canonical donor registry and asserts
                      # GT 12.78 / PURE 23.02 / gap +10.24 / CI [8.88, 11.62]
make check-consistency
# → 40+ invariants: registry summary ↔ entry counts; accounting family sums;
#   paper tokens (round-34 title, terminology, dev-probe gaps, split
#   reconciliation, family mean/SD); JSON↔TeX cross-checks. Exit 0 required.
```

`make core-audit` is L1 only: it recomputes selected statistics from
committed per-item decodes; it does not retrain models or decode from
checkpoint weights. L3 (re-decode from weights) and L4 (from-scratch
retraining, ≈6 GPU-h per seed) require the full environment and the
licensed SLRTP2025/PHOENIX-2014T data (see licensing).

Round-34 additions:
- `results/dev_probe_eval.json` — frozen-dev-split probe gaps for the
  clip-ladder checkpoints (scripts/eval_dev_probe.py);
- `results/split_reconciliation.json` — ID-level classification of the
  36/4/1 official→released split difference (scripts/split_reconciliation.py);
- multi-seed clip-ladder replicates (seeds 43/44 at clip 2/3/5) in
  `results/faithful_steps_eval.json` and SI Sup. V.

## Data licensing (composition — read before reuse)

| Component | Terms | In this artifact? |
|---|---|---|
| PHOENIX-2014T (RWTH Aachen) | Academic research licence. | No (not redistributed). Obtain from RWTH. |
| SLRTP2025 pose bundle + released BT checkpoint | SLRTP2025 challenge terms (research use). | Yes — `checkpoints/released/` verbatim copy under the same terms. Other trained weights inherit the same research-use restriction. |
| Czehmann et al. (2026) human back-translations | CC BY-NC-SA 4.0. | Yes — `data/sacrebird/test_subset_backtranslations_sacrebirdphoenix.csv` and `test_full_annotations_sacrebirdphoenix.csv` are redistributed under CC BY-NC-SA 4.0, with attribution and the share-alike / non-commercial boundary documented in `data/sacrebird/LICENSE` and `data/sacrebird/NOTICE`. |
| CSL-Daily | Provider-specific academic terms. | No (not redistributed). |

- Root `/LICENSE` is **MIT** and applies **only** to the RCP audit software
  (scripts, src, Makefile, configs we authored, paper sources, tests); the
  preamble enumerates the scope.
- `/data/sacrebird/LICENSE` (CC BY-NC-SA 4.0) governs the two CSVs in that
  directory and is not overridden by the root MIT Licence.
- `/checkpoints/released/` is governed by SLRTP2025 challenge terms; the
  MIT Licence does not extend to it.

## Provenance and integrity

- Canonical donor registry SHA-256 `9170a53026ab3263b451a3632a9e023187
  45acabf12f0b679921477afbefa301`; the frozen dev-split registry SHA is
  recorded in `results/dev_probe_eval.json`.
- The one byte-identical checkpoint collision (`cf_202` ≡ `ls_202`) is
  asserted by the registry builder and documented in SI Sup. D.
- `artifact_ledger.json` carries the file inventory; `claim_manifest.json`
  maps each paper headline number to its source file and verify command.

## History note (artifact integrity)

A host compromise on 2026-07-14 was cleaned; the released BT checkpoint,
SLRTP2025 raw dataset, decoded BT cells, canonical checkpoint registry, and
paper sources are intact. **Most trained checkpoints were rebuilt from
scratch after the incident** (identical protocols; the retraining pipeline
is deterministic — fresh re-runs reproduce published best checkpoints
byte-identically). Five checkpoint families listed in earlier manual
registries could not be recovered from disk and are omitted from the
disk-scanned registry (SI Sup. D, Table D footnote; main-text Limitations).
The 30-rater DGS evaluation raw response CSVs are preserved under `评分/`,
but the video stimuli and UUID-to-system manifest were lost, so that
feasibility study is not reported in the paper. The host is treated as
untrusted until rebuilt; credentials from before 2026-07-14 should be
considered compromised.

## Correspondence

Paper sources in `paper/` correspond to the LRE submission. SI appendix
labels (Sup. A–V) in `supplementary.tex` match the references in
`paper/main_lre.tex`. Every headline number in the paper is covered by
`claim_manifest.json` and verified by `make check-consistency`.
