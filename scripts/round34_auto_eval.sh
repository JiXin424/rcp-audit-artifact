#!/usr/bin/env bash
# Round-34 multi-seed clip-ladder auto-eval: wait for ALL six trainings to
# finalize (training_log.json), then evaluate them SEQUENTIALLY on one GPU
# (round-33 GOTCHA: concurrent eval_step_ckpts.py runs clobber
# faithful_steps_eval.json last-writer-wins; serialize to avoid it).
# Each run gets the full eval (test REC/PURE/gap + 7,060 readout) plus the
# frozen-dev-probe eval (scripts/eval_dev_probe.py).
set -u
cd /ssd/xkb4/RCP

RUNS="clip2p0_seed43 clip2p0_seed44 clip3p0_seed43 clip3p0_seed44 clip5p0_seed43 clip5p0_seed44"

for r in $RUNS; do
  while [ ! -f "checkpoints/ctc_clip_sens/$r/training_log.json" ]; do sleep 300; done
  echo "[round34] $r finalized at $(date +%H:%M:%S)"
done
echo "[round34] all six trainings finalized; starting sequential evals at $(date +%H:%M:%S)"

for r in $RUNS; do
  CUDA_VISIBLE_DEVICES=7 python3 scripts/eval_step_ckpts.py \
      --run-dir "checkpoints/ctc_clip_sens/$r" --ckpts best \
      > "logs/eval_round34_${r}.log" 2>&1
  echo "[round34] full eval $r done: $(grep -o 'done.*' logs/eval_round34_${r}.log | tail -1)"
  CUDA_VISIBLE_DEVICES=7 python3 scripts/eval_dev_probe.py \
      --run-dir "checkpoints/ctc_clip_sens/$r" --ckpts best \
      >> "logs/eval_round34_${r}.log" 2>&1
  echo "[round34] dev-probe eval $r done: $(grep -o 'done.*' logs/eval_round34_${r}.log | tail -1)"
done
echo "[round34] all evals done at $(date +%H:%M:%S)"
touch results/.round34_evals_complete
