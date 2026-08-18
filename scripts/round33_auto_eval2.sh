#!/usr/bin/env bash
# Round-33 confirmatory auto-eval v2: wait for each of the 5 confirmatory
# trainings to finalize (training_log.json written), then evaluate best
# checkpoints one GPU each. Marker-based so the long clipinf_steps run
# does not delay the evals.
set -u
cd /ssd/xkb4/RCP

wait_for() {  # wait_for <run_dir>
  local d="checkpoints/$1"
  while [ ! -f "$d/training_log.json" ]; do sleep 120; done
  echo "[driver2] $1 finalized at $(date +%H:%M:%S)"
}

for spec in \
  "ctc_clip_sens/clipinf_seed43:best" \
  "ctc_clip_sens/clipinf_seed44:best" \
  "ctc_clip_sens/clip3p0_seed42:best" \
  "ctc_clip_sens/clip5p0_seed42:best" \
  "ctc_clip_sens/signjoeyexact_seed42:best" \
  "ctc_clip_sens/clipinf_steps_seed42:step_1820,step_2828"; do
  dir="${spec%%:*}"; cks="${spec##*:}"
  tag=$(echo "$dir" | tr '/' '_')
  (
    wait_for "$dir"
    g=$( ($(date +%s) / 60) % 8 )   # pick a GPU by the minute to avoid collisions
    CUDA_VISIBLE_DEVICES=$g python scripts/eval_step_ckpts.py \
        --run-dir "checkpoints/$dir" --ckpts "$cks" \
        > "logs/eval_${tag}.log" 2>&1
    echo "[driver2] eval $dir done"
  ) &
done
wait
echo "[driver2] all evals done at $(date +%H:%M:%S)"
touch results/.round33_evals2_complete
