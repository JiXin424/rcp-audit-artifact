#!/usr/bin/env bash
# Round-33 auto-eval driver: wait for all 8 trainings to finish, then
# evaluate every remaining checkpoint (best/final) across free GPUs.
set -u
cd /ssd/xkb4/RCP

echo "[driver] waiting for 8 trainings to finish..."
while true; do
  n=$(pgrep -fc "src.training.train_faithful" || true)
  [ "$n" -eq 0 ] && break
  sleep 120
done
echo "[driver] all trainings done at $(date +%H:%M:%S)"

# Evaluate best+final for the three step-matched runs and the five
# ctc/clip variants, one GPU each.
g=0
for spec in \
  "faithful_steps/seed_42:best,final" \
  "faithful_steps/seed_43:best,final" \
  "faithful_steps/seed_44:best,final" \
  "ctc_clip_sens/ctcgtok_seed42:best,final" \
  "ctc_clip_sens/ctcframe_seed42:best,final" \
  "ctc_clip_sens/ctcsum_seed42:best,final" \
  "ctc_clip_sens/clip2p0_seed42:best,final" \
  "ctc_clip_sens/clipinf_seed42:best,final"; do
  dir="${spec%%:*}"; cks="${spec##*:}"
  tag=$(echo "$dir" | tr '/' '_')
  CUDA_VISIBLE_DEVICES=$g python scripts/eval_step_ckpts.py \
      --run-dir "checkpoints/$dir" --ckpts "$cks" \
      > "logs/eval_${tag}.log" 2>&1 &
  g=$(( (g+1) % 8 ))
done
wait
echo "[driver] all evals done at $(date +%H:%M:%S)"
touch results/.round33_evals_complete
