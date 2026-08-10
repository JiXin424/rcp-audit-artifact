#!/usr/bin/env bash
# Launch 15 distillation students across 8 GPUs (one per GPU at a time;
# reconstructions are already using GPU 0-6 with 2 tasks each).
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=(
    "0.0:101" "0.0:202" "0.0:303"
    "0.25:101" "0.25:202" "0.25:303"
    "0.5:101" "0.5:202" "0.5:303"
    "0.75:101" "0.75:202" "0.75:303"
    "1.0:101" "1.0:202" "1.0:303"
)
NUM_GPUS=8

mkdir -p checkpoints/distillation

for i in "${!TASKS[@]}"; do
    T=${TASKS[$i]}
    A=${T%:*}
    S=${T##*:}
    GPU=$((i % NUM_GPUS))
    OUTPUT="checkpoints/distillation/alpha_${A}_seed_${S}"
    LOG="checkpoints/distillation/alpha_${A}_seed_${S}.log"

    if [ -f "${OUTPUT}/best.ckpt" ]; then
        echo "[SKIP] alpha=$A seed=$S already done"
        continue
    fi

    echo "[GPU $GPU] Launching alpha=$A seed=$S -> $OUTPUT"
    nohup python scripts/train_distillation.py \
        --alpha "$A" --seed "$S" --gpu "$GPU" --epochs 300 \
        --batch-size 32 --grad-accum 8 \
        --output "$OUTPUT" > "$LOG" 2>&1 &
    sleep 5
done

echo "Launched ${#TASKS[@]} distillation students."
jobs -p
