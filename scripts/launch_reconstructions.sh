#!/usr/bin/env bash
# Launch 14 reconstruction seeds across 7 GPUs (2 per GPU; GPU 7 reserved for distillation).
# Usage: bash scripts/launch_reconstructions.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=(101 202 303 404 505 606 707 808 909 1001 1102 1203 1304 1405)
NUM_GPUS=7  # reserve GPU 7 for distillation

mkdir -p checkpoints/reconstructions

for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=$((i % NUM_GPUS))
    OUTPUT="checkpoints/reconstructions/seed_${SEED}"
    LOG="checkpoints/reconstructions/seed_${SEED}.log"

    if [ -f "${OUTPUT}/best.ckpt" ]; then
        echo "[SKIP] seed $SEED already has best.ckpt"
        continue
    fi

    echo "[GPU $GPU] Launching seed $SEED -> $OUTPUT"
    nohup python scripts/train_reconstruction.py \
        --seed "$SEED" --gpu "$GPU" --epochs 300 \
        --batch-size 128 --grad-accum 2 \
        --output "$OUTPUT" > "$LOG" 2>&1 &
    sleep 3  # short stagger
done

echo "Launched ${#SEEDS[@]} reconstruction seeds across $NUM_GPUS GPUs."
echo "Monitor with: tail -f checkpoints/reconstructions/seed_*.log"
echo "PID list:"
jobs -p
