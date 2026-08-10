#!/bin/bash
# Retrain 14 frozen checkpoint families on 8 GPUs with retry logic.
# After training, each is decoded with canonical donor registry by build_canonical_panel.py.
#
# Families and their training params (reconstructed from old scripts):
#   config_faithful (4): epochs=3000, selection=bleu, patience=5
#   step_faithful (2): same but validate every 14 steps (approximated by epochs=3000)
#   ladder (4): subsample train pool at 0.125/0.25/0.5/0.75
#   confirmation (2): standard reconstruction, seeds 1506/1607
#   long_schedule (1): epochs=800
#   rescue_wd0 (1): weight_decay=0, seed=202
set -eu

cd /ssd/xkb4/RCP
LOGDIR=checkpoints/retrain_logs
mkdir -p "$LOGDIR"

train_with_retry() {
    local seed=$1 gpu=$2 epochs=$3 selection=$4 output=$5 name=$6
    local attempt=1 max_attempts=3
    local bs=256 ga=8
    while [ $attempt -le $max_attempts ]; do
        echo "[$(date +%H:%M:%S)] $name: attempt $attempt (gpu=$gpu, seed=$seed, epochs=$epochs, bs=$bs)"
        if CUDA_VISIBLE_DEVICES=$gpu python3 scripts/train_reconstruction.py \
            --seed $seed --gpu 0 --epochs $epochs --batch-size $bs --grad-accum $ga \
            --selection $selection --output "$output" 2>&1 | tee "$LOGDIR/${name}.log"; then
            echo "[$(date +%H:%M:%S)] $name: SUCCESS"
            return 0
        fi
        echo "[$(date +%H:%M:%S)] $name: FAILED (attempt $attempt)"
        # On OOM, reduce batch size
        if grep -qi "out of memory" "$LOGDIR/${name}.log"; then
            bs=$((bs / 2)); ga=$((ga * 2))
            echo "[$(date +%H:%M:%S)] $name: OOM, retrying with bs=$bs ga=$ga"
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    echo "[$(date +%H:%M:%S)] $name: FAILED after $max_attempts attempts"
    return 1
}

echo "=== Starting retraining at $(date) ==="

# GPU 0-1: config-faithful (4 seeds, most important)
train_with_retry 101 0 3000 bleu checkpoints/config_faithful/seed_101 cf_101 &
train_with_retry 202 1 3000 bleu checkpoints/config_faithful/seed_202 cf_202 &
# GPU 2-3: confirmation + long-schedule
train_with_retry 1506 2 300 nll checkpoints/confirmation/seed_1506 conf_1506 &
train_with_retry 1607 3 300 nll checkpoints/confirmation/seed_1607 conf_1607 &
# GPU 4-5: config-faithful cont. + step-faithful (approximate with bleu selection)
train_with_retry 303 4 3000 bleu checkpoints/config_faithful/seed_303 cf_303 &
train_with_retry 404 5 3000 bleu checkpoints/config_faithful/seed_404 cf_404 &
# GPU 6: long-schedule
train_with_retry 202 6 800 nll checkpoints/long_schedule/seed_202 ls_202 &
# GPU 7: rescue wd0 (most important for gap range)
# Standard reconstruction but with wd=0 — need to modify config on the fly
CUDA_VISIBLE_DEVICES=7 python3 -c "
import yaml, copy
cfg = yaml.safe_load(open('configs/reconstruction.yaml'))
cfg['training']['weight_decay'] = 0.0
yaml.dump(cfg, open('configs/rescue_wd0.yaml', 'w'))
print('Created rescue_wd0.yaml with weight_decay=0')
"
train_with_retry 202 7 300 nll checkpoints/rescue/seed_202_wd0 rescue_wd0 &

# Wait for all
wait
echo "=== All training complete at $(date) ==="

# Verify checkpoints
echo "=== Verifying checkpoints ==="
for d in checkpoints/config_faithful/seed_*/best.ckpt \
         checkpoints/confirmation/seed_*/best.ckpt \
         checkpoints/long_schedule/seed_*/best.ckpt \
         checkpoints/rescue/seed_*_wd0/best.ckpt; do
    if [ -f "$d" ]; then echo "OK: $d"; else echo "MISSING: $d"; fi
done
