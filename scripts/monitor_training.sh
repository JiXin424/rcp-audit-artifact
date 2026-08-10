#!/usr/bin/env bash
# Quick status of all running training tasks.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Reconstruction seeds ==="
for d in checkpoints/reconstructions/seed_*/; do
    name=$(basename "$d")
    logf="checkpoints/reconstructions/${name}.log"
    [ -f "$logf" ] || continue
    last=$(grep -oE "epoch [0-9]+/300.*best=[0-9.]+@[0-9]+" "$logf" 2>/dev/null | tail -1)
    if [ -z "$last" ]; then
        nlines=$(wc -l < "$logf")
        echo "  $name: starting (log $nlines lines)"
    else
        echo "  $name: $last"
    fi
done

echo ""
echo "=== Distillation students ==="
for d in checkpoints/distillation/alpha_*/; do
    name=$(basename "$d")
    logf="checkpoints/distillation/${name}.log"
    [ -f "$logf" ] || continue
    last=$(grep -oE "epoch [0-9]+/300.*best=[0-9.]+@[0-9]+" "$logf" 2>/dev/null | tail -1)
    if [ -z "$last" ]; then
        nlines=$(wc -l < "$logf")
        echo "  $name: starting (log $nlines lines)"
    else
        echo "  $name: $last"
    fi
done

echo ""
echo "=== GPU usage ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

echo ""
echo "=== Process count ==="
echo "train_matched: $(ps -eo cmd | grep -c 'src.training.train_matched')"
echo "train_distillation: $(ps -eo cmd | grep -c 'src.training.train_distillation')"
