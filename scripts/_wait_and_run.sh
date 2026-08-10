#!/usr/bin/env bash
# Wait for all training processes to finish, then run the post-training pipeline.
set -uo pipefail
cd /ssd/xkb4/RCP
LOG=checkpoints/_wait_monitor.log
echo "=== Wait monitor started at $(date) ===" >> "$LOG"

while true; do
    # Count active training processes (subprocess invocations of src.training.train_*)
    N_RECO=$(pgrep -fa "src.training.train_matched" 2>/dev/null | wc -l)
    N_DISTILL=$(pgrep -fa "src.training.train_distillation" 2>/dev/null | wc -l)
    N_TOTAL=$((N_RECO + N_DISTILL))
    
    echo "[$(date '+%H:%M:%S')] active processes: reco=$N_RECO distill=$N_DISTILL total=$N_TOTAL" >> "$LOG"
    
    if [ "$N_TOTAL" -eq 0 ]; then
        echo "[$(date)] All training processes finished." >> "$LOG"
        break
    fi
    
    sleep 120  # check every 2 minutes
done

echo "=== Wait monitor complete at $(date) ===" >> "$LOG"
echo "Now running post-training pipeline..." >> "$LOG"
bash /ssd/xkb4/RCP/scripts/run_post_training.sh >> "$LOG" 2>&1
