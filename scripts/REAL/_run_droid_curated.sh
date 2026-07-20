#!/bin/bash
# Wrapper: activate venv, run the 1-GPU curated DROID pretrain, tee to a
# timestamped log. Launch inside a detached screen session.
set -uo pipefail
cd /home/khanh/work/Seer
source .venv/bin/activate
mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
LOG="logs/droid_curated_${ts}.log"
ln -sf "droid_curated_${ts}.log" logs/droid_curated_latest.log
echo "=== launch $(date -Is) ===" | tee "$LOG"
echo "python: $(which python)" | tee -a "$LOG"
bash scripts/REAL/single_node_1gpu_curated.sh 2>&1 | tee -a "$LOG"
echo "=== exit ${PIPESTATUS[0]} at $(date -Is) ===" | tee -a "$LOG"
