#!/bin/bash
# Wrapper: activate the uv venv, run pretrain.sh from the repo root, tee to a
# timestamped log. Meant to be launched inside a detached screen session.
set -uo pipefail

cd /home/khanh/work/Seer
source .venv/bin/activate

mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
LOG="logs/pretrain_${ts}.log"
ln -sf "pretrain_${ts}.log" logs/pretrain_latest.log

echo "=== launch $(date -Is) ===" | tee "$LOG"
echo "python: $(which python)" | tee -a "$LOG"
bash scripts/LIBERO_LONG/Seer/pretrain.sh 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
echo "=== exit $code at $(date -Is) ===" | tee -a "$LOG"
