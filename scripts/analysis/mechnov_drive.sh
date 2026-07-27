#!/bin/bash
# Sequential driver for the remaining MECHNOV pair-verify cells.
# Each cell is a short-lived process that serialises its own result immediately,
# so a login-node reap costs at most one cell. Retries a reaped cell up to 3 times.
# Runs detached (setsid) so it is not tied to any interactive shell lifetime.
cd /data/jehc223/RGCL || exit 1
source ~/.bashrc 2>/dev/null
conda activate HateVideo 2>/dev/null
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
LOG=scripts/analysis/.mechnov_drive.log
: > "$LOG"

run_cell () {   # $1=dataset $2=space
  local part="scripts/analysis/mechnov_parts/$1_$2.json"
  for try in 1 2 3; do
    if [ -s "$part" ]; then echo "[drive] $1/$2 already present" >> "$LOG"; return 0; fi
    echo "[drive] $1/$2 attempt $try $(date +%T)" >> "$LOG"
    python scripts/analysis/mechnov_pairverify_runner.py --dataset "$1" --space "$2" >> "$LOG" 2>&1
    echo "[drive] $1/$2 attempt $try exit $? $(date +%T)" >> "$LOG"
  done
  [ -s "$part" ]
}

for cell in "zh fused" "hatemm text" "hatemm img" "zh text" "zh img" "en text" "en img"; do
  run_cell $cell
done

for d in hatemm zh en; do
  python scripts/analysis/mechnov_pairverify_runner.py --dataset "$d" --merge >> "$LOG" 2>&1
done
echo "[drive] ALL CELLS DONE $(date +%T)" >> "$LOG"
