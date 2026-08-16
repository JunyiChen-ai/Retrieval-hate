#!/usr/bin/env bash
# LIKELIHOOD PROBE -- formal run, single submission. Arms are run in sequence so only one
# 7B model is resident at a time.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
cd /home/jehc223/Retrieval-hate/idea-stage/likelihood_probe
for ARM in A1 A2 B1 C1; do
  for SET in ctrl eval; do
    echo "=== $(date +%H:%M:%S) START $ARM/$SET ==="
    python run_likelihood.py --arm "$ARM" --set "$SET" 2>&1 | grep -v "it/s\]"
    echo "=== $(date +%H:%M:%S) END $ARM/$SET rc=$? ==="
  done
done
echo "=== $(date +%H:%M:%S) ALL ARMS DONE ==="
python score_likelihood.py --arms A1 A2 B1 C1 2>&1
echo "=== $(date +%H:%M:%S) SCORED ==="
