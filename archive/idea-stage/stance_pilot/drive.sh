#!/usr/bin/env bash
# STANCE PILOT driver: wait for submit -> poll batch -> fetch -> score. Idempotent-ish.
set -u
TAG="$1"
cd /home/jehc223/Retrieval-hate
source ~/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
D=idea-stage/stance_pilot
echo "[drive] waiting for batch_meta_${TAG}.json"
until [ -f "$D/batch_meta_${TAG}.json" ]; do
  pgrep -f "run_pilot.py submit .*--tag ${TAG}" >/dev/null || { echo "[drive] submit died"; exit 1; }
  sleep 20
done
echo "[drive] submitted: $(cat $D/batch_meta_${TAG}.json)"
python $D/run_pilot.py poll  --tag "$TAG"
python $D/run_pilot.py fetch --tag "$TAG"
if [ -f "$D/pred_${TAG}.jsonl" ]; then
  python $D/score.py --tag "$TAG"
fi
echo "[drive] done"
