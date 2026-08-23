#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=cvoi_formal
#SBATCH --output=logging/slurm/%x_%j.out
# Intentionally no --time. Formal execution remains locked until C1--C15 pass.
set -euo pipefail
cd /home/jehc223/Retrieval-hate
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
test -n "${CVOI_FROZEN_CONFIG:-}"
test -n "${CVOI_COMPLETENESS:-}"
test -n "${CVOI_RUN_OUT:-}"
PREFLIGHT="${CVOI_RUN_OUT}.preflight.json"
RESOURCES="${CVOI_RUN_OUT}.resources.jsonl"
python -m scripts.cvoi_acq.preflight --completeness "$CVOI_COMPLETENESS" \
  --projected-bytes 8589934592 --out "$PREFLIGHT" --enforce
python -m scripts.cvoi_acq.formal_runner \
  --outer artifacts/cvoi_acq/premetric-v2/groups-v4/outer_folds.json \
  --inner artifacts/cvoi_acq/premetric-v2/groups-v4/inner_folds.json \
  --out "$CVOI_RUN_OUT" --frozen-config "$CVOI_FROZEN_CONFIG" \
  --completeness "$CVOI_COMPLETENESS" &
FORMAL_PID=$!
GUARD_PID=""
cleanup() {
  [[ -z "$GUARD_PID" ]] || kill "$GUARD_PID" 2>/dev/null || true
  kill "$FORMAL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
python -m scripts.cvoi_acq.resource_guard --pid "$FORMAL_PID" --out "$RESOURCES" \
  --interval 30 --min-disk-bytes 38654705664 --max-rss-bytes 60129542144 \
  --max-gpu-bytes 32212254720 &
GUARD_PID=$!
set +e
wait -n -p FINISHED_PID "$FORMAL_PID" "$GUARD_PID"
FIRST_RC=$?
if [[ "$FINISHED_PID" == "$GUARD_PID" ]]; then
  kill "$FORMAL_PID" 2>/dev/null || true
  wait "$FORMAL_PID" 2>/dev/null
  FORMAL_RC=$?
  GUARD_RC=$FIRST_RC
  [[ "$GUARD_RC" -ne 0 ]] || GUARD_RC=97
else
  FORMAL_RC=$FIRST_RC
  wait "$GUARD_PID"
  GUARD_RC=$?
fi
set -e
trap - EXIT INT TERM
if [[ "$FORMAL_RC" -ne 0 ]]; then exit "$FORMAL_RC"; fi
exit "$GUARD_RC"
