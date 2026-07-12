#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lbscgp_g0_gpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=realfold|replay}
RUN_ID=${RUN_ID:?set frozen RUN_ID}
DATASET=${DATASET:?set DATASET=MHC_zh}
OUTER_FOLD=${OUTER_FOLD:?set OUTER_FOLD=4}
CONFIG=${CONFIG:-configs/lb_scgp/lb_scgp_v1.json}

if [[ "$DATASET" != "MHC_zh" || "$OUTER_FOLD" != "4" ]]; then
  echo "Refusing unauthorized G0 GPU task" >&2
  exit 2
fi

# Round3 contract: realfold/replay use whole-video memory only. Segment caches
# are not inputs; lambda_seg must remain 0 and segment_cache must remain None.
case "$TASK" in
  realfold)
    python scripts/analysis/lb_scgp_g0.py --config "$CONFIG" --task realfold \
      --dataset "$DATASET" --outer-fold "$OUTER_FOLD" --run-id "$RUN_ID"
    ;;
  replay)
    EXPECTED_REPLAY_RUN_ID="$(python - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
run_id = cfg.get("lineage", {}).get("run_ids", {}).get("replay")
if not isinstance(run_id, str) or not run_id:
    raise SystemExit("missing lineage.run_ids.replay")
print(run_id)
PY
)"
    if [[ "$RUN_ID" != "$EXPECTED_REPLAY_RUN_ID" ]]; then
      echo "Refusing unauthorized real replay RUN_ID for CONFIG" >&2
      exit 2
    fi
    python scripts/analysis/lb_scgp_real_replay.py --config "$CONFIG" \
      --task replay --run-id "$RUN_ID"
    ;;
  *)
    echo "Refusing unauthorized G0 GPU task" >&2
    exit 2
    ;;
esac
