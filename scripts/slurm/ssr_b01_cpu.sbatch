#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=ssr_b01_cpu
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=compile|validate|sanity|freeze|mine|preflight|aggregate|audit|oracle|verify}
RUN_ID=${RUN_ID:?set RUN_ID}
CONFIG=${CONFIG:-configs/ssr/ssr_v1.yaml}
DATASET=${DATASET:-}
FAMILY=${FAMILY:-}

case "$TASK" in
  compile)
    python -m py_compile \
      scripts/analysis/ssr_common.py \
      scripts/analysis/ssr_make_folds.py \
      scripts/analysis/ssr_oof.py \
      scripts/analysis/ssr_mine_pairs.py \
      scripts/analysis/ssr_extract_relations.py \
      scripts/analysis/ssr_aggregate.py \
      scripts/analysis/ssr_b1_gate.py \
      scripts/analysis/ssr_b1_preflight.py \
      scripts/analysis/ssr_exact_shuffle.py \
      scripts/analysis/ssr_verify_b1.py \
      scripts/analysis/ssr_b0_static_sanity.py
    bash -n scripts/slurm/ssr_b01_cpu.sbatch \
      scripts/slurm/ssr_b01_oof.sbatch \
      scripts/slurm/ssr_b01_extract.sbatch \
      scripts/slurm/ssr_b01_b1.sbatch
    ;;
  validate)
    python -m json.tool TARGET_STATE.json >/dev/null
    python -m json.tool artifacts/ssr/v1/b1/preflight_oracle_upper_bound.json >/dev/null
    python -m json.tool artifacts/ssr/v1/B1_DECISION.json >/dev/null
    test "$(python -c 'import json; print(json.load(open("artifacts/ssr/v1/B1_DECISION.json"))["B1_DECISION"])')" = "STOP"
    ;;
  sanity)
    python scripts/analysis/ssr_b0_static_sanity.py --config "$CONFIG" --run-id "$RUN_ID"
    ;;
  freeze)
    EXTRA=()
    if [[ "${ALLOW_UNFROZEN:-0}" == "1" ]]; then EXTRA+=(--allow-unfrozen); fi
    python scripts/analysis/ssr_make_folds.py --config "$CONFIG" --dataset "${DATASET:-all}" "${EXTRA[@]}"
    ;;
  mine)
    python scripts/analysis/ssr_mine_pairs.py --config "$CONFIG" --dataset "$DATASET" --run-id "$RUN_ID"
    ;;
  preflight)
    python scripts/analysis/ssr_b1_preflight.py --config "$CONFIG" --run-id "$RUN_ID"
    ;;
  aggregate)
    python scripts/analysis/ssr_aggregate.py --config "$CONFIG" --dataset "$DATASET" --run-id "$RUN_ID"
    ;;
  audit)
    python scripts/analysis/ssr_b1_gate.py --config "$CONFIG" --gate audit --dataset "$DATASET" --family "$FAMILY" --run-id "$RUN_ID"
    ;;
  oracle)
    python scripts/analysis/ssr_b1_gate.py --config "$CONFIG" --gate oracle --dataset "$DATASET" --family "$FAMILY" --run-id "$RUN_ID"
    ;;
  verify)
    python scripts/analysis/ssr_verify_b1.py --config "$CONFIG" --run-id "$RUN_ID"
    ;;
  *) echo "unknown TASK=$TASK" >&2; exit 2 ;;
esac
