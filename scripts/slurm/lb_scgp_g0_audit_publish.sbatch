#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --job-name=lbscgp_g0_audit_publish
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time: project policy.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
bash scripts/disk_guard.sh || true

TASK=${TASK:?set TASK=audit-publish}
RUN_ID=${RUN_ID:?set RUN_ID matching CONFIG lineage.run_ids.code_audit}
CONFIG=${CONFIG:?set CONFIG=configs/lb_scgp/lb_scgp_v4.json}
REVIEW=${REVIEW:?set REVIEW=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md}
REVIEW_RECORD=${REVIEW_RECORD:?set REVIEW_RECORD=refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json}

if [[ "$TASK" != "audit-publish" ]]; then
  echo "Refusing unauthorized task: $TASK" >&2
  exit 2
fi

EXPECTED_RUN_ID="$(jq -r '.lineage.run_ids.code_audit // empty' "$CONFIG")"
EXPECTED_VERSION="$(jq -r '.lineage.version // empty' "$CONFIG")"
EXPECTED_REVIEW="$(jq -r '.lineage.review_report_path // empty' "$CONFIG")"
EXPECTED_REVIEW_RECORD="$(jq -r '.lineage.review_record_path // empty' "$CONFIG")"

if [[ "$EXPECTED_VERSION" != "v4" && "$EXPECTED_VERSION" != "v5" ]]; then
  echo "Refusing unauthorized code-audit lineage for CONFIG" >&2
  exit 2
fi
if [[ -z "$EXPECTED_RUN_ID" || "$RUN_ID" != "$EXPECTED_RUN_ID" ]]; then
  echo "Refusing unauthorized code-audit RUN_ID for CONFIG" >&2
  exit 2
fi
if [[ "$REVIEW" != "$EXPECTED_REVIEW" || "$REVIEW_RECORD" != "$EXPECTED_REVIEW_RECORD" ]]; then
  echo "Refusing non-config review or review-record path" >&2
  exit 2
fi

python scripts/analysis/lb_scgp_independent_verify.py --config "$CONFIG" \
  --task audit-publish --run-id "$RUN_ID" \
  --review "$REVIEW" --review-record "$REVIEW_RECORD"
