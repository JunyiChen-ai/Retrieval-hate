#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=gen_archive_v2
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Archive prompt v2 (AUDIT_archive_faithfulness.md fixes: target-groups recall
# + no-evidence-no-mechanism). Writes ONLY to data/Archive/<DS>/v2/ and
# data/CLIP_Embedding/<DS>/v2/ -- v1 artefacts are never touched.
#
# Usage:
#   smoke : LIMIT=30 SPLITS=train STAGE=generate sbatch scripts/slurm/gen_archive_v2.sbatch MHC_zh
#   full  : sbatch scripts/slurm/gen_archive_v2.sbatch MHC_zh
#           sbatch scripts/slurm/gen_archive_v2.sbatch MHC
#
# Env knobs: LIMIT (0=all), SPLITS (train,val,test), STAGE (generate|reparse|
# encode|all), RETRY_FAILED=1, MAX_NEW_TOKENS (600), PUSH_B2=1.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASETS=${1:-MHC_zh}
LIMIT=${LIMIT:-0}
SPLITS=${SPLITS:-train,val,test}
RETRY_FLAG=""
if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    RETRY_FLAG="--retry_failed"
fi

echo "[gen_archive_v2] DATASETS=${DATASETS} SPLITS=${SPLITS} LIMIT=${LIMIT} STAGE=${STAGE:-all} RETRY=${RETRY_FAILED:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IFS=',' read -ra DS_ARR <<< "$DATASETS"
for DS in "${DS_ARR[@]}"; do
    echo "[gen_archive_v2] === dataset ${DS} ==="
    python src/utils/generate_video_archive_HF.py \
        --dataset "$DS" \
        --prompt_version v2 \
        --stage "${STAGE:-all}" \
        --splits "$SPLITS" \
        --limit "$LIMIT" \
        --max_new_tokens "${MAX_NEW_TOKENS:-600}" \
        --device cuda \
        $RETRY_FLAG

    if [[ "${PUSH_B2:-0}" == "1" ]]; then
        bash scripts/b2_push.sh "data/Archive/$DS/v2" "archives/$DS/v2"
        for PT in data/CLIP_Embedding/"$DS"/v2/*_archive_*_HF.pt; do
            [[ -e "$PT" ]] || continue
            bash scripts/b2_push.sh "$PT" "embeddings/$DS/v2/$(basename "$PT")"
        done
    fi
done

echo "[gen_archive_v2] done for DATASETS=${DATASETS}"
