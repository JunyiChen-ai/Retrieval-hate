#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=gen_archive
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# DESIGN_iter3 E0b (MLLM Role 1): structured video archives with Qwen2.5-VL-7B
# + CLIP text-encoder retrieval keys.
#
# Usage:
#   smoke : LIMIT=5 SPLITS=train sbatch scripts/slurm/gen_archive.sbatch MHC,MHC_zh
#   full  : sbatch scripts/slurm/gen_archive.sbatch MHC
#           sbatch scripts/slurm/gen_archive.sbatch MHC_zh
#
# Env knobs:
#   LIMIT   (default 0 = all)   items per split (smoke test)
#   SPLITS  (default train,val,test)
#   RETRY_FAILED=1              re-generate parse-failed entries
#   MAX_NEW_TOKENS (default 600)  raise for retry of truncated JSON outputs
#   STAGE   (default all)       generate | reparse | encode | all
#   PUSH_B2=1                   push archives + .pt to B2 on success (full runs)
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASETS=${1:-MHC}
LIMIT=${LIMIT:-0}
SPLITS=${SPLITS:-train,val,test}
RETRY_FLAG=""
if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
    RETRY_FLAG="--retry_failed"
fi

echo "[gen_archive] DATASETS=${DATASETS} SPLITS=${SPLITS} LIMIT=${LIMIT} RETRY=${RETRY_FAILED:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IFS=',' read -ra DS_ARR <<< "$DATASETS"
for DS in "${DS_ARR[@]}"; do
    echo "[gen_archive] === dataset ${DS} ==="
    python src/utils/generate_video_archive_HF.py \
        --dataset "$DS" \
        --stage "${STAGE:-all}" \
        --splits "$SPLITS" \
        --limit "$LIMIT" \
        --max_new_tokens "${MAX_NEW_TOKENS:-600}" \
        --device cuda \
        $RETRY_FLAG

    if [[ "${PUSH_B2:-0}" == "1" ]]; then
        bash scripts/b2_push.sh "data/Archive/$DS" "archives/$DS"
        for PT in data/CLIP_Embedding/"$DS"/*_archive_*_HF.pt; do
            [[ -e "$PT" ]] || continue
            bash scripts/b2_push.sh "$PT" "embeddings/$DS/$(basename "$PT")"
        done
    fi
done

echo "[gen_archive] done for DATASETS=${DATASETS}"
