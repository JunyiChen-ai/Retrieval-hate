#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mhc_segasr
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# Segment-aligned Whisper ASR for the multimodal sub-clip keys
# (EXP_mm_segment_keys phase 1). Writes ONLY data/ASR/<DS>/*.jsonl —
# no existing cache is touched. Resume-safe (skips already-done ids).
#
# Usage:
#   sbatch scripts/slurm/gen_segment_asr.sbatch                 # both datasets, full
#   LIMIT=20 sbatch scripts/slurm/gen_segment_asr.sbatch        # smoke
#   DATASETS=MHC SPLITS=train sbatch scripts/slurm/gen_segment_asr.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

DATASETS=${DATASETS:-MHC,MHC_zh}
SPLITS=${SPLITS:-train,val,test}
LIMIT=${LIMIT:-0}
MODEL=${MODEL:-openai/whisper-large-v3}
BATCH=${BATCH:-8}

GEN_MM=${GEN_MM:-0}   # 1 = chain the multimodal sub-clip cache build (phase 2)

IFS=',' read -ra DS_ARR <<< "$DATASETS"
for DS in "${DS_ARR[@]}"; do
    echo "[segasr] DATASET=$DS SPLITS=$SPLITS LIMIT=$LIMIT MODEL=$MODEL"
    python src/utils/generate_segment_asr_HF.py \
        --dataset "$DS" \
        --model "$MODEL" \
        --splits "$SPLITS" \
        --limit "$LIMIT" \
        --batch_size "$BATCH" \
        --timestamps word \
        --device cuda
done

for DS in "${DS_ARR[@]}"; do
    for SP in train dev_seen test_seen; do
        if [ -f "data/ASR/$DS/${SP}_asrK4_whisper-large-v3.jsonl" ]; then
            python scripts/analysis/asr_qc.py --dataset "$DS" --split "$SP" --samples 2 || true
        fi
    done
done

if [ "$GEN_MM" = "1" ]; then
    for DS in "${DS_ARR[@]}"; do
        echo "[segasr->submm] DATASET=$DS SPLITS=$SPLITS"
        python src/utils/generate_subclip_mm_embedding_HF.py \
            --dataset "$DS" \
            --splits "$SPLITS" \
            --device cuda
        bash scripts/b2_push.sh "data/CLIP_Embedding/$DS" "embeddings/$DS" || true
    done
fi

echo "[segasr] done DATASETS=$DATASETS LIMIT=$LIMIT GEN_MM=$GEN_MM"
