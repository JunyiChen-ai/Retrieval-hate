#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p6_loc
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
# P6 span-free MLLM localization on HateClipSeg (research-wiki/EXP_p6_mllm_localization.md).
#   STAGE=asr   -> Whisper K=30 window ASR
#   STAGE=score -> frozen P3 MLLM scorer at K=30/M=120 (needs the ASR output)
# K=30, M=120 = density-matched window alignment (only --num_frames/--num_subclips differ
# from P3; prompt/model/decoding frozen).
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

STAGE=${STAGE:?set STAGE=asr or score}
K=${K:-30}
M=${M:-120}
LIMIT=${LIMIT:-0}
echo "[p6] STAGE=$STAGE K=$K M=$M LIMIT=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ "$STAGE" == "asr" ]]; then
  python src/utils/generate_segment_asr_HF.py \
    --dataset HateClipSeg --splits test \
    --num_frames "$M" --num_subclips "$K" \
    --timestamps word --limit "$LIMIT" --device cuda
elif [[ "$STAGE" == "score" ]]; then
  python scripts/analysis/score_segments_mllm.py \
    --dataset HateClipSeg --splits test \
    --num_frames "$M" --num_subclips "$K" \
    --asr_tag "asrK${K}_whisper-large-v3" \
    --limit "$LIMIT" --device cuda
else
  echo "unknown STAGE=$STAGE"; exit 1
fi
echo "[p6] done STAGE=$STAGE"
