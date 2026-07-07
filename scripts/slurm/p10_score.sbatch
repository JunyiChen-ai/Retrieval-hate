#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p10_score
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: no --time (JobHeldUser auto-release).
# P10 localization-amplification MLLM window scorer (research-wiki/EXP_p10_loc_amplify.md).
# Calibration = HateMM hateful videos (gt_p10hate); test = HateClipSeg (frozen P6 harness).
#   SCRIPT   score_segments_mllm.py (anchor, P6 prompt) | p10_score_segments.py (variants)
#   DS HateMM|HateClipSeg  SPLITS  GTDIR  K  M  ASRTAG  MODEL  PROMPT(p10 scorer only)  QUANT
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

SCRIPT=${SCRIPT:-scripts/analysis/score_segments_mllm.py}
DS=${DS:-HateMM}
SPLITS=${SPLITS:-train,val,test}
GTDIR=${GTDIR:-./data/gt_p10hate}
K=${K:-30}; M=${M:-120}
ASRTAG=${ASRTAG:-asrK30_whisper-large-v3}
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
PROMPT=${PROMPT:-p6}
QUANT=${QUANT:-none}
LIMIT=${LIMIT:-0}
echo "[p10_score] SCRIPT=$SCRIPT DS=$DS SPLITS=$SPLITS GTDIR=$GTDIR K=$K M=$M ASR=$ASRTAG MODEL=$MODEL PROMPT=$PROMPT QUANT=$QUANT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA=""
if [[ "$SCRIPT" == *"p10_score_segments.py" ]]; then EXTRA="--prompt $PROMPT --quant $QUANT"; fi

python "$SCRIPT" \
  --dataset "$DS" --splits "$SPLITS" --gt_dir "$GTDIR" \
  --num_frames "$M" --num_subclips "$K" \
  --asr_tag "$ASRTAG" --model "$MODEL" --limit "$LIMIT" $EXTRA
echo "[p10_score] done"
