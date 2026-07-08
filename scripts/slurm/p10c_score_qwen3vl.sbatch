#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=p10c_q3vl
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: no --time (JobHeldUser auto-release).
# P10-c: score a NEW-GENERATION open VLM (Qwen3-VL family) on the HateMM
# hateful-train calibration set at K30 (fine) then K4 (coarse), so A-fuse
# (0.5*K30+0.5*K4) is built on the SAME model (matches the 7B/32B/72B recipe).
# One GPU, two passes. Frozen recipe: only the scorer model changes.
#   MODEL Qwen/Qwen3-VL-32B-Instruct | Qwen/Qwen3-VL-30B-A3B-Instruct
#   TAG   p10c-qwen3vl-32b           | p10c-qwen3vl-30ba3b
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
# P10-c runs in the SEPARATE cloned env (transformers 4.57.6, torch 2.6 cu124
# preserved) so the HateVideo env (transformers 4.49) is untouched.
conda activate HateVideoVLM
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:?set MODEL=Qwen/Qwen3-VL-32B-Instruct or -30B-A3B-Instruct}
TAG=${TAG:?set TAG=p10c-qwen3vl-32b or p10c-qwen3vl-30ba3b}
DS=${DS:-HateMM}
SPLITS=${SPLITS:-train}
GTDIR=${GTDIR:-./data/gt_p10hate}
ASRTAG_K30=${ASRTAG_K30:-asrK30_whisper-large-v3}
ASRTAG_K4=${ASRTAG_K4:-asrK4_whisper-large-v3}
LIMIT=${LIMIT:-0}
echo "[p10c] MODEL=$MODEL TAG=$TAG DS=$DS SPLITS=$SPLITS GTDIR=$GTDIR LIMIT=$LIMIT"
python -c "import transformers; print('[p10c] transformers', transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_k () {  # $1=K  $2=M  $3=asr_tag
  echo "=== scoring K=$1 M=$2 (asr=$3) ==="
  python scripts/analysis/p10c_score_segments.py \
    --dataset "$DS" --splits "$SPLITS" --gt_dir "$GTDIR" \
    --num_frames "$2" --num_subclips "$1" \
    --asr_tag "$3" --model "$MODEL" --tag "$TAG" --limit "$LIMIT"
}

# K30/M120 and K4/M16 = the EXACT round-1/round-2 recipes (frozen).
run_k 30 120 "$ASRTAG_K30"
run_k 4  16  "$ASRTAG_K4"
echo "[p10c] done MODEL=$MODEL TAG=$TAG"
