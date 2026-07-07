#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=p9train
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# P9 (research-wiki/EXP_p9_lmm_rgcl_video.md): LoRA-SFT via RA-HMD's sft_classifier stage
# (rgcl OFF = runnable). Usage:
#   smoke : CONFIG=<abs yaml> MAX_STEPS=20 sbatch scripts/slurm/p9_train.sbatch
#   full  : CONFIG=<abs yaml> sbatch scripts/slurm/p9_train.sbatch
set -euo pipefail

RGCL_ROOT=/data/jehc223/RGCL
LF_ROOT=${RGCL_ROOT}/RA-HMD/LLAMA-FACTORY-Ver202512
CONFIG=${CONFIG:?set CONFIG=<absolute path to yaml>}
MAX_STEPS=${MAX_STEPS:-}

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
export DISABLE_VERSION_CHECK=1
export CUDA_HOME=${LF_ROOT}/.cuda_home_shim
# 8-frame video vision-tower forward is memory-heavy; reduce fragmentation OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ${RGCL_ROOT}/scripts/disk_guard.sh || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# smoke: cap steps via max_steps ONLY (a NEW key — never duplicate an existing key, which
# yaml rejects with ConstructorError). HF Trainer: positive max_steps overrides num_train_epochs.
RUN_CONFIG="${CONFIG}"
if [ -n "${MAX_STEPS}" ]; then
    RUN_CONFIG=$(mktemp --suffix=.yaml)
    cp "${CONFIG}" "${RUN_CONFIG}"
    echo "max_steps: ${MAX_STEPS}" >> "${RUN_CONFIG}"
    echo "[p9train] SMOKE max_steps=${MAX_STEPS} config=${RUN_CONFIG}"
fi

cd ${LF_ROOT}
echo "[p9train] python src/train.py ${RUN_CONFIG}"
time python src/train.py "${RUN_CONFIG}"
echo "[p9train] DONE"
