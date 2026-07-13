#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=sav_f0
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# NOTE: intentionally NO --time (project policy). 1 GPU (frozen forward only) + 16 CPU
# (the F-G1 U-1 100,352-d LogisticRegressionCV is the CPU bottleneck; n_jobs=-1).
#
# SAV (C2) F-G0 extraction -> two-tier reproduction guard -> F-G1 statistics engine.
# Authority: research-wiki/experiments/exp-sav-f0.md (Rev-2a, APPROVED).
# SINGLE-SUBMIT-PER-LINEAGE ceremony: this sbatch runs the whole F-G0/F-G1 chain serially;
# the guard's PRIMARY min-cosine>=0.999 gates the probe (fail-closed jq between stages).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

RUN_ID=SAV-F0-FG0-FG1
EXPECTED=SAV-F0-FG0-FG1
DATASETS=HateMM,MHC,MHC_zh

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RUN_ID="$RUN_ID" EXPECTED="$EXPECTED" DATASETS="$DATASETS" LIMIT=0 \
  bash scripts/wrappers/sav_f0.sh

echo "======== sav_f0 ALL DONE (${SLURM_JOB_ID:-nojob}) ========"
