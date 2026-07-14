#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=b5probe
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.  NO --gres (CPU-only; zero formal GPU).
#
# B5 operating-point conversion probe (G0-cond).  CPU-only offline analysis:
# reloads the 11 safekept heads from refine-logs/b5_ckpt_snapshot/, replays kNN
# retrieval off cached MHC_zh features (CPU faiss IndexFlatIP), runs the STRICT-ORDER
# probe (G-repro -> freeze tau -> oracle kill-switch -> honest preview -> D3 bootstrap).
# Wrapped in sbatch because the >1-min sustained CPU load must not run on the login node
# (login node reaps non-SLURM compute).  Applies NO pass/fail interpretation.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # hard CPU-only

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
python ./scripts/analysis/b5_conv_probe.py
echo "end=$(date -u +%FT%TZ) exit=$?"
