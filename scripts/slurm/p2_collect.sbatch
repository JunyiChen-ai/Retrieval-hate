#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=p2_collect
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
# CPU-only: gate (25% val threshold) + top-60 retrieval + floor reproduction gate
# + dedup (query,neighbor) pairs for the P2 neighbor-rerank experiment.
# (research-wiki/EXP_p2_neighbor_rerank.md)
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

python scripts/analysis/p2_rerank_eval.py --mode collect \
    --datasets "${DATASETS:-MHC,MHC_zh}" \
    --cache_dir /data/jehc223/RGCL/scripts/analysis/p2_out

echo "[p2_collect] done"
