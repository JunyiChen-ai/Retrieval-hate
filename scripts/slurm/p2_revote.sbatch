#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=p2_revote
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
# CPU-only: apply A/B/C/D conditions with the frozen revote rule over the cached
# retrieval + MLLM verdicts; write results JSON + markdown block.
# (research-wiki/EXP_p2_neighbor_rerank.md)
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

python scripts/analysis/p2_rerank_eval.py --mode revote \
    --datasets "${DATASETS:-MHC,MHC_zh}" \
    --cache_dir /data/jehc223/RGCL/scripts/analysis/p2_out \
    --out_json /data/jehc223/RGCL/scripts/analysis/p2_out/p2_results.json \
    --out_block /data/jehc223/RGCL/scripts/analysis/p2_out/p2_results_block.md

echo "[p2_revote] done"
