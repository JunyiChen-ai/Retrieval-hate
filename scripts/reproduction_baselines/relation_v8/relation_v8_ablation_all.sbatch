#!/usr/bin/env bash
#SBATCH --job-name=relation-v8-ablate
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/reproduction/relation_v8/ablation-slurm-%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
bash scripts/reproduction_baselines/relation_v8/run_ablation_all.sh
