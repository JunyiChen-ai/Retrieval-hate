#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=c04_a0t_small_v1_v4
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally no time directive; one GPU and one strictly serial process.
# No array, dependency, chained submission, release, or automatic resubmission.
set -euo pipefail

cd /data/jehc223/RGCL
bash scripts/wrappers/c04_a0t_small_v1_v4.sh
