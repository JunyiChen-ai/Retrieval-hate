#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=c04_a0t_small_v1_v7_reconcile
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# Intentionally CPU-only and without a time directive.
# No GPU, array, dependency, chained submission, release, or resubmission.
set -euo pipefail

cd /data/jehc223/RGCL
bash scripts/wrappers/c04_a0t_small_v1_v7_reconcile.sh
