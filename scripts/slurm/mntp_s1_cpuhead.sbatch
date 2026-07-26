#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=mntp_s1_cpuhead
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. NO --gres: this job requests NO GPU, so it does
# not consume either of the user's 2 GPU slots. 8 CPUs (never 16 -- the aggregate
# submit-time cap wedge).
#
# KS-MNTP-2 (refine-logs/MNTP_FORENSIC_RECON.md §5.2): CPU head dev screen for the S1
# bidir+meanpool arm. 3 arms x 2 datasets x 3 seeds = 18 head trainings, ~52 s each.
# Paired CPU-to-CPU against a CPU-trained causal floor (recon §4.3: CPU-vs-CUDA head
# drift is -0.0031, so a CPU arm may never be paired against a banked GPU floor).
# DEV ONLY. ZERO test-touch (three independent belts inside the python script).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export MNTP_THREADS=6

# MNTP_ARMS (optional, comma-separated) restricts which arms run. The causal and bidir
# arms need no S1 cache, so they can be measured while the S1 extraction is still queued;
# a later run with MNTP_ARMS=meanpool merges into the same OUT json.
python scripts/analysis/mntp_s1_cpuhead.py 2>&1

echo "[mntp_s1_cpuhead] ALL DONE"
