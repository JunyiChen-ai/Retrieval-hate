#!/usr/bin/env bash
# Run-time check of the bias-table gradient (README section 6). Submit on the target host:
#   ssh <host> 'cd ~/Retrieval-hate && sbatch -p local-<node> -J sacma_bench \
#     -o runs/20261031_scale_aware_cma/bench_bias_%j.out experiments/20261031_scale_aware_cma/launch/sbatch_bench.sh'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$HOME/Retrieval-hate"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds)"
exec "$HOME/miniconda3/envs/HateVideo/bin/python" experiments/20261031_scale_aware_cma/bench_bias.py
