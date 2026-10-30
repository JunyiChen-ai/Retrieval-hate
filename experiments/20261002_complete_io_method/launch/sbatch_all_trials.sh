#!/usr/bin/env bash
# Complete-combination evaluation of every trial (val + test per setting), no training.
# Submit on the target host: sbatch -p <partition> -J <name> -o runs/20261002_complete_io_method/all_trials/slurm_<corpus>_<seed>_%j.out \
#   experiments/20261002_complete_io_method/launch/sbatch_all_trials.sh <corpus> [seed]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -euo pipefail
cd "$HOME/Retrieval-hate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date -Iseconds) | corpus $1 seed ${2:-all}"
SOURCE=runs/20261001_inside_outside_backbone
[ "$1" = dehate ] && SOURCE=runs/20261002_complete_io_method
exec "$HOME/miniconda3/envs/HateVideo/bin/python" \
  experiments/20261002_complete_io_method/launch/run_corpus.py --corpus "$1" ${2:+--seed "$2"} \
  --source-root "$SOURCE" --out-root runs/20261002_complete_io_method/all_trials --all-trials
