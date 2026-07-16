#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
# CPU-ONLY: intentionally NO --gres (audio decode + openSMILE are pure CPU/ffmpeg; zero GPU).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=apx_egemaps
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/apx_egemaps_%j.log
# NOTE: intentionally NO --time limit (project policy). PENDING (JobHeldUser) -> wait for
# auto-release, never force. Per-video checkpointing (data/audio/HateMM/egemaps_v02/<id>.npy)
# makes a reap/requeue resume cleanly from where it left off.
#
# APX audio gate — Stage A: extract openSMILE eGeMAPSv02 88-d functionals for HateMM train U val.
# Design refine-logs/WAVE3_CANDIDATES.md CANDIDATE 3 (d). Raw videos stay LOCAL; features-only output.
#   sbatch scripts/slurm/apx_egemaps_extract.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
# ffmpeg lives in the ExMRD env on the shared filesystem; APPEND it to PATH (CPU decode only).
# MUST append (not prepend): prepending shadows HateVideo's `python`/opensmile with ExMRD's.
export PATH="$PATH:/data/jehc223/miniconda3/envs/ExMRD/bin"
# Call the HateVideo interpreter by ABSOLUTE path so the env is never ambiguous.
PY=/data/jehc223/miniconda3/envs/HateVideo/bin/python
export PYTHONUNBUFFERED=1
export APX_NPROC=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "================================================================================"
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "ffmpeg=$(command -v ffmpeg)  python=$PY"
echo "================================================================================"
# Fail fast with a clear message if the interpreter is missing the deps (guards the 13200 regression).
"$PY" -c "import opensmile, soundfile; print('preflight ok opensmile', opensmile.__version__)"
"$PY" scripts/analysis/apx_extract_egemaps.py
echo "apx eGeMAPS extraction complete jobid=${SLURM_JOB_ID:-none} end=$(date -u +%FT%TZ)"
