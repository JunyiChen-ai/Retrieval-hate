#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=s2s_extract
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/s2s_extract_%j.log
# NOTE: intentionally NO --time limit (project policy). Single GPU. One job runs BOTH
# datasets sequentially. PENDING (JobHeldUser) -> wait for auto-release, never force.
#
# S2S Stage-E extraction (prereg exp-s2s-r3.md r1; spec S2S_PROBE_DESIGN.md r1 §3).
# Extracts per-frame-group Qwen2.5-VL vectors for HateMM + MHC-EN (train/val/test), runs
# the HARD gates inline (A1 temporal control + grid gate, G-decomp, G-recon), and HALTs on
# any violation. NO submission is authorised until the three scripts pass independent code
# review and are hash-pinned (S2S_PROBE_DESIGN.md §10).
#
# Smoke first (throwaway path, --limit 1, no real-path artifact, G-recon skipped):
#   SMOKE=1 sbatch scripts/slurm/s2s_extract.sbatch
# Real run (default):
#   sbatch scripts/slurm/s2s_extract.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

NUM_FRAMES=${NUM_FRAMES:-8}
SMOKE=${SMOKE:-0}
EXTRACT=scripts/analysis/s2s_extract.py

echo "================================================================================"
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "[s2s_extract] script sha256:"
sha256sum "$EXTRACT" scripts/slurm/s2s_extract.sbatch
echo "[s2s_extract] NUM_FRAMES=${NUM_FRAMES} SMOKE=${SMOKE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "================================================================================"

if [ "$SMOKE" = "1" ]; then
    SMOKE_OUT="/data/jehc223/RGCL/slurm/logs/s2s_smoke_out_${SLURM_JOB_ID:-local}"
    echo "[s2s_extract] SMOKE run: --limit 1 -> throwaway ${SMOKE_OUT} (G-recon skipped)"
    for DS in HateMM MHC; do
        python "$EXTRACT" --dataset "$DS" --splits train,val,test \
            --num_frames "$NUM_FRAMES" --limit 1 --out_root "$SMOKE_OUT" --device cuda
    done
    echo "[s2s_extract] SMOKE done. Inspect gate lines above; throwaway artifacts under ${SMOKE_OUT}."
else
    for DS in HateMM MHC; do
        echo "[s2s_extract] === dataset ${DS} ==="
        python "$EXTRACT" --dataset "$DS" --splits train,val,test \
            --num_frames "$NUM_FRAMES" --device cuda
    done
    echo "[s2s_extract] real run complete for HateMM + MHC (${NUM_FRAMES} frames)."
fi

echo "end=$(date -u +%FT%TZ) exit=$?"
