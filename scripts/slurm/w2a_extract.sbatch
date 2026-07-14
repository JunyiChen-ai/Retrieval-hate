#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
# gres: node advertises Gres=gpu:a100:8 on slurmpartition and the banked-cache producer
# (gen_embed_mllm.sbatch) + the S2S extractor both used gpu:a100:1 and ran — so gpu:a100:1 is a
# schedulable type here (bf16 A100 kernel parity for the G-recon-IMG banked anchor).
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=w2a_extract
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/w2a_extract_%j.log
# NOTE: intentionally NO --time limit (project policy). Single GPU. One job runs BOTH datasets
# sequentially. PENDING (JobHeldUser) -> wait for auto-release, never force.
#
# W2-A Stage-E' grounded extraction (prereg exp-w2a-grounded.md r1, cb59a94).
# Per video TWO frozen Qwen2.5-VL-7B forwards (transcript-first GROUNDED + video-first IMG-CONTROL)
# for HateMM + MHC-EN (train/val/test), running the HARD gates inline:
#   gate 0 grid+vision-pad contiguity (both forwards) | gate 1 G-recon-IMG (fresh img-control ==
#   banked img_feats) | gate 4 length-parity ; the aggregate gate 2 grounding-live VOID and gate 3
#   placebo VOID are recorded in the per-split gatelog. HALT (fail-loud) on any gate-0/1/4 breach.
# NO submission is authorised until the three W2-A scripts pass independent code review and are
# hash-pinned (prereg §16 / status block).
#
# Smoke first (throwaway path, --limit 1) — STILL exercises the IMG-control-vs-banked G-recon gate
# (gate 1 runs under --limit; the banked cache is read from the real, unchanged path; only --out_root
# is a throwaway). The >=50-video placebo (gate 3) needs many videos and is thus a real-run gate; the
# CPU self_test + probe synthetic self-test exercise its code path.
#   SMOKE=1 sbatch scripts/slurm/w2a_extract.sbatch
# Real run (default):
#   sbatch scripts/slurm/w2a_extract.sbatch
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
EXTRACT=scripts/analysis/w2a_extract.py

echo "================================================================================"
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "[w2a_extract] script sha256:"
sha256sum "$EXTRACT" scripts/slurm/w2a_extract.sbatch
echo "[w2a_extract] parity-by-import source sha256:"
sha256sum src/utils/generate_VideoMLLM_embedding_HF.py
echo "[w2a_extract] NUM_FRAMES=${NUM_FRAMES} SMOKE=${SMOKE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "================================================================================"

if [ "$SMOKE" = "1" ]; then
    SMOKE_OUT="/data/jehc223/RGCL/slurm/logs/w2a_smoke_out_${SLURM_JOB_ID:-local}"
    echo "[w2a_extract] SMOKE run: --limit 1 -> throwaway ${SMOKE_OUT} (gate 1 G-recon-IMG DOES run)"
    for DS in HateMM MHC; do
        python "$EXTRACT" --dataset "$DS" --splits train,val,test \
            --num_frames "$NUM_FRAMES" --limit 1 --out_root "$SMOKE_OUT" --device cuda
    done
    echo "[w2a_extract] SMOKE done. Inspect gate lines above; throwaway artifacts under ${SMOKE_OUT}."
else
    for DS in HateMM MHC; do
        echo "[w2a_extract] === dataset ${DS} ==="
        python "$EXTRACT" --dataset "$DS" --splits train,val,test \
            --num_frames "$NUM_FRAMES" --device cuda
    done
    echo "[w2a_extract] real run complete for HateMM + MHC (${NUM_FRAMES} frames)."
fi

# with `set -e` a python failure exits before this line, so reaching here means success.
echo "end=$(date -u +%FT%TZ)"
