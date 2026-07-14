#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=b2_dl32b
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (project rule; JobHeldUser auto-release).
# NOTE: intentionally NO --gres (CPU-only download job).
#
# B2 Stage-D: download Qwen/Qwen2.5-VL-32B-Instruct weights (~66G bf16, 18 safetensors
# shards among ~32 repo files) into the default HF cache. Transient: deleted in Stage-C
# after Stage-E extraction is G-dims-verified. Pre-registration:
# research-wiki/experiments/exp-encoder-32b-b2.md; review: refine-logs/B2_PREREG_REVIEW.md.
# NOTE: HF_HUB_ENABLE_HF_TRANSFER intentionally NOT set — hf_transfer is not installed in
# the HateVideo env (verified 2026-07-14: ModuleNotFoundError).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1

MODEL_ID="Qwen/Qwen2.5-VL-32B-Instruct"
CACHE_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct"

echo "[b2_stage_d] df BEFORE download:"
df -h /data | tail -2

echo "[b2_stage_d] downloading ${MODEL_ID} ..."
huggingface-cli download "$MODEL_ID"

echo "[b2_stage_d] df AFTER download:"
df -h /data | tail -2

# ---- blob verification -----------------------------------------------------
echo "[b2_stage_d] verification: snapshot listing"
SNAP_DIR=$(ls -d "$CACHE_DIR"/snapshots/*/ | head -1)
echo "  snapshot dir: $SNAP_DIR"
ls -lh "$SNAP_DIR"

N_ST=$(ls "$SNAP_DIR"/*.safetensors 2>/dev/null | wc -l)
echo "[b2_stage_d] safetensors shard count: $N_ST (expect 18)"

echo "[b2_stage_d] total cache size:"
du -sh "$CACHE_DIR"

echo "[b2_stage_d] per-shard sizes (resolved blobs):"
du -Lh "$SNAP_DIR"/*.safetensors | sort -k2

if [ "$N_ST" -ne 18 ]; then
    echo "[b2_stage_d] WARNING: shard count $N_ST != 18 — verify against the HF repo file list before Stage-E."
fi

echo "[b2_stage_d] done."
