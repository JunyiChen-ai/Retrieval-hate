#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=mhc_embed_impli
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Generate video-CLIP embeddings for the ImpliHateVid dataset.
#
# The ~50GB of mp4 videos do NOT live locally: they sit on Backblaze B2 under
# b2:junyi-data/ImpliHateVid. This job mounts B2 read-only with rclone, symlinks
# the needed videos into data/video/ImpliHateVid/All/, runs the generator, then
# unmounts (via an EXIT trap) and pushes the resulting embeddings back to B2.
#
# Usage: sbatch scripts/slurm/gen_embed_impli.sbatch
#   Env overrides: NUM_FRAMES (default 8), BATCH (default 32).
set -euo pipefail

cd /data/jehc223/RGCL

# Activate the HateVideo conda environment.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo

# Disk quota watchdog: reclaim space (B2-verified prune) before the run.
# Run as a subprocess so it can never abort this job; it is a no-op below the
# threshold. NOTE: this job also spins up a 25G rclone VFS cache below, which
# disk_guard purges (~/.cache/rclone/vfs*) on the NEXT run if space is tight.
# See scripts/disk_guard.sh and research-wiki/DISK_POLICY.md.
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

# CLIP model is pre-cached; stay offline. Disable wandb.
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

# ---------------------------------------------------------------------------
# 1. Mount the B2 bucket read-only.
#    rclone cache/temp live under /data/jehc223/home (290G quota), so bound the
#    VFS cache with --vfs-cache-max-size to avoid blowing the disk quota.
# ---------------------------------------------------------------------------
MNT=$(mktemp -d)

# Always unmount + clean up the mountpoint, no matter how the job exits.
trap 'fusermount -u "$MNT" 2>/dev/null || fusermount3 -u "$MNT" 2>/dev/null; rmdir "$MNT"' EXIT

echo "[gen_embed_impli] mounting b2:junyi-data/ImpliHateVid at $MNT"
rclone mount b2:junyi-data/ImpliHateVid "$MNT" \
    --read-only \
    --vfs-cache-mode full \
    --vfs-cache-max-size 25G \
    --daemon

# Wait for the FUSE mount to become ready (up to ~30s). The bucket root should
# expose the three category folders once mounted.
mounted=0
for _ in $(seq 1 30); do
    if ls "$MNT" 2>/dev/null | grep -q "Non Hate Videos"; then
        mounted=1
        break
    fi
    sleep 1
done
if [[ "$mounted" -ne 1 ]]; then
    echo "[gen_embed_impli] ERROR: B2 mount at $MNT never became ready (no 'Non Hate Videos' folder after 30s)" >&2
    exit 1
fi
echo "[gen_embed_impli] mount ready:"
ls "$MNT"

# ---------------------------------------------------------------------------
# 2. Build symlinks: data/video/ImpliHateVid/All/<id>.mp4 -> $MNT/<b2 relpath>
#    The mapping file is TSV: "<id>\t<b2 relative path, may contain spaces>".
#    Read with IFS set to a literal tab so the space-containing relpath stays
#    intact in a single field.
# ---------------------------------------------------------------------------
mkdir -p data/video/ImpliHateVid/All

MAP=data/video/ImpliHateVid/_id2b2path.tsv
if [[ ! -f "$MAP" ]]; then
    echo "[gen_embed_impli] ERROR: mapping file not found: $MAP" >&2
    exit 1
fi

nlinks=0
while IFS=$'\t' read -r id rel; do
    # Skip blank lines / malformed rows.
    [[ -z "$id" || -z "$rel" ]] && continue
    ln -sfn "$MNT/$rel" "data/video/ImpliHateVid/All/$id.mp4"
    nlinks=$((nlinks + 1))
done < "$MAP"
echo "[gen_embed_impli] created $nlinks symlinks in data/video/ImpliHateVid/All/"

# ---------------------------------------------------------------------------
# 3. Run the embedding generator.
#    It reads data/gt/ImpliHateVid/{train,val,test}.jsonl and
#    data/video/ImpliHateVid/All/<id>.mp4, and substitutes a zero vector for any
#    missing/corrupt video, so a transient B2 read hiccup will not crash the run.
# ---------------------------------------------------------------------------
echo "[gen_embed_impli] DATASET=ImpliHateVid NUM_FRAMES=${NUM_FRAMES:-8} BATCH=${BATCH:-32}"
python src/utils/generate_VideoCLIP_embedding_HF.py \
    --dataset ImpliHateVid \
    --num_frames "${NUM_FRAMES:-8}" \
    --batch_size "${BATCH:-32}" \
    --device cuda

# ---------------------------------------------------------------------------
# 4. On success, push the embeddings to B2.
# ---------------------------------------------------------------------------
bash scripts/b2_push.sh data/CLIP_Embedding/ImpliHateVid embeddings/ImpliHateVid

echo "[gen_embed_impli] done for DATASET=ImpliHateVid"

# 5. The EXIT trap unmounts B2 and removes the temporary mountpoint.
