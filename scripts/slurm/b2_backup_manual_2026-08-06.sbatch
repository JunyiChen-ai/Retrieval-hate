#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=b2backup0806
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.  NO --gres (CPU-only cloud transfer).
# The login node reaps sustained non-SLURM transfers, so rclone runs under sbatch
# (same precedent as scripts/slurm/b2_backup_manual.sbatch, job 13157, 2026-07-14).
#
# 2026-08-06 gap backup.  Survey found these local assets ABSENT from B2:
#   (1) data/CLIP_Embedding : 34 files missing from the canonical embeddings/ mirror
#       (c02den NAT/RFULL/RW1-4 + nullop2merge caches + HateClipSeg subclip + SPANSTATS)
#   (2) logging/lora        : 549 files / 6.6G -- B2 logs/lora holds ONLY optimizer.pt +
#       scheduler.pt (disk_guard pushed those); every adapter_model.safetensors and
#       tokenizer/config file is local-only.
#   (3) the rest of the repo tree (artifacts/, data/ non-embedding, refine-logs ckpt
#       snapshots, slurm/logs, scripts, src, .git) -- no B2 copy at all.
#   (4) /data/jehc223/baselines/MoRE -- baseline derived features + trained .pth, no B2 copy.
#
# (1) and (2) go to the CANONICAL disk_guard mirror paths so scripts/b2_pull.sh and
# disk_guard.sh keep working.  (3) and (4) go to a dated manual_backup prefix.
# (3) EXCLUDES logging/lora/** and data/CLIP_Embedding/** to avoid duplicating (1)+(2).
set -uo pipefail

cd /data/jehc223/RGCL

RCLONE=/data/jehc223/home/.local/bin/rclone
BASE=b2:junyi-data/RGCL_video
SNAP=$BASE/manual_backup_2026-08-06
JID=${SLURM_JOB_ID:-manual}
LOGDIR=/data/jehc223/RGCL/slurm/logs

echo "host=$(hostname) jobid=$JID start=$(date -u +%FT%TZ)"
echo "rclone=$($RCLONE version | head -1)"

# copy_one <tag> <src> <dest> [extra rclone args...]
copy_one() {
    local tag="$1"; shift
    local src="$1"; shift
    local dest="$1"; shift
    local rlog="$LOGDIR/b2_backup_0806_${JID}_${tag}.rclone.log"
    local clog="$LOGDIR/b2_backup_0806_${JID}_${tag}.check.log"

    echo "=== COPY [$tag] $src -> $dest  ($(date -u +%FT%TZ)) ==="
    $RCLONE copy "$src" "$dest" \
        --transfers 8 --checkers 16 --b2-chunk-size 96M \
        --log-level INFO --log-file "$rlog" "$@"
    echo "copy_exit[$tag]=$?  (rclone log: $rlog)"

    echo "=== CHECK [$tag] one-way (every source file must exist byte-identical on dest) ==="
    $RCLONE check "$src" "$dest" --one-way \
        --log-level INFO --log-file "$clog" "$@"
    echo "check_exit[$tag]=$?  (check log: $clog)"
}

# --- (1) canonical embeddings mirror: adds the 34 missing feature caches ---
copy_one embeddings data/CLIP_Embedding "$BASE/embeddings"

# --- (2) canonical lora mirror: adds the 549 adapter/config files (purely additive;
#         the 54 optimizer.pt/scheduler.pt already on B2 have no local counterpart) ---
copy_one lora logging/lora "$BASE/logs/lora"

# --- (3) dated full-repo snapshot, minus the two canonically-mirrored trees and
#         minus the data/_src_Multihateclip symlink (-> /data/jehc223/Multihateclip, 27G raw video) ---
copy_one repo /data/jehc223/RGCL "$SNAP/RGCL" \
    --exclude "logging/lora/**" \
    --exclude "data/CLIP_Embedding/**" \
    --exclude "data/_src_Multihateclip/**"

# --- (4) MoRE baseline: derived features (2.8G) + trained checkpoints (2.5G) ---
copy_one baselines_MoRE /data/jehc223/baselines/MoRE "$SNAP/baselines_MoRE"

echo "=== CHECK SUMMARY TAILS ==="
for tag in embeddings lora repo baselines_MoRE; do
    echo "--- $tag ---"
    tail -n 6 "$LOGDIR/b2_backup_0806_${JID}_${tag}.check.log" 2>/dev/null || echo "(no check log)"
done

echo "end=$(date -u +%FT%TZ)"
