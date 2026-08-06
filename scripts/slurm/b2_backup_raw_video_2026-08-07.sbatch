#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=b2rawvid
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.  NO --gres (CPU-only cloud transfer).
# The login node reaps sustained non-SLURM transfers, so rclone runs under sbatch.
#
# 2026-08-07 RAW VIDEO backup, under the user ruling
# "把这几个数据集上面的原始视频都上传到B2" (upload the raw videos of these datasets to B2).
# Closes DISK_POLICY §5 items P2 and P3, which had stood at "None have been done".
#
# Destination follows the convention DISK_POLICY §5 ALREADY documents, so its restore
# line (`scripts/b2_pull.sh raw/Multihateclip /data/jehc223/Multihateclip`) becomes true:
#     b2:junyi-data/RGCL_video/raw/{Multihateclip,HateMM,HateClipSeg}
# Verified EMPTY before this run -> purely additive, nothing can be overwritten.
#
# WHAT IS UPLOADED: the video files + the small annotation/split metadata needed to make
# the mirror restorable on its own (the exact paths src/ reads: <ds>/video, splits/,
# 'annotation(new).json', HateClipSeg/lexicons.json + Dataset/).
#
# WHAT IS EXCLUDED, and why it is safe:
#   HateMM/frames/  (34,567 jpg, 2.4G)   HateMM/quad/ (8,643 jpg, 1.1G)
#   Multihateclip/*/quad/ (13,229 jpg)   Multihateclip/*/audios/ (1,604 wav, 1.6G)
# None of these has a consumer in this repo -- verified by grep over src/ scripts/ configs/.
# The pipeline decodes frames straight from the video at run time with decord
# (src/utils/generate_video_archive_HF.py:319 `_decode_with_decord(video_path, num_frames)`,
# default --num_frames 8), and the audio route re-extracts from video
# (scripts/analysis/apx_extract_egemaps.py, scripts/analysis/laud_extract_whisper.py).
# So every excluded byte is re-derivable from a byte that IS in this backup.
#
# DELIBERATELY NOT RUN: scripts/disk_guard.sh.  Usage is 287G against the guard's 250G
# default threshold, so sourcing it would trigger its push-verify-PRUNE path and delete
# local checkpoints.  This job is backup-only; the ruling says nothing is deleted locally.
set -uo pipefail

# conda activate (per CLAUDE.md); rclone does not need it, so never let it abort the job.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate HateVideo 2>/dev/null
echo "conda_env=${CONDA_DEFAULT_ENV:-<none>} (informational only; rclone is env-independent)"

RCLONE=/data/jehc223/home/.local/bin/rclone
DEST=b2:junyi-data/RGCL_video/raw
JID=${SLURM_JOB_ID:-manual}
# Transfer logs live under RGCL/slurm/logs -- OUTSIDE all three copied trees
# (/data/jehc223/{Multihateclip,HateMM,HateClipSeg}), per the forward-fix noted in
# refine-logs/DISK_BACKUP_RECORD_2026-08-06.md §3: a tree containing its own transfer
# log can never self-verify in one pass.
LOGDIR=/data/jehc223/RGCL/slurm/logs

echo "host=$(hostname) jobid=$JID start=$(date -u +%FT%TZ)"
echo "rclone=$($RCLONE version | head -1)"
echo "DEST=$DEST"

# backup_one <name> <src> [rclone filter args...]
backup_one() {
    local name="$1"; shift
    local src="$1"; shift
    local rlog="$LOGDIR/b2_rawvid_${JID}_${name}.rclone.log"
    local clog="$LOGDIR/b2_rawvid_${JID}_${name}.check.log"

    echo "=== COPY [$name] $src -> $DEST/$name  ($(date -u +%FT%TZ)) ==="
    $RCLONE copy "$src" "$DEST/$name" \
        --transfers 8 --checkers 16 --b2-chunk-size 96M \
        --log-level INFO --log-file "$rlog" "$@"
    echo "copy_exit[$name]=$?  (rclone log: $rlog)"

    echo "=== CHECK [$name] one-way, SHA1 ($(date -u +%FT%TZ)) ==="
    $RCLONE check "$src" "$DEST/$name" --one-way \
        --log-level INFO --log-file "$clog" "$@"
    echo "check_exit[$name]=$?  (check log: $clog)"
}

# Multihateclip: keep English/{video,video_mp4}, Chinese/video, + annotation/splits
backup_one Multihateclip /data/jehc223/Multihateclip \
    --exclude "*/quad/**" --exclude "*/audios/**"

# HateMM: keep video/ + annotation(new).json + splits/
backup_one HateMM /data/jehc223/HateMM \
    --exclude "frames/**" --exclude "quad/**"

# HateClipSeg: whole tree (videos/ + 8.5M of Dataset/Images/pilot/lexicons metadata)
backup_one HateClipSeg /data/jehc223/HateClipSeg

echo "=== CHECK SUMMARY TAILS ==="
for name in Multihateclip HateMM HateClipSeg; do
    echo "--- $name ---"
    tail -n 6 "$LOGDIR/b2_rawvid_${JID}_${name}.check.log" 2>/dev/null || echo "(no check log)"
done

echo "end=$(date -u +%FT%TZ)"
