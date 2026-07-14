#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=b2backup
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.  NO --gres (CPU-only cloud transfer).
# Compute nodes have outbound internet (HF downloads / disk_guard B2 pushes run here);
# the login node reaps sustained non-SLURM compute, so the multi-hour rclone copy of
# ~144G must run under sbatch.
#
# Manual backup-then-delete of logging/lora_p9 (83G) and logging/Retrieval (61G) to a
# DISTINCT B2 subpath, separate from disk_guard's own RGCL_video/{logs,embeddings,...}
# mirror layout, so nothing collides.  Per user ruling 2026-07-14 (see
# refine-logs/DISK_BACKUP_RECORD_2026-07-14.md).  Deletion is performed by a separate
# tracked waiter ONLY after this job's rclone check reports 0 differences.
set -uo pipefail

cd /data/jehc223/RGCL

RCLONE=/data/jehc223/home/.local/bin/rclone
DEST=b2:junyi-data/RGCL_video/manual_backup_2026-07-14
JID=${SLURM_JOB_ID:-manual}
LOGDIR=/data/jehc223/RGCL/slurm/logs

echo "host=$(hostname) jobid=$JID start=$(date -u +%FT%TZ)"
echo "rclone=$($RCLONE version | head -1)"
echo "DEST=$DEST"

backup_one() {
    local name="$1"
    local src="/data/jehc223/RGCL/logging/$name"
    local rlog="$LOGDIR/b2_backup_manual_${JID}_${name}.rclone.log"
    local clog="$LOGDIR/b2_backup_manual_${JID}_${name}.check.log"

    echo "=== COPY $name -> $DEST/$name  ($(date -u +%FT%TZ)) ==="
    $RCLONE copy "$src" "$DEST/$name" \
        --transfers 8 --checkers 16 --b2-chunk-size 96M \
        --log-level INFO --log-file "$rlog"
    local cp=$?
    echo "copy_exit[$name]=$cp  (rclone log: $rlog)"

    echo "=== CHECK $name (one-way, source must fully exist on dest)  ($(date -u +%FT%TZ)) ==="
    $RCLONE check "$src" "$DEST/$name" --one-way \
        --log-level INFO --log-file "$clog"
    local ck=$?
    echo "check_exit[$name]=$ck  (check log: $clog)"
}

backup_one lora_p9
backup_one Retrieval

echo "=== CHECK SUMMARY TAILS ==="
for name in lora_p9 Retrieval; do
    clog="$LOGDIR/b2_backup_manual_${JID}_${name}.check.log"
    echo "--- $name check tail ---"
    tail -n 10 "$clog" 2>/dev/null || echo "(no check log for $name)"
done

echo "end=$(date -u +%FT%TZ)"
