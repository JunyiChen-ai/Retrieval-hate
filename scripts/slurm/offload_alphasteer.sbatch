#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=offload_alphasteer
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit. CPU-only (no --gres) reap-safe transfer.
#
# Copy AlphaSteer to B2, then verify. Copy-first, verify-only. This job NEVER
# deletes local data; deletion is a separate manual step gated on a clean check.
set -uo pipefail

SRC=/data/jehc223/AlphaSteer
DST=b2:junyi-data/AlphaSteer
CHECK_OUT=/data/jehc223/RGCL/slurm/logs/alphasteer_rclone_check.txt
STATUS_OUT=/data/jehc223/RGCL/slurm/logs/alphasteer_offload_status.txt

echo "[offload] start $(date -Is) on $(hostname)"
echo "[offload] SRC=$SRC DST=$DST"

# 1) Copy (resumable / idempotent).
rclone copy "$SRC" "$DST" --transfers 8 --fast-list --progress --stats 60s
COPY_RC=$?
echo "[offload] rclone copy exit=$COPY_RC"
if [ "$COPY_RC" -ne 0 ]; then
    echo "COPY_FAILED rc=$COPY_RC" > "$STATUS_OUT"
    echo "[offload] copy failed, aborting before check"
    exit "$COPY_RC"
fi

# 2) Verify: full checksum/size/count comparison. Capture output verbatim.
echo "[offload] running rclone check $(date -Is)"
rclone check "$SRC" "$DST" 2>&1 | tee "$CHECK_OUT"
CHECK_RC=${PIPESTATUS[0]}
echo "[offload] rclone check exit=$CHECK_RC"

# 3) Record source/dest sizes for the human report.
echo "=== rclone size SRC ===" >> "$CHECK_OUT"
rclone size "$SRC" 2>&1 | tee -a "$CHECK_OUT"
echo "=== rclone size DST ===" >> "$CHECK_OUT"
rclone size "$DST" 2>&1 | tee -a "$CHECK_OUT"

if [ "$CHECK_RC" -eq 0 ]; then
    echo "CHECK_CLEAN" > "$STATUS_OUT"
    echo "[offload] VERIFY CLEAN — safe to delete local copy"
else
    echo "CHECK_FAILED rc=$CHECK_RC" > "$STATUS_OUT"
    echo "[offload] VERIFY FAILED — DO NOT delete local copy"
fi

echo "[offload] done $(date -Is)"
exit "$CHECK_RC"
