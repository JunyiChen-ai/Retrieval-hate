#!/usr/bin/env bash
# Push a local artifact to Backblaze B2 via rclone.
#
# Usage: b2_push.sh <local_path> <b2_subpath> [--move]
#   <local_path>  local file or directory to upload
#   <b2_subpath>  destination path under the RGCL_video base prefix
#   --move        use `rclone move` instead of `rclone copy`
set -euo pipefail

B2_BASE="b2:junyi-data/RGCL_video"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <local_path> <b2_subpath> [--move]" >&2
    exit 1
fi

LOCAL_PATH="$1"
B2_SUBPATH="$2"
MOVE_FLAG="${3:-}"

DEST="${B2_BASE}/${B2_SUBPATH}"

RCLONE_CMD="copy"
if [[ "${MOVE_FLAG}" == "--move" ]]; then
    RCLONE_CMD="move"
elif [[ -n "${MOVE_FLAG}" ]]; then
    echo "Unknown option: ${MOVE_FLAG} (expected --move)" >&2
    exit 1
fi

echo "[b2_push] ${RCLONE_CMD}: ${LOCAL_PATH} -> ${DEST}"

# When LOCAL_PATH is a regular file, rclone copy/move treats the remote
# argument as a directory and puts the file *inside* it, creating an
# unwanted extra level of nesting (e.g. ckpt/model.pt/model.pt).
# Fix: pass the *parent* of DEST as the remote directory so the file lands
# at exactly DEST.  For directories we keep the existing behaviour (rclone
# copies the contents into DEST).
if [[ -f "${LOCAL_PATH}" ]]; then
    DEST_DIR="$(dirname -- "${DEST}")"
    rclone "${RCLONE_CMD}" "${LOCAL_PATH}" "${DEST_DIR}/" \
        --transfers 8 --b2-hard-delete --progress
else
    rclone "${RCLONE_CMD}" "${LOCAL_PATH}" "${DEST}" \
        --transfers 8 --b2-hard-delete --progress
fi

echo "[b2_push] done -> ${DEST}"
