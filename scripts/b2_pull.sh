#!/usr/bin/env bash
# Pull an artifact from Backblaze B2 to a local path via rclone.
#
# Usage: b2_pull.sh <b2_subpath> <local_path>
#   <b2_subpath>  source path under the RGCL_video base prefix
#   <local_path>  local destination file or directory
set -euo pipefail

B2_BASE="b2:junyi-data/RGCL_video"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <b2_subpath> <local_path>" >&2
    exit 1
fi

B2_SUBPATH="$1"
LOCAL_PATH="$2"

SRC="${B2_BASE}/${B2_SUBPATH}"

# Ensure the local parent directory exists.
PARENT_DIR="$(dirname "${LOCAL_PATH}")"
mkdir -p "${PARENT_DIR}"

echo "[b2_pull] copy: ${SRC} -> ${LOCAL_PATH}"
rclone copy "${SRC}" "${LOCAL_PATH}" \
    --transfers 8 --progress

echo "[b2_pull] done -> ${LOCAL_PATH}"
