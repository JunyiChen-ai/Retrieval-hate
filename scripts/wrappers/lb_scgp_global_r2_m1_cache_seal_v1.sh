#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

# CPU seal wrapper. Reads both dataset cache banks, re-verifies, and publishes the
# GO/STOP seal decision. No temp file of its own; any temp would go in-repo under
# slurm/tmp/ (realbank-v1 $TMPDIR landmine lesson).
RUN_ID=${RUN_ID:-LBSCGP-GLOBAL-M1-CACHE-SEAL-v1}
CONFIG=${CONFIG:-configs/lb_scgp_global_r2/m1_cache_seal_v1.json}
EXPECTED=LBSCGP-GLOBAL-M1-CACHE-SEAL-v1
REPO_TMPDIR="/data/jehc223/RGCL/slurm/tmp"
mkdir -p "$REPO_TMPDIR"

ARTIFACT="artifacts/lb_scgp_global/v1/m1/cache_seal_decision.json"
COMPLETE=0
PROSPECTIVE_OUTPUTS=("$ARTIFACT" "$ARTIFACT.publish.lock")

cleanup_on_exit() {
  status=$?
  # A published STOP decision is a real artifact and is kept; only clean when the
  # seal aborted before publishing (COMPLETE stays 0 and no decision on disk).
  if [[ "$COMPLETE" -ne 1 && ! -f "$ARTIFACT" ]]; then
    rm -f "${PROSPECTIVE_OUTPUTS[@]}"
  fi
  exit "$status"
}
signal_exit() { exit "$1"; }
trap cleanup_on_exit EXIT
trap 'signal_exit 129' HUP
trap 'signal_exit 130' INT
trap 'signal_exit 143' TERM

if [[ "$RUN_ID" != "$EXPECTED" ]]; then
  echo "Refusing unauthorized RUN_ID=$RUN_ID" >&2
  exit 2
fi
CONFIG_RUN_ID=$(jq -r '.run.run_id' "$CONFIG")
CONFIG_ARTIFACT=$(jq -r '.run.artifact_path' "$CONFIG")
if [[ "$CONFIG_RUN_ID" != "$RUN_ID" ]]; then
  echo "Config run ID mismatch: $CONFIG_RUN_ID" >&2
  exit 2
fi
if [[ "$CONFIG_ARTIFACT" != "$ARTIFACT" ]]; then
  echo "Config artifact path mismatch: $CONFIG_ARTIFACT" >&2
  exit 2
fi

python scripts/analysis/lb_scgp_global_r2_m1_cache_seal_v1.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID"

jq -e '.decision == "GO"' "$ARTIFACT" >/dev/null
COMPLETE=1
