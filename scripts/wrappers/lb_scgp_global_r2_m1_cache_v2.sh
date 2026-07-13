#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

# RUN_ID / CONFIG / DATASET are provided by the per-dataset sbatch. The producer
# self-verifies config <-> machine-plan <-> code-constant bindings; this wrapper adds
# the RUN_ID / config-field guards and a fail-closed no-clobber cleanup trap. Any
# temp file MUST live in-repo under slurm/tmp/ (realbank-v1 $TMPDIR landmine lesson);
# the producer's atomic publish already uses an explicit in-repo dir=, and this
# wrapper creates no temp of its own.
RUN_ID=${RUN_ID:?RUN_ID required}
CONFIG=${CONFIG:?CONFIG required}
DATASET=${DATASET:?DATASET required}
REPO_TMPDIR="/data/jehc223/RGCL/slurm/tmp"
mkdir -p "$REPO_TMPDIR"

ARTIFACT_ROOT="artifacts/lb_scgp_global/v1/m1/cache/$DATASET"
COMPLETE=0
PROSPECTIVE_OUTPUTS=(
  "$ARTIFACT_ROOT/cache.jsonl"
  "$ARTIFACT_ROOT/cache.jsonl.publish.lock"
  "$ARTIFACT_ROOT/cache_manifest.json"
  "$ARTIFACT_ROOT/cache_manifest.json.publish.lock"
  "$ARTIFACT_ROOT/access_ledger.json"
  "$ARTIFACT_ROOT/access_ledger.json.publish.lock"
)

cleanup_on_exit() {
  status=$?
  if [[ "$COMPLETE" -ne 1 ]]; then
    rm -f "${PROSPECTIVE_OUTPUTS[@]}"
  fi
  exit "$status"
}
signal_exit() { exit "$1"; }
trap cleanup_on_exit EXIT
trap 'signal_exit 129' HUP
trap 'signal_exit 130' INT
trap 'signal_exit 143' TERM

CONFIG_RUN_ID=$(jq -r '.run.run_id' "$CONFIG")
CONFIG_DATASET=$(jq -r '.run.dataset' "$CONFIG")
CONFIG_ARTIFACT=$(jq -r '.run.artifact_path' "$CONFIG")
if [[ "$CONFIG_RUN_ID" != "$RUN_ID" ]]; then
  echo "Config run ID mismatch: $CONFIG_RUN_ID != $RUN_ID" >&2
  exit 2
fi
if [[ "$CONFIG_DATASET" != "$DATASET" ]]; then
  echo "Config dataset mismatch: $CONFIG_DATASET != $DATASET" >&2
  exit 2
fi
if [[ "$CONFIG_ARTIFACT" != "$ARTIFACT_ROOT/cache.jsonl" ]]; then
  echo "Config artifact path mismatch: $CONFIG_ARTIFACT" >&2
  exit 2
fi

python scripts/analysis/lb_scgp_global_r2_m1_cache_producer_v2.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --dataset "$DATASET"

jq -e '.terminal_state == "CACHE_PRODUCED_PENDING_SEAL"' "$ARTIFACT_ROOT/cache_manifest.json" >/dev/null
COMPLETE=1
