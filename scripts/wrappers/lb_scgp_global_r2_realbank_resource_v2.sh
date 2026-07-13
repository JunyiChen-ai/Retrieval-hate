#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

RUN_ID=${RUN_ID:-LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2}
CONFIG=${CONFIG:-configs/lb_scgp_global_r2/m0_realbank_resource_v2.json}
EXPECTED=LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2
ARTIFACT_ROOT=artifacts/lb_scgp_global/v1/m0/realbank_resource
COMPLETE=0
VALIDATION_JSON=""
PROSPECTIVE_OUTPUTS=(
  "$ARTIFACT_ROOT/decision.json"
  "$ARTIFACT_ROOT/decision.json.publish.lock"
  "$ARTIFACT_ROOT/source_manifest.json"
  "$ARTIFACT_ROOT/source_manifest.json.publish.lock"
  "$ARTIFACT_ROOT/access_ledger.json"
  "$ARTIFACT_ROOT/access_ledger.json.publish.lock"
  "$ARTIFACT_ROOT/semantic_verification.json"
  "$ARTIFACT_ROOT/semantic_verification.json.publish.lock"
)

cleanup_on_exit() {
  status=$?
  if [[ -n "$VALIDATION_JSON" ]]; then
    rm -f "$VALIDATION_JSON"
  fi
  if [[ "$COMPLETE" -ne 1 ]]; then
    rm -f "${PROSPECTIVE_OUTPUTS[@]}"
  fi
  exit "$status"
}

signal_exit() {
  local code=$1
  exit "$code"
}
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
if [[ "$CONFIG_ARTIFACT" != "artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json" ]]; then
  echo "Config artifact path mismatch: $CONFIG_ARTIFACT" >&2
  exit 2
fi

# v2 fix (REALBANK_FULLCHAIN_STATIC_AUDIT.md §5.4, direction (a)): the validator->producer handoff
# JSON must live INSIDE the repo, because the producer reads it through canonical_root_path
# (in-repo-only). The v1 wrapper wrote it to $TMPDIR (out-of-repo), which the producer's guard
# rejected. slurm/tmp/ is outside all five old_protected roots and every no-clobber / git-diff /
# relevant_git_status pathspec (audit §5.1), so the transient file perturbs no gate.
REALBANK_TMPDIR="/data/jehc223/RGCL/slurm/tmp"
mkdir -p "$REALBANK_TMPDIR"
VALIDATION_JSON=$(mktemp "$REALBANK_TMPDIR/lbscgp_global_r2_realbank_resource_v2_validation.XXXXXX.json")
python scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_validate.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --json-out "$VALIDATION_JSON"

python scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_producer.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --validation-json "$VALIDATION_JSON"

python scripts/analysis/lb_scgp_global_r2_realbank_resource_v2_independent_verify.py \
  --config "$CONFIG" \
  --manifest "$ARTIFACT_ROOT/decision.json" \
  --out "$ARTIFACT_ROOT/semantic_verification.json"

jq -e '.decision == "PASS"' "$ARTIFACT_ROOT/semantic_verification.json" >/dev/null
COMPLETE=1
