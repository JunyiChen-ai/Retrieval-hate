#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

RUN_ID=${RUN_ID:-LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1}
CONFIG=${CONFIG:-configs/lb_scgp_global_r2/m0_contract_freeze_v1.json}
EXPECTED=LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1

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
if [[ "$CONFIG_ARTIFACT" != "artifacts/lb_scgp_global/v1/m0/contract_freeze.json" ]]; then
  echo "Config artifact path mismatch: $CONFIG_ARTIFACT" >&2
  exit 2
fi

VALIDATION_JSON=$(mktemp "${TMPDIR:-/tmp}/lbscgp_global_r2_run1_validation.XXXXXX.json")
python scripts/analysis/lb_scgp_global_r2_validate.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --json-out "$VALIDATION_JSON"

python scripts/analysis/lb_scgp_global_r2_contract_freeze.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --validation-json "$VALIDATION_JSON"
