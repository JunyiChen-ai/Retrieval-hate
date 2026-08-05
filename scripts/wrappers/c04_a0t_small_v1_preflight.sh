#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

RUN_ID=${RUN_ID:-C04-A0T-SMALL-v1}
CONFIG=${CONFIG:-configs/c04/c04_a0t_small_v1.json}
EXPECTED_RUN_ID=C04-A0T-SMALL-v1

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "C04 preflight requires SLURM" >&2
  exit 2
fi
if [[ "$RUN_ID" != "$EXPECTED_RUN_ID" ]]; then
  echo "Unexpected C04 run id: $RUN_ID" >&2
  exit 2
fi
if [[ "$(jq -r '.authorization.preflight_materialization_authorized' "$CONFIG")" != "true" ]]; then
  echo "Prospective config blocks preflight until fresh code/resource authorization" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || -n "${SLURM_JOB_DEPENDENCY:-}" ]]; then
  echo "C04 arrays and dependencies are forbidden" >&2
  exit 2
fi

python scripts/analysis/c04_a0t_small_v1_preflight.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --mode self-test

python scripts/analysis/c04_a0t_small_v1_preflight.py \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --mode freeze

echo "C04 CPU preflight hash-freeze complete; GPU remains blocked pending payload review."
