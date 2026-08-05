#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

readonly RUN_ID=C04-A0T-SMALL-v1
readonly CONFIG=configs/c04/c04_a0t_small_v1_v3.json
readonly PYTHON_BIN=/data/jehc223/miniconda3/envs/HateVideo/bin/python
readonly PREFLIGHT_SCRIPT=scripts/analysis/c04_a0t_small_v1_v3_preflight.py

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "C04 preflight requires SLURM" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || -n "${SLURM_JOB_DEPENDENCY:-}" ]]; then
  echo "C04 arrays and dependencies are forbidden" >&2
  exit 2
fi
if [[ "$(jq -r '.authorization.preflight_materialization_authorized' "$CONFIG")" != "true" ]]; then
  echo "Prospective config blocks preflight pending code/resource GO" >&2
  exit 2
fi

"$PYTHON_BIN" "$PREFLIGHT_SCRIPT" --mode self-test
"$PYTHON_BIN" "$PREFLIGHT_SCRIPT" --mode freeze

echo "C04 impl-v3 CPU preflight complete; GPU remains blocked pending payload/GPU GO."
