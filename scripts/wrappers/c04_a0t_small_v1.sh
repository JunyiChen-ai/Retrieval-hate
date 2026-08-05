#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL
read -r C04_ALLOCATION_UPTIME _ < /proc/uptime
C04_ALLOCATION_START_SECONDS=${C04_ALLOCATION_UPTIME%%.*}

RUN_ID=${RUN_ID:-C04-A0T-SMALL-v1}
CONFIG=${CONFIG:-configs/c04/c04_a0t_small_v1.json}
EXPECTED_RUN_ID=C04-A0T-SMALL-v1
LOCK_PATH=artifacts/c04/a0t_small_v1/resource/serial_gpu.lock

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "C04 producer requires SLURM" >&2
  exit 2
fi
if [[ "$RUN_ID" != "$EXPECTED_RUN_ID" ]]; then
  echo "Unexpected C04 run id: $RUN_ID" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || -n "${SLURM_JOB_DEPENDENCY:-}" ]]; then
  echo "C04 arrays and dependencies are forbidden" >&2
  exit 2
fi
for field in teacher_authorized gpu_authorized slurm_authorized small_tranche_execution_authorized; do
  if [[ "$(jq -r ".authorization.$field" "$CONFIG")" != "true" ]]; then
    echo "Prospective config blocks $field until fresh code/resource authorization" >&2
    exit 2
  fi
done
for field in chain_authorized release_authorized resubmit_authorized; do
  if [[ "$(jq -r ".authorization.$field" "$CONFIG")" != "false" ]]; then
    echo "Forbidden authorization enabled: $field" >&2
    exit 2
  fi
done

TICKET=$(jq -r '.paths.resource_ticket' "$CONFIG")
if [[ ! -f "$TICKET" ]]; then
  echo "Missing sealed resource ticket: $TICKET" >&2
  exit 2
fi
REMAINING_SECONDS=$(jq -r '.remaining_seconds' "$TICKET")
WATCHDOG_SECONDS=$(jq -r '.watchdog_seconds' "$TICKET")
MINIMUM_SECONDS=$(jq -r '.resources.minimum_submit_remaining_seconds' "$CONFIG")
RESERVE_SECONDS=$(jq -r '.resources.watchdog_reserve_seconds' "$CONFIG")
KILL_AFTER_SECONDS=$(jq -r '.resources.watchdog_term_then_kill_seconds' "$CONFIG")
if (( REMAINING_SECONDS <= MINIMUM_SECONDS )); then
  echo "HALT_RESOURCE_CAP: remaining seconds <= minimum" >&2
  exit 3
fi
if (( WATCHDOG_SECONDS != REMAINING_SECONDS - RESERVE_SECONDS )); then
  echo "HALT_RESOURCE_CAP: watchdog reserve mismatch" >&2
  exit 3
fi
read -r C04_CURRENT_UPTIME _ < /proc/uptime
C04_CURRENT_SECONDS=${C04_CURRENT_UPTIME%%.*}
C04_STARTUP_SECONDS=$(( C04_CURRENT_SECONDS - C04_ALLOCATION_START_SECONDS ))
C04_ACTIVE_WATCHDOG_SECONDS=$(( WATCHDOG_SECONDS - C04_STARTUP_SECONDS ))
if (( C04_ACTIVE_WATCHDOG_SECONDS <= 0 )); then
  echo "HALT_RESOURCE_CAP: allocation-start watchdog budget exhausted" >&2
  exit 3
fi

mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "HALT_RESOURCE_CAP: another C04 GPU allocation holds the serial lock" >&2
  exit 3
fi

set +e
C04_WATCHDOG_SECONDS="$C04_ACTIVE_WATCHDOG_SECONDS" timeout \
  --signal=TERM \
  --kill-after="${KILL_AFTER_SECONDS}s" \
  "${C04_ACTIVE_WATCHDOG_SECONDS}s" \
  python scripts/analysis/c04_a0t_small_v1_producer.py \
    --config "$CONFIG" \
    --run-id "$RUN_ID"
status=$?
set -e

if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
  marker=artifacts/c04/a0t_small_v1/resource/watchdog_event.json
  if [[ ! -e "$marker" ]]; then
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg job_id "$SLURM_JOB_ID" \
      --argjson status "$status" \
      '{schema_version:"c04_watchdog_event_v1",run_id:$run_id,slurm_job_id:$job_id,status:$status,terminal_state:"HALT_RESOURCE_CAP",resubmit_authorized:false}' \
      > "$marker"
  fi
  echo "HALT_RESOURCE_CAP: watchdog terminated C04; another job is forbidden" >&2
  exit "$status"
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

SEAL=$(jq -r '.paths.seal_manifest' "$CONFIG")
jq -e '
  .terminal_state == "SEALED_PRELABEL_RELIABILITY_PASS"
  or .terminal_state == "KILL_C04_TEACHER_SEMANTIC_RELIABILITY"
' "$SEAL" >/dev/null
echo "C04 pre-label seal complete. This wrapper did not submit, chain, release, or resubmit any job."
