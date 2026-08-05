#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

readonly RUN_ID=C04-A0T-SMALL-v1
readonly CONFIG=configs/c04/c04_a0t_small_v1_v7.json
readonly PYTHON_BIN=/data/jehc223/miniconda3/envs/HateVideo/bin/python
readonly LEDGER_SCRIPT=scripts/analysis/c04_a0t_small_v1_v7_gpu_ledger.py
readonly PRODUCER_SCRIPT=scripts/analysis/c04_a0t_small_v1_v7_producer.py
readonly LOCK_PATH=artifacts/c04/a0t_small_v1_impl_v7/resource/serial_gpu.lock
readonly BREACH_RECORD=artifacts/c04/a0t_small_v1_impl_v7/resource/budget_breach.json

read -r C04_ALLOCATION_UPTIME _ < /proc/uptime
readonly C04_ALLOCATION_START_SECONDS=${C04_ALLOCATION_UPTIME%%.*}
C04_FINAL_STATUS=255

mark_allocation_exit() {
  local trap_status=$?
  local recorded_status=$C04_FINAL_STATUS
  if [[ "$recorded_status" -eq 255 ]]; then
    recorded_status=$trap_status
  fi
  "$PYTHON_BIN" "$LEDGER_SCRIPT" \
    --mode mark-exit \
    --allocation-start-uptime-seconds "$C04_ALLOCATION_START_SECONDS" \
    --exit-code "$recorded_status" || true
}
trap mark_allocation_exit EXIT

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "C04 producer requires SLURM" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || -n "${SLURM_JOB_DEPENDENCY:-}" ]]; then
  echo "C04 arrays and dependencies are forbidden" >&2
  exit 2
fi
# Authorization is checked BEFORE the namespace is touched. The preflight
# wrapper has such a gate and the reconcile wrapper has a nine-clause one; this
# wrapper had none, so a submission made in the repository's normal
# (`gpu_authorized: false`) state would `mkdir` the no-clobber namespace and
# write the entry marker before any code read an authorization flag -- after
# which the CPU preflight refuses that namespace forever and the whole
# implementation version has to be rebuilt. The same "irreversible step before
# the rejecting check" shape this implementation exists to remove.
jq -e '
  .run.run_id == "C04-A0T-SMALL-v1"
  and .run.implementation_version == "v7_prospective"
  and .authorization.implementation_authorized == true
  and .authorization.teacher_authorized == true
  and .authorization.gpu_authorized == true
  and .authorization.slurm_authorized == true
  and .authorization.small_tranche_execution_authorized == true
  and .authorization.preflight_materialization_authorized == false
  and .authorization.post_job_reconciliation_authorized == false
  and .authorization.dev_authorized == false
  and .authorization.test_authorized == false
  and .authorization.ocr_authorized == false
  and .authorization.external_api_authorized == false
  and .authorization.network_authorized == false
  and .authorization.cross_dataset_authorized == false
  and .authorization.label_value_authorized_before_seal == false
  and .authorization.chain_authorized == false
  and .authorization.release_authorized == false
  and .authorization.resubmit_authorized == false
  and .review.payload_hash_verdict == "GO"
  and .review.gpu_execution_verdict == "GO"
  and (.review.payload_review_sha256 | test("^[0-9a-f]{64}$"))
  and (.review.gpu_execution_authorization_sha256 | test("^[0-9a-f]{64}$"))
  and (.prompt_hashes | to_entries | all(.value | test("^[0-9a-f]{64}$")))
' "$CONFIG" >/dev/null

# The CPU preflight must already have frozen this namespace. If it has not, the
# allocation is refused here rather than creating a namespace that would then
# make the preflight impossible.
readonly PREFLIGHT_MANIFEST=artifacts/c04/a0t_small_v1_impl_v7/freeze/preflight_manifest.json
if [[ ! -f "$PREFLIGHT_MANIFEST" ]]; then
  echo "HALT_REVIEW_LINEAGE: no frozen preflight manifest; refusing to enter the namespace" >&2
  exit 2
fi

readonly ENTRY_MARKER=artifacts/c04/a0t_small_v1_impl_v7/resource/allocation_entry_marker.json
mkdir -p "$(dirname "$ENTRY_MARKER")"
if [[ -e "$ENTRY_MARKER" ]]; then
  jq -e \
    --arg job_id "$SLURM_JOB_ID" \
    --argjson started "$C04_ALLOCATION_START_SECONDS" \
    '.slurm_job_id == $job_id and .allocation_entry_uptime_seconds == $started' \
    "$ENTRY_MARKER" >/dev/null
else
  (
    set -o noclobber
    umask 027
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg job_id "$SLURM_JOB_ID" \
      --argjson started "$C04_ALLOCATION_START_SECONDS" \
      '{
        schema_version:"c04_allocation_entry_marker_v7",
        run_id:$run_id,
        slurm_job_id:$job_id,
        allocation_entry_uptime_seconds:$started,
        claim_completed:false,
        exit_marker_recorded:false
      }' > "$ENTRY_MARKER"
  )
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Pinned HateVideo Python is unavailable" >&2
  exit 2
fi

# This is the first Python operation in the allocation. It persistently marks
# allocation entry, reconciles prior terminal/partial sacct rows, consumes the
# single-use ticket, and reserves the cap before model/data work.
C04_ACTIVE_WATCHDOG_SECONDS=$(
  "$PYTHON_BIN" "$LEDGER_SCRIPT" \
    --mode claim \
    --allocation-start-uptime-seconds "$C04_ALLOCATION_START_SECONDS"
)
readonly C04_ACTIVE_WATCHDOG_SECONDS

if [[ ! "$C04_ACTIVE_WATCHDOG_SECONDS" =~ ^[0-9]+$ ]] \
  || (( C04_ACTIVE_WATCHDOG_SECONDS <= 0 )); then
  echo "HALT_RESOURCE_CAP: invalid claimed watchdog" >&2
  exit 3
fi
readonly KILL_AFTER_SECONDS=30

mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "HALT_RESOURCE_CAP: another C04 GPU process holds the serial lock" >&2
  exit 3
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

set +e
C04_WATCHDOG_SECONDS="$C04_ACTIVE_WATCHDOG_SECONDS" timeout \
  --signal=TERM \
  --kill-after="${KILL_AFTER_SECONDS}s" \
  "${C04_ACTIVE_WATCHDOG_SECONDS}s" \
  "$PYTHON_BIN" "$PRODUCER_SCRIPT"
C04_FINAL_STATUS=$?
set -e

if [[ "$C04_FINAL_STATUS" -eq 124 || "$C04_FINAL_STATUS" -eq 137 \
   || "$C04_FINAL_STATUS" -eq 143 ]]; then
  echo "HALT_RESOURCE_CAP: watchdog terminated C04; resubmission is forbidden" >&2
  exit "$C04_FINAL_STATUS"
fi
# Exit 40 is the in-job budget guard, and it is deliberately distinct from an
# engineering failure: the guard stopped before an item began, no output was
# truncated or altered, no seal exists, and an accounting-only breach record was
# written. It is still a terminal state -- resubmission remains forbidden.
readonly BUDGET_BREACH_EXIT_CODE=40
if [[ "$C04_FINAL_STATUS" -eq "$BUDGET_BREACH_EXIT_CODE" ]]; then
  echo "HALT_RESOURCE_CAP: C04 stopped at the frozen tranche ceiling before an item;" >&2
  echo "  accounting-only breach record: ${BREACH_RECORD}" >&2
  jq -e '.terminal_state == "HALT_RESOURCE_CAP_TRANCHE_CEILING"
         and .seal_published == false
         and .outputs_truncated_or_altered == 0' "$BREACH_RECORD" >/dev/null
  exit "$BUDGET_BREACH_EXIT_CODE"
fi
if [[ "$C04_FINAL_STATUS" -ne 0 ]]; then
  exit "$C04_FINAL_STATUS"
fi
if [[ -e "$BREACH_RECORD" ]]; then
  echo "HALT_INVALID_FREEZE: a breach record exists on a zero-exit run" >&2
  exit 3
fi

readonly SEAL=artifacts/c04/a0t_small_v1_impl_v7/seal/seal_manifest.json
jq -e '
  .terminal_state == "SEALED_PRELABEL_RELIABILITY_PASS"
  or .terminal_state == "KILL_C04_TEACHER_SEMANTIC_RELIABILITY"
' "$SEAL" >/dev/null
echo "C04 impl-v7 pre-label seal complete; terminal CPU sacct reconciliation is still required."
