#!/usr/bin/env bash
set -euo pipefail

cd /data/jehc223/RGCL

readonly CONFIG=configs/c04/c04_a0t_small_v1_v6.json
readonly PYTHON_BIN=/data/jehc223/miniconda3/envs/HateVideo/bin/python
readonly LEDGER_SCRIPT=scripts/analysis/c04_a0t_small_v1_v6_gpu_ledger.py

if [[ -z "${SLURM_JOB_ID:-}" || ! "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "C04 terminal reconciliation requires a numeric SLURM_JOB_ID" >&2
  exit 2
fi
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" || -n "${SLURM_JOB_DEPENDENCY:-}" ]]; then
  echo "C04 reconciliation arrays and dependencies are forbidden" >&2
  exit 2
fi
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|-1|NoDevFiles) ;;
  *)
    echo "C04 reconciliation must be CPU-only" >&2
    exit 2
    ;;
esac
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" != "0" ]]; then
  echo "C04 reconciliation received a GPU allocation" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Pinned HateVideo Python is unavailable" >&2
  exit 2
fi
jq -e '
  .run.run_id == "C04-A0T-SMALL-v1"
  and .run.implementation_version == "v6_prospective"
  and .authorization.implementation_authorized == true
  and .authorization.post_job_reconciliation_authorized == true
  and (
    [
      .authorization
      | to_entries[]
      | select(
          .key != "implementation_authorized"
          and .key != "post_job_reconciliation_authorized"
        )
      | .value
    ]
    | all(. == false)
  )
  and .review.resource_reconciliation_verdict == "GO"
  and (.review.resource_reconciliation_authorization_sha256 | test("^[0-9a-f]{64}$"))
  and .review.downstream_review_requires_terminal_resource_state == true
' "$CONFIG" >/dev/null

readonly GPU_LEDGER=artifacts/c04/a0t_small_v1_impl_v6/resource/gpu_ledger.json
readonly FINAL_STATE=artifacts/c04/a0t_small_v1_impl_v6/resource/resource_final_state.json
set +e
"$PYTHON_BIN" "$LEDGER_SCRIPT" --mode reconcile-terminal
C04_RECONCILE_STATUS=$?
set -e
if [[ "$C04_RECONCILE_STATUS" -ne 0 ]]; then
  if [[ -f "$GPU_LEDGER" && ! -e "$FINAL_STATE" ]] \
    && jq -e '
      .state == "SACCT_TERMINAL_RECONCILED"
      and .requires_terminal_reconciliation == false
      and (.jobs | length) == 1
      and .jobs[0].status == "SACCT_TERMINAL"
      and .jobs[0].reserved_gpu_seconds == 0
    ' "$GPU_LEDGER" >/dev/null; then
    echo "C04 retrying final-state publication in the same CPU allocation." >&2
    "$PYTHON_BIN" "$LEDGER_SCRIPT" --mode reconcile-terminal
  else
    exit "$C04_RECONCILE_STATUS"
  fi
fi

jq -e '
  .requires_terminal_reconciliation == false
  and .reserved_gpu_seconds == 0
  and .single_gpu_allocation_count == 1
  and .second_gpu_allocation_authorized == false
  and .downstream_review_resource_gate_satisfied == true
' "$FINAL_STATE" >/dev/null
echo "C04 impl-v6 terminal sacct reconciliation complete; no GPU/submit/chain performed."
