#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=sq_docs_verify
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# Intentionally no --time.
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
jq empty TARGET_STATE.json
test "$(jq -r '.gate_4_s0_execution.decision' TARGET_STATE.json)" = "STOP"
test "$(jq -r '.gate_4_s0_execution.S1_unlocked' TARGET_STATE.json)" = "false"
test "$(jq -r '.gate_4_s0_execution.segment_gold_used' TARGET_STATE.json)" = "false"
test "$(jq -r '.gate_4_s0_execution.new_teacher_call_count' TARGET_STATE.json)" = "0"
grep -q 'COMPLETED_STOP job 12730' refine-logs/sq/EXPERIMENT_TRACKER.md
grep -q 'SQ-RGCL S0 terminal provenance/audit fast-fail' TARGET_LOOP.md
grep -q 'q_signal_status = PROXY_ONLY_CHEAP_FORMAT' research-wiki/experiments/exp-sq-s0.md
echo '{"status":"PASS","run_id":"SQ-DOCS-VERIFY-v1","segment_gold_used":false,"new_teacher_call_count":0}'
