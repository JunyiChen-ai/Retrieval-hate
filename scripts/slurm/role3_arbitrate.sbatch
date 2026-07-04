#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=role3_arb
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Role 3: confidence-gated selective reasoning -- Qwen2.5-VL-7B arbitration of
# margin-gate-deferred samples (gate plans from scripts/role3/gate_margin.py).
#
# Self-gating: phase 1 arbitrates the first 10 deferred EN test samples and
# hard-gates on strict-JSON rate (>=8/10, scripts/role3/smoke_check.py); only
# then does phase 2 run the full RUNS list (resume-by-id skips phase-1 items).
# Verdict-plausibility is checked post-hoc from the smoke lines in this log.
#
# Usage:
#   full (with in-job smoke): sbatch scripts/slurm/role3_arbitrate.sbatch
#   smoke only              : SMOKE_ONLY=1 sbatch scripts/slurm/role3_arbitrate.sbatch
#
# Env knobs: RUNS (comma list ds:variant:split:mode), LIMIT (0=all),
#            NUM_FRAMES (16), MAX_NEW_TOKENS (250), SMOKE_ONLY=1,
#            PV (prompt version v1|v2; v2 outputs get the _pv2 suffix).
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUNS_DEFAULT="MHC:base:val:frames,MHC:base:test:frames,MHC:clean:test:frames,MHC:base:test:textonly,MHC_zh:base:val:frames,MHC_zh:base:test:frames,MHC_zh:base:test:textonly"
RUNS=${RUNS:-$RUNS_DEFAULT}
LIMIT=${LIMIT:-0}
PV=${PV:-v1}
SUFFIX=""
if [[ "$PV" != "v1" ]]; then SUFFIX="_p${PV}"; fi

echo "[role3_arb] RUNS=${RUNS} LIMIT=${LIMIT} PV=${PV}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[role3_arb] === phase 1: smoke (10 EN deferred test samples) ==="
python scripts/role3/arbitrate_qwen.py \
    --runs "MHC:base:test:frames" \
    --limit 10 \
    --prompt_version "$PV" \
    --num_frames "${NUM_FRAMES:-16}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-250}"

python scripts/role3/smoke_check.py \
    "scripts/role3/out/arb_MHC_base_test_frames${SUFFIX}.jsonl" --n 10 --min_json 8

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo "[role3_arb] SMOKE_ONLY=1 -> stopping after smoke."
    exit 0
fi

echo "[role3_arb] === phase 2: full runs ==="
python scripts/role3/arbitrate_qwen.py \
    --runs "$RUNS" \
    --limit "$LIMIT" \
    --num_frames "${NUM_FRAMES:-16}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-250}"

echo "[role3_arb] done"
