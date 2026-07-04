#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=role3_arb_v3
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Role 3 v3: task-calibrated LoRA arbiter (logging/lora/<DS>, one-word-answer
# SFT on MHClip train). Same harness/deferred queues/JSON-first contract as
# v1/v2 (prompt = v2 policy block); bare-verdict word fallback is counted
# separately. EN and ZH use their own adapters -> two invocations.
#
# Self-gating: phase 1 = 10 EN deferred test samples, hard gate on usable
# verdict rate >= 8/10 (JSON or word fallback), then full EN + ZH runs.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

NUM_FRAMES=${NUM_FRAMES:-16}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-250}

echo "[role3_v3] === phase 1: smoke (10 EN deferred test samples, EN adapter) ==="
python scripts/role3/arbitrate_qwen.py \
    --runs "MHC:base:test:frames" \
    --limit 10 \
    --prompt_version v2 --tag v3 \
    --adapter logging/lora/MHC \
    --num_frames "$NUM_FRAMES" \
    --max_new_tokens "$MAX_NEW_TOKENS"

python scripts/role3/smoke_check.py \
    scripts/role3/out/arb_MHC_base_test_frames_pv3.jsonl --n 10 --min_json 8

echo "[role3_v3] === phase 2: EN full (adapter logging/lora/MHC) ==="
python scripts/role3/arbitrate_qwen.py \
    --runs "MHC:base:val:frames,MHC:base:test:frames" \
    --prompt_version v2 --tag v3 \
    --adapter logging/lora/MHC \
    --num_frames "$NUM_FRAMES" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo "[role3_v3] === phase 3: ZH full (adapter logging/lora/MHC_zh) ==="
python scripts/role3/arbitrate_qwen.py \
    --runs "MHC_zh:base:val:frames,MHC_zh:base:test:frames" \
    --prompt_version v2 --tag v3 \
    --adapter logging/lora/MHC_zh \
    --num_frames "$NUM_FRAMES" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo "[role3_v3] done"
