#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=c02_density_extract
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
#SBATCH --error=/data/jehc223/RGCL/slurm/logs/%x_%j.err
# NOTE: intentionally NO --time limit (project rule). No dependency, no array, no
# singleton, no requeue, no chain. ONE submission only.
#
# C02 Stage-0 bounded extraction (authority:
# TARGET_STATE.json::iteration_8_stage0_bounded_extraction_amendment; prereg:
# refine-logs/C02_A0_V9_RECORD.md).
#
# Extracts the TEXT-STREAM density-view orbit {NAT, RFULL, RW1..RW4} for
#   HateMM  @ Qwen2.5-VL-7B-Instruct-LoRA-curric_HF   (adapter logging/lora/HateMM_curric)
#   MHC_zh  @ Qwen2.5-VL-7B-Instruct-LoRA_HF          (adapter logging/lora/MHC_zh)
# on train + dev_seen ONLY. Test is never named, never opened, and is refused by an
# assertion in the extractor. 8 CPU / 1 GPU / 64 G matches the historical
# gen_embed_* extraction jobs, so the 16-CPU aggregate-cap wedge cannot occur.
#
# Budget: the amendment caps the candidate at 4.0 GPU-hours total; actual elapsed is
# read back from sacct and recorded in refine-logs/C02_A0_V9_RECORD.md.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

RUN_ID=C02-DEN-v1
MAN_DIR=/data/jehc223/RGCL/artifacts/c02_edq/v1/extract/${RUN_ID}
mkdir -p "$MAN_DIR"

# ---------------------------------------------------------------- GPU-budget guard
# Amendment condition (f): an over-budget run is a PROTOCOL VIOLATION and its result is
# VOID.  CLAUDE.md forbids --time, so the job enforces the cap itself.  ONE absolute
# deadline is computed here, at job start, and passed to BOTH dataset invocations, so the
# second inherits whatever the first left.  The guard may only ever STOP work: view
# caches and the manifest are written only after a whole split completes, so a breach
# leaves the in-progress dataset entirely unwritten and any completed dataset intact.
BUDGET_CAP_SECONDS=14400        # 4.0 GPU-hours, the amendment cap
BUDGET_MARGIN_SECONDS=600       # halt 10 min early so the cap cannot be reached
JOB_START_EPOCH=${SLURM_JOB_START_TIME:-$(date +%s)}
BUDGET_DEADLINE=$(( JOB_START_EPOCH + BUDGET_CAP_SECONDS - BUDGET_MARGIN_SECONDS ))
echo "[c02_density_extract] budget: cap ${BUDGET_CAP_SECONDS}s, margin ${BUDGET_MARGIN_SECONDS}s,"
echo "[c02_density_extract] job start ${JOB_START_EPOCH}, guard deadline ${BUDGET_DEADLINE} ($(date -d @${BUDGET_DEADLINE} 2>/dev/null || echo n/a))"

echo "[c02_density_extract] job=${SLURM_JOB_ID:-NA} run_id=${RUN_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "########## HateMM ##########"
python src/utils/generate_c02_density_view_text_embedding_HF.py \
    --dataset HateMM \
    --lora_dir logging/lora/HateMM_curric \
    --base_tag Qwen2.5-VL-7B-Instruct-LoRA-curric_HF \
    --splits train,val \
    --run_id "$RUN_ID" \
    --manifest "$MAN_DIR/manifest_HateMM.json" \
    --budget_deadline_epoch "$BUDGET_DEADLINE" \
    --device cuda

echo "########## MHC_zh ##########"
python src/utils/generate_c02_density_view_text_embedding_HF.py \
    --dataset MHC_zh \
    --lora_dir logging/lora/MHC_zh \
    --base_tag Qwen2.5-VL-7B-Instruct-LoRA_HF \
    --splits train,val \
    --run_id "$RUN_ID" \
    --manifest "$MAN_DIR/manifest_MHC_zh.json" \
    --budget_deadline_epoch "$BUDGET_DEADLINE" \
    --device cuda

NOW=$(date +%s)
echo "[c02_density_extract] elapsed $(( NOW - JOB_START_EPOCH ))s of the ${BUDGET_CAP_SECONDS}s cap"
echo "[c02_density_extract] done"
