#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=judge_mem
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# Semantic vote for the automatic memory-repair experiment
# (research-wiki/EXP_auto_memory_repair.md). Qwen2.5-VL-7B judges each TRAIN memory
# entry's archive JSON + label -> {SUPPORT, CONTRADICT, UNSURE}. Seed-independent;
# text-only; greedy (temperature 0). Reused across all seeds by the CPU eval.
#
# Usage:
#   smoke : LIMIT=6 sbatch scripts/slurm/judge_memory_archive.sbatch
#   full  : sbatch scripts/slurm/judge_memory_archive.sbatch
#
# Env knobs:
#   LIMIT     (default 0 = all)          entries per (dataset, version) for a smoke test
#   DATASETS  (default MHC,MHC_zh)
#   VERSIONS  (default v1,v2)            v1 = primary (reproduction-consistent); v2 = robustness
#   OUT       (default memory_repair_out/verdicts.json under scripts/analysis)
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-MHC,MHC_zh}
VERSIONS=${VERSIONS:-v1,v2}
OUTDIR=/data/jehc223/RGCL/scripts/analysis/memory_repair_out
mkdir -p "$OUTDIR"
OUT=${OUT:-$OUTDIR/verdicts$([[ "$LIMIT" != "0" ]] && echo "_smoke" || echo "").json}

DIAG_FLAG=""
if [[ "${DIAG:-0}" == "1" ]]; then
    DIAG_FLAG="--diag"
    OUT="$OUTDIR/diag.json"
fi

echo "[judge_memory_archive] DATASETS=$DATASETS VERSIONS=$VERSIONS LIMIT=$LIMIT DIAG=${DIAG:-0} -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/analysis/judge_memory_archive.py \
    --datasets "$DATASETS" \
    --versions "$VERSIONS" \
    --limit "$LIMIT" \
    $DIAG_FLAG \
    --device cuda \
    --out "$OUT"

echo "[judge_memory_archive] done -> $OUT"
