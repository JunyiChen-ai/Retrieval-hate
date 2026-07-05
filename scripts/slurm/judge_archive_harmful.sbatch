#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=judge_harm
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
#
# P1 step 1 (research-wiki/EXP_p1_zerolabel_recal.md): label-FREE HARMFUL/BENIGN verdict
# over each video's structured archive (Qwen2.5-VL-7B, greedy). Text-only, ~1600 archives
# for both languages / one version. Feeds the zero-label prior-recalibration analysis
# (scripts/analysis/p1_prior_recal.py, CPU).
#
# Usage:
#   diag  : DIAG=1 sbatch scripts/slurm/judge_archive_harmful.sbatch
#   smoke : LIMIT=6 sbatch scripts/slurm/judge_archive_harmful.sbatch
#   full  : sbatch scripts/slurm/judge_archive_harmful.sbatch
#
# Env knobs:
#   LIMIT     (default 0 = all)       entries per (dataset, version) for a smoke test
#   DATASETS  (default MHC,MHC_zh)
#   VERSIONS  (default v2)            v2 = primary archive version (per team-lead)
#   OUT       (default p1_out/harmful_verdicts.json under scripts/analysis)
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
VERSIONS=${VERSIONS:-v2}
OUTDIR=/data/jehc223/RGCL/scripts/analysis/p1_out
mkdir -p "$OUTDIR"
OUT=${OUT:-$OUTDIR/harmful_verdicts$([[ "$LIMIT" != "0" ]] && echo "_smoke" || echo "").json}

DIAG_FLAG=""
if [[ "${DIAG:-0}" == "1" ]]; then
    DIAG_FLAG="--diag"
    OUT="$OUTDIR/diag_harmful.json"
fi

echo "[judge_archive_harmful] DATASETS=$DATASETS VERSIONS=$VERSIONS LIMIT=$LIMIT DIAG=${DIAG:-0} -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/analysis/judge_archive_harmful.py \
    --datasets "$DATASETS" \
    --versions "$VERSIONS" \
    --limit "$LIMIT" \
    $DIAG_FLAG \
    --device cuda \
    --out "$OUT"

echo "[judge_archive_harmful] done -> $OUT"
