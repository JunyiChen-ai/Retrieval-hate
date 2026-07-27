#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=nullop2merge
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit.
#
# KS-MNTP-0b SYMMETRIC — the same-path DOUBLE-MERGE causal floor probe (MNTP_S1_RECORD.md §6d.4
# belt 4, recon §4.3). The S2a arm carries TWO merges, so it is strictly more exposed to the F87
# bf16 merge-drift artifact, which once manufactured a -0.0268 phantom "method effect" that was
# pure path drift. This probe isolates the path: identical runner, identical double
# merge_and_unload, MNTP adapter replaced by a FRESHLY-INITIALISED LoRA of identical shape (PEFT
# zero-inits lora_B, so B@A == 0 and the merge is a MATHEMATICAL null-op -- only the PATH differs),
# and bidir OFF so the arm is directly comparable to the banked CAUSAL cache.
#
# BAR: mean per-item cosine vs the banked causal cache >= 0.9999 on all cells. If it FAILS, the
# same-path double-merge causal floor becomes BINDING and must be extracted in full before any
# S2a accuracy is read.
#
# Subset (--limit 60/split) is sufficient: merge drift is a numerical property of the PATH, not of
# particular items. A failure escalates to a full-split floor extraction.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

ROWS=(
  "HateMM  logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric-nullop2merge_HF"
  "MHC_zh  logging/lora/MHC_zh        Qwen2.5-VL-7B-Instruct-LoRA-nullop2merge_HF"
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for row in "${ROWS[@]}"; do
  read -r DS LORA_DIR OUT_TAG <<< "$row"
  echo "[nullop2merge] DATASET=${DS} OUT_TAG=${OUT_TAG} (zero-init LoRA, bidir OFF, limit 60)"
  python src/utils/generate_VideoMLLM_embedding_bidir_mntp_HF.py \
      --dataset "$DS" \
      --lora_dir "$LORA_DIR" \
      --out_model_tag "$OUT_TAG" \
      --nullop_lora \
      --no_bidir \
      --splits train,val \
      --limit 60 \
      --num_frames 8 \
      --device cuda
done

echo "[nullop2merge] ALL DONE"
