#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_embed_bidir_mntp
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# MNTP staged plan S2a (refine-logs/MNTP_S1_RECORD.md §6d, declared before the fork was
# built): bidirectional attention + the PUBLISHED McGill MNTP LoRA TRANSPLANT, ZERO training.
# S1/S1b closed the readout route (the vision/text mixture happens INSIDE attention, before any
# pooling), so the surviving hypothesis is the actual MNTP claim: bidirectional attention needs
# WEIGHT ADAPTATION. S2a tests it at zero training cost.
#
# READOUT IS THE DEPLOYED ONE, UNTOUCHED -- so this runner is the F72 bidir arm plus exactly one
# adapter load/merge, and the frozen KS-MNTP-1 bars stay valid without reinterpretation.
#
# Arm (order DECLARED = the cheap order, recon §4.2):
#   base VL -> task LoRA -> merge -> MNTP LoRA (bound to model.model) -> merge -> bidir -> extract
# The second adapter binds the BARE DECODER, not the outer ForConditionalGeneration: the McGill
# keys are base_model.model.layers.N..., which only aligns there, and outer binding would also
# have wrapped 96 VISION-TOWER modules.
#
#   (1) HateMM  adapter logging/lora/HateMM_curric -> Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-mntp_HF
#   (2) MHC_zh  adapter logging/lora/MHC_zh        -> Qwen2.5-VL-7B-Instruct-LoRA-bidir-mntp_HF
#
# DEV-GATED: --splits train,val ONLY. ZERO test-touch. Distinct "-bidir-mntp" out-tags => no
# banked cache can be clobbered. Smoke mode: S1_LIMIT=<N> also appends "-smoke" to the out-tag.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

MNTP_DIR="/data/jehc223/home/.cache/huggingface/hub/models--McGill-NLP--LLM2Vec-Qwen25-7B-Instruct-mntp/snapshots/2c7b5e8605db3caa2593d1ae8afb09c826c0d887"
[ -d "$MNTP_DIR" ] || { echo "[bidir_mntp] FATAL: MNTP adapter dir missing: $MNTP_DIR" >&2; exit 2; }

LIMIT="${S1_LIMIT:-0}"
# Digits-only: bash arithmetic and python's int() disagree on e.g. "1_0", which could
# yield a truncated cache WITHOUT the protective -smoke suffix. Reject anything else.
case "$LIMIT" in
  ''|*[!0-9]*) echo "[bidir_mntp] FATAL: S1_LIMIT must be digits only (got '${LIMIT}')" >&2; exit 2 ;;
esac
# Length cap: an all-digit value beyond bash's signed-int range passes the regex above but
# makes `-gt` fail with "integer expression expected". That failure sits in an `if`
# condition, so `set -e` does NOT fire and the run would silently take the FULL out-tag
# while python truncated the split -- a partial cache wearing a full cache's name.
if [ "${#LIMIT}" -gt 9 ]; then
  echo "[bidir_mntp] FATAL: S1_LIMIT out of range (got '${LIMIT}')" >&2; exit 2
fi
if [ "$LIMIT" -gt 0 ]; then
  TAG_SUFFIX="-smoke"
  echo "[bidir_mntp] SMOKE MODE: limit=${LIMIT} items/split, out-tags get '${TAG_SUFFIX}'"
else
  TAG_SUFFIX=""
  echo "[bidir_mntp] FULL MODE: all items, splits train,val"
fi

# KS-MNTP-0a installation belt: bidir self-test must PASS before any GPU work.
echo "[bidir_mntp] KS-MNTP-0a: bidir_patch.bidir_self_test()"
python -c "import sys; sys.path.insert(0,'src/utils'); from bidir_patch import bidir_self_test; bidir_self_test()"

# Rows: "DATASET  LORA_DIR  OUT_MODEL_TAG_BASE"
# HateMM FIRST: it is the S1 primary dataset (recon §5.1, FN3 is the target cluster),
# so if anything aborts mid-job the primary arm is already banked.
ROWS=(
  "HateMM  logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-mntp_HF"
  "MHC_zh  logging/lora/MHC_zh        Qwen2.5-VL-7B-Instruct-LoRA-bidir-mntp_HF"
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for row in "${ROWS[@]}"; do
  read -r DS LORA_DIR OUT_TAG_BASE <<< "$row"
  OUT_TAG="${OUT_TAG_BASE}${TAG_SUFFIX}"
  echo "[bidir_mntp] DATASET=${DS} LORA_DIR=${LORA_DIR} OUT_TAG=${OUT_TAG} NUM_FRAMES=8 (BIDIR + MNTP TRANSPLANT)"
  python src/utils/generate_VideoMLLM_embedding_bidir_mntp_HF.py \
      --dataset "$DS" \
      --mntp_dir "$MNTP_DIR" \
      --lora_dir "$LORA_DIR" \
      --out_model_tag "$OUT_TAG" \
      --splits train,val \
      --limit "$LIMIT" \
      --num_frames 8 \
      --device cuda
  if [ "$LIMIT" -eq 0 ]; then
    # Push ONLY this arm's two cache files, BY NAME. Never the whole directory: that
    # directory holds pre-existing test_seen_*.pt caches from other arms, and a recursive
    # rclone copy would open and upload them. Copying bytes leaks no test information into
    # any gate, but this arm's record asserts zero test-touch literally, so the backup is
    # scoped to match the assertion instead of weakening it.
    # A backup hiccup must not abort the job and cost the second dataset's GPU time;
    # the local caches are already written and valid at this point.
    for SP in train dev_seen; do
      F="data/CLIP_Embedding/${DS}/${SP}_${OUT_TAG}.pt"
      [ -f "$F" ] || { echo "[bidir_mntp] WARN: expected cache missing: $F"; continue; }
      # In FILE mode b2_push.sh treats arg 2 as the full destination path INCLUDING the
      # filename (it uploads into dirname(dest)/). So the basename must be appended, or the
      # file lands in embeddings/ instead of embeddings/<DS>/.
      bash scripts/b2_push.sh "$F" "embeddings/$DS/$(basename "$F")" \
        || echo "[bidir_mntp] WARN: b2_push failed for ${F}; local cache is written, continuing."
    done
  fi
  echo "[bidir_mntp] done DATASET=${DS} (OUT_TAG=${OUT_TAG})"
done

echo "[bidir_mntp] ALL DONE"
