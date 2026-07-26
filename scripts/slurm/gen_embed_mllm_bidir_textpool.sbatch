#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_embed_bidir_tp
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# MNTP staged plan S1b (refine-logs/MNTP_S1_RECORD.md §6b, declared before the fork was
# built): bidirectional attention + TEXT-POSITIONS-ONLY mean-pool readout. No training,
# no adapter beyond the banked task LoRA. This is the CLEAN test of H2 that S1 did not
# run: S1 pooled all ~900 positions, ~80% of which are vision tokens from the same 8
# frames the img stream pools, so its "text" vector collapsed onto the img vector
# (cos 0.9273/0.9320) and stopped being a text readout.
#
# vs the banked causal arms, EXACTLY TWO things differ: the attention mask (bidir, via
# the frozen src/utils/bidir_patch.py sha 36cedbac...) and the TEXT pooling span (mean
# over positions whose token id is NOT <|video_pad|>/<|image_pad|>, instead of the
# trailing <|im_start|>assistant\n header). The IMG readout delegates to the frozen
# causal `_encode` object verbatim and must remain an exact null-op.
#
#   (1) MHC_zh   adapter logging/lora/MHC_zh        (== the 13150 generic-LoRA arm)
#                -> out-tag Qwen2.5-VL-7B-Instruct-LoRA-bidir-textpool_HF
#   (2) HateMM   adapter logging/lora/HateMM_curric (== the 13241 curric arm)
#                -> out-tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-textpool_HF
#
# DEV-GATED: --splits train,val ONLY. The held-out test split is NOT extracted at S1
# (the extractor asserts 'test' is absent from --splits). ZERO test-touch.
# DISTINCT "-bidir-textpool" out-tags => no banked cache (causal, -bidir, -ro, -moka,
# -zhp, -um) is ever clobbered.
#
# Smoke mode: S1_LIMIT=<N> processes only the first N items per split AND appends a
# "-smoke" suffix to the out-tag, so a smoke run can never overwrite a real cache.
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

LIMIT="${S1_LIMIT:-0}"
# Digits-only: bash arithmetic and python's int() disagree on e.g. "1_0", which could
# yield a truncated cache WITHOUT the protective -smoke suffix. Reject anything else.
case "$LIMIT" in
  ''|*[!0-9]*) echo "[bidir_tp] FATAL: S1_LIMIT must be digits only (got '${LIMIT}')" >&2; exit 2 ;;
esac
# Length cap: an all-digit value beyond bash's signed-int range passes the regex above but
# makes `-gt` fail with "integer expression expected". That failure sits in an `if`
# condition, so `set -e` does NOT fire and the run would silently take the FULL out-tag
# while python truncated the split -- a partial cache wearing a full cache's name.
if [ "${#LIMIT}" -gt 9 ]; then
  echo "[bidir_tp] FATAL: S1_LIMIT out of range (got '${LIMIT}')" >&2; exit 2
fi
if [ "$LIMIT" -gt 0 ]; then
  TAG_SUFFIX="-smoke"
  echo "[bidir_tp] SMOKE MODE: limit=${LIMIT} items/split, out-tags get '${TAG_SUFFIX}'"
else
  TAG_SUFFIX=""
  echo "[bidir_tp] FULL MODE: all items, splits train,val"
fi

# KS-MNTP-0a installation belt: bidir self-test must PASS before any GPU work.
echo "[bidir_tp] KS-MNTP-0a: bidir_patch.bidir_self_test()"
python -c "import sys; sys.path.insert(0,'src/utils'); from bidir_patch import bidir_self_test; bidir_self_test()"

# Rows: "DATASET  LORA_DIR  OUT_MODEL_TAG_BASE"
# HateMM FIRST: it is the S1 primary dataset (recon §5.1, FN3 is the target cluster),
# so if anything aborts mid-job the primary arm is already banked.
ROWS=(
  "HateMM  logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-textpool_HF"
  "MHC_zh  logging/lora/MHC_zh        Qwen2.5-VL-7B-Instruct-LoRA-bidir-textpool_HF"
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for row in "${ROWS[@]}"; do
  read -r DS LORA_DIR OUT_TAG_BASE <<< "$row"
  OUT_TAG="${OUT_TAG_BASE}${TAG_SUFFIX}"
  echo "[bidir_tp] DATASET=${DS} LORA_DIR=${LORA_DIR} OUT_TAG=${OUT_TAG} NUM_FRAMES=8 (BIDIR + MEANPOOL)"
  python src/utils/generate_VideoMLLM_embedding_bidir_textpool_HF.py \
      --dataset "$DS" \
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
      [ -f "$F" ] || { echo "[bidir_tp] WARN: expected cache missing: $F"; continue; }
      # In FILE mode b2_push.sh treats arg 2 as the full destination path INCLUDING the
      # filename (it uploads into dirname(dest)/). So the basename must be appended, or the
      # file lands in embeddings/ instead of embeddings/<DS>/.
      bash scripts/b2_push.sh "$F" "embeddings/$DS/$(basename "$F")" \
        || echo "[bidir_tp] WARN: b2_push failed for ${F}; local cache is written, continuing."
    done
  fi
  echo "[bidir_tp] done DATASET=${DS} (OUT_TAG=${OUT_TAG})"
done

echo "[bidir_tp] ALL DONE"
