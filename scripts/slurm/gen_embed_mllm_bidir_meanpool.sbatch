#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=mllm_embed_bidir_mp
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# MNTP staged plan S1 (refine-logs/MNTP_FORENSIC_RECON.md §6): bidirectional attention
# + LLM2Vec MEAN-POOL text readout, NO training, NO adapter beyond the banked task LoRA.
# The cheapest experiment separating H1 ("bidir breaks the causally-trained weights")
# from H2 ("the deployed EOS-class text readout is mismatched to bidirectional topology").
#
# vs the banked causal arms, EXACTLY TWO things differ: the attention mask (bidir, via
# the frozen src/utils/bidir_patch.py sha 36cedbac...) and the TEXT pooling span
# (mean over ALL non-padding positions instead of the trailing <|im_start|>assistant\n
# header). The IMG readout delegates to the frozen causal `_encode` object verbatim.
#
#   (1) MHC_zh   adapter logging/lora/MHC_zh        (== the 13150 generic-LoRA arm)
#                -> out-tag Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF
#   (2) HateMM   adapter logging/lora/HateMM_curric (== the 13241 curric arm)
#                -> out-tag Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF
#
# DEV-GATED: --splits train,val ONLY. The held-out test split is NOT extracted at S1
# (the extractor asserts 'test' is absent from --splits). ZERO test-touch.
# DISTINCT "-bidir-meanpool" out-tags => no banked cache (causal, -bidir, -ro, -moka,
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
  ''|*[!0-9]*) echo "[bidir_mp] FATAL: S1_LIMIT must be digits only (got '${LIMIT}')" >&2; exit 2 ;;
esac
if [ "$LIMIT" -gt 0 ]; then
  TAG_SUFFIX="-smoke"
  echo "[bidir_mp] SMOKE MODE: limit=${LIMIT} items/split, out-tags get '${TAG_SUFFIX}'"
else
  TAG_SUFFIX=""
  echo "[bidir_mp] FULL MODE: all items, splits train,val"
fi

# KS-MNTP-0a installation belt: bidir self-test must PASS before any GPU work.
echo "[bidir_mp] KS-MNTP-0a: bidir_patch.bidir_self_test()"
python -c "import sys; sys.path.insert(0,'src/utils'); from bidir_patch import bidir_self_test; bidir_self_test()"

# Rows: "DATASET  LORA_DIR  OUT_MODEL_TAG_BASE"
# HateMM FIRST: it is the S1 primary dataset (recon §5.1, FN3 is the target cluster),
# so if anything aborts mid-job the primary arm is already banked.
ROWS=(
  "HateMM  logging/lora/HateMM_curric Qwen2.5-VL-7B-Instruct-LoRA-curric-bidir-meanpool_HF"
  "MHC_zh  logging/lora/MHC_zh        Qwen2.5-VL-7B-Instruct-LoRA-bidir-meanpool_HF"
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for row in "${ROWS[@]}"; do
  read -r DS LORA_DIR OUT_TAG_BASE <<< "$row"
  OUT_TAG="${OUT_TAG_BASE}${TAG_SUFFIX}"
  echo "[bidir_mp] DATASET=${DS} LORA_DIR=${LORA_DIR} OUT_TAG=${OUT_TAG} NUM_FRAMES=8 (BIDIR + MEANPOOL)"
  python src/utils/generate_VideoMLLM_embedding_bidir_meanpool_HF.py \
      --dataset "$DS" \
      --lora_dir "$LORA_DIR" \
      --out_model_tag "$OUT_TAG" \
      --splits train,val \
      --limit "$LIMIT" \
      --num_frames 8 \
      --device cuda
  if [ "$LIMIT" -eq 0 ]; then
    # A backup hiccup must not abort the job and cost the second dataset's GPU time;
    # the local cache is already written and valid at this point.
    bash scripts/b2_push.sh "data/CLIP_Embedding/$DS" "embeddings/$DS" \
      || echo "[bidir_mp] WARN: b2_push failed for ${DS}; local cache is written, continuing."
  fi
  echo "[bidir_mp] done DATASET=${DS} (OUT_TAG=${OUT_TAG})"
done

echo "[bidir_mp] ALL DONE"
