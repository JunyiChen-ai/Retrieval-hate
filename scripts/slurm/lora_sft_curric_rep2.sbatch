#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=lora_sft_curric_rep2
#SBATCH --output=/data/jehc223/RGCL/logging/slurm/lora_sft_curric_rep2_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# CAND-2 curriculum LoRA-SFT DRAW-2 REPLICATION (prereg refine-logs/CAND2_REP2_PREREG.md).
# HateMM ONLY. This is a SECOND independent SFT draw of the SAME frozen curriculum multiset
# (train_curric.json sha 73307ef2...), differing from the draw-1 arm (job 13238) by exactly
# the SFT seed (draw-1 = implicit HF default 42; draw-2 = explicit seed:1 baked into the
# rep2 config). Everything else — recipe, curriculum data, extraction/head code — is
# byte-identical. It converts the draw-1 F0.2 single-curriculum-draw caveat into a 2-draw
# measurement. ZH is NOT re-run (tied both protocols in draw-1; prereg F56/F0.7).
#   1) STEP 1a: builds the GENERIC word-variant ShareGPT data (idempotent) via
#      src/utils/build_lora_sft_data.py, THEN
#   2) STEP 1b: rebuilds the CURRICULUM train set (RNG-free deterministic builder; re-emits
#      train_curric.json bit-exact, sha 73307ef2...) via src/utils/build_curriculum_sft_data.py
#      -> proves draw-2 trains on the IDENTICAL curriculum data, THEN
#   3) STEP 2: LoRA-SFT of Qwen2.5-VL-7B-Instruct on the curriculum set via LLaMA-Factory
#      using the rep2 config (seed:1, output HateMM_curric_rep2). The ONLY manipulated
#      variable vs draw-1 is the SFT seed.
set -euo pipefail

RGCL_ROOT=/data/jehc223/RGCL
LF_ROOT=${RGCL_ROOT}/RA-HMD/LLAMA-FACTORY-Ver202512
DATASET=HateMM
CONFIG=${LF_ROOT}/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft_rep2.yaml
OUTDIR=${RGCL_ROOT}/logging/lora/HateMM_curric_rep2
echo "[lora_sft_curric_rep2] DATASET=${DATASET} CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

# ---- disk guard: abort if < 20G free on the FS holding /data/jehc223/RGCL ---
AVAIL_G=$(df --output=avail -BG "${RGCL_ROOT}" | tail -1 | tr -dc '0-9')
echo "[lora_sft_curric_rep2] free space on FS holding ${RGCL_ROOT}: ${AVAIL_G}G"
if [ -z "${AVAIL_G}" ] || [ "${AVAIL_G}" -lt 20 ]; then
  echo "[lora_sft_curric_rep2] ABORT: free space ${AVAIL_G}G < 20G threshold"; exit 3
fi
# Best-effort project quota watchdog (never aborts this job).
bash ${RGCL_ROOT}/scripts/disk_guard.sh || true

# ---- environment ------------------------------------------------------------
# NOTE: source conda.sh directly (NOT ~/.bashrc): the system /etc/bashrc
# references an unbound var (BASHRCSOURCED) which aborts under `set -u`.
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
# ---- HateVideo env compatibility shims for LLaMA-Factory-Ver202512 ----------
# LF's dependency range check wants trl<=0.9.6 (env pins trl 0.16.1); our stage is
# LoRA-SFT so skip the non-mandatory version check.
export DISABLE_VERSION_CHECK=1
# LF eagerly imports trl's DPOTrainer -> deepspeed, whose op-builder probe reads
# $CUDA_HOME/bin/nvcc -V at import time; point CUDA_HOME at the nvcc shim (no CUDA op
# is compiled during LoRA-SFT).
export CUDA_HOME=${LF_ROOT}/.cuda_home_shim
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- STEP 1a: generic build (frames + word train/val.json + generic reg; idempotent) -
echo "[lora_sft_curric_rep2] building GENERIC data for ${DATASET} (idempotent pre-step) ..."
cd ${RGCL_ROOT}
python src/utils/build_lora_sft_data.py --dataset "${DATASET}"

# ---- STEP 1b: curriculum reweight (RNG-free; re-emits train_curric.json bit-exact) --
echo "[lora_sft_curric_rep2] rebuilding CURRICULUM data for ${DATASET} (idempotent; sha must == 73307ef2...) ..."
python src/utils/build_curriculum_sft_data.py --dataset "${DATASET}" --mode softconf

# ---- STEP 2: LoRA-SFT via LLaMA-Factory on the curriculum set (draw-2 seed) ----------
cd ${LF_ROOT}
echo "[lora_sft_curric_rep2] launching training: python src/train.py ${CONFIG}"
python src/train.py "${CONFIG}"
echo "[lora_sft_curric_rep2] DONE DATASET=${DATASET} (draw-2)"
