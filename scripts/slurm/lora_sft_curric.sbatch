#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=lora_sft_curric
#SBATCH --output=/data/jehc223/RGCL/logging/slurm/lora_sft_curric_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# CAND-2 retrieval-confusion CURRICULUM LoRA-SFT (prereg refine-logs/CAND2_CURRICULUM_PREREG.md).
# Usage: sbatch scripts/slurm/lora_sft_curric.sbatch [DATASET]
#   DATASET in {MHC_zh, HateMM}; defaults to MHC_zh.
#   1) builds the GENERIC word-variant ShareGPT data (idempotent; frames + train/val.json +
#      generic dataset_info registration) via src/utils/build_lora_sft_data.py, THEN
#   2) builds the CURRICULUM train set (reweighted multiset, cost-neutral == generic N) via
#      src/utils/build_curriculum_sft_data.py -> data/lora_sft/<DS>/train_curric.json +
#      registers <prefix>_lora_curric_train, THEN
#   3) LoRA-SFT of Qwen2.5-VL-7B-Instruct on the curriculum set via LLaMA-Factory. The recipe
#      is BYTE-IDENTICAL to the generic arm (r16 a32, vision+proj frozen, lr1e-4/3ep/cosine/
#      eff-bs8) — the ONLY manipulated variable is the train example distribution.
set -euo pipefail

RGCL_ROOT=/data/jehc223/RGCL
LF_ROOT=${RGCL_ROOT}/RA-HMD/LLAMA-FACTORY-Ver202512
DATASET=${1:-MHC_zh}

# ---- config + output mapping ------------------------------------------------
case "${DATASET}" in
  MHC_zh)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/mhc_zh_qwen25vl_lora_curric_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/MHC_zh_curric
    ;;
  HateMM)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/hatemm_qwen25vl_lora_curric_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/HateMM_curric
    ;;
  *)
    echo "[lora_sft_curric] unknown DATASET='${DATASET}' (expected MHC_zh or HateMM)"; exit 2 ;;
esac
echo "[lora_sft_curric] DATASET=${DATASET} CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

# ---- disk guard: abort if < 20G free on the FS holding /data/jehc223/RGCL ---
AVAIL_G=$(df --output=avail -BG "${RGCL_ROOT}" | tail -1 | tr -dc '0-9')
echo "[lora_sft_curric] free space on FS holding ${RGCL_ROOT}: ${AVAIL_G}G"
if [ -z "${AVAIL_G}" ] || [ "${AVAIL_G}" -lt 20 ]; then
  echo "[lora_sft_curric] ABORT: free space ${AVAIL_G}G < 20G threshold"; exit 3
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
echo "[lora_sft_curric] building GENERIC data for ${DATASET} (idempotent pre-step) ..."
cd ${RGCL_ROOT}
python src/utils/build_lora_sft_data.py --dataset "${DATASET}"

# ---- STEP 1b: curriculum reweight (train_curric.json + curric registration) ---------
echo "[lora_sft_curric] building CURRICULUM data for ${DATASET} ..."
python src/utils/build_curriculum_sft_data.py --dataset "${DATASET}" --mode softconf

# ---- STEP 2: LoRA-SFT via LLaMA-Factory on the curriculum set -----------------------
cd ${LF_ROOT}
echo "[lora_sft_curric] launching training: python src/train.py ${CONFIG}"
python src/train.py "${CONFIG}"
echo "[lora_sft_curric] DONE DATASET=${DATASET}"
