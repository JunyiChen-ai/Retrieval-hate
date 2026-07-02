#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=lora_sft
#SBATCH --output=/data/jehc223/RGCL/logging/slurm/lora_sft_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# Usage: sbatch scripts/slurm/lora_sft.sbatch [DATASET]
#   DATASET in {MHC, MHC_zh}; defaults to MHC.
#   1) builds LLaMA-Factory multi-image ShareGPT data (8-frame JPGs + train/val.json)
#      via src/utils/build_lora_sft_data.py (idempotent; CPU pre-step)
#   2) LoRA-SFT of Qwen2.5-VL-7B-Instruct via LLaMA-Factory (python src/train.py).
set -euo pipefail

RGCL_ROOT=/data/jehc223/RGCL
LF_ROOT=${RGCL_ROOT}/RA-HMD/LLAMA-FACTORY-Ver202512
DATASET=${1:-MHC}

# ---- config + output mapping ------------------------------------------------
case "${DATASET}" in
  MHC)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/mhc_qwen25vl_lora_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/MHC
    ;;
  MHC_zh)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/mhc_zh_qwen25vl_lora_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/MHC_zh
    ;;
  *)
    echo "[lora_sft] unknown DATASET='${DATASET}' (expected MHC or MHC_zh)"; exit 2 ;;
esac
echo "[lora_sft] DATASET=${DATASET} CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

# ---- disk guard: abort if < 20G free on the FS holding /data/jehc223/RGCL ---
AVAIL_G=$(df --output=avail -BG "${RGCL_ROOT}" | tail -1 | tr -dc '0-9')
echo "[lora_sft] free space on FS holding ${RGCL_ROOT}: ${AVAIL_G}G"
if [ -z "${AVAIL_G}" ] || [ "${AVAIL_G}" -lt 20 ]; then
  echo "[lora_sft] ABORT: free space ${AVAIL_G}G < 20G threshold"; exit 3
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
# The HateVideo env pins trl==0.16.1 (RGCL pipeline). LLaMA-Factory's dependency
# range check wants trl<=0.9.6, so skip its non-mandatory version checks. Our
# stage is LoRA-SFT (transformers 4.49 / peft 0.14 already satisfy LF's ranges).
export DISABLE_VERSION_CHECK=1
# LF eagerly imports trl's DPOTrainer -> deepspeed, whose op-builder probe reads
# $CUDA_HOME/bin/nvcc -V at import time. The login=compute node has no real CUDA
# toolkit, so point CUDA_HOME at a minimal nvcc shim reporting CUDA 12.4 (matches
# torch 2.6.0+cu124). No CUDA op is compiled during LoRA-SFT.
export CUDA_HOME=${LF_ROOT}/.cuda_home_shim
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- STEP 1: build data (frames + ShareGPT json + dataset_info registration)-
echo "[lora_sft] building data for ${DATASET} ..."
cd ${RGCL_ROOT}
python src/utils/build_lora_sft_data.py --dataset "${DATASET}"

# ---- STEP 2: LoRA-SFT via LLaMA-Factory ------------------------------------
# llamafactory is NOT pip-installed in HateVideo; it imports from ./src, so the
# entrypoint is `python src/train.py <config.yaml>` (calls run_exp()).
cd ${LF_ROOT}
echo "[lora_sft] launching training: python src/train.py ${CONFIG}"
python src/train.py "${CONFIG}"
echo "[lora_sft] DONE DATASET=${DATASET}"
