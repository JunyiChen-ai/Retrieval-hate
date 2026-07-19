#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=lora_sft_vis
#SBATCH --output=/data/jehc223/RGCL/logging/slurm/lora_sft_vis_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# VISION-UNFREEZE LoRA-SFT (prereg refine-logs/VISION_UNFREEZE_PREREG.md).
# Usage: sbatch scripts/slurm/lora_sft_vis.sbatch [DATASET]
#   DATASET in {MHC, HateMM}; defaults to MHC.
#   1) builds the GENERIC word-variant ShareGPT data (idempotent; frames + train/val.json +
#      generic dataset_info registration) via src/utils/build_lora_sft_data.py — the SAME
#      data the banked generic-LoRA comparator trained on (byte-identical: K-V2 is a clean
#      superset), THEN
#   2) LoRA-SFT of Qwen2.5-VL-7B-Instruct via LLaMA-Factory. The recipe is BYTE-IDENTICAL to
#      the generic arm EXCEPT two flipped lines (freeze_vision_tower: false + lora_target: all)
#      + output_dir — LoRA now reaches the Qwen2.5-VL ViT blocks (projector stays frozen).
#      The ONLY manipulated variable vs generic is the LoRA target set (LLM ⊕ ViT).
set -euo pipefail

RGCL_ROOT=/data/jehc223/RGCL
LF_ROOT=${RGCL_ROOT}/RA-HMD/LLAMA-FACTORY-Ver202512
DATASET=${1:-MHC}

# ---- config + output mapping ------------------------------------------------
case "${DATASET}" in
  MHC)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/mhc_qwen25vl_lora_vis_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/MHC_vis
    ;;
  HateMM)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/hatemm_qwen25vl_lora_vis_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/HateMM_vis
    ;;
  *)
    echo "[lora_sft_vis] unknown DATASET='${DATASET}' (expected MHC or HateMM)"; exit 2 ;;
esac
echo "[lora_sft_vis] DATASET=${DATASET} CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

# ---- disk guard: abort if < 20G free on the FS holding /data/jehc223/RGCL ---
AVAIL_G=$(df --output=avail -BG "${RGCL_ROOT}" | tail -1 | tr -dc '0-9')
echo "[lora_sft_vis] free space on FS holding ${RGCL_ROOT}: ${AVAIL_G}G"
if [ -z "${AVAIL_G}" ] || [ "${AVAIL_G}" -lt 20 ]; then
  echo "[lora_sft_vis] ABORT: free space ${AVAIL_G}G < 20G threshold"; exit 3
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

# ---- STEP 1: build data (frames + ShareGPT json + dataset_info registration)-
# Idempotent generic word-variant build == the data the banked generic-LoRA comparator
# trained on (no curriculum, no reweight; the vis arm only changes the LoRA reach).
echo "[lora_sft_vis] building data for ${DATASET} ..."
cd ${RGCL_ROOT}
python src/utils/build_lora_sft_data.py --dataset "${DATASET}"

# ---- STEP 2: vision-unfreeze LoRA-SFT via LLaMA-Factory ---------------------
cd ${LF_ROOT}
echo "[lora_sft_vis] launching training: python src/train.py ${CONFIG}"
python src/train.py "${CONFIG}"
echo "[lora_sft_vis] DONE DATASET=${DATASET}"
