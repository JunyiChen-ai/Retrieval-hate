#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=lora_sft_moka
#SBATCH --output=/data/jehc223/RGCL/logging/slurm/lora_sft_moka_%j.out
# NOTE: intentionally NO --time limit (jobs may run unbounded).
#
# MOKA cell — stage 1 of 2 (refine-logs/MOKA_PREREG.md).
# Clone of scripts/slurm/lora_sft.sbatch. The ONLY differences are:
#   (a) entry point  python src/train.py <yaml>  ->  python src/moka/train_moka.py <yaml>
#       (which monkey-patches llamafactory.model.adapter.get_peft_model and then calls the very
#        same run_exp(); ZERO lines of the vendored LLaMA-Factory tree are edited);
#   (b) the yaml (identical to the deployed ZH recipe except output_dir; `diff` = 1 line);
#   (c) a >=25G disk preflight instead of >=20G (MokA adapters are 1.4489x, ~234 MB each x 4 saves);
#   (d) a $0 post-run KS-MOKA-2 readout (per-layer ||A_v - A_t||_F / ||A_t||_F) + a key-count assert.
#
# Usage: sbatch scripts/slurm/lora_sft_moka.sbatch [DATASET]    # DATASET = MHC_zh (default)
set -euo pipefail

RGCL_ROOT=/data/jehc223/RGCL
LF_ROOT=${RGCL_ROOT}/RA-HMD/LLAMA-FACTORY-Ver202512
DATASET=${1:-MHC_zh}

# ---- config + output mapping ------------------------------------------------
case "${DATASET}" in
  MHC_zh)
    CONFIG=${LF_ROOT}/my_configs/hatevideo/mhc_zh_qwen25vl_lora_moka_sft.yaml
    OUTDIR=${RGCL_ROOT}/logging/lora/MHC_zh_moka
    ;;
  HateMM)
    echo "[moka_sft] HateMM is the STAGE-2 hold-the-pass leg and is AUTO-DEFUNDED by KS-MOKA-1."
    echo "[moka_sft] It requires (i) a surviving ZH verdict, (ii) a prereg amendment, (iii) a new"
    echo "[moka_sft] hatemm_qwen25vl_lora_curric_moka_sft.yaml. Refusing to run."; exit 2 ;;
  *)
    echo "[moka_sft] unknown DATASET='${DATASET}' (this prereg authorises MHC_zh only)"; exit 2 ;;
esac
echo "[moka_sft] DATASET=${DATASET} CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

# ---- disk guard: abort if < 25G free on the FS holding /data/jehc223/RGCL ----
AVAIL_G=$(df --output=avail -BG "${RGCL_ROOT}" | tail -1 | tr -dc '0-9')
echo "[moka_sft] free space on FS holding ${RGCL_ROOT}: ${AVAIL_G}G (need >=25G; MokA adapter set ~1.0G)"
if [ -z "${AVAIL_G}" ] || [ "${AVAIL_G}" -lt 25 ]; then
  echo "[moka_sft] ABORT: free space ${AVAIL_G}G < 25G threshold"; exit 3
fi
bash ${RGCL_ROOT}/scripts/disk_guard.sh || true

# ---- collision guard: never clobber a banked adapter ------------------------
if [ -e "${OUTDIR}/adapter_model.safetensors" ]; then
  echo "[moka_sft] ABORT: ${OUTDIR}/adapter_model.safetensors already exists (prior partial run)."
  exit 4
fi

# ---- environment (verbatim from lora_sft.sbatch) ----------------------------
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export DISABLE_VERSION_CHECK=1
export CUDA_HOME=${LF_ROOT}/.cuda_home_shim
# MokA routing: 'dense' is the FROZEN deployed formulation (bit-exact identity control); strict=1
# makes a missing/mismatched modality mask a hard error instead of a silent plain-LoRA null-op.
export MOKA_ROUTE_IMPL=dense
export MOKA_STRICT=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- STEP 1: build data (idempotent; identical to the deployed floor) --------
echo "[moka_sft] building data for ${DATASET} ..."
cd ${RGCL_ROOT}
python src/utils/build_lora_sft_data.py --dataset "${DATASET}"

# ---- STEP 2: MokA-routed LoRA-SFT via the LLaMA-Factory entry ---------------
cd ${LF_ROOT}
echo "[moka_sft] launching training: python ${RGCL_ROOT}/src/moka/train_moka.py ${CONFIG}"
python ${RGCL_ROOT}/src/moka/train_moka.py "${CONFIG}"

# ---- STEP 3: $0 post-run asserts + KS-MOKA-2 routing readout ----------------
cd ${RGCL_ROOT}
python - "${OUTDIR}" <<'PY'
import json, os, statistics, sys
import torch
from safetensors.torch import load_file
out = sys.argv[1]
sd = load_file(os.path.join(out, "adapter_model.safetensors"))
n_a  = sum(1 for k in sd if ".lora_A." in k)
n_av = sum(1 for k in sd if ".lora_A_v." in k)
n_b  = sum(1 for k in sd if ".lora_B." in k)
tot  = sum(v.numel() for v in sd.values())
print("[moka_sft] adapter keys: lora_A {} | lora_A_v {} | lora_B {} | total tensors {} | params {}".format(
    n_a, n_av, n_b, len(sd), tot))
assert n_a == n_av == n_b == 196, "expected 28 layers x 7 projections = 196 of each"
assert tot == 58490880, "expected 58,490,880 MokA params (1.4489x the deployed 40,370,176), got {}".format(tot)
# KS-MOKA-2: per-layer ||A_v - A_t||_F / ||A_t||_F  (routing-is-real, $0, no test touch)
rows = []
for k in sorted(sd):
    if ".lora_A." not in k:
        continue
    a_t = sd[k].float(); a_v = sd[k.replace(".lora_A.", ".lora_A_v.")].float()
    den = torch.linalg.norm(a_t).item()
    rows.append({"module": k, "rel_fro_diff": (torch.linalg.norm(a_v - a_t).item() / den) if den else float("nan")})
vals = sorted(r["rel_fro_diff"] for r in rows)
# TRUE median: n == 196 is EVEN, so the median is (vals[97] + vals[98]) / 2, not vals[98].
# `statistics.median` implements exactly that (mean of the two middle order statistics for even n,
# the middle element for odd n) on the already-sorted list.
med = statistics.median(vals)
print("[moka_sft] KS-MOKA-2 rel ||A_v-A_t||_F/||A_t||_F : min {:.4f} median {:.4f} max {:.4f}".format(
    vals[0], med, vals[-1]))
print("[moka_sft] KS-MOKA-2 median >= 0.05 ? {}".format(med >= 0.05))
json.dump({"rows": rows, "median": med, "min": vals[0], "max": vals[-1]},
          open(os.path.join("refine-logs", "MOKA_KS2_routing_report.json"), "w"), indent=1)
PY

echo "[moka_sft] DONE DATASET=${DATASET} -> ${OUTDIR}"
