#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=p2b_judge
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/%x_%j.out
# NOTE: intentionally NO --time limit (job auto-releases from JobHeldUser).
# P2b configurable comparability judge over a pairs file (TRAIN benchmark or TEST pairs).
#   CONFIG in {C0,C1,C2,C3,C4,C5}; SET in {train,test}; both languages inside.
#   train: CONFIG=C1 SET=train sbatch scripts/slurm/p2b_judge.sbatch
#   test : CONFIG=C3 SET=test  sbatch scripts/slurm/p2b_judge.sbatch
set -euo pipefail
cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1

CONFIG=${CONFIG:?set CONFIG=C0..C5}
SET=${SET:-train}
OUT=/data/jehc223/RGCL/scripts/analysis/p2_out
LIMIT=${LIMIT:-0}

case "$CONFIG" in
  C0) MODEL="Qwen/Qwen2.5-VL-7B-Instruct"; MT=vl;     EV=archive;            PR=orig ;;
  C1) MODEL="Qwen/Qwen2.5-VL-7B-Instruct"; MT=vl;     EV=archive;            PR=flip ;;
  C2) MODEL="Qwen/Qwen2.5-VL-7B-Instruct"; MT=vl;     EV=archive_transcript; PR=orig ;;
  C3) MODEL="Qwen/Qwen2.5-VL-7B-Instruct"; MT=vl;     EV=archive_transcript; PR=flip ;;
  C4) MODEL="Qwen/Qwen2.5-32B-Instruct";   MT=causal; EV=archive_transcript; PR=flip ;;
  C5) MODEL="Qwen/Qwen2.5-32B-Instruct";   MT=causal; EV=archive_transcript; PR=orig ;;
  *) echo "unknown CONFIG=$CONFIG"; exit 1 ;;
esac

echo "[p2b_judge] CONFIG=$CONFIG SET=$SET MODEL=$MODEL EV=$EV PR=$PR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for DS in MHC MHC_zh; do
  if [[ "$SET" == "train" ]]; then
    PAIRS="$OUT/trainbench_${DS}.jsonl"
    OUTF="$OUT/tb_verdicts_${DS}_${CONFIG}.jsonl"
  else
    PAIRS="$OUT/pairs_${DS}.jsonl"
    OUTF="$OUT/verdicts_${DS}_${CONFIG}.jsonl"
  fi
  echo "=== $DS ($SET) -> $OUTF ==="
  python scripts/analysis/judge_neighbor_compat_p2b.py \
    --dataset "$DS" --pairs_file "$PAIRS" --out_file "$OUTF" \
    --model "$MODEL" --model_type "$MT" --evidence "$EV" --prompt "$PR" \
    --limit "$LIMIT"
done
echo "[p2b_judge] done CONFIG=$CONFIG SET=$SET"
