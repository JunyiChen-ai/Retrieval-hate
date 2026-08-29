#!/usr/bin/env bash
#SBATCH --job-name=v9-hatemm-evidence
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/reproduction/relation_v9/hatemm-evidence-%j.out
set -euo pipefail
cd /home/jehc223/Hate-follow-up
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONPATH="$PWD/scripts/reproduction_baselines"
mode="${EVIDENCE_MODE:?submit with --export=ALL,EVIDENCE_MODE=macil or vera}"
if [[ "$mode" == macil ]]; then
  for seed in 234 2025 3407; do
    python scripts/reproduction_baselines/relation_v9/produce_hatemm_macil_train.py --seed "$seed" --device cuda
  done
elif [[ "$mode" == vera ]]; then
  raw=results/reproduction/official_val/final/vera/hatemm/seed_234/train_infer/raw
  scores=results/reproduction/official_val/final/vera/hatemm/seed_234/train_infer/scores.jsonl
  prompt=results/reproduction/official_val/tuning/vera/hatemm/selected_prompt.json
  python scripts/reproduction_baselines/relation_v9/vera_hatemm_train_label_free.py \
    --raw-dir "$raw" --out "$scores" --prompt-json "$prompt" --batch-size 1
  python scripts/reproduction_baselines/relation_v9/finalize_hatemm_vera_train.py
else
  echo "invalid EVIDENCE_MODE=$mode" >&2
  exit 2
fi
