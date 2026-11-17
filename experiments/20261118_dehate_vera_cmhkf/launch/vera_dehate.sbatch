#!/usr/bin/env bash
# VERA on DeHate (README section 3): prompt-bank selection on the 668 validation videos (video labels, video AP),
# then the frozen test pass (10 s window every second, InternVL2-8B), post-processing, and the baseline evaluator.
# Every stage is resumable (per-video files), so a job stopped by the 1-day limit is resubmitted as is.
# Runs on uoa-lab2 (DeHate videos in ~/data/DeHate, InternVL2-8B in the Hugging Face cache):
#   cd ~/Retrieval-hate && mkdir -p runs/20261118_dehate_vera_cmhkf/vera && \
#       sbatch experiments/20261118_dehate_vera_cmhkf/launch/vera_dehate.sbatch
#SBATCH -J rh-vera-dehate
#SBATCH -p local-sc474399
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/20261118_dehate_vera_cmhkf/vera/slurm_%j.out
set -uo pipefail
cd "$HOME/Retrieval-hate"
V=runs/20261118_dehate_vera_cmhkf/vera
mkdir -p "$V/selection" "$V/final/raw"
PY=$HOME/miniconda3/envs/HateVideo/bin/python
# InternVL2-8B's remote code (InternLM2ForCausalLM without GenerationMixin) has no generate() under transformers
# >= 4.50; the HateVideo environment has 4.57.6, so this job loads transformers 4.45.2 from a separate directory.
export PYTHONPATH="$HOME/Retrieval-hate/third_party/_venv/vera_transformers_4.45.2${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
A=experiments/20261118_dehate_vera_cmhkf/vera_dehate.py
echo "$(date -Is) host $(hostname) job ${SLURM_JOB_ID:-none} vera dehate"
rm -f "$V/vera.DONE" "$V/vera.FAILED"
if $PY -u "$A" select --corpus dehate --out-dir "$V/selection" \
   && $PY -u "$A" infer --corpus dehate --split test --out-dir "$V/final/raw" \
        --prompt-json "$V/selection/selected_prompt.json" \
   && $PY -u "$A" postprocess --corpus dehate --split test --raw-dir "$V/final/raw" --out "$V/final/scores.jsonl" \
   && $PY -u scripts/reproduction_baselines/eval_baseline_scores.py --corpus dehate --split test \
        --scores "$V/final/scores.jsonl" --require-full-coverage --json-out "$V/final/frame_eval.json"; then
  echo "$(date -Is) DONE" | tee "$V/vera.DONE"
else
  echo "$(date -Is) FAILED rc=$?" | tee "$V/vera.FAILED"
fi
