#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=laud_extract
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/laud_extract_%j.log
# NOTE: intentionally NO --time limit (project policy). PENDING (JobHeldUser) -> wait for
# auto-release, never force. Per-video checkpointing (data/audio/<DS>/whisper_<tag>/<id>.npz)
# makes a reap/requeue resume cleanly from where it left off.
#
# LEARNED-AUDIO axis — Stage A: extract Whisper-large-v3 ENCODER hidden-state embeddings
# (mean(+)max pooled, 2560-d video vector + K=4 segment variant) for HateMM / MHC(EN) / MHC_zh(ZH),
# train (union) val only (ZERO test-touch). Design: refine-logs/AUDIO_AXIS_FORENSIC_RECON.md sec 2-3.
# Raw videos stay LOCAL (CLAUDE.md hard rule); features-only .pt output. GPU-light (recon: ~1-2 GPU-h).
#   sbatch scripts/slurm/laud_extract.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
# Whisper weights are already on disk (openai/whisper-large-v3, snapshot 06f233fe...) -> stay OFFLINE.
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# The PyAV decode path needs NO ffmpeg binary and NO torchaudio, so NO aux env is appended to PATH
# (this structurally avoids the 13200 python-shadow trap documented in APX_GATE_RECORD.md sec 1.1).
# Call the HateVideo interpreter by ABSOLUTE path so the env is never ambiguous.
PY=/data/jehc223/miniconda3/envs/HateVideo/bin/python

echo "================================================================================"
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "python=$PY"
echo "================================================================================"
# Fail fast with a clear message if the interpreter is missing deps (guards the 13200-class regression).
"$PY" -c "import av, transformers, torch; \
print('preflight ok  av', av.__version__, ' transformers', transformers.__version__, \
' torch', torch.__version__, ' cuda', torch.cuda.is_available())"
"$PY" scripts/analysis/laud_extract_whisper.py
echo "laud whisper-encoder extraction complete jobid=${SLURM_JOB_ID:-none} end=$(date -u +%FT%TZ)"
