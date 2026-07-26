#!/usr/bin/env bash
#SBATCH --partition=slurmpartition
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=clap_extract
#SBATCH --output=/data/jehc223/RGCL/slurm/logs/clap_extract_%j.log
# NOTE: intentionally NO --time limit (project policy). PENDING (JobHeldUser) -> wait for
# auto-release, NEVER force. Per-video checkpointing (data/audio/HateMM/clap_<tag>/<id>.npz)
# makes a reap/requeue resume cleanly from where it left off.
#
# NOTE: CPU-ONLY on purpose — NO --gres. Smoke measured 2.2 s/video single-pass on login-node CPU
# (88 s for 20 videos WITH the D-1 double-encode), so 1066 videos ~ 40 min. The CLAP audio tower
# (HTSAT, 0.78 GB) does not need a GPU, and keeping this lane off the GPUs leaves the 2-GPU user cap
# free for the other live lanes. 8 CPU (NOT 16) per the standing infra rule against two concurrent
# 16-CPU jobs.
#
# CLAP general-audio axis — Stage A: extract CLAP (laion/larger_clap_general) audio embeddings for
# HateMM. Per video: PyAV-decode 48 kHz mono -> consecutive non-overlapping 10 s windows ->
# per-window forward -> mean(+)max over windows. Two blocks banked from ONE forward:
#   proj   1024-d (L2-normalised projected joint-space, == get_audio_features)  [BINDING PRIMARY]
#   hidden 2048-d (pre-projection HTSAT pooler_output)                          [SECONDARY]
# Splits: train+val -> clap_<tag>_trainval.pt (the ONLY file the gate opens);
#         test       -> clap_<tag>_test.pt (written, then untouched by this lane).
# Design FROZEN in refine-logs/CLAP_GATE_SPEC_2026-07-27.md (commit 6c8929d) sec 1-2.
# Raw videos/audio stay LOCAL (CLAUDE.md hard rule); features-only output. ZERO GPU.
#   sbatch scripts/slurm/clap_extract.sbatch
set -euo pipefail

cd /data/jehc223/RGCL
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
bash /data/jehc223/RGCL/scripts/disk_guard.sh || true
# CLAP weights are already on disk (laion/larger_clap_general, snapshot ada0c23a...) -> stay OFFLINE.
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# The PyAV decode path needs NO ffmpeg binary and NO torchaudio, so NO aux env is appended to PATH
# (this structurally avoids the 13200 python-shadow trap documented in APX_GATE_RECORD.md sec 1.1).
# Call the HateVideo interpreter by ABSOLUTE path so the env is never ambiguous.
PY=/data/jehc223/miniconda3/envs/HateVideo/bin/python

echo "================================================================================"
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "python=$PY  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "================================================================================"
# Fail fast with a clear message if the interpreter is missing deps (guards the 13200-class regression).
"$PY" -c "import av, transformers, torch; from transformers import ClapModel, ClapFeatureExtractor; \
print('preflight ok  av', av.__version__, ' transformers', transformers.__version__, \
' torch', torch.__version__, ' ClapModel ok')"
"$PY" scripts/analysis/clap_extract.py --splits all --device cpu --batch_windows 8
echo "clap extraction complete jobid=${SLURM_JOB_ID:-none} end=$(date -u +%FT%TZ)"
