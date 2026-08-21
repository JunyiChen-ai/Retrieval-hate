#!/usr/bin/env bash
# REPRO campaign Wave 2 — T3AL launcher.  Parks in the GPU queue until the card is
# free, then runs the whole pipeline (t3al_stage.sh).  Detach with:
#
#   setsid nohup bash scripts/repro_campaign/t3al_launch.sh \
#     >> logging/runs/repro_t3al/run.log 2>&1 < /dev/null &
#   echo $! > logging/runs/repro_t3al/run.pid
set -u
cd /home/jehc223/Retrieval-hate
mkdir -p logging/runs/repro_t3al
echo "[launch] $(date -Is) waiting for the GPU queue"
exec bash scripts/repro_campaign/gpu_queue.sh t3al \
  bash scripts/repro_campaign/t3al_stage.sh
