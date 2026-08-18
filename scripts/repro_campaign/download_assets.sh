#!/usr/bin/env bash
# Phase A / route 2: download model assets for the label-free frame-level repro campaign.
# All downloads go to the default HF cache (~/.cache/huggingface/hub) unless noted.
# Gated meta-llama repos are replaced by ungated byte-identical mirrors (NousResearch).
export HF_HUB_ENABLE_HF_TRANSFER=1
set -u
LOG=/home/jehc223/Retrieval-hate/logging/runs/repro_assets/run.log
mkdir -p "$(dirname "$LOG")"
CKPT=/home/jehc223/Retrieval-hate/third_party/_ckpt
mkdir -p "$CKPT"

hfd () {  # repo, then allow patterns
  local repo="$1"; shift
  echo "[$(date +%F_%T)] START $repo $*" >>"$LOG"
  huggingface-cli download "$repo" "$@" --quiet >>"$LOG" 2>&1 \
    && echo "[$(date +%F_%T)] DONE  $repo" >>"$LOG" \
    || echo "[$(date +%F_%T)] FAIL  $repo (rc=$?)" >>"$LOG"
}

hfd Salesforce/blip2-opt-6.7b-coco --exclude "*.bin" "*.msgpack" "*.h5"
hfd NousResearch/Llama-2-13b-chat-hf --exclude "*.bin" "*.msgpack" "*.h5"
hfd NousResearch/Meta-Llama-3.1-8B-Instruct --exclude "*.pth"
hfd NousResearch/Meta-Llama-3.1-8B-Instruct --include "original/*"
hfd DAMO-NLP-SG/VideoLLaMA3-7B
hfd DAMO-NLP-SG/VL3-SigLIP-NaViT

# ImageBind huge (direct URL, no HF)
if [ ! -s "$CKPT/imagebind_huge.pth" ]; then
  echo "[$(date +%F_%T)] START imagebind_huge" >>"$LOG"
  wget -q -O "$CKPT/imagebind_huge.pth" \
    https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth >>"$LOG" 2>&1 \
    && echo "[$(date +%F_%T)] DONE  imagebind_huge" >>"$LOG" \
    || echo "[$(date +%F_%T)] FAIL  imagebind_huge" >>"$LOG"
fi

echo "[$(date +%F_%T)] ALL DONE" >>"$LOG"
