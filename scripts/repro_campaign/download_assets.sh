#!/usr/bin/env bash
# Phase A / route 2: download model assets for the label-free frame-level repro campaign.
# Big model weights go to the default HF cache (~/.cache/huggingface/hub).
# Gated meta-llama repos are replaced by ungated mirrors (NousResearch).
# hf_fetch.py wraps huggingface-cli with a stall watchdog (hf_transfer hangs
# every few GB from this host).
export HF_HUB_ENABLE_HF_TRANSFER=1
set -u
LOG=/home/jehc223/Retrieval-hate/logging/runs/repro_assets/run.log
FETCH=/home/jehc223/Retrieval-hate/scripts/repro_campaign/hf_fetch.py
mkdir -p "$(dirname "$LOG")"
CKPT=/home/jehc223/Retrieval-hate/third_party/_ckpt
mkdir -p "$CKPT"

log () { echo "[$(date +%F_%T)] $*" >>"$LOG"; }

hfd () {
  local repo="$1"; shift
  log "START $repo $*"
  python "$FETCH" "$repo" "$@" >>"$LOG" 2>&1 && log "DONE  $repo" || log "FAIL  $repo"
}

hfd Salesforce/blip2-opt-6.7b-coco --exclude "*.bin" "*.msgpack" "*.h5"
hfd DAMO-NLP-SG/VideoLLaMA3-7B
hfd DAMO-NLP-SG/VL3-SigLIP-NaViT
hfd NousResearch/Meta-Llama-3.1-8B-Instruct --exclude "*.pth"
hfd NousResearch/Llama-2-13b-chat-hf --exclude "*.bin" "*.msgpack" "*.h5"

# ImageBind huge (direct URL, no HF)
if [ ! -s "$CKPT/imagebind_huge.pth" ]; then
  log "START imagebind_huge"
  wget -q --tries=20 --continue --timeout=60 -O "$CKPT/imagebind_huge.pth" \
    https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth >>"$LOG" 2>&1 \
    && log "DONE  imagebind_huge" || log "FAIL  imagebind_huge"
fi

log "STREAM1 ALL DONE"
