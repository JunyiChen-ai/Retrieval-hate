#!/usr/bin/env bash
# Phase A / route 2, stream 2: smaller method-specific checkpoints.
export HF_HUB_ENABLE_HF_TRANSFER=1
set -u
LOG=/home/jehc223/Retrieval-hate/logging/runs/repro_assets/run2.log
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

# AV2A: LanguageBind towers  (UniTime LoRA already fetched)
hfd LanguageBind/LanguageBind_Image
hfd lb203/LanguageBind_Image

# SeViLA localizer + answerer
if [ ! -s "$CKPT/sevila/sevila_pretrained.pth" ]; then
  mkdir -p "$CKPT/sevila"; log "START sevila_pretrained.pth"
  wget -q --tries=20 --continue --timeout=60 -O "$CKPT/sevila/sevila_pretrained.pth" \
    https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth >>"$LOG" 2>&1 \
    && log "DONE  sevila_pretrained.pth" || log "FAIL  sevila_pretrained.pth"
fi

# LaGoVAD pretrained ckpts (Google Drive)
if [ ! -e "$CKPT/lagovad_ckpts.zip" ]; then
  log "START lagovad gdrive"
  gdown -O "$CKPT/lagovad_ckpts.zip" \
    161T7EkR64Px1cveP7_dG1xupT8IEICbM >>"$LOG" 2>&1 \
    && log "DONE  lagovad gdrive" || log "FAIL  lagovad gdrive"
fi

# OpenAI CLIP ViT-L/14-336 raw weights (LaGoVAD feature extractor format)
if [ ! -s "$CKPT/ViT-L-14-336px.pt" ]; then
  log "START ViT-L-14-336px.pt"
  wget -q --tries=20 --continue --timeout=60 -O "$CKPT/ViT-L-14-336px.pt" \
    https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt >>"$LOG" 2>&1 \
    && log "DONE  ViT-L-14-336px.pt" || log "FAIL  ViT-L-14-336px.pt"
fi

log "STREAM2 ALL DONE"
