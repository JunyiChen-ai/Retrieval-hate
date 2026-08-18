#!/usr/bin/env bash
# Phase A / route 2, stream 2: smaller method-specific checkpoints.
export HF_HUB_ENABLE_HF_TRANSFER=1
set -u
LOG=/home/jehc223/Retrieval-hate/logging/runs/repro_assets/run2.log
mkdir -p "$(dirname "$LOG")"
CKPT=/home/jehc223/Retrieval-hate/third_party/_ckpt
mkdir -p "$CKPT"

log () { echo "[$(date +%F_%T)] $*" >>"$LOG"; }

hfd () {
  local repo="$1"; shift
  log "START $repo $*"
  huggingface-cli download "$repo" "$@" --quiet >>"$LOG" 2>&1 \
    && log "DONE  $repo" || log "FAIL  $repo"
}

# UniTime LoRA adapter (Qwen2-VL-7B-Instruct base already local)
hfd zeqianli/UniTime --local-dir "$CKPT/unitime"

# AV2A: LanguageBind towers
hfd LanguageBind/LanguageBind_Video_FT
hfd LanguageBind/LanguageBind_Audio_FT
hfd LanguageBind/LanguageBind_Image
hfd lb203/LanguageBind_Image

# SeViLA localizer + answerer
if [ ! -s "$CKPT/sevila/sevila_pretrained.pth" ]; then
  mkdir -p "$CKPT/sevila"; log "START sevila_pretrained.pth"
  wget -q -O "$CKPT/sevila/sevila_pretrained.pth" \
    https://huggingface.co/Shoubin/SeViLA/resolve/main/sevila_pretrained.pth >>"$LOG" 2>&1 \
    && log "DONE  sevila_pretrained.pth" || log "FAIL  sevila_pretrained.pth"
fi

# LaGoVAD pretrained ckpts (Google Drive folder archive)
if [ ! -e "$CKPT/lagovad_ckpts.zip" ]; then
  log "START lagovad gdrive"
  gdown --fuzzy -O "$CKPT/lagovad_ckpts.zip" \
    "https://drive.google.com/file/d/161T7EkR64Px1cveP7_dG1xupT8IEICbM/view?usp=sharing" >>"$LOG" 2>&1 \
    && log "DONE  lagovad gdrive" || log "FAIL  lagovad gdrive"
fi

# CLIP ViT-L/14-336 raw OpenAI weights (LaGoVAD feature extractor uses the openai-CLIP format)
if [ ! -s "$CKPT/ViT-L-14-336px.pt" ]; then
  log "START ViT-L-14-336px.pt"
  wget -q -O "$CKPT/ViT-L-14-336px.pt" \
    https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt >>"$LOG" 2>&1 \
    && log "DONE  ViT-L-14-336px.pt" || log "FAIL  ViT-L-14-336px.pt"
fi

log "STREAM2 ALL DONE"
