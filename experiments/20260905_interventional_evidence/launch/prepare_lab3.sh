#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Retrieval-hate"
run=runs/20260905_interventional_evidence/prepare_lab3
mkdir -p "$run"
echo "$$" > "$run/run.pid"
echo "host=$(hostname) date=$(date -Is)"
ssh uoa-lab3 'mkdir -p ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct ~/data/HateClipSeg ~/Retrieval-hate/data/ASR ~/Retrieval-hate/data/video/HateClipSeg'
scp -rq "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/refs" "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots" uoa-lab3:.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/
scp -rq "$HOME/data/HateClipSeg/videos" uoa-lab3:data/HateClipSeg/
scp -rq data/ASR/HateClipSeg uoa-lab3:Retrieval-hate/data/ASR/
ssh uoa-lab3 'test -e ~/Retrieval-hate/data/video/HateClipSeg/All || ln -s ~/data/HateClipSeg/videos ~/Retrieval-hate/data/video/HateClipSeg/All'
"$HOME/miniconda3/envs/HateVideo/bin/python" -c 'import json,pathlib; pathlib.Path("runs/20260905_interventional_evidence/prepare_lab3/completion.json").write_text(json.dumps({"state":"TRANSFER_FINISHED"}))'
