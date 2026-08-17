#!/usr/bin/env bash
set -u
cd /home/jehc223/Retrieval-hate
source /home/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate HateVideo
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=idea-stage/qwen_smoke_tmp
freemb(){ U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); echo $((32607-U)); }

go(){ # tag mode frames cutoff need_mb wait_s
  local tag=$1 m=$2 f=$3 c=$4 need=$5 ws=$6
  local out=$T/res_${tag}.json
  if [ -f "$out" ]; then echo "[skip] $out"; return; fi
  echo "=== WAIT ${need}MB for $tag (max ${ws}s)  $(date +%H:%M:%S) ==="
  local dl=$(( $(date +%s) + ws ))
  while [ "$(freemb)" -lt "$need" ]; do
    if [ $(date +%s) -gt $dl ]; then echo "[TIMEOUT] $tag never got ${need}MB (free=$(freemb)MB)"; return; fi
    sleep 45
  done
  echo "=== RUN $tag mode=$m frames=$f cutoff=$c free=$(freemb)MB $(date +%H:%M:%S) ==="
  timeout 3000 python $T/smoke.py --mode $m --frames $f --batch-size 1 --cutoff-len $c \
      --steps 5 --out $out 2>&1 | grep -avE "Loading checkpoint|warnings.warn|^  " | tail -14
}

# Phase A: fits in the ~10 GB window we have right now (full 28-layer model, QLoRA)
go qlora_f2 qlora 2 2048  9500  900
go qlora_f4 qlora 4 3072 10200  900

# Phase B: the real operating points -- need the card to actually free up
go qlora_f8  qlora  8 4096 17000 10800
go qlora_f16 qlora 16 8192 24000  3600
go bf16_f8   bf16    8 4096 28000  3600
go bf16_f16  bf16   16 8192 31000  1800
echo "=== ORCH3 DONE $(date +%H:%M:%S) ==="
