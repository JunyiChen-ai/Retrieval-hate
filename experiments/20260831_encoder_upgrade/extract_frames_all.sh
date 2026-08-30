#!/bin/bash
# E1: 1-fps frame extraction for every split video lacking a cached frame dir.
# Parallel ffmpeg (N workers). Idempotent: skips dirs that already exist.
set -u
REPO=/home/jehc223/Retrieval-hate
N=${N:-10}
list() {
  python - <<'EOF'
import os, sys
sys.path.insert(0, '/home/jehc223/Retrieval-hate/scripts/reproduction_baselines')
from hate_common import data as hdata
M = {"hatemm":("HateMM","HateMM/video"),
     "mhclip_en":("MHC","Multihateclip/English/videos"),
     "mhclip_zh":("MHC_zh","Multihateclip/Chinese/videos"),
     "hateclipseg":("HateClipSeg","HateClipSeg/videos")}
for corpus,(d,vd) in M.items():
    vids = set()
    for split in ("train","val","test"):
        try: vids |= set(hdata.load_split(corpus, split))
        except Exception: pass
    for v in sorted(vids):
        if not os.path.isdir(f'/home/jehc223/Retrieval-hate/data/frames_1fps/{d}/{v}'):
            print(d, vd, v)
EOF
}
extract_one() {
  d=$1; vd=$2; v=$3
  out="$REPO/data/frames_1fps/$d/$v"
  src=""
  for ext in mp4 mkv webm avi mov m4v flv; do
    [ -f "/home/jehc223/data/$vd/$v.$ext" ] && src="/home/jehc223/data/$vd/$v.$ext" && break
  done
  [ -z "$src" ] && echo "MISSING_SRC $d/$v" && return
  mkdir -p "$out"
  ffmpeg -nostdin -loglevel error -i "$src" -vf fps=1 -start_number 0 "$out/%06d.jpg" \
    || { echo "FFMPEG_FAIL $d/$v"; rm -rf "$out"; }
}
export -f extract_one
export REPO
list | xargs -P "$N" -n 3 bash -c 'extract_one "$@"' _
echo "E1_DONE"
