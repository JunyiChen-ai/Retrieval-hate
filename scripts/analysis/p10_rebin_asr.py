#!/usr/bin/env python
"""P10 — re-bin existing word-level ASR to an arbitrary window count K (CPU).

The HateMM/HateClipSeg ASR JSONL (generate_segment_asr_HF.py, --timestamps word)
stores raw word chunks + duration, so any K-window transcript can be produced
WITHOUT re-running Whisper. Reuses the exact window_time_bounds +
assign_chunks_to_windows contract so the re-binned windows align to the P3 scorer
frame windows. Writes data/ASR/<DS>/<split>_asrK<K>_whisper-large-v3.jsonl.
"""
import argparse
import json
import os
import sys

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))
from utils.generate_segment_asr_HF import (  # noqa: E402
    window_time_bounds, assign_chunks_to_windows)

SPLIT_OUT = {"train": "train", "val": "dev_seen", "test": "test_seen",
             "dev_seen": "dev_seen", "test_seen": "test_seen"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", default="train,dev_seen,test_seen")
    ap.add_argument("--src_K", type=int, default=4)
    ap.add_argument("--dst_K", type=int, required=True)
    ap.add_argument("--dst_M", type=int, required=True, help="frame count for window bounds")
    ap.add_argument("--model_tag", default="whisper-large-v3")
    args = ap.parse_args()
    adir = os.path.join(ROOT, "data/ASR", args.dataset)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        out = SPLIT_OUT.get(split, split)
        src = os.path.join(adir, "{}_asrK{}_{}.jsonl".format(out, args.src_K, args.model_tag))
        dst = os.path.join(adir, "{}_asrK{}_{}.jsonl".format(out, args.dst_K, args.model_tag))
        if not os.path.exists(src):
            print("[skip] no source {}".format(src)); continue
        n = 0
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                dur = r.get("duration") or 1.0
                chunks = r.get("chunks") or []
                bounds = window_time_bounds(dur if dur > 0 else 1.0, args.dst_M, args.dst_K)
                wt = assign_chunks_to_windows(
                    chunks, bounds, word_level=(r.get("timestamps") == "word"))
                r["window_bounds"] = bounds
                r["window_text"] = wt
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        print("[{}] re-binned {} videos K{}->K{} -> {}".format(split, n, args.src_K, args.dst_K, dst))


if __name__ == "__main__":
    main()
