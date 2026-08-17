#!/usr/bin/env python
"""R11-SEG: build the canonical K=30 window grid + gold labels for HateClipSeg.

Canonical grid = the `window_bounds` field already stored in
`data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl`.  Those bounds are
derived from exactly the same 120-frame linspace / 30-group partition that produced
`data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_*.pt`, so window k of the ASR
file and window k of the CLIP tensor are the same interval.  Verified for all 395
videos: contiguous, starts at 0.0, ends within 2 s of the ffprobe duration.

Outputs `idea-stage/r11_seg/out/grid_labels.npz`:
  video_ids        (395,)  str
  bounds           (395,30,2) float64   window [start,end) seconds
  y_win            (395,30) int8   window binary offensive label (duration-majority)
  frac_off         (395,30) float32 offensive fraction of the window
  y_ts             list of (T_i,) int8  per-timestamp gold at 0.25 s stride
  win_of_ts        list of (T_i,) int16 which window each timestamp falls in
  y_video          (395,) int8   video-level: any offensive segment
  y_multi          (395,30,5) int8  per-window multi-hot for the 5 offensive classes
  n_bnd            (395,30) int8   number of gold segment boundaries inside the window
  y_change         (395,30) int8   1 iff y_win[k] != y_win[k-1]  (k=0 -> 0)

Class order in gold_segments.json: 0 normal, 1 hateful, 2 insulting, 3 sexual,
4 violence, 5 harm.  "Offensive" = any of classes 1..5 (the paper's binary online task).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
K = 30
STRIDE = 0.25
OUT = ROOT / "idea-stage/r11_seg/out/grid_labels.npz"


def main() -> None:
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    asr = {}
    with open(ROOT / "data/ASR/HateClipSeg/test_seen_asrK30_whisper-large-v3.jsonl") as fh:
        for ln in fh:
            d = json.loads(ln)
            asr[d["id"]] = d

    clip = __import__("torch").load(
        ROOT / "data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
        map_location="cpu",
    )
    vids = list(clip["video_ids"])  # canonical order = CLIP tensor order
    assert len(vids) == 395

    bounds = np.zeros((len(vids), K, 2), dtype=np.float64)
    frac_off = np.zeros((len(vids), K), dtype=np.float32)
    y_multi = np.zeros((len(vids), K, 5), dtype=np.int8)
    n_bnd = np.zeros((len(vids), K), dtype=np.int8)
    y_video = np.zeros(len(vids), dtype=np.int8)
    y_ts_list, win_of_ts_list = [], []

    for i, vid in enumerate(vids):
        g = gold[vid]
        segs = g["segments"]  # [start, end, multi-hot(6)]
        wb = np.asarray(asr[vid]["window_bounds"], dtype=np.float64)
        bounds[i] = wb
        dur = float(wb[-1, 1])
        y_video[i] = int(any(sum(s[2][1:]) > 0 for s in segs))

        # --- per-window duration-weighted overlap with each gold segment
        for k in range(K):
            s0, s1 = wb[k]
            wlen = max(s1 - s0, 1e-9)
            off = 0.0
            for a, b, mh in segs:
                ov = min(s1, b) - max(s0, a)
                if ov <= 0:
                    continue
                if sum(mh[1:]) > 0:
                    off += ov
                for c in range(5):
                    if mh[c + 1]:
                        y_multi[i, k, c] = 1
            frac_off[i, k] = off / wlen
        # --- boundary count per window (internal gold boundaries only)
        for a, b, _mh in segs[1:]:
            t = a
            if t <= 0 or t >= dur:
                continue
            k = int(np.searchsorted(wb[:, 1], t, side="right"))
            if 0 <= k < K:
                n_bnd[i, k] = min(127, int(n_bnd[i, k]) + 1)

        # --- per-timestamp gold at 0.25 s
        ts = np.arange(0.0, dur, STRIDE) + STRIDE / 2.0
        yts = np.zeros(len(ts), dtype=np.int8)
        starts = np.array([s[0] for s in segs])
        for j, t in enumerate(ts):
            si = int(np.searchsorted(starts, t, side="right") - 1)
            si = max(0, min(si, len(segs) - 1))
            yts[j] = 1 if sum(segs[si][2][1:]) > 0 else 0
        wof = np.clip(np.searchsorted(wb[:, 1], ts, side="right"), 0, K - 1).astype(np.int16)
        y_ts_list.append(yts)
        win_of_ts_list.append(wof)

    y_win = (frac_off >= 0.5).astype(np.int8)
    y_change = np.zeros_like(y_win)
    y_change[:, 1:] = (y_win[:, 1:] != y_win[:, :-1]).astype(np.int8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        video_ids=np.array(vids),
        bounds=bounds,
        y_win=y_win,
        frac_off=frac_off,
        y_video=y_video,
        y_multi=y_multi,
        n_bnd=n_bnd,
        y_change=y_change,
        y_ts=np.array(y_ts_list, dtype=object),
        win_of_ts=np.array(win_of_ts_list, dtype=object),
        allow_pickle=True,
    )
    print(f"wrote {OUT}")

    # ---- TRAIN-ONLY descriptive statistics (design-time; no arm metric computed)
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    idx = {v: i for i, v in enumerate(vids)}
    for name in ("train",):
        ii = np.array([idx[v] for v in split[name]])
        print(f"[{name}] n_videos={len(ii)}")
        print(f"  window offensive base rate      = {y_win[ii].mean():.4f}")
        print(f"  window offensive-fraction mean  = {frac_off[ii].mean():.4f}")
        print(f"  label-change rate (win k vs k-1)= {y_change[ii][:, 1:].mean():.4f}")
        print(f"  windows containing >=1 boundary = {(n_bnd[ii] > 0).mean():.4f}")
        print(f"  mean gold boundaries per window = {n_bnd[ii].mean():.4f}")
        ts_all = np.concatenate([y_ts_list[j] for j in ii])
        print(f"  timestamp (0.25 s) base rate    = {ts_all.mean():.4f}  n={len(ts_all)}")
        print(f"  video-level any-offensive rate  = {y_video[ii].mean():.4f}")
        for c, nm in enumerate(["hateful", "insulting", "sexual", "violence", "harm"]):
            print(f"    per-window {nm:10s} rate = {y_multi[ii][:, :, c].mean():.4f}")


if __name__ == "__main__":
    main()
