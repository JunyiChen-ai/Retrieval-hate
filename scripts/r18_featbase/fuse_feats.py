#!/usr/bin/env python
"""R18-FEATBASE: build the concatenated feature dirs for the R18 arms.

  dense4fps_mat/<vid>.npy    (T, 3200) = VideoMAEv2-g | A | T
  dense4fps_mvat/<vid>.npy   (T, 4224) = VideoMAEv2-g | CLIP | A | T

Stream order and the "raw, unnormalised concatenation" convention are carried over from
`scripts/r17_ocrv/build_ocr_feats.py` (V | A | T | O), with the new visual stream prepended.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
EMB = ROOT / "data/CLIP_Embedding/HateClipSeg"
V = EMB / "dense4fps_clipL336"
A = EMB / "dense4fps_w2vemo"
T = EMB / "dense4fps_bertbase"
M = EMB / "dense4fps_vmaev2g"
MAT = EMB / "dense4fps_mat"
MVAT = EMB / "dense4fps_mvat"


def main() -> None:
    MAT.mkdir(parents=True, exist_ok=True)
    MVAT.mkdir(parents=True, exist_ok=True)
    vids = sorted(json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text()).keys())
    t0 = time.time()
    rms = {k: [] for k in "MVAT"}
    for n, vid in enumerate(vids):
        arrs = {k: np.load(d / f"{vid}.npy").astype(np.float32)
                for k, d in (("M", M), ("V", V), ("A", A), ("T", T))}
        n_t = {k: a.shape[0] for k, a in arrs.items()}
        assert len(set(n_t.values())) == 1, (vid, n_t)
        for k, a in arrs.items():
            rms[k].append(float((a ** 2).mean()))
        for out, keys in ((MAT, "MAT"), (MVAT, "MVAT")):
            p = out / f"{vid}.npy"
            if p.exists():
                continue
            X = np.concatenate([arrs[k] for k in keys], axis=1)
            tmp = out / f".{vid}.tmp.npy"
            np.save(tmp, X)
            os.replace(tmp, p)
        if (n + 1) % 50 == 0:
            print(f"[fuse] {n+1}/{len(vids)} t={time.time()-t0:.0f}s", flush=True)
    print({k: round(float(np.sqrt(np.mean(v))), 4) for k, v in rms.items()}, flush=True)
    print(f"[fuse] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
