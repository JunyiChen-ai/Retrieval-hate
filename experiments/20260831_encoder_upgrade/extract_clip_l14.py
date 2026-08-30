#!/usr/bin/env python3
"""E2: CLIP-L/14-336 image embeddings at 1 fps from cached frames.

Per video: read data/frames_1fps/<dir>/<vid>/*.jpg in second order, embed with
openai/clip-vit-large-patch14-336 (image tower, 768-d, batched), truncate/pad
to the vggish duration T (same length convention as every other channel).
Missing frame dir -> skipped (counted); missing individual seconds -> nearest
available frame's embedding.

Usage: python extract_clip_l14.py --corpus hateclipseg
Output: results/reproduction/features/clip_l14_336_1fps/<corpus>/<vid>.npy
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402

FRAME_DIRS = {"hatemm": "HateMM", "mhclip_en": "MHC", "mhclip_zh": "MHC_zh",
              "hateclipseg": "HateClipSeg"}
OUT_ROOT = os.path.join(REPO, "results", "reproduction", "features",
                        "clip_l14_336_1fps")
RUN_DIR = os.path.join(REPO, "runs", "20260831_encoder_upgrade")
VGGISH = os.path.join(REPO, "results", "reproduction", "features", "vggish_1s")
MODEL_ID = "openai/clip-vit-large-patch14-336"
BATCH = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()
    corpus = args.corpus
    os.makedirs(os.path.join(OUT_ROOT, corpus), exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    proc = CLIPImageProcessor.from_pretrained(MODEL_ID)
    model = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16).to("cuda").eval()

    vids = set()
    for split in ("train", "val", "test"):
        try:
            vids |= set(hdata.load_split(corpus, split))
        except Exception:
            pass

    n_ok = n_nodir = 0
    for i, vid in enumerate(sorted(vids)):
        out_path = os.path.join(OUT_ROOT, corpus, vid + ".npy")
        if os.path.exists(out_path):
            continue
        try:
            T = int(np.load(os.path.join(VGGISH, corpus, vid + ".npy"),
                            mmap_mode="r").shape[0])
        except FileNotFoundError:
            continue
        fdir = os.path.join(REPO, "data", "frames_1fps", FRAME_DIRS[corpus], vid)
        frames = sorted(glob.glob(os.path.join(fdir, "*.jpg")))
        if not frames:
            n_nodir += 1
            continue
        sec_of = {int(os.path.basename(p)[:6]): p for p in frames}
        secs = sorted(sec_of)
        embs = {}
        for b in range(0, len(secs), BATCH):
            chunk = secs[b:b + BATCH]
            imgs = [Image.open(sec_of[s]).convert("RGB") for s in chunk]
            with torch.no_grad():
                inp = proc(images=imgs, return_tensors="pt").to("cuda")
                out = model(pixel_values=inp.pixel_values.half()).image_embeds
            for s, e in zip(chunk, out.float().cpu().numpy()):
                embs[s] = e
        feats = np.zeros((T, 768), dtype=np.float32)
        avail = np.asarray(secs)
        for t in range(T):
            s = int(avail[np.abs(avail - t).argmin()])
            feats[t] = embs[s]
        np.save(out_path, feats)
        n_ok += 1
        if (i + 1) % 100 == 0:
            print("PROGRESS %s %d/%d" % (corpus, i + 1, len(vids)), flush=True)
    report = {"corpus": corpus, "videos": len(vids), "written": n_ok,
              "no_frame_dir": n_nodir, "model": MODEL_ID}
    with open(os.path.join(RUN_DIR, "e2_%s.json" % corpus), "w") as fh:
        json.dump(report, fh, indent=1)
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
