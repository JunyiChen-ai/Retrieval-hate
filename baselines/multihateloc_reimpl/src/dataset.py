#!/usr/bin/env python3
"""Dataset + collate for MultiHateLoc reimplementation.

Loads per-video tri-modal features (fv[T,768], fa[T,128], ft[T,768]) produced by
extract_features.py, plus the video-level binary label from our HateMM splits.
Training truncates T to --max_t (default 256s) to bound the O(N^2) CM-Contrast
memory; eval runs full length.
"""
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset

FEAT_DIR = "/data/jehc223/RGCL/data/multihateloc_feats/HateMM"
GT_DIR = "/data/jehc223/RGCL/data/gt/HateMM"


def read_split(split):
    path = os.path.join(GT_DIR, f"{split}.jsonl")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append((d["id"], int(d["label"])))
    return items


class HateMMFeats(Dataset):
    def __init__(self, split, max_t=0, feat_dir=FEAT_DIR):
        self.max_t = max_t
        self.feat_dir = feat_dir
        raw = read_split(split)
        self.items = [(vid, y) for (vid, y) in raw
                      if os.path.exists(os.path.join(feat_dir, f"{vid}.npz"))]
        missing = len(raw) - len(self.items)
        if missing:
            print(f"[{split}] {missing} videos have no features (skipped); "
                  f"kept {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, y = self.items[i]
        z = np.load(os.path.join(self.feat_dir, f"{vid}.npz"))
        fv = z["fv"].astype(np.float32)
        fa = z["fa"].astype(np.float32)
        ft = z["ft"].astype(np.float32)
        T = fv.shape[0]
        if self.max_t and T > self.max_t:
            fv, fa, ft, T = fv[:self.max_t], fa[:self.max_t], ft[:self.max_t], self.max_t
        return {
            "vid": vid, "y": y, "T": T,
            "fv": torch.from_numpy(fv), "fa": torch.from_numpy(fa),
            "ft": torch.from_numpy(ft),
        }


def collate(batch):
    B = len(batch)
    Tmax = max(b["T"] for b in batch)
    fv = torch.zeros(B, Tmax, 768)
    fa = torch.zeros(B, Tmax, 128)
    ft = torch.zeros(B, Tmax, 768)
    mask = torch.zeros(B, Tmax)
    y = torch.zeros(B)
    vids, lens = [], []
    for i, b in enumerate(batch):
        T = b["T"]
        fv[i, :T] = b["fv"]; fa[i, :T] = b["fa"]; ft[i, :T] = b["ft"]
        mask[i, :T] = 1.0
        y[i] = b["y"]; vids.append(b["vid"]); lens.append(T)
    return {"fv": fv, "fa": fa, "ft": ft, "mask": mask, "y": y,
            "vids": vids, "lens": lens}
