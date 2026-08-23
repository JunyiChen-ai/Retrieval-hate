#!/usr/bin/env python
"""Scan CLIP feature caches for degenerate (byte-identical / all-zero / non-finite) img_feats.

Read-only. Reports groups of ids sharing a byte-identical image feature vector.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import torch

DEFAULT_ROOT = "/home/jehc223/Retrieval-hate/data/CLIP_Embedding"
MODEL = "openai_clip-vit-large-patch14-336_HF"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(path):
    d = torch.load(path, map_location="cpu")
    ids = list(d["ids"])
    if len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    img = d["img_feats"]
    if isinstance(img, list):
        img = torch.stack([torch.as_tensor(x) for x in img])
    img = torch.as_tensor(img)
    if img.dim() == 3 and img.shape[0] == 1:
        img = img[0]
    txt = d.get("text_feats")
    if txt is not None:
        txt = torch.as_tensor(txt)
        if txt.dim() == 3 and txt.shape[0] == 1:
            txt = txt[0]
    labels = d.get("labels")
    if labels is not None:
        labels = torch.as_tensor(labels).flatten().tolist()

    groups = defaultdict(list)
    for i, vid in enumerate(ids):
        key = hashlib.sha256(img[i].contiguous().numpy().tobytes()).hexdigest()
        groups[key].append(i)

    dup = {k: v for k, v in groups.items() if len(v) > 1}
    zero = [ids[i] for i in range(len(ids)) if float(img[i].abs().sum()) == 0.0]
    nonfinite = [ids[i] for i in range(len(ids)) if not bool(torch.isfinite(img[i]).all())]

    out = {
        "path": path,
        "sha256": sha256_file(path),
        "n": len(ids),
        "dim": list(img.shape[1:]),
        "n_dup_groups": len(dup),
        "n_dup_items": sum(len(v) for v in dup.values()),
        "zero_vectors": zero,
        "nonfinite_vectors": nonfinite,
        "groups": [],
    }
    for k, idxs in sorted(dup.items(), key=lambda kv: -len(kv[1])):
        g = {
            "feat_sha256": k[:16],
            "size": len(idxs),
            "ids": [ids[i] for i in idxs],
            "labels": [labels[i] for i in idxs] if labels is not None else None,
            "img_norm": round(float(img[idxs[0]].norm()), 6),
            "txt_identical": bool(
                txt is not None
                and all(torch.equal(txt[idxs[0]], txt[j]) for j in idxs[1:])
            ),
        }
        out["groups"].append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "val", "dev_seen", "test_seen", "test"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = a.datasets or sorted(
        x for x in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, x))
    )
    res = []
    for d in ds:
        for sp in a.splits:
            p = os.path.join(a.root, d, f"{sp}_{a.model}.pt")
            if not os.path.exists(p):
                continue
            r = scan(p)
            r["dataset"] = d
            r["split"] = sp
            res.append(r)
            print(
                f"{d:15s} {sp:10s} n={r['n']:5d} dup_groups={r['n_dup_groups']:3d} "
                f"dup_items={r['n_dup_items']:3d} zeros={len(r['zero_vectors'])} "
                f"nonfinite={len(r['nonfinite_vectors'])}",
                flush=True,
            )
            for g in r["groups"]:
                print(f"    [{g['size']}] norm={g['img_norm']} txt_same={g['txt_identical']} {g['ids']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
