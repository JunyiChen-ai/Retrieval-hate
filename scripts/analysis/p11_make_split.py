#!/usr/bin/env python
"""P11 — freeze the HateClipSeg train/val/test split (pre-registration).

60/10/30 by VIDEO, seed 0, stratified by whether the video has >=1 toxic gold
second. Written once to data/gt/HateClipSeg/p11_split.json and committed BEFORE
any P11 head is trained, so the test set is fixed and touched exactly once.
"""
import json
import os

import numpy as np

ROOT = "/data/jehc223/RGCL"
GOLD = os.path.join(ROOT, "data/gt/HateClipSeg/gold_segments.json")
OUT = os.path.join(ROOT, "data/gt/HateClipSeg/p11_split.json")
SEED = 0
FRAC = (0.60, 0.10, 0.30)  # train / val / test


def has_toxic(g):
    return any(sum(l[1:]) > 0 for _, _, l in g["segments"])


def main():
    gold = json.load(open(GOLD))
    vids = sorted(gold.keys())
    tox = np.array([has_toxic(gold[v]) for v in vids])
    rng = np.random.RandomState(SEED)

    split = {"train": [], "val": [], "test": []}
    for stratum in (True, False):
        ids = [v for v, t in zip(vids, tox) if t == stratum]
        ids = list(np.array(ids)[rng.permutation(len(ids))])
        n = len(ids)
        n_tr = int(round(FRAC[0] * n))
        n_va = int(round(FRAC[1] * n))
        split["train"] += ids[:n_tr]
        split["val"] += ids[n_tr:n_tr + n_va]
        split["test"] += ids[n_tr + n_va:]

    for k in split:
        split[k] = sorted(split[k])
    # integrity
    allv = split["train"] + split["val"] + split["test"]
    assert len(allv) == len(vids) == len(set(allv)), "split not a partition"

    meta = {
        "seed": SEED, "frac": list(FRAC), "stratify": "has_toxic_second",
        "n_total": len(vids),
        "n_train": len(split["train"]), "n_val": len(split["val"]),
        "n_test": len(split["test"]),
        "tox_rate": {
            k: round(float(np.mean([has_toxic(gold[v]) for v in split[k]])), 4)
            for k in ("train", "val", "test")},
    }
    json.dump({"meta": meta, **split}, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
