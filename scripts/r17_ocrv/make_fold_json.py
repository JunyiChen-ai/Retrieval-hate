#!/usr/bin/env python
"""R17-OCRV: build the three cross-fitting annotation JSONs.

Frozen in `idea-stage/R17_OCRV_FREEZE.md` §2: the 237 train videos are partitioned into 3 folds
by *sorted video id modulo 3* -- deterministic, no seed.  For fold f the JSON marks

  subset 'train' : the 158 train videos NOT in fold f
  subset 'val'   : the 39 frozen val videos (epoch and threshold selection only, as in R16)
  subset 'oof'   : the 79 held-out train videos of fold f

and the 119 test videos are written with subset 'unused' so that no dataset object can ever
load them.  The builder asserts that no test id carries subset train/val/oof.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT_DIR = ROOT / "third_party/actionformer/data/hateclipseg"
FPS = 4.0
NFOLD = 3


def rawseg_of(segs):
    return [(s, e) for s, e, mh in segs if sum(mh[1:]) > 0]


def main() -> None:
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    train = sorted(split["train"])
    val = set(split["val"])
    test = set(split["test"])
    assert len(train) == 237 and len(val) == 39 and len(test) == 119

    folds = [[v for i, v in enumerate(train) if i % NFOLD == f] for f in range(NFOLD)]
    print("[folds] sizes = " + str([len(f) for f in folds]))

    for f in range(NFOLD):
        held = set(folds[f])
        db, n = {}, {"train": 0, "val": 0, "oof": 0, "unused": 0}
        for vid, g in gold.items():
            if vid in test:
                sub = "unused"
            elif vid in val:
                sub = "val"
            elif vid in held:
                sub = "oof"
            else:
                sub = "train"
            ivs = rawseg_of(g["segments"])
            n[sub] += len(ivs)
            db[vid] = {"subset": sub, "duration": float(g["duration"]), "fps": FPS,
                       "annotations": [{"segment": [float(s), float(e)],
                                        "label": "offensive", "label_id": 0} for s, e in ivs]}
        assert all(db[v]["subset"] == "unused" for v in test), "test id leaked into a used subset"
        assert sum(1 for v in db if db[v]["subset"] == "oof") == len(held)
        assert sum(1 for v in db if db[v]["subset"] == "train") == 237 - len(held)
        out = OUT_DIR / f"hateclipseg_rawseg_fold{f}.json"
        out.write_text(json.dumps({"version": f"HateClipSeg-395-p11-r17-fold{f}",
                                   "database": db}))
        print(f"fold {f}: {out}  videos train/val/oof/unused = "
              f"{237-len(held)}/{len(val)}/{len(held)}/{len(test)}  "
              f"instances {n['train']}/{n['val']}/{n['oof']}")


if __name__ == "__main__":
    main()
