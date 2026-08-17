#!/usr/bin/env python
"""R16-DETBASE: build ActionFormer annotation JSONs for HateClipSeg.

Two ground-truth conventions, both pre-declared in `idea-stage/R16_DETBASE_FREEZE.md`:

  blocks  (PRIMARY)  maximal contiguous runs of offensive gold segments, merged.
                     This is the convention every prior number in this project uses
                     (`scripts/r14_loc/recon_decode.py:blocks_of`), so it is the one on
                     which the detector and the per-window score curve are comparable.
  rawseg  (SECONDARY) every offensive gold segment is its own instance, un-merged.
                     The paper says only that the five offensive *labels* are merged into
                     one class; it does not say whether adjacent offensive segments are
                     merged into one instance.  Reported as a protocol-sensitivity arm.

Split = the frozen 237/39/119 `data/gt/HateClipSeg/p11_split.json`.
Subsets are written as 'train' / 'val' / 'test'.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/home/jehc223/Retrieval-hate")
OUT_DIR = ROOT / "third_party/actionformer/data/hateclipseg"
FPS = 4.0


def blocks_of(segs):
    out, cur = [], None
    for s, e, mh in segs:
        if sum(mh[1:]) > 0:
            cur = [s, e] if cur is None else [cur[0], e]
        else:
            if cur:
                out.append(tuple(cur))
                cur = None
    if cur:
        out.append(tuple(cur))
    return out


def rawseg_of(segs):
    return [(s, e) for s, e, mh in segs if sum(mh[1:]) > 0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    subset_of = {}
    for name in ("train", "val", "test"):
        for v in split[name]:
            subset_of[v] = name
    assert len(subset_of) == 395

    for conv, fn in (("blocks", blocks_of), ("rawseg", rawseg_of)):
        db = {}
        n_inst = {"train": 0, "val": 0, "test": 0}
        for vid, g in gold.items():
            sub = subset_of[vid]
            ivs = fn(g["segments"])
            n_inst[sub] += len(ivs)
            db[vid] = {
                "subset": sub,
                "duration": float(g["duration"]),
                "fps": FPS,
                "annotations": [
                    {"segment": [float(s), float(e)], "label": "offensive", "label_id": 0}
                    for s, e in ivs
                ],
            }
        out = OUT_DIR / f"hateclipseg_{conv}.json"
        out.write_text(json.dumps({"version": "HateClipSeg-395-p11", "database": db}))
        print(f"{conv}: {out}  instances train/val/test = "
              f"{n_inst['train']}/{n_inst['val']}/{n_inst['test']}")


if __name__ == "__main__":
    main()
