#!/usr/bin/env python
"""P8b — build the vision-summary text cache (CPU; reuses P8's encode_single).

Reads data/Summaries_vision/<DS>/<split>.jsonl (from generate_vision_summary.py),
single-chunk CLIP-text encodes each summary EXACTLY as P8's B arm (max_content=75,
one CLIPTextModel forward, pooler), and writes a drop-in floor-swap cache
  <split>_p8vsum_HF.pt   (copies floor img/ids/labels VERBATIM; only text_feats replaced)
so it plugs into P8's probe/training harness as the B_vision condition, apples-to-
apples with A (floor) and C (p8trunc). Never mutates existing caches.
"""
import argparse
import json
import os
import sys

import torch

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))
from p8_generate_summaries import encode_single, MODEL_TAG, SPLIT_TO_OUTNAME  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC_zh")
    ap.add_argument("--summaries_dir", default="data/Summaries_vision")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from transformers import CLIPTokenizer, CLIPTextModel
    device = torch.device(a.device)
    tok = CLIPTokenizer.from_pretrained(a.clip_model)
    tmodel = CLIPTextModel.from_pretrained(a.clip_model).to(device).eval()

    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        ds_dir = os.path.join(ROOT, "data/CLIP_Embedding", ds)
        for split in ("train", "val", "test"):
            outname = SPLIT_TO_OUTNAME[split]
            floor = torch.load(os.path.join(ds_dir, "{}_{}.pt".format(outname, MODEL_TAG)),
                               map_location="cpu")
            floor_ids = [i for sub in floor["ids"] for i in sub]
            spath = os.path.join(ROOT, a.summaries_dir, ds, split + ".jsonl")
            if not os.path.exists(spath):
                print("[WARN] missing {}; skip".format(spath)); continue
            sums = {}
            for line in open(spath):
                line = line.strip()
                if line:
                    o = json.loads(line); sums[str(o["id"])] = o
            miss, ntrunc, B = 0, 0, []
            for vid in floor_ids:
                s = sums.get(vid)
                if s is None:
                    s = {"summary": ""}; miss += 1
                b, tb = encode_single(s.get("summary", ""), tok, tmodel, device, max_content=75)
                B.append(b); ntrunc += tb
            B = torch.stack(B, 0)
            d = dict(floor)
            d["text_feats"] = B.float().contiguous()
            assert torch.equal(d["img_feats"], floor["img_feats"])
            assert d["ids"] == floor["ids"] and torch.equal(d["labels"], floor["labels"])
            op = os.path.join(ds_dir, "{}_p8vsum_HF.pt".format(outname))
            torch.save(d, op)
            print("[{}/{}] N={} trunc(>75tok)={} missing={} -> {}".format(
                ds, split, len(floor_ids), ntrunc, miss, op))


if __name__ == "__main__":
    main()
