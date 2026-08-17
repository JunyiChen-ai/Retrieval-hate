#!/usr/bin/env python
"""R12-IMG -- build the 8 arm caches for the image-stream readout pilot.

Frozen design: idea-stage/R12_FREEZE.md section 2.4.

Every vector comes from caches produced on THIS machine in ONE extraction pass each:
  <split>_<BASE>-tp.pt   text spans {A0,TXT,...} x layers {28,24}   (R10 extraction)
  <split>_<BASE>-ip.pt   img spans {PRE,VIS,INS,STD,RA,RB} at layer 28 (R12 extraction)
No banked A100 cache enters any arm, so every contrast is within one machine and one
pass (the standing R10 deviation-D1 rule).  No labels are read for any decision here.

The TEXT stream is CAT = [n(A0_28) | n(TXT_28)] in EVERY arm (7168-d), byte-identical
across arms, so it cancels in every contrast.  Only img_feats differ:

  I0      n(PRE)                    3584   reference -- the deployed readout
  ISPLIT  [n(VIS) | n(INS)]         7168   candidate B2
  I2M     [n(PRE) | n(STD)]         7168   candidate B1
  IRSPLIT [n(RA)  | n(RB)]          7168   control -- random positional split
  IRW     [n(PRE) | n(PRE @ R)]     7168   control -- matched width
  IVIS    n(VIS)                    3584   diagnostic, non-selectable
  IINS    n(INS)                    3584   diagnostic, non-selectable
  ISTD    n(STD)                    3584   diagnostic, non-selectable

R is the SAME fixed Gaussian used by idea-stage/r6_readout/build_arms.py and
r10_tokpos/build_arms.py; its sha256 is asserted against r6_readout/build_meta.json.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "dev_seen", "test_seen"]
PREFIX = "R12IM"
DIM = 3584
RNG_SEED = 20260817  # the r6_readout R matrix seed -- unchanged


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC_zh")
    ap.add_argument("--base", default="Qwen2.5-VL-7B-Instruct-LoRA_HF")
    a = ap.parse_args()
    DS, BASE = a.dataset, a.base

    rng = np.random.default_rng(RNG_SEED)
    R_np = rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM)).astype(np.float32)
    R = torch.from_numpy(R_np)
    r_sha = hashlib.sha256(np.ascontiguousarray(R_np).tobytes()).hexdigest()
    ref = json.load(open(os.path.join(ROOT, "idea-stage", "r6_readout", "build_meta.json")))
    assert r_sha == ref["R_sha256_float32"], (
        "R matrix differs from the r6_readout one: %s vs %s" % (r_sha, ref["R_sha256_float32"]))
    print("R sha256 = %s (matches r6_readout)" % r_sha)

    meta = {"freeze": "idea-stage/R12_FREEZE.md 2.4", "dataset": DS, "base": BASE,
            "R_sha256_float32": r_sha, "files": {}}

    for split in SPLITS:
        tp = torch.load(os.path.join(EMB, DS, "%s_%s-tp.pt" % (split, BASE)),
                        map_location="cpu", weights_only=False)
        ip = torch.load(os.path.join(EMB, DS, "%s_%s-ip.pt" % (split, BASE)),
                        map_location="cpu", weights_only=False)

        ids_tp, ids_ip = tp["ids"][0], ip["ids"][0]
        assert ids_tp == ids_ip, "id order differs between -tp and -ip for %s" % split
        assert torch.equal(tp["labels"], ip["labels"]), "labels differ for %s" % split

        t = tp["spans"]["28"]
        text_cat = torch.cat([l2norm(t["A0"].float()), l2norm(t["TXT"].float())], dim=1)

        s = ip["spans"]["28"]
        PRE = l2norm(s["PRE"].float())
        VIS = l2norm(s["VIS"].float())
        INS = l2norm(s["INS"].float())
        STD = l2norm(s["STD"].float())
        RA = l2norm(s["RA"].float())
        RB = l2norm(s["RB"].float())
        PRE_R = l2norm(PRE @ R)

        arms = {
            "I0": PRE,
            "ISPLIT": torch.cat([VIS, INS], dim=1),
            "I2M": torch.cat([PRE, STD], dim=1),
            "IRSPLIT": torch.cat([RA, RB], dim=1),
            "IRW": torch.cat([PRE, PRE_R], dim=1),
            "IVIS": VIS,
            "IINS": INS,
            "ISTD": STD,
        }

        for name, img in arms.items():
            obj = {"ids": [ids_tp], "labels": ip["labels"],
                   "img_feats": img.contiguous().float(),
                   "text_feats": text_cat.contiguous().float()}
            op = os.path.join(EMB, DS, "%s_%s-%s.pt" % (split, PREFIX, name))
            torch.save(obj, op)
            meta["files"]["%s/%s" % (split, name)] = {
                "path": op, "img_dim": int(img.shape[1]),
                "text_dim": int(text_cat.shape[1]), "n": int(img.shape[0]),
                "img_sha256": sha_tensor(img), "text_sha256": sha_tensor(text_cat)}
            print("  %-8s %s img %s text %s" % (name, split, tuple(img.shape),
                                                tuple(text_cat.shape)))

    with open(os.path.join(HERE, "build_meta_%s.json" % DS), "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote build_meta_%s.json" % DS)


if __name__ == "__main__":
    main()
