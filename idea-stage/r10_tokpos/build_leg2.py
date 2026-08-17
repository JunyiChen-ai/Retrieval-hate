#!/usr/bin/env python
"""R10 Task B leg 2 -- does the token-position gain stack on the L24+L28 configuration?

Frozen design: idea-stage/R10_TOKPOS_FREEZE.md section 2.6, as amended by
idea-stage/R10_TOKPOS_DEVIATION_D1.md clause 4 (C0 must be rebuilt from THIS pass, not
taken from the A100-extracted R6RO-CAT on disk).

  R10L2-C0   img = [n(img28) | n(img24)]   text = [n(A0_28) | n(A0_24)]
  R10L2-C1   img = [n(img28) | n(img24)]   text = C0 text  ++  W's L28 and L24 blocks

W is passed on the command line and must be the leg-1 arm with the higher DEV macro-F1
(never test).  Runs only if leg 1 returned GO.
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
DATASET = "MHC_zh"
BASE = "Qwen2.5-VL-7B-Instruct-LoRA_HF"
SPLITS = ["train", "dev_seen", "test_seen"]


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", required=True, choices=["TXT", "CAT"],
                    help="leg-1 arm with the higher DEV macro-F1")
    a = ap.parse_args()

    meta = {"freeze": "idea-stage/R10_TOKPOS_FREEZE.md 2.6",
            "deviation": "idea-stage/R10_TOKPOS_DEVIATION_D1.md clause 4",
            "winner": a.winner, "files": {}}

    for split in SPLITS:
        tp = torch.load(os.path.join(EMB, DATASET, "%s_%s-tp.pt" % (split, BASE)),
                        map_location="cpu", weights_only=False)
        ro28 = torch.load(os.path.join(EMB, DATASET, "%s_%s-ro_L28.pt" % (split, BASE)),
                          map_location="cpu", weights_only=False)
        ro24 = torch.load(os.path.join(EMB, DATASET, "%s_%s-ro_L24.pt" % (split, BASE)),
                          map_location="cpu", weights_only=False)
        ids = tp["ids"][0]
        assert ids == ro28["ids"][0] == ro24["ids"][0]

        img = torch.cat([l2norm(ro28["img_feats"].float()),
                         l2norm(ro24["img_feats"].float())], dim=1)

        def blk(layer, span):
            return l2norm(tp["spans"][layer][span].float())

        c0_txt = torch.cat([blk("28", "A0"), blk("24", "A0")], dim=1)
        if a.winner == "TXT":
            extra = [blk("28", "TXT"), blk("24", "TXT")]
        else:  # CAT already contains A0; only TXT is new relative to C0
            extra = [blk("28", "TXT"), blk("24", "TXT")]
        c1_txt = torch.cat([c0_txt] + extra, dim=1)

        for arm, tx in [("C0", c0_txt), ("C1", c1_txt)]:
            obj = {"ids": [ids], "img_feats": img.contiguous(),
                   "text_feats": tx.contiguous(), "labels": ro28["labels"]}
            op = os.path.join(EMB, DATASET, "%s_R10L2-%s.pt" % (split, arm))
            torch.save(obj, op)
            meta["files"]["%s/%s" % (split, arm)] = {
                "path": os.path.relpath(op, ROOT), "rows": int(tx.shape[0]),
                "img_dim": int(img.shape[1]), "text_dim": int(tx.shape[1]),
                "text_sha256": sha_tensor(tx)}
            print("wrote %-30s img=%s text=%s" % (os.path.basename(op),
                                                  tuple(img.shape), tuple(tx.shape)))

    json.dump(meta, open(os.path.join(HERE, "leg2_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
