#!/usr/bin/env python
"""R10 Task B -- build the arm caches for the token-position readout pilot.

Frozen design: idea-stage/R10_TOKPOS_FREEZE.md section 2.3, as amended by
idea-stage/R10_TOKPOS_DEVIATION_D1.md.

Every arm keeps img_feats = the banked -ro_L28 img stream (constant across all arms,
so it cancels in every contrast); only text_feats differ.  All text vectors come from
ONE extraction pass on this machine (dev_seen/train/test_seen_<BASE>-tp.pt).

  R10TP-A0    text = n(A0_28)                                 3584   control
  R10TP-TXT   text = n(TXT_28)                                3584
  R10TP-CAT   text = [n(A0_28) | n(TXT_28)]                   7168
  R10TP-RAND  text = [n(A0_28) | n(A0_28 @ R)]                7168   width control
  R10TP-SEG   text = [n(A0_28) | n(S1_28) | ... | n(S4_28)]  17920   exploratory

R is the SAME fixed Gaussian used by idea-stage/r6_readout/build_arms.py; its sha256 is
verified against idea-stage/r6_readout/build_meta.json.  No labels are read for any
decision here.
"""
import hashlib
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--dataset", default="MHC_zh")
_ap.add_argument("--base", default="Qwen2.5-VL-7B-Instruct-LoRA_HF")
_A = _ap.parse_args()
DATASET = _A.dataset
BASE = _A.base
SPLITS = ["train", "dev_seen", "test_seen"]
DIM = 3584
RNG_SEED = 20260817
PREFIX = "R10TP"


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


def main():
    rng = np.random.default_rng(RNG_SEED)
    R_np = rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM)).astype(np.float32)
    R = torch.from_numpy(R_np)
    r_sha = hashlib.sha256(np.ascontiguousarray(R_np).tobytes()).hexdigest()
    ref = json.load(open(os.path.join(ROOT, "idea-stage", "r6_readout", "build_meta.json")))
    assert r_sha == ref["R_sha256_float32"], (
        "R matrix differs from the r6_readout one: %s vs %s" % (r_sha, ref["R_sha256_float32"]))
    print("R sha256 = %s (matches r6_readout)" % r_sha)

    meta = {"freeze": "idea-stage/R10_TOKPOS_FREEZE.md 2.3",
            "deviation": "idea-stage/R10_TOKPOS_DEVIATION_D1.md",
            "R_sha256_float32": r_sha, "files": {}, "parity": {}}

    for split in SPLITS:
        tp = torch.load(os.path.join(EMB, DATASET, "%s_%s-tp.pt" % (split, BASE)),
                        map_location="cpu", weights_only=False)
        ro28 = torch.load(os.path.join(EMB, DATASET, "%s_%s-ro_L28.pt" % (split, BASE)),
                          map_location="cpu", weights_only=False)

        ids_tp, ids_ro = tp["ids"][0], ro28["ids"][0]
        assert ids_tp == ids_ro, "id order differs for %s" % split
        assert torch.equal(tp["labels"], ro28["labels"]), "labels differ for %s" % split

        sp = tp["spans"]["28"]
        A0 = l2norm(sp["A0"].float())
        TXT = l2norm(sp["TXT"].float())
        SEG = [l2norm(sp["S%d" % k].float()) for k in (1, 2, 3, 4)]
        RND = l2norm(A0 @ R)

        # ---- cross-hardware drift record (DESCRIPTIVE; deviation D2) ----
        # The gating belt is the EXACT internal one in R10_TOKPOS_DEVIATION_D2.md: A0 is
        # bit-identical (max abs diff 0.0, 12/12) to the frozen deployed
        # _pool_span(span="response") run on the same forward on this machine.  The
        # comparison below is against an A100-extracted cache and therefore measures
        # platform drift, not the operator; it is gated only by a loose floor that a wrong
        # span (which reads ~0.45) would fail and drift (~0.996) passes.
        ref28 = ro28["text_feats"].float()
        nz = (ref28.norm(dim=1) > 0) & (A0.norm(dim=1) > 0)
        cos = torch.nn.functional.cosine_similarity(A0[nz], l2norm(ref28[nz]), dim=1)
        meta["parity"][split] = {
            "note": "descriptive cross-hardware drift; gating belt is the bit-exact "
                    "internal one (R10_TOKPOS_DEVIATION_D2.md)",
            "n_compared": int(nz.sum()), "cos_min": float(cos.min()),
            "cos_mean": float(cos.mean()), "cos_median": float(cos.median()),
            "frac_ge_0.99": float((cos >= 0.99).float().mean()),
            "frac_ge_0.95": float((cos >= 0.95).float().mean())}
        print("%-10s drift vs banked ro_L28 (A100): n=%d cos mean=%.6f min=%.6f "
              "frac>=.99=%.4f frac>=.95=%.4f"
              % (split, int(nz.sum()), cos.mean(), cos.min(),
                 (cos >= 0.99).float().mean(), (cos >= 0.95).float().mean()))
        if float(cos.mean()) < 0.95 or float(cos.min()) < 0.90:
            raise SystemExit("HALT: bug floor tripped on %s (mean=%.4f min=%.4f) -- this is "
                             "a span error, not drift" % (split, cos.mean(), cos.min()))

        img = ro28["img_feats"]
        arms = {
            "A0": A0,
            "TXT": TXT,
            "CAT": torch.cat([A0, TXT], dim=1),
            "RAND": torch.cat([A0, RND], dim=1),
            "SEG": torch.cat([A0] + SEG, dim=1),
        }
        for arm, tx in arms.items():
            obj = {"ids": [ids_tp], "img_feats": img.contiguous(),
                   "text_feats": tx.contiguous(), "labels": ro28["labels"]}
            op = os.path.join(EMB, DATASET, "%s_%s-%s.pt" % (split, PREFIX, arm))
            torch.save(obj, op)
            meta["files"]["%s/%s" % (split, arm)] = {
                "path": os.path.relpath(op, ROOT), "rows": int(tx.shape[0]),
                "img_dim": int(img.shape[1]), "text_dim": int(tx.shape[1]),
                "text_sha256": sha_tensor(tx)}
            print("   wrote %-34s img=%s text=%s"
                  % (os.path.basename(op), tuple(img.shape), tuple(tx.shape)))

        # descriptive geometry, no decision attached
        meta.setdefault("geometry", {})[split] = {
            "cos_A0_TXT": float(torch.nn.functional.cosine_similarity(A0, TXT, dim=1).mean()),
            "cos_A0_ALL": float(torch.nn.functional.cosine_similarity(
                A0, l2norm(sp["ALL"].float()), dim=1).mean()),
            "cos_A0_img": float(torch.nn.functional.cosine_similarity(
                A0, l2norm(img.float()), dim=1).mean()),
            "cos_TXT_img": float(torch.nn.functional.cosine_similarity(
                TXT, l2norm(img.float()), dim=1).mean()),
        }

    mp = os.path.join(HERE, "build_meta_%s.json" % DATASET)
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", mp)
    print(json.dumps(meta["geometry"], indent=1))


if __name__ == "__main__":
    main()
