#!/usr/bin/env python
"""CAT close-out -- build A0 / CAT / RAND arm caches from a -cc extraction.

Frozen design: idea-stage/CAT_CLOSEOUT_FREEZE.md 2.4 / 3.3.

Unlike idea-stage/r10_tokpos/build_arms.py (which carried img_feats over from the
banked A100 -ro_L28 cache), BOTH streams here come from the SAME -cc pass on this
machine, so every arm is end-to-end re-derived from raw video.

  <PREFIX>-A0     text = n(A0)                    3584   control
  <PREFIX>-CAT    text = [n(A0) | n(TXT)]         7168   the entry
  <PREFIX>-RAND   text = [n(A0) | n(A0 @ R)]      7168   matched-width control

R is the SAME fixed Gaussian used by idea-stage/r6_readout/build_arms.py and
idea-stage/r10_tokpos/build_arms.py; its sha256 is verified against
idea-stage/r6_readout/build_meta.json.  No label is read for any decision here.

Also emits the section 2.3 numerical comparison table and applies BELT B1
(adapter/span identity vs the banked deployed cache).
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
DIM = 3584
RNG_SEED = 20260817
BELT_COS_MEAN = 0.95
BELT_COS_MIN = 0.90


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


def cos_stats(A, B):
    nz = (A.norm(dim=1) > 0) & (B.norm(dim=1) > 0)
    c = torch.nn.functional.cosine_similarity(l2norm(A[nz]), l2norm(B[nz]), dim=1)
    return {"n": int(nz.sum()), "cos_mean": float(c.mean()), "cos_min": float(c.min()),
            "cos_median": float(c.median()),
            "frac_ge_0.999": float((c >= 0.999).float().mean()),
            "frac_ge_0.99": float((c >= 0.99).float().mean())}


def diff_stats(A, B):
    d = (A - B).abs()
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "frac_rows_bit_identical": float((d.amax(dim=1) == 0).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", required=True, help="deployed cache tag for this dataset")
    ap.add_argument("--prefix", required=True, help="arm cache prefix, e.g. CCA / CCB")
    ap.add_argument("--compare_tp", action="store_true",
                    help="also compare against the banked R10 -tp cache (MHC_zh only)")
    a = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    R_np = rng.normal(0.0, 1.0 / np.sqrt(DIM), size=(DIM, DIM)).astype(np.float32)
    R = torch.from_numpy(R_np)
    r_sha = hashlib.sha256(np.ascontiguousarray(R_np).tobytes()).hexdigest()
    ref = json.load(open(os.path.join(ROOT, "idea-stage", "r6_readout", "build_meta.json")))
    assert r_sha == ref["R_sha256_float32"], "R matrix differs: %s vs %s" % (
        r_sha, ref["R_sha256_float32"])
    print("R sha256 = %s (matches r6_readout)" % r_sha)

    meta = {"freeze": "idea-stage/CAT_CLOSEOUT_FREEZE.md 2.4/3.3", "dataset": a.dataset,
            "prefix": a.prefix, "R_sha256_float32": r_sha,
            "files": {}, "belt_B1": {}, "vs_banked_tp": {}, "geometry": {}}
    belt_fail = []

    for split in SPLITS:
        cc = torch.load(os.path.join(EMB, a.dataset, "%s_%s-cc.pt" % (split, a.base)),
                        map_location="cpu", weights_only=False)
        ids = cc["ids"][0]
        A0 = l2norm(cc["spans"]["28"]["A0"].float())
        TXT = l2norm(cc["spans"]["28"]["TXT"].float())
        img = l2norm(cc["img28"].float())
        RND = l2norm(A0 @ R)

        # ---- BELT B1: adapter / span identity against the banked deployed cache ----
        dep_path = os.path.join(EMB, a.dataset, "%s_%s.pt" % (split, a.base))
        dep = torch.load(dep_path, map_location="cpu", weights_only=False)
        dep_ids = dep["ids"][0]
        pos = {v: i for i, v in enumerate(dep_ids)}
        assert set(ids) == set(dep_ids), "id set differs vs banked deployed cache"
        sel = torch.tensor([pos[i] for i in ids], dtype=torch.long)
        b_img = dep["img_feats"].float()[sel]
        b_txt = dep["text_feats"].float()[sel]
        ci = cos_stats(img, b_img)
        ct = cos_stats(A0, b_txt)
        meta["belt_B1"][split] = {"note": "cross-hardware drift vs the banked deployed "
                                          "(A100) cache; gates encoder identity only",
                                  "img": ci, "text_A0": ct}
        print("%-10s BELT B1 vs banked deployed: img cos mean=%.5f min=%.5f | "
              "A0 cos mean=%.5f min=%.5f"
              % (split, ci["cos_mean"], ci["cos_min"], ct["cos_mean"], ct["cos_min"]))
        for name, c in (("img", ci), ("text_A0", ct)):
            if c["cos_mean"] < BELT_COS_MEAN or c["cos_min"] < BELT_COS_MIN:
                belt_fail.append((split, name, c["cos_mean"], c["cos_min"]))

        # ---- section 2.3: same-hardware determinism vs the banked R10 -tp cache ----
        if a.compare_tp:
            tp = torch.load(os.path.join(EMB, a.dataset, "%s_%s-tp.pt" % (split, a.base)),
                            map_location="cpu", weights_only=False)
            assert tp["ids"][0] == ids, "id order differs vs -tp"
            ent = {}
            for s in ("A0", "TXT", "S1", "S2", "S3", "S4", "ALL"):
                old = tp["spans"]["28"][s].float()
                new = cc["spans"]["28"][s].float()
                ent[s] = dict(diff_stats(new, old), **cos_stats(new, old))
            ro = torch.load(os.path.join(EMB, a.dataset, "%s_%s-ro_L28.pt" % (split, a.base)),
                            map_location="cpu", weights_only=False)
            assert ro["ids"][0] == ids, "id order differs vs -ro_L28"
            ent["img_vs_ro_L28_A100"] = dict(diff_stats(img, l2norm(ro["img_feats"].float())),
                                             **cos_stats(img, ro["img_feats"].float()))
            meta["vs_banked_tp"][split] = ent
            print("%-10s vs banked -tp (same GPU): A0 maxdiff=%.3e bitident=%.3f | "
                  "TXT maxdiff=%.3e bitident=%.3f"
                  % (split, ent["A0"]["max_abs_diff"], ent["A0"]["frac_rows_bit_identical"],
                     ent["TXT"]["max_abs_diff"], ent["TXT"]["frac_rows_bit_identical"]))

        arms = {"A0": A0, "CAT": torch.cat([A0, TXT], dim=1),
                "RAND": torch.cat([A0, RND], dim=1)}
        for arm, tx in arms.items():
            obj = {"ids": [ids], "img_feats": img.contiguous(),
                   "text_feats": tx.contiguous(), "labels": cc["labels"]}
            op = os.path.join(EMB, a.dataset, "%s_%s-%s.pt" % (split, a.prefix, arm))
            torch.save(obj, op)
            meta["files"]["%s/%s" % (split, arm)] = {
                "path": os.path.relpath(op, ROOT), "rows": int(tx.shape[0]),
                "img_dim": int(img.shape[1]), "text_dim": int(tx.shape[1]),
                "img_sha256": sha_tensor(img), "text_sha256": sha_tensor(tx)}
            print("   wrote %-34s img=%s text=%s"
                  % (os.path.basename(op), tuple(img.shape), tuple(tx.shape)))

        meta["geometry"][split] = {
            "cos_A0_TXT": float(torch.nn.functional.cosine_similarity(A0, TXT, dim=1).mean()),
            "cos_A0_img": float(torch.nn.functional.cosine_similarity(A0, img, dim=1).mean()),
            "cos_TXT_img": float(torch.nn.functional.cosine_similarity(TXT, img, dim=1).mean())}

    meta["belt_B1_pass"] = not belt_fail
    mp = os.path.join(HERE, "build_meta_%s_%s.json" % (a.prefix, a.dataset))
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", mp)
    print(json.dumps(meta["geometry"], indent=1))
    if belt_fail:
        raise SystemExit("HALT: BELT B1 failed: %s" % belt_fail)


if __name__ == "__main__":
    main()
