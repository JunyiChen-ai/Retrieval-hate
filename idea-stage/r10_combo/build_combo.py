#!/usr/bin/env python
"""R10-COMBO -- build the 10 arm caches for the block-combination pilot.

Frozen design: idea-stage/R10_COMBO_FREEZE.md section 2.

All vectors come from caches that already exist on disk:
  <split>_<BASE>-tp.pt        spans {A0,TXT,S1..S4,ALL} x layers {28,24}   (text stream)
  <split>_<BASE>-ro_L28.pt    img_feats at layer 28                        (img stream)
  <split>_<BASE>-ro_L24.pt    img_feats at layer 24                        (img stream)
No GPU, no extraction, no labels are read for any decision here.

n(.) = row L2-norm.  P512(.) = PCA to 512 components, mean fit on the TRAIN split
only, applied unchanged to dev_seen/test_seen, followed by row L2-norm.

  A0   img i28        text a28                                   3584   control (deployed)
  LL   img [i28|i24]  text [a28|a24]                             7168   control (layer axis)
  CAT  img i28        text [a28|t28]                             7168   control (token axis)
  PC0  img i28        text P512(a28)                              512   control (PCA family)
  K1   img [i28|i24]  text P512([a28|a24|t28|t24])                512   candidate
  K2   img i28        text P512([a28|t28])                        512   candidate
  K3   img [i28|i24]  text [a28|t28]                             7168   candidate
  K4   img i28        text [a28|t24]                             7168   candidate
  K5   img i28        text n(a28+t28)                            3584   candidate
  K6   img [i28|i24]  text P512(all 7 spans x 2 layers)           512   candidate
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
PREFIX = "R10CB"
PCA_DIM = 512
SPANS_ALL = ["A0", "TXT", "S1", "S2", "S3", "S4", "ALL"]
LAYERS = ["28", "24"]


def l2norm(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sha_tensor(t):
    return hashlib.sha256(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float32)).tobytes()).hexdigest()


class TrainPCA(object):
    """PCA fit on the train split only.  Mean-centred, no scaling, no whitening."""

    def __init__(self, X_train, d):
        X = X_train.astype(np.float64, copy=False)
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        n = Xc.shape[0]
        # Gram trick: n << D here, so eigendecompose the n x n Gram matrix.
        G = Xc @ Xc.T
        w, U = np.linalg.eigh(G)
        order = np.argsort(w)[::-1]
        w, U = w[order], U[:, order]
        keep = min(d, int((w > 1e-9 * max(w[0], 1e-30)).sum()))
        self.d = keep
        s = np.sqrt(np.maximum(w[:keep], 0.0))
        self.W = (Xc.T @ U[:, :keep]) / np.maximum(s, 1e-12)   # (D, keep), orthonormal cols
        self.evr = float(w[:keep].sum() / max(w.sum(), 1e-30))

    def transform(self, X):
        Z = (X.astype(np.float64, copy=False) - self.mu) @ self.W
        return torch.from_numpy(Z.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    DS, BASE = a.dataset, a.base

    raw = {}
    for split in SPLITS:
        tp = torch.load(os.path.join(EMB, DS, "%s_%s-tp.pt" % (split, BASE)),
                        map_location="cpu", weights_only=False)
        ro28 = torch.load(os.path.join(EMB, DS, "%s_%s-ro_L28.pt" % (split, BASE)),
                          map_location="cpu", weights_only=False)
        ro24 = torch.load(os.path.join(EMB, DS, "%s_%s-ro_L24.pt" % (split, BASE)),
                          map_location="cpu", weights_only=False)
        ids = tp["ids"][0]
        assert ids == ro28["ids"][0] == ro24["ids"][0], "id order differs on %s" % split
        assert torch.equal(tp["labels"], ro28["labels"]), "labels differ on %s" % split
        blk = {}
        for L in LAYERS:
            for s in SPANS_ALL:
                blk["%s_%s" % (s, L)] = l2norm(tp["spans"][L][s].float())
        blk["i28"] = l2norm(ro28["img_feats"].float())
        blk["i24"] = l2norm(ro24["img_feats"].float())
        raw[split] = {"ids": ids, "labels": ro28["labels"], "blk": blk}
        print("%-10s n=%d blocks=%d" % (split, len(ids), len(blk)))

    def B(split, name):
        return raw[split]["blk"][name]

    def cat(split, names):
        return torch.cat([B(split, n) for n in names], dim=1)

    # ---- PCA bases, all fit on TRAIN ONLY ----
    pca_sets = {
        "PC0": ["A0_28"],
        "K1": ["A0_28", "A0_24", "TXT_28", "TXT_24"],
        "K2": ["A0_28", "TXT_28"],
        "K6": ["%s_%s" % (s, L) for L in LAYERS for s in SPANS_ALL],
    }
    pcas, pca_meta = {}, {}
    for key, names in pca_sets.items():
        Xtr = cat("train", names).numpy()
        p = TrainPCA(Xtr, PCA_DIM)
        pcas[key] = p
        pca_meta[key] = {"blocks": names, "in_dim": int(Xtr.shape[1]),
                         "out_dim": int(p.d), "explained_var_ratio": p.evr}
        print("PCA %-4s %5d -> %3d  evr=%.4f  blocks=%s"
              % (key, Xtr.shape[1], p.d, p.evr, ",".join(names)))

    meta = {"freeze": "idea-stage/R10_COMBO_FREEZE.md 2",
            "dataset": DS, "base": BASE, "pca_dim": PCA_DIM,
            "pca": pca_meta, "files": {}}

    for split in SPLITS:
        ids, labels = raw[split]["ids"], raw[split]["labels"]
        i1 = B(split, "i28")
        i2 = torch.cat([B(split, "i28"), B(split, "i24")], dim=1)

        def P(key):
            return l2norm(pcas[key].transform(cat(split, pca_sets[key]).numpy()))

        arms = {
            "A0":  (i1, B(split, "A0_28")),
            "LL":  (i2, cat(split, ["A0_28", "A0_24"])),
            "CAT": (i1, cat(split, ["A0_28", "TXT_28"])),
            "PC0": (i1, P("PC0")),
            "K1":  (i2, P("K1")),
            "K2":  (i1, P("K2")),
            "K3":  (i2, cat(split, ["A0_28", "TXT_28"])),
            "K4":  (i1, cat(split, ["A0_28", "TXT_24"])),
            "K5":  (i1, l2norm(B(split, "A0_28") + B(split, "TXT_28"))),
            "K6":  (i2, P("K6")),
        }
        for arm, (img, tx) in arms.items():
            obj = {"ids": [ids], "img_feats": img.contiguous(),
                   "text_feats": tx.contiguous(), "labels": labels}
            op = os.path.join(EMB, DS, "%s_%s-%s.pt" % (split, PREFIX, arm))
            torch.save(obj, op)
            meta["files"]["%s/%s" % (split, arm)] = {
                "path": os.path.relpath(op, ROOT), "rows": int(tx.shape[0]),
                "img_dim": int(img.shape[1]), "text_dim": int(tx.shape[1]),
                "img_sha256": sha_tensor(img), "text_sha256": sha_tensor(tx)}
            print("   %-8s %-14s img=%-12s text=%s"
                  % (split, arm, tuple(img.shape), tuple(tx.shape)))

    # ---- belts (no labels, no metrics) ----
    a0_new = torch.load(os.path.join(EMB, DS, "train_%s-A0.pt" % PREFIX),
                        map_location="cpu", weights_only=False)
    a0_old_p = os.path.join(EMB, DS, "train_R10TP-A0.pt")
    if os.path.exists(a0_old_p):
        a0_old = torch.load(a0_old_p, map_location="cpu", weights_only=False)
        dmax = float((a0_new["text_feats"] - a0_old["text_feats"]).abs().max())
        imax = float((a0_new["img_feats"].float() - a0_old["img_feats"].float()).abs().max())
        meta["belt_A0_vs_R10TP"] = {"text_max_abs_diff": dmax, "img_max_abs_diff": imax}
        print("BELT A0 vs R10TP-A0: text max|d|=%.3e img max|d|=%.3e" % (dmax, imax))
        assert dmax < 1e-5 and imax < 1e-5, "HALT: A0 arm is not the R10 A0 arm"

    catn = torch.load(os.path.join(EMB, DS, "train_%s-CAT.pt" % PREFIX),
                      map_location="cpu", weights_only=False)
    cat_old_p = os.path.join(EMB, DS, "train_R10TP-CAT.pt")
    if os.path.exists(cat_old_p):
        cat_old = torch.load(cat_old_p, map_location="cpu", weights_only=False)
        dmax = float((catn["text_feats"] - cat_old["text_feats"]).abs().max())
        meta["belt_CAT_vs_R10TP"] = {"text_max_abs_diff": dmax}
        print("BELT CAT vs R10TP-CAT: text max|d|=%.3e" % dmax)
        assert dmax < 1e-5, "HALT: CAT arm is not the R10 CAT arm"

    lln = torch.load(os.path.join(EMB, DS, "train_%s-LL.pt" % PREFIX),
                     map_location="cpu", weights_only=False)
    ll_old_p = os.path.join(EMB, DS, "train_R10L2-C0.pt")
    if os.path.exists(ll_old_p):
        ll_old = torch.load(ll_old_p, map_location="cpu", weights_only=False)
        dmax = float((lln["text_feats"] - ll_old["text_feats"]).abs().max())
        imax = float((lln["img_feats"] - ll_old["img_feats"]).abs().max())
        meta["belt_LL_vs_R10L2C0"] = {"text_max_abs_diff": dmax, "img_max_abs_diff": imax}
        print("BELT LL vs R10L2-C0: text max|d|=%.3e img max|d|=%.3e" % (dmax, imax))
        assert dmax < 1e-5 and imax < 1e-5, "HALT: LL arm is not the R10 leg-2 C0 arm"

    mp = os.path.join(HERE, "build_meta_%s.json" % DS)
    json.dump(meta, open(mp, "w"), indent=1)
    print("wrote", mp)


if __name__ == "__main__":
    main()
