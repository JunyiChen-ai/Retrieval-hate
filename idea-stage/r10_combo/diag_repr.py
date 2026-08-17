#!/usr/bin/env python
"""R10-COMBO diagnostic 2 -- representation redundancy between the token axis
and the layer axis.  TRAIN SPLIT ONLY, NO LABELS, no verdict power
(idea-stage/R10_COMBO_FREEZE.md 4.2).

Blocks (all row-L2-normed, from <split>_<BASE>-tp.pt):
    a28 = A0  at layer 28   (the deployed assistant-header readout)
    a24 = A0  at layer 24   (what the layer axis adds)
    t28 = TXT at layer 28   (what the token axis adds)
    t24 = TXT at layer 24

Read-outs:
 1. linear CKA between every pair of blocks.
 2. 5-fold cross-validated R^2 of a kernel-ridge map between blocks (n < d, so
    the dual form is exact and in-sample partialling would be degenerate -- the
    residuals below are therefore OUT-OF-FOLD).
 3. partialled CKA: CKA( resid(t28 | a28), resid(a24 | a28) ) on out-of-fold
    residuals.  This is the quantity the "same pool of signal" claim is about:
    how aligned is the part of the token axis that a28 cannot explain with the
    part of the layer axis that a28 cannot explain.
"""
import argparse
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))
N_FOLDS = 5
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RNG_SEED = 20260817


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    num = float((K * L).sum())
    den = float(np.sqrt((K * K).sum()) * np.sqrt((L * L).sum()))
    return num / max(den, 1e-30)


def kridge_oof(X, Y, folds, lam_scale):
    """Out-of-fold predictions of Y from X via kernel (dual) ridge, one lambda."""
    out = {}
    K = X @ X.T
    for lam in LAMBDAS:
        P = np.zeros_like(Y)
        for f in range(N_FOLDS):
            te = folds == f
            tr = ~te
            Ktr = K[np.ix_(tr, tr)]
            ymu = Y[tr].mean(0, keepdims=True)
            A = np.linalg.solve(Ktr + lam * lam_scale * np.eye(Ktr.shape[0]),
                                Y[tr] - ymu)
            P[te] = K[np.ix_(te, tr)] @ A + ymu
        ss_res = float(((Y - P) ** 2).sum())
        # baseline: train-fold mean, evaluated out of fold
        base = np.zeros_like(Y)
        for f in range(N_FOLDS):
            te = folds == f
            base[te] = Y[~te].mean(0, keepdims=True)
        ss_tot = float(((Y - base) ** 2).sum())
        out[lam] = (1.0 - ss_res / max(ss_tot, 1e-30), P)
    best = max(out, key=lambda l: out[l][0])
    return best, out[best][0], out[best][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tp = torch.load(os.path.join(EMB, a.dataset, "train_%s-tp.pt" % a.base),
                    map_location="cpu", weights_only=False)
    blk = {"a28": l2n(tp["spans"]["28"]["A0"].float().numpy().astype(np.float64)),
           "a24": l2n(tp["spans"]["24"]["A0"].float().numpy().astype(np.float64)),
           "t28": l2n(tp["spans"]["28"]["TXT"].float().numpy().astype(np.float64)),
           "t24": l2n(tp["spans"]["24"]["TXT"].float().numpy().astype(np.float64))}
    n = blk["a28"].shape[0]
    rng = np.random.default_rng(RNG_SEED)
    folds = rng.permutation(np.arange(n) % N_FOLDS)

    res = {"what": "R10-COMBO representation redundancy (train only, no labels)",
           "dataset": a.dataset, "n_train": int(n), "n_folds": N_FOLDS,
           "cka": {}, "ridge_oof_r2": {}, "cosine_mean": {}, "partialled": {}}

    names = ["a28", "a24", "t28", "t24"]
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            res["cka"]["%s|%s" % (A, B)] = cka(blk[A], blk[B])
            res["cosine_mean"]["%s|%s" % (A, B)] = float(
                (blk[A] * blk[B]).sum(1).mean())

    resid = {}
    for (X, Y) in [("a28", "t28"), ("a28", "a24"), ("a24", "t28"),
                   ("t28", "a24"), ("a28", "t24")]:
        lam_scale = float(np.trace(blk[X] @ blk[X].T) / n)
        lam, r2, P = kridge_oof(blk[X], blk[Y], folds, lam_scale)
        res["ridge_oof_r2"]["%s->%s" % (X, Y)] = {"r2": float(r2), "lambda": lam}
        resid[(X, Y)] = blk[Y] - P
        print("ridge %s -> %s : oof R2 = %+.4f (lambda=%g)" % (X, Y, r2, lam))

    rt = resid[("a28", "t28")]
    rl = resid[("a28", "a24")]
    rnd = rng.normal(size=rt.shape)
    res["partialled"] = {
        "cka_resid_t28_vs_resid_a24_given_a28": cka(rt, rl),
        "cka_resid_t28_vs_random_floor": cka(rt, rnd),
        "resid_energy_frac_t28_given_a28": float(
            (rt ** 2).sum() / ((blk["t28"] - blk["t28"].mean(0)) ** 2).sum()),
        "resid_energy_frac_a24_given_a28": float(
            (rl ** 2).sum() / ((blk["a24"] - blk["a24"].mean(0)) ** 2).sum()),
    }

    print()
    for k, v in res["cka"].items():
        print("CKA %-10s = %.4f   (mean cos %.4f)" % (k, v, res["cosine_mean"][k]))
    print("partialled CKA(resid t28|a28 , resid a24|a28) = %.4f  (random floor %.4f)"
          % (res["partialled"]["cka_resid_t28_vs_resid_a24_given_a28"],
             res["partialled"]["cka_resid_t28_vs_random_floor"]))

    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
