#!/usr/bin/env python
"""R10-COMBO diagnostic 2b -- POST-HOC addition to diag_repr.py.

Declared as an addition in idea-stage/R10_COMBO_RESULT.md: the frozen
diag_repr.py compares the partialled CKA against a Gaussian floor, which does
not preserve the residuals' own spectra.  This script adds a row-permutation
null -- shuffle the item order of one side, which destroys the item
correspondence while keeping both spectra exactly -- for the partialled CKA and
for every raw block pair.

Still DIAGNOSTIC ONLY: train split, no labels, no verdict power.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from diag_repr import cka, kridge_oof, l2n, N_FOLDS, RNG_SEED  # noqa: E402

import torch  # noqa: E402

EMB = "/home/jehc223/Retrieval-hate/data/CLIP_Embedding"
N_PERM = 200


def perm_null(X, Y, rng, n_perm=N_PERM):
    v = [cka(X, Y[rng.permutation(Y.shape[0])]) for _ in range(n_perm)]
    v = np.asarray(v)
    return float(v.mean()), float(v.std(ddof=1)), float(np.percentile(v, 97.5))


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

    res = {"what": "R10-COMBO permutation null for CKA (post-hoc)",
           "dataset": a.dataset, "n_perm": N_PERM, "raw": {}, "partialled": {}}

    names = ["a28", "a24", "t28", "t24"]
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            obs = cka(blk[A], blk[B])
            m, s, p975 = perm_null(blk[A], blk[B], rng)
            res["raw"]["%s|%s" % (A, B)] = {
                "cka": obs, "perm_null_mean": m, "perm_null_p97.5": p975,
                "z": (obs - m) / max(s, 1e-12)}
            print("%-10s CKA=%.4f perm-null=%.4f (p97.5=%.4f) z=%.1f"
                  % ("%s|%s" % (A, B), obs, m, p975, (obs - m) / max(s, 1e-12)))

    lam_scale = float(np.trace(blk["a28"] @ blk["a28"].T) / n)
    _, _, Pt = kridge_oof(blk["a28"], blk["t28"], folds, lam_scale)
    _, _, Pl = kridge_oof(blk["a28"], blk["a24"], folds, lam_scale)
    rt, rl = blk["t28"] - Pt, blk["a24"] - Pl
    obs = cka(rt, rl)
    m, s, p975 = perm_null(rt, rl, rng)
    res["partialled"] = {"cka": obs, "perm_null_mean": m, "perm_null_p97.5": p975,
                         "z": (obs - m) / max(s, 1e-12)}
    print("partialled CKA(resid t28|a28, resid a24|a28) = %.4f  perm-null=%.4f "
          "(p97.5=%.4f) z=%.1f" % (obs, m, p975, (obs - m) / max(s, 1e-12)))

    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
