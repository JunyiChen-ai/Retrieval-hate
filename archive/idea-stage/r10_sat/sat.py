#!/usr/bin/env python
"""Task A -- spectral saturation index S(K) = erank(pooled within-class cov)/K.

Implements arXiv 2606.24903v2 (Gupta, "The Geometry of Saturation"), Definitions
3.1-3.3 and the sweep protocol of section 4.1, on this project's deployed frozen
feature caches.

Frozen design: idea-stage/R10_TOKPOS_FREEZE.md section 1.

TEST IS NEVER OPENED.  Only train + dev_seen are loaded; the split-file map below
has no test entry and the loader raises if one is requested.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = "/home/jehc223/Retrieval-hate"
EMB = os.path.join(ROOT, "data", "CLIP_Embedding")
HERE = os.path.dirname(os.path.abspath(__file__))

# deployed tag per dataset (freeze section 0)
DATASETS = {
    "HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    "MHC": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    "ImpliHateVid": "Qwen2.5-VL-7B-Instruct_HF",
}
SPLITS = ["train", "dev_seen"]          # NO test.  Hard-coded, not a parameter.
VIEWS = ["concat", "img", "text"]
GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N_TRIALS = 50
PCA_DIM = 50
TAU = 0.02
BASE_SEED = 20260817


def load_trainval(ds, tag):
    Xi, Xt, Y = [], [], []
    for sp in SPLITS:
        assert "test" not in sp, "test split requested -- forbidden"
        p = os.path.join(EMB, ds, "%s_%s.pt" % (sp, tag))
        d = torch.load(p, map_location="cpu", weights_only=False)
        Xi.append(d["img_feats"].float().numpy())
        Xt.append(d["text_feats"].float().numpy())
        Y.append(d["labels"].numpy())
    return (np.concatenate(Xi, 0), np.concatenate(Xt, 0),
            np.concatenate(Y, 0).astype(int))


def erank(Sigma):
    """exp(Shannon entropy of the normalised eigenvalue spectrum)  (Def 3.1)."""
    lam = np.linalg.eigvalsh(Sigma)
    lam = np.clip(lam, 0.0, None)
    tr = lam.sum()
    if tr <= 0:
        return 1.0
    p = lam / tr
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    return float(np.exp(H))


def pooled_within_spectrum(X, y):
    """Non-zero eigenvalues of Sigma_W^(K) = (1/N) sum_c Sigma_c^(K)  (Def 3.2).

    With equal per-class support K, stacking the class-centred blocks into A gives
    Sigma_W = A^T A / (N (K-1)).  The non-zero eigenvalues of A^T A (d x d) and of
    A A^T (NK x NK) are identical, and zero eigenvalues contribute nothing to the
    Shannon entropy, so erank is unchanged.  We take whichever side is smaller --
    exact, not an approximation.
    """
    classes = np.unique(y)
    N = len(classes)
    blocks = []
    for c in classes:
        Z = X[y == c]
        blocks.append(Z - Z.mean(0, keepdims=True))
        K = Z.shape[0]
    A = np.concatenate(blocks, 0)
    scale = 1.0 / (N * (K - 1))
    if A.shape[0] <= A.shape[1]:
        M = (A @ A.T) * scale
    else:
        M = (A.T @ A) * scale
    lam = np.linalg.eigvalsh(M)
    return np.clip(lam, 0.0, None)


def erank_from_spectrum(lam):
    tr = lam.sum()
    if tr <= 0:
        return 1.0
    p = lam / tr
    p = p[p > 0]
    return float(np.exp(float(-(p * np.log(p)).sum())))


def one_trial(Xall, yall, K, rng, do_pca):
    idx = []
    for c in np.unique(yall):
        pool = np.where(yall == c)[0]
        idx.append(rng.choice(pool, size=K, replace=False))
    idx = np.concatenate(idx)
    X, y = Xall[idx], yall[idx]
    if do_pca:
        # StandardScaler + PCA-50, basis fit on the support set only (paper 4.1)
        X = StandardScaler().fit_transform(X)
        ncomp = min(PCA_DIM, X.shape[0] - 1, X.shape[1])
        X = PCA(n_components=ncomp, svd_solver="randomized",
                random_state=0).fit_transform(X)
    e = erank_from_spectrum(pooled_within_spectrum(X, y))
    return e, e / float(K)


def main():
    out = {"freeze": "idea-stage/R10_TOKPOS_FREEZE.md section 1",
           "paper": "arXiv 2606.24903v2", "tau": TAU, "n_trials": N_TRIALS,
           "splits_used": SPLITS, "pca_dim": PCA_DIM, "results": {}}

    for ds, tag in DATASETS.items():
        Ximg, Xtxt, y = load_trainval(ds, tag)
        n_per = {int(c): int((y == c).sum()) for c in np.unique(y)}
        Kmax = min(n_per.values())
        ks = [k for k in GRID if k <= Kmax]
        if Kmax not in ks:
            ks.append(Kmax)
        views = {"concat": np.concatenate([Ximg, Xtxt], 1), "img": Ximg, "text": Xtxt}
        print("\n=== %s  n=%d  per-class=%s  Kmax=%d  d=%d/%d  Ks=%s"
              % (ds, len(y), n_per, Kmax, Ximg.shape[1],
                 views["concat"].shape[1], ks), flush=True)

        for vname in VIEWS:
            X = views[vname]
            for arm, do_pca in [("pca50", True), ("native", False)]:
                key = "%s|%s|%s" % (ds, vname, arm)
                rows = []
                for K in ks:
                    er, si = [], []
                    for t in range(N_TRIALS):
                        rng = np.random.default_rng(BASE_SEED + 1000 * t + K)
                        e, s = one_trial(X, y, K, rng, do_pca)
                        er.append(e)
                        si.append(s)
                    rows.append({"K": int(K),
                                 "erank_mean": float(np.mean(er)),
                                 "erank_sd": float(np.std(er, ddof=1)),
                                 "S_mean": float(np.mean(si)),
                                 "S_sd": float(np.std(si, ddof=1))})
                    print("  %-28s K=%-5d erank=%8.3f+-%-7.3f  S=%.5f+-%.5f"
                          % (key, K, rows[-1]["erank_mean"], rows[-1]["erank_sd"],
                             rows[-1]["S_mean"], rows[-1]["S_sd"]), flush=True)
                last = rows[-1]
                out["results"][key] = {
                    "dataset": ds, "view": vname, "arm": arm, "tag": tag,
                    "n_trainval": int(len(y)), "per_class": n_per,
                    "d": int(X.shape[1]), "Kmax": int(Kmax), "sweep": rows,
                    "S_at_Kmax": last["S_mean"],
                    "erank_at_Kmax": last["erank_mean"],
                    "verdict_tau002": ("STOP" if last["S_mean"] < TAU else "CONTINUE"),
                }

    op = os.path.join(HERE, "sat.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", op)

    print("\n=== HEADLINE (PCA-50 arm, K=Kmax, tau=0.02) ===")
    for ds in DATASETS:
        for v in VIEWS:
            r = out["results"]["%s|%s|pca50" % (ds, v)]
            print("%-14s %-7s Kmax=%4d erank=%7.3f S=%.5f -> %s"
                  % (ds, v, r["Kmax"], r["erank_at_Kmax"], r["S_at_Kmax"],
                     r["verdict_tau002"]))
    print("\n=== native-dim arm (monitor band 0.3 -> 0.05) ===")
    for ds in DATASETS:
        for v in VIEWS:
            r = out["results"]["%s|%s|native" % (ds, v)]
            print("%-14s %-7s d=%5d Kmax=%4d erank=%8.3f S=%.5f"
                  % (ds, v, r["d"], r["Kmax"], r["erank_at_Kmax"], r["S_at_Kmax"]))


if __name__ == "__main__":
    sys.exit(main())
