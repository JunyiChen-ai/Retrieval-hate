#!/usr/bin/env python
"""
mechnov_pairverify_diag.py -- POST-HOC MECHANISM DIAGNOSTICS for the MECHNOV
pair-verify pregate. This file adds NO arm and can promote nothing: it exists only
to explain the result the frozen arms produced.

It reuses the frozen machinery in mechnov_pairverify.py (same seeds, same folds,
same PCA, same models) on the PRIMARY space only (fused), and asks one question:

    the verifier's POOLED pair-AUC is computed over all (query, bank) pairs, but the
    DECISION only ever compares candidates WITHIN one query's own shortlist. Are
    these the same discriminative ability?

Diagnostics:
  (a) mean WITHIN-QUERY pair-AUC for cosine vs the verifier (same eval pairs,
      AUC computed per query row and averaged over queries that have both classes);
  (b) two-way variance decomposition of the score matrix S[q, b] into a query main
      effect, a bank main effect and the residual interaction, for cosine and for
      the verifier. A relation scorer that helps only through main effects cannot
      change any within-query ordering.

CPU only, <= 8 threads, zero GPU / SLURM / Modal, train split only.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechnov_pairverify as P      # noqa: E402  frozen arms + helpers


def within_query_auc(S, same):
    """Mean AUC computed inside each query's own row (queries with both classes)."""
    aucs = []
    for r in range(S.shape[0]):
        y = same[r]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, S[r]))
    return float(np.mean(aucs)), len(aucs)


def var_decomp(S):
    """Share of total variance carried by query main effect, bank main effect and
    residual interaction in an additive two-way decomposition."""
    g = S.mean()
    rq = S.mean(1, keepdims=True) - g
    rb = S.mean(0, keepdims=True) - g
    resid = S - g - rq - rb
    tot = float(((S - g) ** 2).sum())
    return {"query_main": round(float((rq ** 2).sum() * S.shape[1]) / tot, 4),
            "bank_main": round(float((rb ** 2).sum() * S.shape[0]) / tot, 4),
            "interaction": round(float((resid ** 2).sum()) / tot, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="hatemm,zh,en",
                    help="run a subset per process (login-node reap window)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(8)

    OUT = {"meta": {"script": os.path.abspath(__file__),
                    "script_sha256": P.sha256_of(os.path.abspath(__file__)),
                    "frozen_arms_sha256": P.sha256_of(
                        os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")),
                    "space": "fused (PRIMARY only)",
                    "status": "POST-HOC DIAGNOSTIC -- adds no arm, promotes nothing"},
           "datasets": {}}
    if os.path.exists(a.out):          # resume: keep datasets already measured
        OUT["datasets"] = json.load(open(a.out)).get("datasets", {})

    for key in a.datasets.split(","):
        cfg = P.DATASETS[key]
        _, img, txt, lab = P.load_cache(cfg["cache_dir"], "train", cfg["model"])
        X = P.build_space(img, txt, "fused")
        skf = StratifiedKFold(n_splits=P.K_FOLDS, shuffle=True, random_state=P.FOLD_SEED)
        rec = {"folds": []}
        for fold, (fit_idx, ho_idx) in enumerate(skf.split(X, lab)):
            fit_idx = np.asarray(fit_idx); ho_idx = np.asarray(ho_idx)
            ncomp = min(P.PCA_DIM, len(fit_idx) - 1, X.shape[1])
            pca = PCA(n_components=ncomp, svd_solver=P.PCA_SOLVER, random_state=0)
            pca.fit(X[fit_idx])
            Zn = P.l2n(pca.transform(X))

            rng = np.random.RandomState(P.PAIR_SUBSAMPLE_SEED + fold)
            pi, pj, _ = P.all_unordered_pairs(fit_idx, rng, P.PAIR_FIT_CAP)
            Phi = P.pair_features(Zn, pi, pj)
            y = (lab[pi] == lab[pj]).astype(int)
            mu = Phi.mean(0); sd = Phi.std(0); sd[sd == 0] = 1.0
            Phi -= mu; Phi /= sd

            qq = np.repeat(ho_idx, len(fit_idx))
            bb = np.tile(fit_idx, len(ho_idx))
            Pe = P.pair_features(Zn, qq, bb); Pe -= mu; Pe /= sd
            same = (lab[qq] == lab[bb]).astype(int).reshape(len(ho_idx), len(fit_idx))
            Scos = (X[ho_idx] @ X[fit_idx].T)

            m = P.fit_mlp(Phi, y)
            Sv = P.predict_mlp(m, Pe).reshape(len(ho_idx), len(fit_idx))
            clf = P.fit_logistic(Phi, y)
            Sl = P.predict_logistic(clf, Pe).reshape(len(ho_idx), len(fit_idx))

            f = {"fold": fold}
            for nm, S in (("cosine", Scos), ("mlp", Sv), ("logistic", Sl)):
                wa, nq = within_query_auc(S, same)
                f[f"within_query_auc_{nm}"] = round(wa, 4)
                f[f"pooled_auc_{nm}"] = round(
                    float(roc_auc_score(same.ravel(), S.ravel())), 4)
                f[f"vardecomp_{nm}"] = var_decomp(S)
                f["n_queries_scored"] = nq
            rec["folds"].append(f)
            print(f"[{key}] fold {fold}: within-query AUC cos={f['within_query_auc_cosine']:.4f} "
                  f"mlp={f['within_query_auc_mlp']:.4f} log={f['within_query_auc_logistic']:.4f} | "
                  f"pooled cos={f['pooled_auc_cosine']:.4f} mlp={f['pooled_auc_mlp']:.4f}",
                  flush=True)
        mean = {}
        for nm in ("cosine", "mlp", "logistic"):
            for q in ("within_query_auc", "pooled_auc"):
                mean[f"{q}_{nm}"] = round(
                    float(np.mean([x[f"{q}_{nm}"] for x in rec["folds"]])), 4)
            for comp in ("query_main", "bank_main", "interaction"):
                mean[f"vardecomp_{nm}_{comp}"] = round(float(np.mean(
                    [x[f"vardecomp_{nm}"][comp] for x in rec["folds"]])), 4)
        for nm in ("mlp", "logistic"):
            mean[f"d_within_query_auc_{nm}"] = round(
                mean[f"within_query_auc_{nm}"] - mean["within_query_auc_cosine"], 4)
            mean[f"d_pooled_auc_{nm}"] = round(
                mean[f"pooled_auc_{nm}"] - mean["pooled_auc_cosine"], 4)
        rec["fold_mean"] = mean
        OUT["datasets"][key] = rec
        json.dump(OUT, open(a.out, "w"), indent=1)
    json.dump(OUT, open(a.out, "w"), indent=1)
    print("DIAG DONE ->", a.out)


if __name__ == "__main__":
    main()
