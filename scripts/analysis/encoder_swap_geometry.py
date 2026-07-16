#!/usr/bin/env python
"""
Encoder-swap mechanism diagnosis — representation-geometry comparison.

Question: why does frozen Qwen2.5-VL-7B beat frozen CLIP by +5.3-5.6 acc on
HateMM (3/3 seeds) yet fail on MHC-EN and MHC-ZH?

This script measures the frozen-feature geometry that the RGCL head rides on,
for CLIP vs Qwen, on all three datasets, using TRAIN + DEV splits only
(zero test-touch). No GPU, no model inference — banked .pt caches only.

Read-outs (all approximate the head's kNN-vote substrate; the head adds only a
shallow per-modality linear map + L2-norm before a top-20 cosine kNN vote):
  * train->dev kNN vote (memory=train, query=dev, k=20, cosine-weighted signed vote)
  * train leave-one-out kNN (k=20)
  * kNN label homogeneity (mean same-label fraction among top-20, train LOO)
  * linear probe (StandardScaler + LogisticRegressionCV, 5-fold, train only)
  * class-centroid cosine separation

Per-modality variants: img-only, text-only, concat (each modality L2-normed
then concatenated, matching the head's fusion which L2-norms each stream).

Delta(Qwen - CLIP) per dataset is the headline: does the geometry gap track the
downstream +5.3 HateMM / fail MHC pattern?
"""
import os, json, sys
import numpy as np
import torch

BASE = "/data/jehc223/RGCL/data/CLIP_Embedding"
DATASETS = {
    "HateMM": "HateMM",
    "MHC-EN": "MHC",
    "MHC-ZH": "MHC_zh",
}
ENCODERS = {
    "CLIP": "openai_clip-vit-large-patch14-336_HF",
    "Qwen": "Qwen2.5-VL-7B-Instruct_HF",
}
K = 20
np.random.seed(0)


def load(ds_dir, enc_tag, split):
    p = f"{BASE}/{ds_dir}/{split}_{enc_tag}.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    img = d["img_feats"].float().numpy()
    txt = d["text_feats"].float().numpy()
    lab = d["labels"].long().numpy()
    ids = d["ids"][0] if isinstance(d["ids"], list) and len(d["ids"]) == 1 else d["ids"]
    return img, txt, lab, np.asarray(ids)


def l2n(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def build_modality(img, txt, mode):
    if mode == "img":
        return l2n(img)
    if mode == "text":
        return l2n(txt)
    if mode == "concat":
        return np.concatenate([l2n(img), l2n(txt)], axis=1)
    raise ValueError(mode)


def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f))


def bal_acc(y, p):
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            accs.append(np.mean(p[m] == c))
    return float(np.mean(accs))


def auc(y, score):
    # rank-based AUC
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ties
    n1 = np.sum(y == 1); n0 = np.sum(y == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def knn_vote(mem_X, mem_y, q_X, k):
    """cosine-weighted signed vote: score = sum_j w_j * (2*y_j-1), w_j=cos.
    returns predicted label (score>0) and the raw score (for AUC)."""
    S = q_X @ mem_X.T  # cosine sims (rows already unit or concat-of-units)
    # normalise rows to true cosine (concat has norm sqrt(2))
    qn = np.linalg.norm(q_X, axis=1, keepdims=True)
    mn = np.linalg.norm(mem_X, axis=1, keepdims=True)
    S = S / (qn + 1e-8) / (mn.T + 1e-8)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    scores = np.zeros(len(q_X))
    homog = np.zeros(len(q_X))
    for i in range(len(q_X)):
        nn = idx[i]
        w = S[i, nn]
        signs = 2 * mem_y[nn] - 1
        scores[i] = np.sum(w * signs)
    pred = (scores > 0).astype(int)
    return pred, scores, idx


def knn_homogeneity(X, y, k, loo=True):
    S = X @ X.T
    n = np.linalg.norm(X, axis=1, keepdims=True)
    S = S / (n + 1e-8) / (n.T + 1e-8)
    if loo:
        np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    frac = np.array([np.mean(y[idx[i]] == y[i]) for i in range(len(y))])
    return float(np.mean(frac))


def loo_knn(X, y, k):
    S = X @ X.T
    n = np.linalg.norm(X, axis=1, keepdims=True)
    S = S / (n + 1e-8) / (n.T + 1e-8)
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    scores = np.zeros(len(y))
    for i in range(len(y)):
        nn = idx[i]
        w = S[i, nn]
        signs = 2 * y[nn] - 1
        scores[i] = np.sum(w * signs)
    pred = (scores > 0).astype(int)
    return pred, scores


def centroid_sep(X, y):
    Xn = l2n(X)
    c0 = Xn[y == 0].mean(0); c1 = Xn[y == 1].mean(0)
    c0 /= (np.linalg.norm(c0) + 1e-8); c1 /= (np.linalg.norm(c1) + 1e-8)
    between = 1 - float(c0 @ c1)  # cosine distance between class means
    # within: mean cosine dist of points to own class mean
    within = []
    for c, cen in [(0, c0), (1, c1)]:
        m = y == c
        within.append(np.mean(1 - Xn[m] @ cen))
    within = float(np.mean(within))
    return between, within, between / (within + 1e-8)


def linear_probe(X, y, seeds=(0, 1, 2, 3, 4)):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    Cs = np.logspace(-4, 3, 8)
    accs, f1s, aucs = [], [], []
    for s in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        yp = np.zeros(len(y)); yscore = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegressionCV(Cs=Cs, cv=4, scoring="balanced_accuracy",
                                       max_iter=2000, n_jobs=1)
            clf.fit(sc.transform(X[tr]), y[tr])
            yp[te] = clf.predict(sc.transform(X[te]))
            yscore[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        accs.append(bal_acc(y, yp)); f1s.append(macro_f1(y, yp)); aucs.append(auc(y, yscore))
    return dict(balacc=float(np.mean(accs)), macrof1=float(np.mean(f1s)),
                auc=float(np.mean(aucs)), balacc_std=float(np.std(accs)))


def main():
    out = {}
    for ds_name, ds_dir in DATASETS.items():
        out[ds_name] = {}
        for enc_name, enc_tag in ENCODERS.items():
            tr_img, tr_txt, tr_y, tr_ids = load(ds_dir, enc_tag, "train")
            dv_img, dv_txt, dv_y, dv_ids = load(ds_dir, enc_tag, "dev_seen")
            rec = {"n_train": int(len(tr_y)), "n_dev": int(len(dv_y)),
                   "train_pos_frac": float(tr_y.mean()), "dev_pos_frac": float(dv_y.mean())}
            for mode in ("img", "text", "concat"):
                Xtr = build_modality(tr_img, tr_txt, mode)
                Xdv = build_modality(dv_img, dv_txt, mode)
                # train->dev kNN
                pred, score, _ = knn_vote(Xtr, tr_y, Xdv, K)
                d2d = dict(acc=float(np.mean(pred == dv_y)), balacc=bal_acc(dv_y, pred),
                           macrof1=macro_f1(dv_y, pred), auc=auc(dv_y, score))
                # train LOO kNN
                lp, ls = loo_knn(Xtr, tr_y, K)
                loo = dict(acc=float(np.mean(lp == tr_y)), balacc=bal_acc(tr_y, lp),
                           macrof1=macro_f1(tr_y, lp), auc=auc(tr_y, ls))
                homog = knn_homogeneity(Xtr, tr_y, K, loo=True)
                b, w, r = centroid_sep(Xtr, tr_y)
                lprobe = linear_probe(Xtr, tr_y) if mode == "concat" else None
                rec[mode] = dict(dev_knn=d2d, train_loo_knn=loo, knn_homog20=homog,
                                 centroid_between=b, centroid_within=w, centroid_ratio=r,
                                 linear_probe=lprobe)
            out[ds_name][enc_name] = rec
            print(f"[{ds_name}/{enc_name}] n_tr={rec['n_train']} n_dv={rec['n_dev']} "
                  f"pos_tr={rec['train_pos_frac']:.3f} "
                  f"concat dev-kNN acc={rec['concat']['dev_knn']['acc']:.4f} "
                  f"mF1={rec['concat']['dev_knn']['macrof1']:.4f} "
                  f"auc={rec['concat']['dev_knn']['auc']:.4f} "
                  f"homog={rec['concat']['knn_homog20']:.3f}")
    # deltas
    print("\n===== Delta(Qwen - CLIP), concat, dev-kNN =====")
    for ds_name in DATASETS:
        q = out[ds_name]["Qwen"]["concat"]; c = out[ds_name]["CLIP"]["concat"]
        print(f"{ds_name:8s} d_acc={q['dev_knn']['acc']-c['dev_knn']['acc']:+.4f} "
              f"d_mF1={q['dev_knn']['macrof1']-c['dev_knn']['macrof1']:+.4f} "
              f"d_auc={q['dev_knn']['auc']-c['dev_knn']['auc']:+.4f} "
              f"d_homog={q['knn_homog20']-c['knn_homog20']:+.3f} "
              f"d_LPauc={q['linear_probe']['auc']-c['linear_probe']['auc']:+.4f}")
    outp = "/data/jehc223/home/tmp/claude-135258174/-data-jehc223-RGCL/e8f03e41-3e21-4cea-b12c-29207373bfca/scratchpad/geometry_out.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", outp)


if __name__ == "__main__":
    main()
