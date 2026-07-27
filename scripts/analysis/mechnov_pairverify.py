#!/usr/bin/env python
"""
mechnov_pairverify.py -- FROZEN implementation for the MECHNOV pair-verification
$0 pregate (record: refine-logs/MECHNOV_PAIRVERIFY_PREGATE.md).

THE IDEA UNDER TEST
    Replace VOTING with VERIFICATION. The deployed decision is a top-20
    rank-weighted signed-cosine kNN vote over the own-train memory bank
    (src/utils/metrics.py:262-301, src/model/evaluate_rac.py:405-465), measured by
    the ERRPAT reports to behave as a local-class-prior estimator. Here retrieval
    only NOMINATES candidate analogues; the decision is made by a TRAINED PAIR
    VERIFIER that scores each (query, candidate) DIFFERENCE-SPACE relation as
    "same-class-like" vs "cross-class-like". Supervision becomes relational:
    n train labels -> ~n^2 pair labels.

WHAT IS FROZEN HERE
    Every constant below was fixed before any treatment number was computed. There
    is no tuning, no early stopping on any held-out quantity, no post-hoc arm. The
    arms declared in the record's section 2 are exactly the arms this file runs.

ARENA
    Banked RAW encoder key spaces (seed-independent), NOT trained head spaces. The
    trained RGCL head memorises its own train split (LOO train acc 0.998, F47), so a
    verifier fitted in head space would be measuring memorisation. The raw space is
    the honest pregate arena; the limitation is stated in the record.

PROTOCOL (item-disjoint, non-negotiable)
    K=5 stratified folds over TRAIN ITEMS ONLY. The verifier is fitted exclusively on
    pairs whose BOTH endpoints lie in the fitting folds. It is evaluated on
    (held-out item) x (in-fold item) pairs. No pair with a held-out endpoint is ever
    seen at fit time. ZERO test-split contact anywhere in this file.

COST
    CPU only, <= 8 threads. Zero GPU, zero SLURM, zero Modal, zero training of any
    deployed arm.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechfix_ops as M          # noqa: E402  frozen, parity-verified deployed vote

# ----------------------------------------------------------------- FROZEN CONSTANTS
K_FOLDS = 5                      # item-disjoint folds over the train split
FOLD_SEED = 0                    # StratifiedKFold(shuffle=True, random_state=0)
PCA_DIM = 256                    # tractability: pair features live in 2*PCA_DIM
PCA_SOLVER = "full"              # deterministic
PAIR_FIT_CAP = 150000            # cap on fitted pairs; subsample is seeded (below)
PAIR_SUBSAMPLE_SEED = 0
MLP_HIDDEN = 128                 # 1 hidden layer
MLP_EPOCHS = 30                  # fixed; NO early stopping on any held-out quantity
MLP_BATCH = 1024
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_SEED = 0
LOGIT_C = 1.0                    # sklearn default L2 strength
LOGIT_MAXITER = 1000
M_PER_CLASS = 10                 # retrieval nominates top-10 of EACH class
TOPK_DEPLOYED = 20               # deployed budget, unchanged
MEAN_TOPQ = 3                    # secondary aggregation: mean of top-3 verified scores
PATHOLOGY_RANK = 5               # ERRPAT pathology population: same-class analogue in top-5

SPACES = ("fused", "text", "img")          # PRIMARY = fused; text/img are SECONDARY
MODELS = ("mlp", "logistic")               # both declared PRIMARY
AGGS = ("max", "mean3")                    # PRIMARY = max; mean3 SECONDARY

DATASETS = {
    "hatemm": dict(ds="HateMM", model="Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
                   cache_dir=os.path.join(REPO, "data/CLIP_Embedding/HateMM")),
    "zh": dict(ds="MHC_zh", model="Qwen2.5-VL-7B-Instruct-LoRA_HF",
               cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC_zh")),
    "en": dict(ds="MHC", model="Qwen2.5-VL-7B-Instruct_HF",
               cache_dir=os.path.join(REPO, "data/CLIP_Embedding/MHC")),
}


# ------------------------------------------------------------------------- helpers
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_cache(cache_dir, split, model):
    d = torch.load(os.path.join(cache_dir, f"{split}_{model}.pt"),
                   map_location="cpu", weights_only=False)
    ids = d["ids"]
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return (list(ids), d["img_feats"].float().numpy().astype("float64"),
            d["text_feats"].float().numpy().astype("float64"),
            np.asarray(d["labels"]).astype(int))


def l2n(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def build_space(img, txt, space):
    """RAW banked key space. 'fused' reproduces the errpat raw-encoder control
    (L2-normalised concat of the two streams); text/img are the single streams.
    Every space is returned L2-normalised, i.e. the deployed pre-index form."""
    if space == "img":
        return l2n(img)
    if space == "text":
        return l2n(txt)
    if space == "fused":
        return l2n(np.concatenate([l2n(img), l2n(txt)], axis=1))
    raise ValueError(space)


def pair_features(Zn, ii, jj):
    """[ |z_i - z_j| , z_i * z_j ] -- the standard difference-space pair encoding.
    Symmetric in (i, j) by construction. NOTE for the audit: with L2-normalised z,
    sum(z_i * z_j) IS the cosine, so a linear model on these features CONTAINS the
    cosine rule; the cosine control is therefore a nested-model comparison."""
    a = Zn[ii]
    b = Zn[jj]
    return np.concatenate([np.abs(a - b), a * b], axis=1).astype("float32")


def all_unordered_pairs(idx, rng, cap):
    n = len(idx)
    ii, jj = np.triu_indices(n, k=1)
    tot = len(ii)
    if tot > cap:
        sel = rng.choice(tot, size=cap, replace=False)
        ii, jj = ii[sel], jj[sel]
    return idx[ii], idx[jj], tot


def fit_logistic(Phi, y):
    clf = LogisticRegression(penalty="l2", C=LOGIT_C, solver="lbfgs",
                             max_iter=LOGIT_MAXITER, n_jobs=1)
    clf.fit(Phi, y)
    return clf


def predict_logistic(clf, Phi):
    return clf.predict_proba(Phi)[:, 1].astype("float64")


class MLP(torch.nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Phi, y):
    torch.manual_seed(MLP_SEED)
    m = MLP(Phi.shape[1], MLP_HIDDEN)
    opt = torch.optim.Adam(m.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    lossf = torch.nn.BCEWithLogitsLoss()
    X = torch.from_numpy(Phi)
    Y = torch.from_numpy(y.astype("float32"))
    rng = np.random.RandomState(MLP_SEED)
    n = X.shape[0]
    m.train()
    for _ in range(MLP_EPOCHS):
        perm = rng.permutation(n)
        for s in range(0, n, MLP_BATCH):
            b = perm[s:s + MLP_BATCH]
            opt.zero_grad()
            out = m(X[b])
            loss = lossf(out, Y[b])
            loss.backward()
            opt.step()
    m.eval()
    return m


def predict_mlp(m, Phi, chunk=200000):
    out = np.empty(Phi.shape[0], dtype="float64")
    with torch.no_grad():
        for s in range(0, Phi.shape[0], chunk):
            x = torch.from_numpy(Phi[s:s + chunk])
            out[s:s + chunk] = torch.sigmoid(m(x)).numpy().astype("float64")
    return out


def acc(y, p):
    return float((np.asarray(y) == np.asarray(p)).mean())


# ------------------------------------------------------------------------ the pregate
def run_space(X, lab, space, log):
    """One dataset x one raw space: 5 item-disjoint folds, all declared arms."""
    n = X.shape[0]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=FOLD_SEED)
    per_fold = []
    # per-item collectors over the union of held-out folds (each item held out once)
    coll = {k: np.full(n, -1, dtype=int) for k in
            ["dep_pred", "cos_shape_pred"] +
            [f"{mo}_{ag}_pred" for mo in MODELS for ag in AGGS]}
    coll["dep_vote"] = np.full(n, np.nan)
    coll["sc_rank"] = np.full(n, -1, dtype=int)

    for fold, (fit_idx, ho_idx) in enumerate(skf.split(X, lab)):
        t0 = time.time()
        fit_idx = np.asarray(fit_idx)
        ho_idx = np.asarray(ho_idx)
        Xf, yf = X[fit_idx], lab[fit_idx]

        # ---- reduction: PCA fitted on FIT-FOLD ITEMS ONLY, then L2-renormalise
        ncomp = min(PCA_DIM, len(fit_idx) - 1, X.shape[1])
        pca = PCA(n_components=ncomp, svd_solver=PCA_SOLVER, random_state=0)
        pca.fit(Xf)
        Zn = l2n(pca.transform(X))
        evr = float(pca.explained_variance_ratio_.sum())

        # ---- FIT pairs: both endpoints in the fitting folds
        rng = np.random.RandomState(PAIR_SUBSAMPLE_SEED + fold)
        pi, pj, tot_pairs = all_unordered_pairs(fit_idx, rng, PAIR_FIT_CAP)
        Phi_fit = pair_features(Zn, pi, pj)
        y_fit = (lab[pi] == lab[pj]).astype(int)
        mu = Phi_fit.mean(0)
        sd = Phi_fit.std(0)
        sd[sd == 0] = 1.0
        Phi_fit -= mu
        Phi_fit /= sd

        # ---- EVAL pairs: held-out item x in-fold item (no held-out endpoint at fit)
        qq = np.repeat(ho_idx, len(fit_idx))
        bb = np.tile(fit_idx, len(ho_idx))
        Phi_ev = pair_features(Zn, qq, bb)
        Phi_ev -= mu
        Phi_ev /= sd
        y_ev = (lab[qq] == lab[bb]).astype(int)
        cos_full = np.einsum("ij,ij->i", X[qq], X[bb])       # deployed metric
        cos_red = np.einsum("ij,ij->i", Zn[qq], Zn[bb])      # reduced-space metric

        fitted, scores = {}, {}
        for mo in MODELS:
            if mo == "logistic":
                obj = fit_logistic(Phi_fit, y_fit)
                scores[mo] = predict_logistic(obj, Phi_ev)
            else:
                obj = fit_mlp(Phi_fit, y_fit)
                scores[mo] = predict_mlp(obj, Phi_ev)
            fitted[mo] = obj

        # ---- CONTROL 1: pair-AUC on the SAME eval pairs
        c1 = {"n_eval_pairs": int(len(y_ev)),
              "same_class_rate": round(float(y_ev.mean()), 4),
              "auc_cosine_fullspace": round(float(roc_auc_score(y_ev, cos_full)), 4),
              "auc_cosine_pcaspace": round(float(roc_auc_score(y_ev, cos_red)), 4)}
        for mo in MODELS:
            c1[f"auc_{mo}"] = round(float(roc_auc_score(y_ev, scores[mo])), 4)
            c1[f"d_auc_{mo}_vs_cos_full"] = round(
                c1[f"auc_{mo}"] - c1["auc_cosine_fullspace"], 4)
            c1[f"d_auc_{mo}_vs_cos_pca"] = round(
                c1[f"auc_{mo}"] - c1["auc_cosine_pcaspace"], 4)

        # ---- CONTROL 4a: verifier pair-prediction balance at 0.5
        for mo in MODELS:
            c1[f"posrate_pairpred_{mo}"] = round(float((scores[mo] >= 0.5).mean()), 4)

        # ---- END-TO-END: deployed vote (LOO form) over the SAME in-fold bank
        dep_v, dep_p, _, _ = M.deployed_vote(X[fit_idx], lab[fit_idx], X[ho_idx],
                                             topk=TOPK_DEPLOYED)
        coll["dep_pred"][ho_idx] = dep_p
        coll["dep_vote"][ho_idx] = dep_v

        # ---- retrieval NOMINATES top-M per class by full-space cosine
        S = X[ho_idx] @ X[fit_idx].T                     # (n_ho, n_fit) cosine
        smap = scores  # alias
        Sv = {mo: smap[mo].reshape(len(ho_idx), len(fit_idx)) for mo in MODELS}
        cls_pos = {c: np.flatnonzero(lab[fit_idx] == c) for c in (0, 1)}
        for c in (0, 1):
            assert len(cls_pos[c]) >= M_PER_CLASS, (c, len(cls_pos[c]))

        nom = {}
        for c in (0, 1):
            sub = S[:, cls_pos[c]]
            top = np.argsort(-sub, axis=1, kind="stable")[:, :M_PER_CLASS]
            nom[c] = cls_pos[c][top]                     # (n_ho, M) indices into fit_idx

        # cosine-shape control (same retrieval + same aggregation, cosine as scorer)
        cs = {}
        for c in (0, 1):
            cs[c] = np.take_along_axis(S, nom[c], axis=1).max(1)
        coll["cos_shape_pred"][ho_idx] = (cs[1] >= cs[0]).astype(int)

        for mo in MODELS:
            for ag in AGGS:
                sc = {}
                for c in (0, 1):
                    v = np.take_along_axis(Sv[mo], nom[c], axis=1)
                    if ag == "max":
                        sc[c] = v.max(1)
                    else:
                        vs = -np.sort(-v, axis=1)[:, :MEAN_TOPQ]
                        sc[c] = vs.mean(1)
                coll[f"{mo}_{ag}_pred"][ho_idx] = (sc[1] >= sc[0]).astype(int)

        # ---- rank of the first same-gold-class bank item (ERRPAT pathology stat)
        order = np.argsort(-S, axis=1, kind="stable")
        bl = lab[fit_idx][order]
        for r, q in enumerate(ho_idx):
            hit = np.flatnonzero(bl[r] == lab[q])
            coll["sc_rank"][q] = int(hit[0]) + 1 if len(hit) else 10 ** 6

        fold_rec = {"fold": fold, "n_fit_items": int(len(fit_idx)),
                    "n_ho_items": int(len(ho_idx)),
                    "pca_dim": int(ncomp), "pca_explained_var": round(evr, 4),
                    "n_pairs_total": int(tot_pairs), "n_pairs_fitted": int(len(y_fit)),
                    "fit_same_class_rate": round(float(y_fit.mean()), 4),
                    "control1": c1,
                    "acc_deployed": round(acc(lab[ho_idx], dep_p), 4),
                    "acc_cos_shape": round(acc(lab[ho_idx],
                                               coll["cos_shape_pred"][ho_idx]), 4),
                    "secs": round(time.time() - t0, 1)}
        for mo in MODELS:
            for ag in AGGS:
                fold_rec[f"acc_{mo}_{ag}"] = round(
                    acc(lab[ho_idx], coll[f"{mo}_{ag}_pred"][ho_idx]), 4)
        per_fold.append(fold_rec)
        log(f"    [{space}] fold {fold}: dep {fold_rec['acc_deployed']:.4f} "
            f"mlp_max {fold_rec['acc_mlp_max']:.4f} log_max {fold_rec['acc_logistic_max']:.4f} "
            f"dAUC(mlp) {c1['d_auc_mlp_vs_cos_full']:+.4f} ({fold_rec['secs']}s)")

    # -------------------------------------------------- pooled over all held-out items
    assert (coll["dep_pred"] >= 0).all()
    pooled = {"acc_deployed": round(acc(lab, coll["dep_pred"]), 4),
              "mF1_deployed": round(M.macro_f1(lab, coll["dep_pred"]), 4),
              "acc_cos_shape": round(acc(lab, coll["cos_shape_pred"]), 4),
              "mF1_cos_shape": round(M.macro_f1(lab, coll["cos_shape_pred"]), 4),
              "posrate_bank": round(float(lab.mean()), 4),
              "posrate_deployed": round(float(coll["dep_pred"].mean()), 4),
              "posrate_cos_shape": round(float(coll["cos_shape_pred"].mean()), 4)}
    for mo in MODELS:
        for ag in AGGS:
            k = f"{mo}_{ag}"
            p = coll[f"{k}_pred"]
            pooled[f"acc_{k}"] = round(acc(lab, p), 4)
            pooled[f"mF1_{k}"] = round(M.macro_f1(lab, p), 4)
            pooled[f"posrate_{k}"] = round(float(p.mean()), 4)          # CONTROL 4
            pooled[f"dacc_{k}_vs_deployed"] = round(
                pooled[f"acc_{k}"] - pooled["acc_deployed"], 4)
            pooled[f"dacc_{k}_vs_cos_shape"] = round(
                pooled[f"acc_{k}"] - pooled["acc_cos_shape"], 4)
            fs = [f[f"acc_{k}"] - f["acc_deployed"] for f in per_fold]
            pooled[f"foldsigns_{k}"] = "".join(
                "+" if v > 0 else ("-" if v < 0 else "0") for v in fs)
            pooled[f"folddeltas_{k}"] = [round(v, 4) for v in fs]

    # control-1 3-fold... 5-fold means and signs
    c1m = {}
    for kk in per_fold[0]["control1"]:
        vals = [f["control1"][kk] for f in per_fold]
        c1m[kk] = round(float(np.mean(vals)), 4)
    for mo in MODELS:
        d = [f["control1"][f"d_auc_{mo}_vs_cos_full"] for f in per_fold]
        c1m[f"foldsigns_dauc_{mo}"] = "".join(
            "+" if v > 0 else ("-" if v < 0 else "0") for v in d)
        c1m[f"folddeltas_dauc_{mo}"] = [round(v, 4) for v in d]

    # -------------------------------------------------- CONTROL 3: mechanism read
    dep_wrong = coll["dep_pred"] != lab
    patho = dep_wrong & (coll["sc_rank"] <= PATHOLOGY_RANK) & (coll["sc_rank"] > 0)
    mech = {"n_items": int(len(lab)),
            "n_deployed_wrong": int(dep_wrong.sum()),
            "n_pathology_pop": int(patho.sum()),
            "median_sc_rank_all": float(np.median(coll["sc_rank"])),
            "median_sc_rank_deployed_wrong": (
                float(np.median(coll["sc_rank"][dep_wrong])) if dep_wrong.any() else None)}
    for mo in MODELS:
        for ag in AGGS:
            k = f"{mo}_{ag}"
            p = coll[f"{k}_pred"]
            fixed = dep_wrong & (p == lab)
            broke = (~dep_wrong) & (p != lab)
            pfix = patho & (p == lab)
            mech[f"{k}_fixed"] = int(fixed.sum())
            mech[f"{k}_broke"] = int(broke.sum())
            mech[f"{k}_net"] = int(fixed.sum()) - int(broke.sum())
            mech[f"{k}_exchange_rate"] = (round(float(fixed.sum()) / float(broke.sum()), 4)
                                          if broke.sum() else None)
            mech[f"{k}_pathology_fixed"] = int(pfix.sum())
            mech[f"{k}_pathology_frac_fixed"] = (round(float(pfix.sum()) / float(patho.sum()), 4)
                                                 if patho.sum() else None)
    return {"per_fold": per_fold, "pooled": pooled, "control1_5foldmean": c1m,
            "control3_mechanism": mech}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(8)
    cfg = DATASETS[a.dataset]
    logf = open(a.out.replace(".json", ".log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    ids, img, txt, lab = load_cache(cfg["cache_dir"], "train", cfg["model"])
    log(f"[{a.dataset}] train items {len(ids)}  pos-rate {lab.mean():.4f}  "
        f"img {img.shape} txt {txt.shape}")

    OUT = {"meta": {
        "script": os.path.abspath(__file__),
        "script_sha256": sha256_of(os.path.abspath(__file__)),
        "mechfix_ops_sha256": sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py")),
        "dataset": a.dataset, "ds": cfg["ds"], "model": cfg["model"],
        "n_train_items": int(len(ids)), "pos_rate": round(float(lab.mean()), 4),
        "frozen": dict(K_FOLDS=K_FOLDS, FOLD_SEED=FOLD_SEED, PCA_DIM=PCA_DIM,
                       PCA_SOLVER=PCA_SOLVER, PAIR_FIT_CAP=PAIR_FIT_CAP,
                       MLP_HIDDEN=MLP_HIDDEN, MLP_EPOCHS=MLP_EPOCHS,
                       MLP_BATCH=MLP_BATCH, MLP_LR=MLP_LR, MLP_WD=MLP_WD,
                       MLP_SEED=MLP_SEED, LOGIT_C=LOGIT_C,
                       LOGIT_MAXITER=LOGIT_MAXITER, M_PER_CLASS=M_PER_CLASS,
                       TOPK_DEPLOYED=TOPK_DEPLOYED, MEAN_TOPQ=MEAN_TOPQ,
                       PATHOLOGY_RANK=PATHOLOGY_RANK, SPACES=list(SPACES),
                       MODELS=list(MODELS), AGGS=list(AGGS)),
        "test_contact": "NONE -- only the train split is loaded by this script",
    }, "spaces": {}}

    for space in SPACES:
        X = build_space(img, txt, space)
        log(f"  space={space} dim={X.shape[1]}")
        OUT["spaces"][space] = run_space(X, lab, space, log)
        json.dump(OUT, open(a.out, "w"), indent=1)
    json.dump(OUT, open(a.out, "w"), indent=1)
    log(f"[{a.dataset}] DONE -> {a.out}")
    logf.close()


if __name__ == "__main__":
    main()
