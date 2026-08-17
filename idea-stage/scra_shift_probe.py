"""
SCRA shift probe -- zero test-label measurement of how much covariate shift is worth.

Decision rules frozen in idea-stage/SCRA_SHIFT_FREEZE.md BEFORE this file existed.

Test labels are NEVER loaded: load_inputs() reads only img/text feature tensors for the test
split and discards the label field.

M1 domain-classifier OOF AUC (full dim + PCA32), M2 density-ratio stats, M3 MMD^2 + permutation,
M4 support coverage, M5 importance-weighted val AUC vs plain val AUC for the deployed head.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4_harness import Head, feat_path, load_split, macro_f1, CLIP  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
LORA = "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"
QWEN = "Qwen2.5-VL-7B-Instruct_HF"
CELLS = [
    ("HateMM", LORA, "LORA"),
    ("MHC", QWEN, "QWEN"),
    ("MHC_zh", LORA, "LORA"),
    ("ImpliHateVid", CLIP, "CLIP"),
]
EPOCHS, LR, BS, WARMUP = 30, 1e-4, 64, 5
NBOOT, NPERM = 200, 200


def load_inputs(dataset, model_tag, split):
    """Test split: inputs only, labels dropped on purpose."""
    d = torch.load(feat_path(dataset, model_tag, split), map_location="cpu",
                   weights_only=False)
    return {"img": d["img_feats"].float(), "txt": d["text_feats"].float()}


def fuse(pack):
    a = torch.nn.functional.normalize(pack["img"], p=2, dim=1)
    b = torch.nn.functional.normalize(pack["txt"], p=2, dim=1)
    return torch.cat([a, b], dim=1).numpy().astype(np.float64)


def oof_domain(A, B, seed=0, dim=None):
    """Cross-fitted logistic domain classifier: label 1 = B (target). Returns (auc, oof_prob_A)."""
    X = np.concatenate([A, B], 0)
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    if dim is not None and X.shape[1] > dim:
        X = PCA(n_components=dim, random_state=seed).fit_transform(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    p = np.zeros(len(y))
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tri], y[tri])
        p[tei] = clf.predict_proba(X[tei])[:, 1]
    return float(roc_auc_score(y, p)), p[:len(A)]


def mmd2_perm(A, B, seed=0):
    X = np.concatenate([A, B], 0)
    n = len(A)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    sig2 = np.median(d2[np.triu_indices(len(X), 1)])
    K = np.exp(-d2 / (sig2 + 1e-12))

    def stat(idx):
        a, b = idx[:n], idx[n:]
        return (K[np.ix_(a, a)].mean() + K[np.ix_(b, b)].mean()
                - 2 * K[np.ix_(a, b)].mean())

    obs = stat(np.arange(len(X)))
    rng = np.random.default_rng(seed)
    null = np.array([stat(rng.permutation(len(X))) for _ in range(NPERM)])
    return float(obs), float((null >= obs).mean())


def coverage(train, test):
    Tr = torch.tensor(train, dtype=torch.float32)
    Te = torch.tensor(test, dtype=torch.float32)
    Tr = torch.nn.functional.normalize(Tr, dim=1)
    Te = torch.nn.functional.normalize(Te, dim=1)
    s_tr = Tr @ Tr.T
    s_tr.fill_diagonal_(-2)
    nn_tr = 1 - s_tr.max(1).values.numpy()          # train->train NN cosine distance
    nn_te = 1 - (Te @ Tr.T).max(1).values.numpy()   # test ->train NN cosine distance
    thr = np.percentile(nn_tr, 95)
    return float(thr), float((nn_te > thr).mean()), float(np.median(nn_te)), \
        float(np.median(nn_tr))


def weighted_auc(y, s, w):
    pos, neg = y > 0.5, y < 0.5
    sp, wp = s[pos], w[pos]
    sn, wn = s[neg], w[neg]
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    M = (sp[:, None] > sn[None, :]).astype(np.float64) + \
        0.5 * (sp[:, None] == sn[None, :]).astype(np.float64)
    W = wp[:, None] * wn[None, :]
    return float((M * W).sum() / W.sum())


def train_deployed(tr, va, te_x, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Head(tr["img"].shape[1], tr["txt"].shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    Xi, Xt, Y = tr["img"].to(DEV), tr["txt"].to(DEV), tr["y"].to(DEV)
    Vi, Vt, yv = va["img"].to(DEV), va["txt"].to(DEV), va["y"].numpy()
    Ti, Tt = te_x["img"].to(DEV), te_x["txt"].to(DEV)
    n = len(Y)
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, None)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=g).to(DEV)
        for k in range(0, n, BS):
            idx = perm[k:k + BS]
            lo = model(Xi[idx], Xt[idx]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(lo, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Vi, Vt).squeeze(1)).cpu().numpy()
            pt = torch.sigmoid(model(Ti, Tt).squeeze(1)).cpu().numpy()
        if ep >= WARMUP:
            sc = macro_f1(yv, pv)
            if sc > best[0]:
                best = (sc, {"ep": ep, "val": pv, "test": pt})
    return best[1]


def run_cell(dataset, model_tag, tag):
    tr = load_split(dataset, model_tag, "train")
    va = load_split(dataset, model_tag, "val")
    te = load_inputs(dataset, model_tag, "test")
    Ftr, Fva, Fte = fuse(tr), fuse(va), fuse(te)
    out = {"dataset": dataset, "model": model_tag, "tag": tag,
           "n": [len(Ftr), len(Fva), len(Fte)], "dim": Ftr.shape[1]}

    # M1 domain classifiers
    out["m1_val_test_full"], sv = oof_domain(Fva, Fte, dim=None)
    out["m1_val_test_pca32"], sv32 = oof_domain(Fva, Fte, dim=32)
    out["m1_train_test_full"], _ = oof_domain(Ftr, Fte, dim=None)
    out["m1_train_test_pca32"], _ = oof_domain(Ftr, Fte, dim=32)

    # M2 density ratios on val (from the PCA32 classifier: better calibrated at these n)
    ratio = (len(Fva) / len(Fte)) * sv32 / np.clip(1 - sv32, 1e-6, None)
    lo, hi = np.percentile(ratio, [1, 99])
    w = np.clip(ratio, lo, hi)
    w = w / w.mean()
    out["m2"] = {"max": float(ratio.max()), "q95": float(np.percentile(ratio, 95)),
                 "ess_frac": float((w.sum() ** 2) / (len(w) * (w ** 2).sum()))}

    # M3 MMD, M4 coverage
    sub = np.random.default_rng(0)
    ia = sub.choice(len(Fva), min(len(Fva), 300), replace=False)
    ib = sub.choice(len(Fte), min(len(Fte), 300), replace=False)
    m, p = mmd2_perm(Fva[ia], Fte[ib])
    out["m3"] = {"mmd2": m, "perm_p": p}
    thr, frac, mte, mtr = coverage(Ftr, Fte)
    out["m4"] = {"thr95_train_nn": thr, "frac_test_outside": frac,
                 "med_nn_test": mte, "med_nn_train": mtr}

    # M5 deployed head: plain vs importance-weighted val AUC
    yv = va["y"].numpy()
    rows = []
    rng = np.random.default_rng(0)
    for s in SEEDS:
        sel = train_deployed(tr, va, te, s)
        sc = sel["val"]
        a_plain = float(roc_auc_score(yv, sc))
        a_iw = weighted_auc(yv, sc, w)
        boots, boots_p = [], []
        for _ in range(NBOOT):
            bi = rng.integers(0, len(yv), len(yv))
            if len(np.unique(yv[bi])) < 2:
                continue
            boots.append(weighted_auc(yv[bi], sc[bi], w[bi]))
            boots_p.append(float(roc_auc_score(yv[bi], sc[bi])))
        rows.append({"seed": s, "epoch": sel["ep"], "auc_plain": a_plain,
                     "auc_iw": a_iw, "delta": a_iw - a_plain,
                     "se_iw": float(np.std(boots)),
                     "se_plain": float(np.std(boots_p))})
    out["m5"] = {"rows": rows,
                 "delta_mean": float(np.mean([r["delta"] for r in rows])),
                 "se_iw_mean": float(np.mean([r["se_iw"] for r in rows])),
                 "auc_plain_mean": float(np.mean([r["auc_plain"] for r in rows])),
                 "auc_iw_mean": float(np.mean([r["auc_iw"] for r in rows]))}
    return out


if __name__ == "__main__":
    res = []
    for ds, mt, tag in CELLS:
        print(f"=== {ds}/{tag}", flush=True)
        r = run_cell(ds, mt, tag)
        print(json.dumps({k: v for k, v in r.items() if k != "m5"}, indent=1), flush=True)
        print("M5", json.dumps(r["m5"], indent=1), flush=True)
        res.append(r)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "scra_shift.json"), "w"), indent=2)


# ---- Deviation D1: null calibration (see SCRA_SHIFT_DEVIATION_D1.md) ----
def null_calibration(dataset, model_tag, nrep=40):
    """Matched-size zero-shift null.

    A = the full val split (labelled, out-of-sample head scores -- IDENTICAL to the object
    used in M5). B = a random subset of TRAIN of size n_test. Under i.i.d. splitting train and
    val are exchangeable, so A-vs-B carries ZERO shift by construction while reproducing the
    exact sample sizes (n_val, n_test), dimensionality and estimator of the real M5 comparison.
    Delta_null is therefore the noise floor of M5's Delta.
    """
    tr = load_split(dataset, model_tag, "train")
    va = load_split(dataset, model_tag, "val")
    te = load_inputs(dataset, model_tag, "test")
    Ftr, Fva = fuse(tr), fuse(va)
    n_te = len(te["img"])
    yv = va["y"].numpy()
    rng = np.random.default_rng(0)
    d_all = []
    for seed in SEEDS:
        sc = train_deployed(tr, va, te, seed)["val"]
        a_plain = float(roc_auc_score(yv, sc))
        for r in range(nrep):
            B = rng.choice(len(Ftr), min(n_te, len(Ftr)), replace=False)
            _, sA = oof_domain(Fva, Ftr[B], seed=r, dim=32)
            ratio = (len(Fva) / len(B)) * sA / np.clip(1 - sA, 1e-6, None)
            lo, hi = np.percentile(ratio, [1, 99])
            w = np.clip(ratio, lo, hi); w = w / w.mean()
            d_all.append(weighted_auc(yv, sc, w) - a_plain)
    d = np.abs(np.array(d_all))
    return {"dataset": dataset, "n_rep": int(len(d)),
            "abs_delta_null_mean": float(d.mean()),
            "abs_delta_null_p50": float(np.percentile(d, 50)),
            "abs_delta_null_p90": float(np.percentile(d, 90)),
            "abs_delta_null_p95": float(np.percentile(d, 95)),
            "abs_delta_null_max": float(d.max())}
