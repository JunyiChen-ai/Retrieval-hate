#!/usr/bin/env python
"""
vga_pregate_emit.py -- PER-ITEM EMITTER for the VGA (C1) / VNQ (C2) $0 pregate.
Record: refine-logs/VGA_PREGATE_RECORD.md.  Spec: refine-logs/LITSWEEP6_RELGEN.md §2.

WHY THIS FILE EXISTS
    LITSWEEP6_RELGEN §0(a): the F95 pair-score matrices are NOT banked --
    scripts/analysis/mechnov_pairverify.py contains zero np.save/torch.save calls and
    only per-cell summary JSON survives.  The S[query, bank] matrices and the per-item
    fixed/broken identities C1 and C2 both need were computed in memory and discarded.
    The sweep record's prescribed remedy is followed here EXACTLY: a NEW emitter that
    IMPORTS the frozen arms module unmodified (sha asserted below), never edits it.

WHAT IT DOES
    Replays the frozen F95 fused-space primary cell (MLP verifier, max aggregation)
    fold by fold and writes, per TRAIN item: gold, fold, the deployed rank-weighted
    top-20 vote + its decision, the F95 adjudicated decision (max and mean-top-3), the
    cosine-shape control decision, the ERRPAT same-class-analogue rank, and three
    declared feature blocks (VGA gate features, F47-control features, kNN-UE features).

PARITY (the correctness gate, in the spirit of MECHFIX's 15/15 floor-parity gates)
    Because the frozen MLP is a deterministic pointwise function of the fitted pairs,
    scoring only the 20 nominated candidates instead of every (held-out x in-fold) pair
    is bit-identical on that subset.  Every pooled quantity this emitter recomputes is
    hard-asserted at 4 dp against the frozen mechnov_pairverify_{ds}_OUT.json fused
    cell: acc_deployed, acc_cos_shape, acc_mlp_max, acc_mlp_mean3, mF1s, positive
    rates, n_deployed_wrong, n_pathology_pop, and the fixed/broke/net/exchange-rate
    mechanism counts.  A mismatch aborts the run.

ARENA AND COST
    Banked RAW fused encoder key space, TRAIN SPLIT ONLY, item-disjoint 5-fold LOO --
    the F95 protocol verbatim.  Zero test contact, zero GPU, zero SLURM, zero Modal,
    CPU <= 8 threads.
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
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechnov_pairverify as P      # noqa: E402  FROZEN F95 arms -- imported, never edited
import mechfix_ops as M             # noqa: E402  FROZEN F89 deployed-vote replay

FROZEN_PAIRVERIFY_SHA = "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d"
FROZEN_MECHFIX_SHA = "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d"

SPACE = "fused"          # F95 PRIMARY space; the C1/C2 spec lives in the primary cell
VERIFIER = "mlp"         # F95 PRIMARY model (the logistic arm fired control 4 in F95)

# ---- declared feature blocks (frozen before any real-data number; see record §2.3)
VGA_FEATS = ["v_max", "v_top3mean", "v_gap", "v_spearman_rho",
             "v_rank_of_cos_top1", "v_disp"]
F47_FEATS = ["abs_vote", "purity_pred", "mean_cos20", "cos_spread",
             "label_ratio", "sub_vote_gap"]
F47_FULL_FEATS = F47_FEATS + ["vote", "sub_vote_text", "sub_vote_img", "sub_agree_ft"]
KNNUE_FEATS = ["max_cos", "mean_cos20", "cos_spread", "label_ratio", "purity_pred"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def rankdata_desc(A):
    """Average ranks (1 = largest) along axis 1, ties averaged."""
    n = A.shape[1]
    order = np.argsort(-A, axis=1, kind="stable")
    ranks = np.empty_like(A, dtype="float64")
    rows = np.arange(A.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :].astype("float64")
    # average ties
    for r in range(A.shape[0]):
        v = A[r]
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            for u in np.flatnonzero(cnt > 1):
                m = inv == u
                ranks[r, m] = ranks[r, m].mean()
    return ranks


def spearman_rows(A, B):
    ra, rb = rankdata_desc(A), rankdata_desc(B)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    den[den == 0] = 1.0
    return num / den


def run_dataset(key, log):
    cfg = P.DATASETS[key]
    ids, img, txt, lab = P.load_cache(cfg["cache_dir"], "train", cfg["model"])
    n = len(lab)
    X = P.build_space(img, txt, SPACE)          # PRIMARY arena
    Xt = P.build_space(img, txt, "text")        # sub-vote channel (F47 control feature)
    Xi = P.build_space(img, txt, "img")         # sub-vote channel (F47 control feature)
    log(f"[{key}] n={n} pos-rate {lab.mean():.4f} dim={X.shape[1]}")

    skf = StratifiedKFold(n_splits=P.K_FOLDS, shuffle=True, random_state=P.FOLD_SEED)

    col = {k: np.full(n, np.nan) for k in
           VGA_FEATS + F47_FULL_FEATS + ["max_cos", "dep_vote"]}
    ipred = {k: np.full(n, -1, dtype=int) for k in
             ["dep_pred", "cos_shape_pred", "adj_max_pred", "adj_mean3_pred"]}
    fold_of = np.full(n, -1, dtype=int)
    sc_rank = np.full(n, -1, dtype=int)
    per_fold_meta = []

    for fold, (fit_idx, ho_idx) in enumerate(skf.split(X, lab)):
        t0 = time.time()
        fit_idx = np.asarray(fit_idx)
        ho_idx = np.asarray(ho_idx)
        fold_of[ho_idx] = fold

        # ---------- frozen F95 verifier for this fold (constants from the frozen module)
        ncomp = min(P.PCA_DIM, len(fit_idx) - 1, X.shape[1])
        pca = PCA(n_components=ncomp, svd_solver=P.PCA_SOLVER, random_state=0)
        pca.fit(X[fit_idx])
        Zn = P.l2n(pca.transform(X))
        evr = float(pca.explained_variance_ratio_.sum())

        rng = np.random.RandomState(P.PAIR_SUBSAMPLE_SEED + fold)
        pi, pj, tot_pairs = P.all_unordered_pairs(fit_idx, rng, P.PAIR_FIT_CAP)
        Phi_fit = P.pair_features(Zn, pi, pj)
        y_fit = (lab[pi] == lab[pj]).astype(int)
        mu = Phi_fit.mean(0)
        sd = Phi_fit.std(0)
        sd[sd == 0] = 1.0
        Phi_fit -= mu
        Phi_fit /= sd
        mdl = P.fit_mlp(Phi_fit, y_fit)     # torch seed frozen inside the frozen module
        del Phi_fit

        # ---------- deployed vote (F89-frozen replay), fused + the two sub-channels
        dep_v, dep_p, dep_I, dep_sim = M.deployed_vote(
            X[fit_idx], lab[fit_idx], X[ho_idx], topk=P.TOPK_DEPLOYED)
        tv, _, _, _ = M.deployed_vote(Xt[fit_idx], lab[fit_idx], Xt[ho_idx],
                                      topk=P.TOPK_DEPLOYED)
        iv, _, _, _ = M.deployed_vote(Xi[fit_idx], lab[fit_idx], Xi[ho_idx],
                                      topk=P.TOPK_DEPLOYED)
        ipred["dep_pred"][ho_idx] = dep_p
        col["dep_vote"][ho_idx] = dep_v

        # ---------- retrieval NOMINATES top-M per class by full-space cosine (frozen)
        S = X[ho_idx] @ X[fit_idx].T
        cls_pos = {c: np.flatnonzero(lab[fit_idx] == c) for c in (0, 1)}
        for c in (0, 1):
            assert len(cls_pos[c]) >= P.M_PER_CLASS, (c, len(cls_pos[c]))
        nom = {}
        for c in (0, 1):
            sub = S[:, cls_pos[c]]
            top = np.argsort(-sub, axis=1, kind="stable")[:, :P.M_PER_CLASS]
            nom[c] = cls_pos[c][top]
        nom_all = np.concatenate([nom[0], nom[1]], axis=1)      # (n_ho, 2M): class0 | class1
        cos_nom = np.take_along_axis(S, nom_all, axis=1)

        # cosine-shape control (F95 control 2b)
        cs0 = np.take_along_axis(S, nom[0], axis=1).max(1)
        cs1 = np.take_along_axis(S, nom[1], axis=1).max(1)
        ipred["cos_shape_pred"][ho_idx] = (cs1 >= cs0).astype(int)

        # ---------- verifier scores on the nominated shortlist only (bit-identical subset)
        qq = np.repeat(ho_idx, nom_all.shape[1])
        bb = fit_idx[nom_all].ravel()
        Phi = P.pair_features(Zn, qq, bb)
        Phi -= mu
        Phi /= sd
        sc = P.predict_mlp(mdl, Phi).reshape(len(ho_idx), nom_all.shape[1])
        del Phi

        v0 = sc[:, :P.M_PER_CLASS]
        v1 = sc[:, P.M_PER_CLASS:]
        s0max, s1max = v0.max(1), v1.max(1)
        ipred["adj_max_pred"][ho_idx] = (s1max >= s0max).astype(int)
        m0 = (-np.sort(-v0, axis=1)[:, :P.MEAN_TOPQ]).mean(1)
        m1 = (-np.sort(-v1, axis=1)[:, :P.MEAN_TOPQ]).mean(1)
        ipred["adj_mean3_pred"][ho_idx] = (m1 >= m0).astype(int)

        # ---------- VGA gate features (record §2 C1 transplant sketch (ii), verbatim list)
        srt = -np.sort(-sc, axis=1)
        col["v_max"][ho_idx] = srt[:, 0]
        col["v_top3mean"][ho_idx] = srt[:, :P.MEAN_TOPQ].mean(1)
        col["v_gap"][ho_idx] = s1max - s0max
        col["v_spearman_rho"][ho_idx] = spearman_rows(cos_nom, sc)
        vrank = rankdata_desc(sc)
        col["v_rank_of_cos_top1"][ho_idx] = vrank[
            np.arange(len(ho_idx)), np.argmax(cos_nom, axis=1)]
        col["v_disp"][ho_idx] = sc.std(1)

        # ---------- F47 / kNN-UE control features (cosine-derived only, NO verifier)
        nlab = lab[fit_idx][dep_I]
        col["vote"][ho_idx] = dep_v
        col["abs_vote"][ho_idx] = np.abs(dep_v)
        col["purity_pred"][ho_idx] = (nlab == dep_p[:, None]).mean(1)
        col["mean_cos20"][ho_idx] = dep_sim.mean(1)
        col["max_cos"][ho_idx] = dep_sim.max(1)
        col["cos_spread"][ho_idx] = dep_sim.std(1)
        col["label_ratio"][ho_idx] = nlab.mean(1)
        col["sub_vote_text"][ho_idx] = tv
        col["sub_vote_img"][ho_idx] = iv
        col["sub_vote_gap"][ho_idx] = np.abs(tv - iv)
        col["sub_agree_ft"][ho_idx] = (np.sign(dep_v) == np.sign(tv)).astype(float)

        # ---------- ERRPAT same-class-analogue rank (frozen definition)
        order = np.argsort(-S, axis=1, kind="stable")
        bl = lab[fit_idx][order]
        for r, q in enumerate(ho_idx):
            hit = np.flatnonzero(bl[r] == lab[q])
            sc_rank[q] = int(hit[0]) + 1 if len(hit) else 10 ** 6

        per_fold_meta.append({"fold": fold, "n_fit_items": int(len(fit_idx)),
                              "n_ho_items": int(len(ho_idx)), "pca_dim": int(ncomp),
                              "pca_explained_var": round(evr, 4),
                              "n_pairs_total": int(tot_pairs),
                              "n_pairs_fitted": int(len(y_fit)),
                              "secs": round(time.time() - t0, 1)})
        log(f"    fold {fold}: dep {P.acc(lab[ho_idx], dep_p):.4f} "
            f"adj_max {P.acc(lab[ho_idx], ipred['adj_max_pred'][ho_idx]):.4f} "
            f"({per_fold_meta[-1]['secs']}s)")

    assert (ipred["dep_pred"] >= 0).all() and (fold_of >= 0).all()
    for k in col:
        assert np.isfinite(col[k]).all(), k

    return dict(ids=[str(x) for x in ids], lab=lab, fold=fold_of, sc_rank=sc_rank,
                ipred=ipred, col=col, per_fold=per_fold_meta,
                n=n, cfg=cfg)


def parity_gates(key, R, log):
    """Hard 4-dp parity against the frozen F95 fused cell.  Abort on any mismatch."""
    ref = json.load(open(os.path.join(
        REPO, f"scripts/analysis/mechnov_pairverify_{key}_OUT.json")))
    po = ref["spaces"][SPACE]["pooled"]
    me = ref["spaces"][SPACE]["control3_mechanism"]
    lab, ip = R["lab"], R["ipred"]

    got = {
        "acc_deployed": round(P.acc(lab, ip["dep_pred"]), 4),
        "mF1_deployed": round(M.macro_f1(lab, ip["dep_pred"]), 4),
        "posrate_deployed": round(float(ip["dep_pred"].mean()), 4),
        "acc_cos_shape": round(P.acc(lab, ip["cos_shape_pred"]), 4),
        "mF1_cos_shape": round(M.macro_f1(lab, ip["cos_shape_pred"]), 4),
        "posrate_cos_shape": round(float(ip["cos_shape_pred"].mean()), 4),
        "acc_mlp_max": round(P.acc(lab, ip["adj_max_pred"]), 4),
        "mF1_mlp_max": round(M.macro_f1(lab, ip["adj_max_pred"]), 4),
        "posrate_mlp_max": round(float(ip["adj_max_pred"].mean()), 4),
        "acc_mlp_mean3": round(P.acc(lab, ip["adj_mean3_pred"]), 4),
        "mF1_mlp_mean3": round(M.macro_f1(lab, ip["adj_mean3_pred"]), 4),
        "posrate_mlp_mean3": round(float(ip["adj_mean3_pred"].mean()), 4),
    }
    dep_wrong = ip["dep_pred"] != lab
    patho = dep_wrong & (R["sc_rank"] <= P.PATHOLOGY_RANK) & (R["sc_rank"] > 0)
    gotm = {"n_deployed_wrong": int(dep_wrong.sum()),
            "n_pathology_pop": int(patho.sum()),
            "median_sc_rank_all": float(np.median(R["sc_rank"])),
            "median_sc_rank_deployed_wrong": float(np.median(R["sc_rank"][dep_wrong]))}
    for ag, pk in (("max", "adj_max_pred"), ("mean3", "adj_mean3_pred")):
        p = ip[pk]
        fx = dep_wrong & (p == lab)
        bk = (~dep_wrong) & (p != lab)
        gotm[f"mlp_{ag}_fixed"] = int(fx.sum())
        gotm[f"mlp_{ag}_broke"] = int(bk.sum())
        gotm[f"mlp_{ag}_net"] = int(fx.sum()) - int(bk.sum())
        gotm[f"mlp_{ag}_exchange_rate"] = round(float(fx.sum()) / float(bk.sum()), 4)
        gotm[f"mlp_{ag}_pathology_fixed"] = int((patho & (p == lab)).sum())

    gates = []
    for k, v in got.items():
        gates.append((k, po[k], v, po[k] == v))
    for k, v in gotm.items():
        gates.append((k, me[k], v, me[k] == v))
    npass = sum(1 for g in gates if g[3])
    for k, exp, obs, ok in gates:
        if not ok:
            log(f"    PARITY FAIL {k}: frozen={exp} emitted={obs}")
    log(f"[{key}] PARITY {npass}/{len(gates)}")
    assert npass == len(gates), f"{key}: parity gate failed ({npass}/{len(gates)})"
    return {"n_gates": len(gates), "n_pass": npass,
            "gates": [{"key": k, "frozen": e, "emitted": o, "pass": bool(ok)}
                      for k, e, o, ok in gates]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(P.DATASETS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechnov_pairverify.py")) \
        == FROZEN_PAIRVERIFY_SHA, "FROZEN F95 ARMS MODULE CHANGED -- refusing to run"
    assert sha256_of(os.path.join(REPO, "scripts/analysis/mechfix_ops.py")) \
        == FROZEN_MECHFIX_SHA, "FROZEN F89 OPS MODULE CHANGED -- refusing to run"

    torch.set_num_threads(8)
    logf = open(a.out.replace(".json", ".log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    R = run_dataset(a.dataset, log)
    par = parity_gates(a.dataset, R, log)

    OUT = {"meta": {
        "script": os.path.abspath(__file__),
        "script_sha256": sha256_of(os.path.abspath(__file__)),
        "frozen_pairverify_sha256": FROZEN_PAIRVERIFY_SHA,
        "frozen_mechfix_ops_sha256": FROZEN_MECHFIX_SHA,
        "dataset": a.dataset, "ds": R["cfg"]["ds"], "model": R["cfg"]["model"],
        "space": SPACE, "verifier": VERIFIER,
        "n_train_items": R["n"], "pos_rate": round(float(R["lab"].mean()), 4),
        "test_contact": "NONE -- only the train split is loaded by this script",
        "vga_feats": VGA_FEATS, "f47_feats": F47_FEATS,
        "f47_full_feats": F47_FULL_FEATS, "knnue_feats": KNNUE_FEATS,
        "per_fold": R["per_fold"],
        "parity": par,
    },
        "ids": R["ids"],
        "lab": R["lab"].tolist(),
        "fold": R["fold"].tolist(),
        "sc_rank": R["sc_rank"].tolist(),
        "pred": {k: v.tolist() for k, v in R["ipred"].items()},
        "feat": {k: [float(x) for x in v] for k, v in R["col"].items()},
    }
    json.dump(OUT, open(a.out, "w"), indent=1)
    log(f"[{a.dataset}] EMITTED -> {a.out}")
    logf.close()


if __name__ == "__main__":
    main()
