#!/usr/bin/env python
"""headspace_arena.py -- run the FROZEN F105/VSW operator battery in the DEPLOYED HEAD
KEY SPACE, on the arena that is item-for-item and fold-for-fold identical to the raw
arena F105 was decided in.  Record: refine-logs/HEADSPACE_TRANSFER_PREGATE.md.

THE ONLY VARIABLE IS THE KEY SPACE.
    raw arena (F105)  : X = l2n(concat(l2n(img), l2n(text)))            7168-d, seed-free
    head arena (this) : X = l2n(head_f(img, text))                      1024-d, per (seed, fold)
                        head_f = mlp[:-2](l2n(img_proj(img)) * l2n(text_proj(text)))
                        trained on the FITTING POOL of frozen fold f only
Same 744 train items, same StratifiedKFold(5, shuffle=True, random_state=0) assignment
(asserted against the banked vsw_ckpt), same deployed vote, same verifier, same lambda
grids, same inner-CV selection, same degeneracy controls.

REUSE, NOT REWRITE.  Three frozen modules are imported UNMODIFIED with their sha256
asserted at run time -- mechfix_ops.py (F89), mechnov_pairverify.py (F95),
vsw_pregate.py (F105).  Every treatment quantity is produced by F105's own
`run_arms` / `parity_lambda0` / `selected_arm` / `curve` on an arena dict with exactly
the keys F105's own emitter produces.  The ONLY new code is the emitter, which must be
new because the key matrix is per-fold rather than global.

TEST CONTACT: NONE.  This script opens only the mint .npz files (built from train_*.pt
and dev_seen_*.pt) and the banked vsw_ckpt raw-arena checkpoints.

DETERMINISM: clause DET-1/DET-2.  Thread env asserted, runtime block recorded.
COST: CPU only, <= 8 threads.  Zero GPU, zero SLURM, zero Modal.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))

import mechfix_ops as M          # noqa: E402  frozen F89
import mechnov_pairverify as P   # noqa: E402  frozen F95
import vsw_pregate as V          # noqa: E402  frozen F105
from headspace_mint import det1_assert, runtime_block, sha256_of  # noqa: E402

import torch  # noqa: E402

FROZEN = {
    "mechfix_ops.py": "635c13124e79ba1a299bc13fc1175a03aa11e09924f5413ce51061793c83fc8d",
    "mechnov_pairverify.py": "77b0defd8eaa3688e58b6d5d17202bd55d16cf1f4a5aaafbe4b2b98598b7240d",
    "vsw_pregate.py": "ba9982dba98fb14dd53297ac6c087f2e1a4aa068490879d2121c50cf1f932eea",
}

N_PERM_HEAD = 30          # declared budget for the head-space null (p-resolution 1/31)


def load_mint(mintdir, ds, seed, fold):
    tag = "full" if fold < 0 else str(fold)
    z = np.load(os.path.join(mintdir, "mint_{}_s{}_f{}.npz".format(ds, seed, tag)),
                allow_pickle=True)
    return z, json.loads(str(z["meta"]))


def emit_headspace(mintdir, ds, seed, ck, log, perm_draws=None):
    """The F105 emitter with a PER-FOLD key matrix.  Body mirrors
    vsw_pregate.emit_arena; every numerical step is a frozen-module call."""
    os.makedirs(ck, exist_ok=True)
    K = P.TOPK_DEPLOYED
    z0, m0 = load_mint(mintdir, ds, seed, 0)
    lab = z0["lab"].astype(int)
    n = len(lab)
    skf = StratifiedKFold(n_splits=P.K_FOLDS, shuffle=True, random_state=P.FOLD_SEED)
    splits = list(skf.split(np.zeros((n, 1)), lab))

    for fold, (fit_idx, ho_idx) in enumerate(splits):
        fit_idx = np.asarray(fit_idx)
        ho_idx = np.asarray(ho_idx)
        af = os.path.join(ck, "f{}.npz".format(fold))
        need_arena = not os.path.exists(af)
        need_draws = ([d for d in perm_draws
                       if not os.path.exists(os.path.join(ck, "f{}_d{}.npy".format(fold, d)))]
                      if perm_draws else [])
        if not need_arena and not need_draws:
            continue
        zf, mf = load_mint(mintdir, ds, seed, fold)
        assert np.array_equal(zf["lab"].astype(int), lab)
        assert np.array_equal(np.sort(zf["fit_idx"]), np.sort(fit_idx)), \
            "mint fitting pool != frozen fold {} fitting pool".format(fold)
        X = P.l2n(zf["K_train"])          # deployed pre-index form of the head key
        t0 = time.time()
        Zn, evr, ncomp, pi, pj, tot_pairs, Phi_fit, mu, sd = V._fold_setup(
            X, lab, fit_idx, fold)
        dep_v, dep_p, dep_I, dep_sim = M.deployed_vote(
            X[fit_idx], lab[fit_idx], X[ho_idx], topk=K)
        qq_d = np.repeat(ho_idx, K)
        bb_d = fit_idx[dep_I].ravel()
        Phi_d = P.pair_features(Zn, qq_d, bb_d)
        Phi_d -= mu
        Phi_d /= sd
        log("    s{} fold {} setup {:.1f}s (arena={} draws={})".format(
            seed, fold, time.time() - t0, need_arena, len(need_draws)))

        if need_arena:
            t1 = time.time()
            S = X[ho_idx] @ X[fit_idx].T
            cls_pos = {c: np.flatnonzero(lab[fit_idx] == c) for c in (0, 1)}
            nom = {}
            for c in (0, 1):
                assert len(cls_pos[c]) >= P.M_PER_CLASS
                sub = S[:, cls_pos[c]]
                top = np.argsort(-sub, axis=1, kind="stable")[:, :P.M_PER_CLASS]
                nom[c] = cls_pos[c][top]
            cs0 = np.take_along_axis(S, nom[0], axis=1).max(1)
            cs1 = np.take_along_axis(S, nom[1], axis=1).max(1)
            cos_shape = (cs1 >= cs0).astype(int)
            nom_all = np.concatenate([nom[0], nom[1]], axis=1)
            Phi_n = P.pair_features(Zn, np.repeat(ho_idx, nom_all.shape[1]),
                                    fit_idx[nom_all].ravel())
            Phi_n -= mu
            Phi_n /= sd
            y_fit = (lab[pi] == lab[pj]).astype(int)
            mdl = P.fit_mlp(Phi_fit, y_fit)
            nbr_p = P.predict_mlp(mdl, Phi_d).reshape(len(ho_idx), K)
            scn = P.predict_mlp(mdl, Phi_n).reshape(len(ho_idx), nom_all.shape[1])

            # ---- CONTROL 1 (F95): pair-AUC on the FULL held-out x in-fold pair matrix,
            #      plus the IN-SAMPLE fit-pair AUC (the head-space memorisation read).
            qq = np.repeat(ho_idx, len(fit_idx))
            bb = np.tile(fit_idx, len(ho_idx))
            Phi_ev = P.pair_features(Zn, qq, bb)
            Phi_ev -= mu
            Phi_ev /= sd
            y_ev = (lab[qq] == lab[bb]).astype(int)
            s_ev = P.predict_mlp(mdl, Phi_ev)
            cos_full = np.einsum("ij,ij->i", X[qq], X[bb])
            s_fit = P.predict_mlp(mdl, Phi_fit)
            cos_fit = np.einsum("ij,ij->i", X[pi], X[pj])
            c1 = np.array([
                roc_auc_score(y_ev, s_ev), roc_auc_score(y_ev, cos_full),
                roc_auc_score(y_fit, s_fit), roc_auc_score(y_fit, cos_fit),
                float(y_ev.mean()), float(y_fit.mean()),
                float((s_ev >= 0.5).mean()), float(len(y_ev)),
            ], dtype="float64")
            del Phi_ev, s_ev, Phi_n, mdl

            v0, v1 = scn[:, :P.M_PER_CLASS], scn[:, P.M_PER_CLASS:]
            adj_max = (v1.max(1) >= v0.max(1)).astype(int)
            m0_ = (-np.sort(-v0, axis=1)[:, :P.MEAN_TOPQ]).mean(1)
            m1_ = (-np.sort(-v1, axis=1)[:, :P.MEAN_TOPQ]).mean(1)
            adj_mean3 = (m1_ >= m0_).astype(int)
            order = np.argsort(-S, axis=1, kind="stable")
            bl = lab[fit_idx][order]
            scr = np.empty(len(ho_idx), dtype=int)
            for r, q in enumerate(ho_idx):
                hit = np.flatnonzero(bl[r] == lab[q])
                scr[r] = int(hit[0]) + 1 if len(hit) else 10 ** 6

            # ---- MECHFIX (F89) eval-time operators, same bank, same queries, head space
            _, t1p, _, _ = M.t1_class_balanced(X[fit_idx], lab[fit_idx], X[ho_idx])
            r_hub = M.bank_hubness(X[fit_idx])
            _, t2ap, _, _ = M.t2a_csls(X[fit_idx], lab[fit_idx], X[ho_idx], r_hub)
            wmu, W, shrink, _ = M.fit_whitener(X[fit_idx])
            Bw = M.apply_whitener(X[fit_idx], wmu, W)
            Qw = M.apply_whitener(X[ho_idx], wmu, W)
            _, t2bp, _, _ = M.deployed_vote(Bw, lab[fit_idx], Qw, topk=K)
            _, t4p, _, _ = M.t1_class_balanced(Bw, lab[fit_idx], Qw)

            V.atomic_savez(
                af, ho_idx=ho_idx, dep_v=dep_v, dep_p=dep_p, nbr_cos=dep_sim,
                nbr_lab=lab[fit_idx][dep_I].astype("float64"), nbr_glb=fit_idx[dep_I],
                nbr_p=nbr_p, cos_shape=cos_shape, adj_max=adj_max, adj_mean3=adj_mean3,
                sc_rank=scr, ctrl1=c1, t1=t1p, t2a=t2ap, t2b=t2bp, t4=t4p,
                meta=np.array([fold, len(fit_idx), len(ho_idx), ncomp, round(evr, 6),
                               tot_pairs, len(pi), round(time.time() - t1, 1), shrink],
                              dtype="float64"))
            log("    s{} fold {} ARENA {:.1f}s dep {:.4f} dAUC {:+.4f}".format(
                seed, fold, time.time() - t1, V.acc(lab[ho_idx], dep_p),
                c1[0] - c1[1]))

        if need_draws:
            loc = np.full(n, -1, dtype=int)
            loc[fit_idx] = np.arange(len(fit_idx))
            pil, pjl = loc[pi], loc[pj]
            for d in need_draws:
                t1 = time.time()
                r2 = np.random.RandomState(V.PERM_SEED + 1000 * int(d) + fold)
                lab_perm = lab[fit_idx][r2.permutation(len(fit_idx))]
                yp = (lab_perm[pil] == lab_perm[pjl]).astype(int)
                mp = P.fit_mlp(Phi_fit, yp)
                sc = P.predict_mlp(mp, Phi_d).reshape(len(ho_idx), K)
                del mp
                tmp = os.path.join(ck, ".tmp_f{}_d{}.npy".format(fold, d))
                np.save(tmp, sc)
                os.replace(tmp, os.path.join(ck, "f{}_d{}.npy".format(fold, d)))
                log("    s{} fold {} draw {} {:.1f}s".format(seed, fold, d,
                                                             time.time() - t1))
        del Phi_fit, Phi_d, Zn

    # ---------------------------------------------------------------- assemble
    A = {"n": n, "lab": lab, "fold": np.full(n, -1, dtype=int),
         "nbr_cos": np.full((n, K), np.nan), "nbr_lab": np.full((n, K), np.nan),
         "nbr_glb": np.full((n, K), -1, dtype=int), "nbr_p": np.full((n, K), np.nan),
         "dep_vote": np.full(n, np.nan), "dep_pred": np.full(n, -1, dtype=int),
         "cos_shape_pred": np.full(n, -1, dtype=int),
         "adj_max_pred": np.full(n, -1, dtype=int),
         "adj_mean3_pred": np.full(n, -1, dtype=int),
         "sc_rank": np.full(n, -1, dtype=int), "per_fold": [], "ctrl1": [],
         "mechfix": {k: np.full(n, -1, dtype=int) for k in ("t1", "t2a", "t2b", "t4")}}
    for fold in range(P.K_FOLDS):
        z = np.load(os.path.join(ck, "f{}.npz".format(fold)))
        h = z["ho_idx"]
        A["fold"][h] = fold
        for k_src, k_dst in (("nbr_cos", "nbr_cos"), ("nbr_lab", "nbr_lab"),
                             ("nbr_glb", "nbr_glb"), ("nbr_p", "nbr_p"),
                             ("dep_v", "dep_vote"), ("dep_p", "dep_pred"),
                             ("cos_shape", "cos_shape_pred"),
                             ("adj_max", "adj_max_pred"),
                             ("adj_mean3", "adj_mean3_pred"), ("sc_rank", "sc_rank")):
            A[k_dst][h] = z[k_src]
        for k in ("t1", "t2a", "t2b", "t4"):
            A["mechfix"][k][h] = z[k]
        c = z["ctrl1"]
        A["ctrl1"].append({"auc_verifier_heldout": round(float(c[0]), 4),
                           "auc_cosine_heldout": round(float(c[1]), 4),
                           "d_auc_vs_cos": round(float(c[0] - c[1]), 4),
                           "auc_verifier_INSAMPLE_fitpairs": round(float(c[2]), 4),
                           "auc_cosine_INSAMPLE_fitpairs": round(float(c[3]), 4),
                           "same_class_rate_heldout": round(float(c[4]), 4),
                           "same_class_rate_fit": round(float(c[5]), 4),
                           "posrate_pairpred": round(float(c[6]), 4),
                           "n_eval_pairs": int(c[7])})
        m = z["meta"]
        A["per_fold"].append({"fold": int(m[0]), "n_fit_items": int(m[1]),
                              "n_ho_items": int(m[2]), "pca_dim": int(m[3]),
                              "pca_explained_var": round(float(m[4]), 4),
                              "n_pairs_total": int(m[5]), "n_pairs_fitted": int(m[6]),
                              "secs": float(m[7]),
                              "lw_shrinkage": round(float(m[8]), 4)})
    assert (A["dep_pred"] >= 0).all() and (A["fold"] >= 0).all()
    assert np.isfinite(A["nbr_p"]).all() and np.isfinite(A["nbr_cos"]).all()
    return A


def membership_overlap(A, ds):
    """K-HST-2: how much does the deployed top-20 MEMBERSHIP change between the raw arena
    F105 decided in and this head arena, for the SAME query, SAME fold, SAME bank?"""
    ckraw = os.path.join(REPO, "scripts/analysis/vsw_ckpt", ds)
    n, K = A["n"], P.TOPK_DEPLOYED
    raw_glb = np.full((n, K), -1, dtype=int)
    raw_lab = np.full((n, K), np.nan)
    raw_dep = np.full(n, -1, dtype=int)
    for fold in range(P.K_FOLDS):
        z = np.load(os.path.join(ckraw, "f{}.npz".format(fold)))
        h = z["ho_idx"]
        raw_glb[h] = z["nbr_glb"]
        raw_lab[h] = z["nbr_lab"]
        raw_dep[h] = z["dep_p"]
    ov = np.array([len(set(raw_glb[i]) & set(A["nbr_glb"][i])) for i in range(n)])
    # label-tuple distance: the vote reads only the retrieved labels (LITSWEEP8 Result A,
    # HEADCOV K-HC-3), so this is the quantity that actually drives non-transfer.
    lab_l1 = np.abs(np.sort(raw_lab, axis=1) - np.sort(A["nbr_lab"], axis=1)).sum(1)
    return {"mean_top20_overlap": round(float(ov.mean()), 4),
            "mean_overlap_frac": round(float(ov.mean() / K), 4),
            "median_overlap": int(np.median(ov)),
            "frac_queries_overlap_0": round(float((ov == 0).mean()), 4),
            "frac_queries_overlap_ge10": round(float((ov >= 10).mean()), 4),
            "mean_abs_diff_n_positive_neighbours": round(float(lab_l1.mean() / 2.0), 4),
            "raw_deployed_acc": round(V.acc(A["lab"], raw_dep), 4),
            "head_deployed_acc": round(V.acc(A["lab"], A["dep_pred"]), 4),
            "raw_vs_head_decision_agreement": round(
                float((raw_dep == A["dep_pred"]).mean()), 4)}


def extra_arms(A):
    """F95 (nominate+verify) and F89 (MECHFIX eval-time operators) in head space --
    the transfer ladder, free from the same emitter."""
    lab, dp = A["lab"], A["dep_pred"]
    out = {}
    for name, pred in (("F95_mlp_max", A["adj_max_pred"]),
                       ("F95_mlp_mean3", A["adj_mean3_pred"]),
                       ("F95_cos_shape", A["cos_shape_pred"]),
                       ("F89_T1_classbal", A["mechfix"]["t1"]),
                       ("F89_T2a_csls", A["mechfix"]["t2a"]),
                       ("F89_T2b_whiten", A["mechfix"]["t2b"]),
                       ("F89_T4_whiten_classbal", A["mechfix"]["t4"])):
        out[name] = V.summarise(A, pred)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["arena", "perm"])
    ap.add_argument("--dataset", required=True, choices=sorted(P.DATASETS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mintdir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nperm", type=int, default=N_PERM_HEAD)
    a = ap.parse_args()

    det1_assert(str(a.threads))
    for f, sha in FROZEN.items():
        got = sha256_of(os.path.join(REPO, "scripts/analysis", f))
        assert got == sha, "FROZEN MODULE CHANGED: {} {}".format(f, got)
    torch.set_num_threads(a.threads)

    logf = open(a.out.replace(".json", ".log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    ck = os.path.join(a.ckpt, "{}_s{}".format(a.dataset, a.seed))
    _, mmeta = load_mint(a.mintdir, a.dataset, a.seed, 0)
    meta = {"script_sha256": sha256_of(os.path.abspath(__file__)),
            "frozen_sha256": FROZEN, "stage": a.stage, "dataset": a.dataset,
            "seed": a.seed, "space": "TRAINED HEAD (1024-d), per-fold re-mint",
            "mint_script_sha256": mmeta["script_sha256"],
            "encoder_model": mmeta["encoder_model"],
            "frozen_vsw": {"INNER_FOLDS": V.INNER_FOLDS, "INNER_SEED": V.INNER_SEED,
                           "PERM_SEED": V.PERM_SEED, "DEG_KILL": V.DEG_KILL,
                           "CLASSBAL_TOL": V.CLASSBAL_TOL,
                           "PRIMARY_FAMILY": V.PRIMARY_FAMILY,
                           "N_PERM_HEAD": a.nperm},
            "test_contact": "NONE -- mint npz (train/dev caches only) + banked vsw_ckpt",
            "runtime": runtime_block()}

    if a.stage == "arena":
        A = emit_headspace(a.mintdir, a.dataset, a.seed, ck, log)
        pg = V.parity_lambda0(A, log, tag="[{} s{}] ".format(a.dataset, a.seed))
        R = V.run_arms(A)
        OUT = {"meta": meta, "per_fold": A["per_fold"], "control1": A["ctrl1"],
               "parity_lambda0": pg, "result": R,
               "membership": membership_overlap(A, a.dataset),
               "extra_arms": extra_arms(A)}
        V.atomic_json(a.out, OUT)
        log("[{} s{}] ARENA COMPLETE dep {:.4f} VSW_pow dacc {:+.4f} -> {}".format(
            a.dataset, a.seed, R["acc_deployed"],
            R["arms"]["VSW_pow"]["dacc"], a.out))
        logf.close()
        return

    draws = list(range(a.nperm))
    A = emit_headspace(a.mintdir, a.dataset, a.seed, ck, log, perm_draws=draws)
    rows = []
    for d in draws:
        ef = os.path.join(ck, "eval_d{}.json".format(d))
        if os.path.exists(ef):
            rows.append(json.load(open(ef)))
            continue
        Pn = np.clip(V.load_perm_table(ck, A, d), V.P_FLOOR, 1.0)
        Rn = V.run_arms(A, Pm=Pn, perm_mode=True)
        r = {"draw": int(d),
             "arms": {k: {kk: Rn["arms"][k][kk] for kk in
                          ("dacc", "fixed", "broke", "exchange_rate", "changed",
                           "posrate", "foldsigns")} for k in Rn["arms"]}}
        V.atomic_json(ef, r)
        rows.append(r)
        log("    draw {}: {:+.4f}".format(d, r["arms"]["VSW_pow"]["dacc"]))
    V.atomic_json(a.out, {"meta": meta, "draws": rows})
    log("[{} s{}] PERM COMPLETE ({} draws) -> {}".format(
        a.dataset, a.seed, len(rows), a.out))
    logf.close()


if __name__ == "__main__":
    main()
