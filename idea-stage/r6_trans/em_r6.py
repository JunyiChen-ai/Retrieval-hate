"""R6-2 step 3 -- the transductive pool-refinement operator (TransCLIP/UNEM style)
and the frozen decision rule of idea-stage/R6_PILOT_FREEZE_2026-08-17.md, Pilot R6-2.

Operator (exactly as implemented):
  inputs  Z in R^{n x d} (pool embeddings, row-wise L2-normalised), p in (0,1)^n
          (inductive positive probability), pi_init = TRAIN split class prior,
          knobs lambda >= 0 (KL anchor to the inductive prediction) and rho in [0,1]
          (class-balance strength).
  init    q_i1 = p_i, q_i0 = 1 - p_i ; pi = pi_init
  repeat up to 20 times:
    M-step   mu_c   = sum_i q_ic z_i / sum_i q_ic                     (c in {0,1})
             sigma2 = sum_i sum_c q_ic ||z_i - mu_c||^2 / (n * d)     (shared, spherical)
             pi_c   = (1 - rho) * pi_c_init + rho * mean_i q_ic
    E-step   log qhat_ic = -||z_i - mu_c||^2 / (2 sigma2) + log pi_c + lambda * log p_ic
             q_ic = softmax_c(log qhat_ic)
    stop when max_ic |q_ic - q_ic_prev| < 1e-5
  output  q_i1, thresholded at the SAME validation-selected threshold used for IND.

Leakage guard: (lambda, rho) is chosen per dataset x seed by running the identical EM
over the dev_seen pool (dev embeddings, dev inductive probabilities) and scoring against
dev labels; the winner is then applied once to the test pool. Test labels enter only at
the final scoring call. The dev-selected threshold is likewise fitted on dev only.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = os.path.join(HERE, "dumps")
DATASETS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]
SEEDS = [0, 1, 2]
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]
RHOS = [0.0, 0.5, 1.0]
MAX_IT = 20
TOL = 1e-5
EPS = 1e-12


def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(out))


def select_threshold(prob, y):
    """Threshold maximising macro-F1 on the given (validation) split.
    Candidate set = midpoints between consecutive distinct probabilities, plus the
    two open ends and 0.5.  Ties are broken toward 0.5."""
    u = np.unique(prob)
    cands = [0.5, float(u[0]) - 1e-6, float(u[-1]) + 1e-6]
    cands += [float((u[i] + u[i + 1]) / 2.0) for i in range(len(u) - 1)]
    cands = sorted(set(cands))
    best_f1, best_t = -1.0, 0.5
    for t in cands:
        f1 = macro_f1(y, (prob >= t).astype(int))
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and
                                    abs(t - 0.5) < abs(best_t - 0.5)):
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def em_refine(Z, p, pi_init_pos, lam, rho):
    """Block-MM EM described in the module docstring.  Returns (q1, n_iter)."""
    n, d = Z.shape
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), EPS)
    pc = np.clip(np.stack([1.0 - p, p], axis=1), 1e-6, 1.0 - 1e-6)  # [n,2] p_i0,p_i1
    logp = np.log(pc)
    pi_init = np.array([1.0 - pi_init_pos, pi_init_pos], dtype=np.float64)
    pi = pi_init.copy()
    q = pc.copy()
    n_it = 0
    for it in range(MAX_IT):
        n_it = it + 1
        # ---- M-step
        nk = q.sum(axis=0)                                    # [2]
        mu = (q.T @ Z) / np.maximum(nk[:, None], EPS)         # [2,d]
        d2 = ((Z ** 2).sum(1)[:, None] - 2.0 * (Z @ mu.T)
              + (mu ** 2).sum(1)[None, :])                    # [n,2] squared distances
        d2 = np.maximum(d2, 0.0)
        sigma2 = float((q * d2).sum() / (n * d))
        sigma2 = max(sigma2, EPS)
        pi = (1.0 - rho) * pi_init + rho * (q.mean(axis=0))
        pi = np.maximum(pi, EPS)
        # ---- E-step
        logq = -d2 / (2.0 * sigma2) + np.log(pi)[None, :] + lam * logp
        logq -= logq.max(axis=1, keepdims=True)
        w = np.exp(logq)
        q_new = w / w.sum(axis=1, keepdims=True)
        delta = float(np.abs(q_new - q).max())
        q = q_new
        if delta < TOL:
            break
    return q[:, 1], n_it


def main():
    results = {"grid": {"lambda": LAMBDAS, "rho": RHOS},
               "max_iter": MAX_IT, "tol": TOL, "datasets": {}}
    for ds in DATASETS:
        results["datasets"][ds] = {"seeds": {}}
        for seed in SEEDS:
            npz = np.load(os.path.join(DUMPS, "%s_s%d.npz" % (ds, seed)),
                          allow_pickle=True)
            meta = json.load(open(os.path.join(DUMPS, "%s_s%d.json" % (ds, seed))))
            dz, dp, dy = npz["dev_z"], npz["dev_prob"], npz["dev_y"]
            tz, tp, ty = npz["test_z"], npz["test_prob"], npz["test_y"]
            prior = float(meta["train_prior_pos"])

            # ---- threshold: validation split only
            thr, dev_f1_ind = select_threshold(dp, dy)
            ind_test = (tp >= thr).astype(int)
            f1_ind = macro_f1(ty, ind_test)

            # ---- (lambda, rho) selection on the dev pool, dev labels only
            grid = []
            best = None
            for lam in LAMBDAS:
                for rho in RHOS:
                    q1, nit = em_refine(dz, dp, prior, lam, rho)
                    f1 = macro_f1(dy, (q1 >= thr).astype(int))
                    grid.append({"lambda": lam, "rho": rho, "dev_macro_f1": f1,
                                 "iters": nit})
                    # ties -> first in frozen grid order (lambda asc, rho asc)
                    if best is None or f1 > best["dev_macro_f1"] + 1e-12:
                        best = grid[-1]
            lam, rho = best["lambda"], best["rho"]

            # ---- TRANS on the test pool
            q_tr, it_tr = em_refine(tz, tp, prior, lam, rho)
            pred_tr = (q_tr >= thr).astype(int)
            f1_trans = macro_f1(ty, pred_tr)

            # ---- SHUF control: test embeddings permuted relative to p
            rng = np.random.default_rng(20260817 + seed)
            perm = rng.permutation(len(tp))
            q_sh, it_sh = em_refine(tz[perm], tp, prior, lam, rho)
            pred_sh = (q_sh >= thr).astype(int)
            f1_shuf = macro_f1(ty, pred_sh)

            results["datasets"][ds]["seeds"][str(seed)] = {
                "epoch": meta["epoch"], "encoder": meta["model"],
                "n_dev": int(len(dy)), "n_test": int(len(ty)),
                "train_prior_pos": prior,
                "threshold": thr,
                "dev_macro_f1_IND": dev_f1_ind,
                "selected_lambda": lam, "selected_rho": rho,
                "dev_macro_f1_selected": best["dev_macro_f1"],
                "test_macro_f1_IND": f1_ind,
                "test_macro_f1_TRANS": f1_trans,
                "test_macro_f1_SHUF": f1_shuf,
                "n_changed_TRANS_vs_IND": int((pred_tr != ind_test).sum()),
                "n_changed_SHUF_vs_IND": int((pred_sh != ind_test).sum()),
                "em_iters_TRANS": it_tr, "em_iters_SHUF": it_sh,
                "grid": grid,
            }
            print("[em] %-13s s%d thr=%.4f lam=%.2f rho=%.1f devF1=%.4f | "
                  "IND=%.4f TRANS=%.4f SHUF=%.4f | chg T=%d S=%d"
                  % (ds, seed, thr, lam, rho, best["dev_macro_f1"], f1_ind,
                     f1_trans, f1_shuf,
                     int((pred_tr != ind_test).sum()), int((pred_sh != ind_test).sum())))

    # ---- aggregate + frozen decision rule
    n_go = 0
    go_datasets = []
    for ds in DATASETS:
        S = results["datasets"][ds]["seeds"]
        d_ti = [S[str(s)]["test_macro_f1_TRANS"] - S[str(s)]["test_macro_f1_IND"]
                for s in SEEDS]
        d_ts = [S[str(s)]["test_macro_f1_TRANS"] - S[str(s)]["test_macro_f1_SHUF"]
                for s in SEEDS]
        agg = {
            "delta_TRANS_IND_per_seed": d_ti,
            "delta_TRANS_IND_mean": float(np.mean(d_ti)),
            "n_seeds_positive_TRANS_IND": int(sum(1 for x in d_ti if x > 0)),
            "delta_TRANS_SHUF_per_seed": d_ts,
            "delta_TRANS_SHUF_mean": float(np.mean(d_ts)),
            "mean_test_IND": float(np.mean([S[str(s)]["test_macro_f1_IND"] for s in SEEDS])),
            "mean_test_TRANS": float(np.mean([S[str(s)]["test_macro_f1_TRANS"] for s in SEEDS])),
            "mean_test_SHUF": float(np.mean([S[str(s)]["test_macro_f1_SHUF"] for s in SEEDS])),
        }
        agg["dataset_passes"] = bool(agg["delta_TRANS_IND_mean"] >= 0.005
                                     and agg["n_seeds_positive_TRANS_IND"] == 3
                                     and agg["delta_TRANS_SHUF_mean"] >= 0.005)
        results["datasets"][ds]["aggregate"] = agg
        if agg["dataset_passes"]:
            n_go += 1
            go_datasets.append(ds)
    verdict = "GO" if n_go >= 2 else ("AMBIGUOUS" if n_go == 1 else "KILL")
    results["n_datasets_passing"] = n_go
    results["passing_datasets"] = go_datasets
    results["verdict"] = verdict
    out = os.path.join(HERE, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("\n[em] datasets passing = %d %s -> VERDICT %s" % (n_go, go_datasets, verdict))
    print("[em] wrote", out)


if __name__ == "__main__":
    main()
