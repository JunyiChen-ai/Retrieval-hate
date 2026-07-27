#!/usr/bin/env python
"""
restrans_pregate_diag.py -- POST-HOC diagnostic for the C1 residual-transport pregate.

Adds no arm, promotes nothing, changes no verdict. It exists to explain the measured
null, in the same role mechnov_pairverify_diag.py played for F95.

Three questions, all on the PRIMARY (fused) raw space, train split only:

  (1) DECOMPOSITION. Write the correction as  2*phat_i - 1 = cbar + delta_i  with
      cbar = mean over the fitting-fold bank. The vote shift is then
          v_res - v_dep = -cbar * (SUM_i cos_i w_i / SUM_i w_i)  -  (SUM_i delta_i cos_i w_i / SUM_i w_i)
      i.e. a near-constant threshold move plus an item-level term. How big is each?

  (2) THE CONSTANT-SHIFT TWIN. Run the identical operator with phat_i replaced by its
      own bank mean -- a pure global threshold shift, which is a MEASURED-DEAD lever.
      If its predictions agree with C1's, LITSWEEP6 bar 3 fires by direct measurement
      rather than by inference from sd(phat).

  (3) THE MHC-EN LOO-INTERCEPT ARTEFACT. On MHC-EN the fitted logistic coefficient on
      the covariate is ~0, so the leave-one-out phat collapses to
      (SUM y - y_i) / (n-1), a strictly DECREASING function of the item's own label --
      which is why AUC(phat, gold) comes out at exactly 0.0 in two folds. Verified,
      not asserted.

CPU only, zero GPU/SLURM/Modal, zero test contact.
"""
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechfix_ops as M            # noqa: E402
import mechnov_pairverify as PV    # noqa: E402
import restrans_pregate as R       # noqa: E402


def main():
    torch_threads = 8
    import torch
    torch.set_num_threads(torch_threads)
    out = {"meta": {"script": os.path.abspath(__file__),
                    "script_sha256": hashlib.sha256(
                        open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                    "role": "POST-HOC; adds no arm, promotes nothing",
                    "space": R.PRIMARY_SPACE, "test_contact": "NONE"},
           "datasets": {}}

    for key in ("hatemm", "zh", "en"):
        cfg = R.DATASETS[key]
        ids, img, txt, lab = PV.load_cache(cfg["cache_dir"], "train", cfg["model"])
        n = len(ids)
        vol = R.load_volume(cfg["gt"], ids, cfg["vol"])
        lv = np.log1p(vol)
        X = PV.build_space(img, txt, R.PRIMARY_SPACE)
        skf = StratifiedKFold(n_splits=R.K_FOLDS, shuffle=True,
                              random_state=R.FOLD_SEED)
        folds = list(skf.split(np.zeros((n, 1)), lab))

        dec, twin, art = [], [], []
        agree_c1_const = np.full(n, -1, dtype=int)
        pred_c1 = np.full(n, -1, dtype=int)
        pred_const = np.full(n, -1, dtype=int)
        dep_all = np.full(n, -1, dtype=int)

        for f, (fit_idx, ho_idx) in enumerate(folds):
            fit_idx = np.asarray(fit_idx); ho_idx = np.asarray(ho_idx)
            yb = lab[fit_idx]
            ph = R.phat_logistic_loo(lv[fit_idx], yb)          # B-a, the primary arm
            c = 2.0 * ph - 1.0
            cbar = float(c.mean())
            delta = c - cbar

            dv, dp, dI, dS = M.deployed_vote(X[fit_idx], yb, X[ho_idx], topk=R.TOPK)
            w = M._rank_weights(R.TOPK)
            cw = (dS * w).sum(1) / w.sum()                      # SUM cos_i w_i / SUM w
            dv_res = -(np.asarray(c)[dI] * dS * w).sum(1) / w.sum()
            const_part = -cbar * cw
            item_part = -(np.asarray(delta)[dI] * dS * w).sum(1) / w.sum()

            dec.append({"fold": f, "cbar": round(cbar, 4),
                        "sd_c": round(float(c.std()), 4),
                        "mean_shift_total": round(float(dv_res.mean()), 4),
                        "sd_shift_total": round(float(dv_res.std()), 4),
                        "mean_const_part": round(float(const_part.mean()), 4),
                        "sd_const_part": round(float(const_part.std()), 4),
                        "sd_item_part": round(float(item_part.std()), 4),
                        "sd_ratio_item_over_const": round(
                            float(item_part.std() / max(abs(cbar), 1e-12)), 4),
                        "mean_topk_cos": round(float(cw.mean()), 6)})

            # ---- (2) the constant-shift twin
            r_c1 = (2.0 * yb - 1.0) - c
            r_cs = (2.0 * yb - 1.0) - cbar
            _, p_c1, _, _ = R.residual_vote(X[fit_idx], yb, X[ho_idx], r_c1)
            _, p_cs, _, _ = R.residual_vote(X[fit_idx], yb, X[ho_idx], r_cs)
            pred_c1[ho_idx] = p_c1
            pred_const[ho_idx] = p_cs
            dep_all[ho_idx] = dp
            agree_c1_const[ho_idx] = (p_c1 == p_cs).astype(int)
            twin.append({"fold": f, "n_ho": int(len(ho_idx)),
                         "agree_c1_vs_constshift": int((p_c1 == p_cs).sum()),
                         "acc_deployed": round(R.acc(lab[ho_idx], dp), 4),
                         "acc_C1": round(R.acc(lab[ho_idx], p_c1), 4),
                         "acc_constshift": round(R.acc(lab[ho_idx], p_cs), 4)})

            # ---- (3) LOO-intercept artefact
            clf = LogisticRegression(penalty="l2", C=R.BA_LOGIT_C, solver="lbfgs",
                                     max_iter=R.BA_MAXITER, n_jobs=1)
            clf.fit(lv[fit_idx].reshape(-1, 1), yb)
            nn = len(yb)
            loo_intercept = (yb.sum() - yb) / (nn - 1.0)
            art.append({"fold": f,
                        "logit_coef_on_log1p_vol": round(float(clf.coef_[0, 0]), 4),
                        "sd_phat": round(float(ph.std()), 6),
                        "sd_pure_loo_intercept": round(float(loo_intercept.std()), 6),
                        "corr_phat_vs_pure_loo_intercept": round(
                            float(np.corrcoef(ph, loo_intercept)[0, 1]), 4),
                        "mean_phat_pos_items": round(float(ph[yb == 1].mean()), 4),
                        "mean_phat_neg_items": round(float(ph[yb == 0].mean()), 4)})

        out["datasets"][key] = {
            "n": n, "decomposition": dec, "constant_shift_twin": twin,
            "loo_intercept_artefact": art,
            "pooled_agree_C1_vs_constshift": int(agree_c1_const.sum()),
            "pooled_agree_frac": round(float(agree_c1_const.mean()), 4),
            "pooled_acc_deployed": round(R.acc(lab, dep_all), 4),
            "pooled_acc_C1": round(R.acc(lab, pred_c1), 4),
            "pooled_acc_constshift": round(R.acc(lab, pred_const), 4),
        }
        d = out["datasets"][key]
        print(f"[{key}] agree(C1, pure-threshold-shift) = "
              f"{d['pooled_agree_C1_vs_constshift']}/{n} = {d['pooled_agree_frac']:.4f}  "
              f"| acc dep {d['pooled_acc_deployed']:.4f} C1 {d['pooled_acc_C1']:.4f} "
              f"const {d['pooled_acc_constshift']:.4f}", flush=True)
        for r in dec:
            print(f"   fold {r['fold']}: cbar {r['cbar']:+.4f} sd(c) {r['sd_c']:.4f} "
                  f"| shift mean {r['mean_const_part']:+.4f} sd(const) "
                  f"{r['sd_const_part']:.4f} sd(item) {r['sd_item_part']:.4f}",
                  flush=True)
        for r in art:
            print(f"   fold {r['fold']}: logit coef {r['logit_coef_on_log1p_vol']:+.4f} "
                  f"sd(phat) {r['sd_phat']:.6f} sd(pure LOO intercept) "
                  f"{r['sd_pure_loo_intercept']:.6f} corr {r['corr_phat_vs_pure_loo_intercept']:+.4f}",
                  flush=True)

    p = os.path.join(REPO, "scripts/analysis/restrans_pregate_diag_OUT.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
