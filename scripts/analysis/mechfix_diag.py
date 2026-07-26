#!/usr/bin/env python
"""
mechfix_diag.py -- mechanism diagnostics for the MECHFIX pregate.

Pure measurement on the SAME heads and the SAME frozen operators as
mechfix_run.py. Adds no arm, changes no config, selects nothing. It answers the
"why" questions the §4 result tables raise:

  * T1  -- is the class-balanced decision actually a DIFFERENT classifier from the
           deployed one, or does the cone-collapsed geometry make them the same
           statistic? (independent float64 numpy re-implementation as a control)
  * T2a -- how much dynamic range does the bank-side hubness term r(x) have, and how
           many retrieved SETS change without a single decision changing?
  * T3  -- is the length axis actually excised? Measured directly as the
           length-organisation of retrieval, rho(query volume, median volume of its
           top-20 bank neighbours), before vs after. Plus the variance share of the
           fitted direction in the key covariance.
  * T2b -- what does whitening do to the saturated cosine, and to the length
           organisation?

CPU ONLY. Final-epoch protocol, all 3 seeds, all 3 datasets.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts", "analysis"))

import faiss  # noqa: E402
import mechfix_ops as M  # noqa: E402
import mechfix_run as R  # noqa: E402


def t1_numpy_control(Etr, tr_lab, Ete):
    """Independent float64 numpy re-implementation of T1 (no faiss) as a coding control."""
    B = Etr.astype("float64"); B /= np.linalg.norm(B, axis=1, keepdims=True)
    Q = Ete.astype("float64"); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    SIM = Q @ B.T
    w = np.arange(1, M.T1_K_PER_CLASS + 1)[::-1].astype("float64")
    sc = {}
    for cl in (0, 1):
        idx = np.flatnonzero(tr_lab == cl)
        part = np.sort(SIM[:, idx], axis=1)[:, ::-1][:, :M.T1_K_PER_CLASS]
        sc[cl] = (part * w).sum(1) / w.sum()
    mg = sc[1] - sc[0]
    return mg, (mg >= 0).astype(int), sc


def main():
    faiss.omp_set_num_threads(8)
    OUT = {"meta": {"cpu_only": True, "gpu_jobs": 0, "protocol": "final-epoch, 3 seeds",
                    "ops_sha256": __import__("hashlib").sha256(
                        open(os.path.join(REPO, "scripts/analysis/mechfix_ops.py"),
                             "rb").read()).hexdigest()}}
    for key in ("hatemm", "zh", "en"):
        c = R.CFG[key]()
        tr_ids, tr_img, tr_txt, tr_lab = R.load_cache(c["cache_dir"], "train", c["model"])
        te_ids, te_img, te_txt, te_lab = R.load_cache(c["cache_dir"], "test_seen", c["model"])
        gt = R.load_gt_text(c["gt_dir"])
        trv = R.volume_scalar(tr_ids, gt, c["vol"])
        tev = R.volume_scalar(te_ids, gt, c["vol"])
        per = {}
        for s in (0, 1, 2):
            ep = c["protocols"]["final"][s]
            if "ckpt_dirs" in c:
                ck = sorted(glob.glob(os.path.join(glob.escape(c["ckpt_dirs"][s]), "ckpt",
                                                   f"epoch_model_{ep}_*.pt")))[0]
            else:
                ck = c["ckpt_files"][s]
            m = R.build_head(R.load_sd(ck))
            Etr, Ete = R.embed(m, tr_img, tr_txt), R.embed(m, te_img, te_txt)

            v, p, I, S = M.deployed_vote(Etr, tr_lab, Ete)

            def lenorg(Inb):
                return float(spearmanr(tev, np.median(trv[Inb], axis=1))[0])

            d = {"epoch": int(ep),
                 "deployed_top1sim_mean": round(float(S[:, 0].mean()), 6),
                 "deployed_top1sim_median": round(float(np.median(S[:, 0])), 6),
                 "deployed_length_organisation_rho": round(lenorg(I), 4)}

            # ---- T1
            mg, p1, _, _ = M.t1_class_balanced(Etr, tr_lab, Ete)
            mg_r, p1_r, sc = t1_numpy_control(Etr, tr_lab, Ete)
            d["T1_faiss_vs_numpy_pred_agree"] = f"{int((p1 == p1_r).sum())}/{len(p1)}"
            d["T1_faiss_vs_numpy_max_margin_diff"] = float(f"{np.abs(mg - mg_r).max():.3e}")
            d["T1_pred_equals_deployed"] = f"{int((p1 == p).sum())}/{len(p)}"
            d["T1_margin_range"] = [round(float(mg.min()), 6), round(float(mg.max()), 6)]
            d["T1_class1_top10_score_median"] = round(float(np.median(sc[1])), 6)
            d["T1_class0_top10_score_median"] = round(float(np.median(sc[0])), 6)
            d["T1_pred_equals_top1_label"] = \
                f"{int((p1 == tr_lab[I[:, 0]]).sum())}/{len(p1)}"
            d["deployed_pred_equals_top1_label"] = \
                f"{int((p == tr_lab[I[:, 0]]).sum())}/{len(p)}"

            # ---- T2a
            r = M.bank_hubness(Etr)
            v2, p2, I2, S2 = M.t2a_csls(Etr, tr_lab, Ete, r)
            d["T2a_r_percentiles_0_1_25_50_75_99_100"] = [
                round(float(x), 6) for x in np.percentile(r, [0, 1, 25, 50, 75, 99, 100])]
            d["T2a_r_iqr"] = float(f"{np.percentile(r, 75) - np.percentile(r, 25):.3e}")
            d["T2a_n_items_topset_changed"] = int(sum(
                1 for i in range(len(te_ids)) if set(I[i]) != set(I2[i])))
            d["T2a_n_pred_changed"] = int((p2 != p).sum())
            d["T2a_length_organisation_rho"] = round(lenorg(I2), 4)

            # ---- T2b
            mu, W, sh, ev = M.fit_whitener(Etr)
            Bw, Qw = M.apply_whitener(Etr, mu, W), M.apply_whitener(Ete, mu, W)
            v3, p3, I3, S3 = M.deployed_vote(Bw, tr_lab, Qw)
            d["T2b_lw_shrinkage"] = round(float(sh), 6)
            d["T2b_eig_min"] = float(f"{ev.min():.3e}")
            d["T2b_eig_max"] = float(f"{ev.max():.3e}")
            d["T2b_eig_condition"] = float(f"{ev.max() / ev.min():.3e}")
            d["T2b_top1sim_mean"] = round(float(S3[:, 0].mean()), 6)
            d["T2b_n_pred_changed"] = int((p3 != p).sum())
            d["T2b_length_organisation_rho"] = round(lenorg(I3), 4)

            # ---- T3
            vh = M.fit_length_direction(Etr, np.log1p(trv))
            Bn = M._norm32(Etr).astype("float64")
            proj = Bn @ vh
            ll = np.log1p(trv)
            d["T3_pearson_proj_loglen_train"] = round(float(pearsonr(proj, ll)[0]), 4)
            d["T3_spearman_proj_loglen_train"] = round(float(spearmanr(proj, ll)[0]), 4)
            d["T3_variance_share_of_direction"] = float(
                f"{np.var(Bn @ vh) / np.var(Bn, axis=0).sum():.3e}")
            Ep_tr, Ep_te = M.remove_direction(Etr, vh), M.remove_direction(Ete, vh)
            projr = M._norm32(Ep_tr).astype("float64") @ vh
            d["T3_residual_abs_proj_max"] = float(f"{np.abs(projr).max():.3e}")
            d["T3_pearson_after_removal"] = round(
                float(pearsonr(projr, ll)[0]) if np.std(projr) > 0 else 0.0, 4)
            v4, p4, I4, S4 = M.deployed_vote(Ep_tr, tr_lab, Ep_te)
            d["T3_n_items_topset_changed"] = int(sum(
                1 for i in range(len(te_ids)) if set(I[i]) != set(I4[i])))
            d["T3_n_pred_changed"] = int((p4 != p).sum())
            d["T3_length_organisation_rho"] = round(lenorg(I4), 4)
            per[f"s{s}"] = d
            print(f"[{c['ds']}] s{s} ep{ep}: T1==dep {d['T1_pred_equals_deployed']}, "
                  f"T2a set-changed {d['T2a_n_items_topset_changed']} pred-changed "
                  f"{d['T2a_n_pred_changed']}, T3 rho {d['deployed_length_organisation_rho']}"
                  f"->{d['T3_length_organisation_rho']}, T2b rho "
                  f"{d['T2b_length_organisation_rho']} top1sim "
                  f"{d['deployed_top1sim_mean']}->{d['T2b_top1sim_mean']}", flush=True)
        OUT[key] = {"n_train": len(tr_ids), "n_test": len(te_ids), "vol_mode": c["vol"],
                    "per_seed": per}
    out_path = os.path.join(REPO, "scripts/analysis/mechfix_diag_OUT.json")
    json.dump(OUT, open(out_path, "w"), indent=1, default=str)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
