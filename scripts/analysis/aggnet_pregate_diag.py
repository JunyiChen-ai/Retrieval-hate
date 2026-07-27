#!/usr/bin/env python
"""
aggnet_pregate_diag.py -- POST-HOC diagnostic for the AGGNET (C3) $0 pregate.
ADDS NO ARM AND PROMOTES NOTHING (the RESTRANS restrans_pregate_diag.py precedent).

It answers one question the main OUT does not carry: what is the ORACLE CEILING of
C3's family, i.e. how many of the deployed errors are even reachable by a
non-negative reweighting of the deployed top-20?

    A non-negative weighting can change sign(v) iff the top-20 contains BOTH classes
    with at least one same-signed and one opposite-signed s_i*cos_i term. Since every
    cos_i > 0 in this space, that is exactly "the top-20 is class-mixed".

So the family's oracle is: every deployed-wrong item with a class-mixed top-20 is
fixable, and no other item is. Reported per dataset per space, alongside the F95
adjudication oracle (+0.0726/+0.0535/+0.0893, VGA 3) so the two ceilings are
directly comparable.

Also reports, for the pathology population, the deployed rank profile of the correct
analogue -- the population LITSWEEP6 0(v) says any live candidate must act on.

TEST-SPLIT CONTACT: NONE. train_*.pt only. CPU, seconds, zero GPU/SLURM/Modal.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))
import mechfix_ops as M          # noqa: E402
import mechnov_pairverify as PV  # noqa: E402
import aggnet_pregate as A       # noqa: E402


def main():
    import torch
    torch.set_num_threads(4)
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    OUT = {"note": "POST-HOC. Adds no arm, promotes nothing. Oracle ceilings only.",
           "aggnet_script_sha256": A.sha256_of(
               os.path.join(REPO, "scripts/analysis/aggnet_pregate.py")),
           "datasets": {}}
    for key, cfg in A.DATASETS.items():
        ids, img, txt, lab = PV.load_cache(cfg["cache_dir"], "train", cfg["model"])
        n = len(ids)
        skf = StratifiedKFold(n_splits=A.K_FOLDS, shuffle=True, random_state=A.FOLD_SEED)
        folds = list(skf.split(np.zeros((n, 1)), lab))
        OUT["datasets"][key] = {"n": n, "spaces": {}}
        for space in A.SPACES:
            X = PV.build_space(img, txt, space)
            dep = np.full(n, -1, dtype=int)
            mixed = np.zeros(n, dtype=bool)
            sc_rank = np.full(n, -1, dtype=int)
            for fit_idx, ho_idx in folds:
                fit_idx, ho_idx = np.asarray(fit_idx), np.asarray(ho_idx)
                Xb, yb = X[fit_idx], lab[fit_idx]
                _, dp, dI, _ = M.deployed_vote(Xb, yb, X[ho_idx], topk=A.TOPK)
                dep[ho_idx] = dp
                nl = yb[dI]
                mixed[ho_idx] = nl.min(1) != nl.max(1)
                S = X[ho_idx] @ Xb.T
                bl = yb[np.argsort(-S, axis=1, kind="stable")]
                for r_, q_ in enumerate(ho_idx):
                    hit = np.flatnonzero(bl[r_] == lab[q_])
                    sc_rank[q_] = int(hit[0]) + 1 if len(hit) else 10 ** 6
            wrong = dep != lab
            patho = wrong & (sc_rank <= A.PATHOLOGY_RANK) & (sc_rank > 0)
            reach = wrong & mixed
            e = {"acc_deployed": round(float((dep == lab).mean()), 4),
                 "n_wrong": int(wrong.sum()),
                 "n_class_mixed": int(mixed.sum()),
                 "frac_class_mixed": round(float(mixed.mean()), 4),
                 "n_wrong_and_mixed": int(reach.sum()),
                 "frac_wrong_reachable": round(float(reach.sum() / max(wrong.sum(), 1)), 4),
                 "ORACLE_dacc_C3_family": round(float(reach.sum() / n), 4),
                 "n_pathology": int(patho.sum()),
                 "n_pathology_mixed": int((patho & mixed).sum()),
                 "ORACLE_dacc_pathology_only": round(float((patho & mixed).sum() / n), 4)}
            OUT["datasets"][key]["spaces"][space] = e
            print(f"{key:7s} {space:6s} floor {e['acc_deployed']:.4f} wrong {e['n_wrong']:3d} "
                  f"mixed {e['frac_class_mixed']:.4f} reachable {e['n_wrong_and_mixed']:3d} "
                  f"({e['frac_wrong_reachable']:.4f} of errors) "
                  f"ORACLE dacc {e['ORACLE_dacc_C3_family']:+.4f}", flush=True)
    json.dump(OUT, open(os.path.join(REPO, "scripts/analysis/aggnet_pregate_diag_OUT.json"),
                        "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
