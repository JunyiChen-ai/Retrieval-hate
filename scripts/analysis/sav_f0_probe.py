"""SAV (C2) F-G1 statistics engine (the deciding cheap gate).

Authority: research-wiki/experiments/exp-sav-f0.md (Rev-2a, APPROVED), §4 F-G1.

Runs, per dataset (MHC=carrying, HateMM=no-harm, MHC_zh=secondary) and per seed 0..4:
  * SAV nearest-centroid head selection on a 20/class few-shot draw (seed-varied), top-k
    swept over {10,20,40}; head-set stability across draws reported as a diagnostic (R1a);
  * a matched-capacity probe g (StandardScaler + L2 LogisticRegressionCV, lambda by 5-fold
    CV in-train only) fit on a stratified 80% probe-train resample (seed-varied), evaluated
    on the full val holdout, for the arms: pooled / SAV / C-pos / C-sparse / U-1 (100,352-d)
    / U-2 (best single head), plus SAV's own majority-vote read-out and an oracle
    (full-train-selected) sanity arm;
  * primary MDL = holdout log-loss bits (clip [1e-6,1-1e-6]); co-primary = capacity-matched
    accuracy; example-level clustered bootstrap (10k draws; per-example cross-seed average
    FIRST; effective n stays n_val) for ΔL / Δacc / Fano-projected gain (R1b/R1c).

Emits a machine-readable JSON verdict (fail-closed: any missing arm => NO_VERDICT).
This module also exposes fit_logreg_probe() imported by sav_f0_guard.py (secondary check).
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav_f0_common as C  # noqa: E402


# --------------------------------------------------------------------------- #
# Matched-capacity probe g: StandardScaler + L2 LogisticRegressionCV           #
#   lambda by 5-fold CV within the probe-train split only; never tuned on val. #
# --------------------------------------------------------------------------- #
def fit_logreg_probe(Xtr, ytr, Xval, seed):
    """Return (proba_pos_on_val [n_val], chosen_lambda). L2 penalty, 5-fold CV lambda."""
    Cs = (1.0 / C.LAMBDAS).tolist()
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=int(seed))
    clf = LogisticRegressionCV(
        Cs=Cs, cv=cv, penalty="l2", solver="lbfgs", scoring="neg_log_loss",
        max_iter=C.PROBE_MAX_ITER, n_jobs=-1, refit=True,
    )
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xval)[:, 1]
    chosen_lambda = float(1.0 / clf.C_[0])
    return proba, chosen_lambda


def score_arm(Xtr, ytr, Xval, yval, seed):
    proba, lam = fit_logreg_probe(Xtr, ytr, Xval, seed)
    bits = C.per_example_bits(proba, yval)               # [n_val]
    pred = (proba >= 0.5).astype(np.int64)
    correct = (pred == yval).astype(np.int64)
    return {
        "bits": bits, "correct": correct,
        "acc": float(correct.mean()), "L_bits": float(bits.sum()),
        "chosen_lambda": lam,
    }


# --------------------------------------------------------------------------- #
# SAV majority-vote read-out (secondary reference, §2 step 3)                  #
# --------------------------------------------------------------------------- #
def sav_majority_vote(head_tr, ytr, head_val, yval, head_idx):
    """Each selected head = nearest-centroid (cosine) classifier; global majority vote.

    Centroids from the probe-train memory (train labels). Returns val accuracy.
    """
    votes = np.zeros((head_val.shape[0], len(head_idx)), dtype=np.int64)
    for j, h in enumerate(head_idx):
        xt = C._l2norm_rows(head_tr[:, h, :].astype(np.float64))
        cents = []
        for c in (0, 1):
            m = ytr == c
            cents.append(xt[m].mean(axis=0) if np.any(m) else np.zeros(xt.shape[1]))
        Cc = C._l2norm_rows(np.stack(cents, 0))
        xv = C._l2norm_rows(head_val[:, h, :].astype(np.float64))
        votes[:, j] = np.argmax(xv @ Cc.T, axis=1)
    pred = (votes.mean(axis=1) >= 0.5).astype(np.int64)
    return float((pred == yval).mean())


# --------------------------------------------------------------------------- #
# Per-dataset F-G1                                                             #
# --------------------------------------------------------------------------- #
def _stratified_indices(labels, frac, rng):
    idx = []
    for c in (0, 1):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        k = max(1, int(round(frac * len(ci))))
        idx.extend(ci[:k].tolist())
    idx = np.asarray(sorted(idx), dtype=np.int64)
    return idx


def _selection_draw(labels, per_class, rng):
    idx = []
    for c in (0, 1):
        ci = np.where(labels == c)[0]
        assert len(ci) >= per_class, "class {} has {} < {} for selection".format(c, len(ci), per_class)
        rng.shuffle(ci)
        idx.extend(ci[:per_class].tolist())
    return np.asarray(sorted(idx), dtype=np.int64)


def run_dataset(dataset):
    print("[probe] loading features for {} ...".format(dataset), flush=True)
    tr = C.load_extracted_split(dataset, "train", with_heads=True)
    va = C.load_extracted_split(dataset, "val", with_heads=True)
    ytr_all = tr["labels"]
    yval = va["labels"]
    n_val = len(yval)

    head_final_tr = tr["head_final"]      # [Ntr,784,128]
    head_span_tr = tr["head_span"]
    head_final_va = va["head_final"]
    head_span_va = va["head_span"]
    pooled_tr, pooled_va = tr["img_pooled"], va["img_pooled"]
    cpos_tr, cpos_va = tr["img_hidden_final"], va["img_hidden_final"]

    # deterministic oracle head selection on the FULL train set (gold train labels; §4 oracle)
    oracle_acc = C.head_nearest_centroid_accuracy(head_final_tr, ytr_all)
    oracle_order = C.rank_heads(oracle_acc)

    # per-example arrays keyed by arm -> [n_val, n_seed] bits/correct
    def _new(): return {"bits": np.zeros((n_val, len(C.SEEDS))), "correct": np.zeros((n_val, len(C.SEEDS)))}
    store = {}
    for a in ["pooled", "C-pos", "U-1"]:
        store[a] = _new()
    for k in C.TOPK_SWEEP:
        for a in ["SAV", "C-sparse", "U-2", "oracle"]:
            store["{}@{}".format(a, k)] = _new()
    mv_acc = {k: [] for k in C.TOPK_SWEEP}                  # SAV majority-vote acc per seed
    lambdas = {}                                            # arm -> [chosen lambda per seed]
    topk_sets = {k: [] for k in C.TOPK_SWEEP}               # per-seed top-k head sets (stability)

    for si, seed in enumerate(C.SEEDS):
        t0 = time.time()
        rng_sel = np.random.default_rng(1000 + seed)
        rng_split = np.random.default_rng(2000 + seed)
        sel = _selection_draw(ytr_all, C.SELECTION_PER_CLASS, rng_sel)
        sel_acc = C.head_nearest_centroid_accuracy(head_final_tr[sel], ytr_all[sel])
        order = C.rank_heads(sel_acc)
        best_head = order[0]
        ptr = _stratified_indices(ytr_all, C.PROBE_TRAIN_FRAC, rng_split)
        ytr = ytr_all[ptr]

        def _fit(name, Xtr, Xval):
            r = score_arm(Xtr, ytr, Xval, yval, seed)
            store[name]["bits"][:, si] = r["bits"]
            store[name]["correct"][:, si] = r["correct"]
            lambdas.setdefault(name, []).append(r["chosen_lambda"])

        _fit("pooled", pooled_tr[ptr], pooled_va)
        _fit("C-pos", cpos_tr[ptr], cpos_va)
        _fit("U-1", head_final_tr[ptr].reshape(len(ptr), -1), head_final_va.reshape(n_val, -1))
        # U-2 (best single head) is k-independent; fit once, replicate into every k slot
        r_u2 = score_arm(head_final_tr[ptr][:, best_head, :], ytr,
                         head_final_va[:, best_head, :], yval, seed)
        lambdas.setdefault("U-2", []).append(r_u2["chosen_lambda"])
        for k in C.TOPK_SWEEP:
            store["U-2@{}".format(k)]["bits"][:, si] = r_u2["bits"]
            store["U-2@{}".format(k)]["correct"][:, si] = r_u2["correct"]

        for k in C.TOPK_SWEEP:
            top = order[:k]
            topk_sets[k].append(set(int(h) for h in top))
            otop = oracle_order[:k]
            _fit("SAV@{}".format(k),
                 head_final_tr[ptr][:, top, :].reshape(len(ptr), -1),
                 head_final_va[:, top, :].reshape(n_val, -1))
            _fit("C-sparse@{}".format(k),
                 head_span_tr[ptr][:, top, :].reshape(len(ptr), -1),
                 head_span_va[:, top, :].reshape(n_val, -1))
            _fit("oracle@{}".format(k),
                 head_final_tr[ptr][:, otop, :].reshape(len(ptr), -1),
                 head_final_va[:, otop, :].reshape(n_val, -1))
            mv_acc[k].append(sav_majority_vote(head_final_tr[ptr], ytr, head_final_va, yval, top))
        print("  [{}] seed {} done in {:.1f}s".format(dataset, seed, time.time() - t0), flush=True)

    # ---- aggregation: seed-average per example, then clustered bootstrap vs pooled ----
    def _compare(arm):
        bp = store["pooled"]["bits"]; ba = store[arm]["bits"]
        cp = store["pooled"]["correct"]; ca = store[arm]["correct"]
        dL_ex = (bp - ba).mean(axis=1)                 # pooled - arm = bits SAVED, per example
        dacc_ex = (ca - cp).mean(axis=1)               # arm - pooled accuracy, per example
        ell_pool = bp.mean(axis=1)                     # seed-avg codelength per example
        ell_arm = ba.mean(axis=1)
        return {
            "acc_mean_arm": float(store[arm]["correct"].mean()),
            "acc_mean_pooled": float(store["pooled"]["correct"].mean()),
            "L_bits_arm_mean": float(store[arm]["bits"].sum(axis=0).mean()),
            "L_bits_pooled_mean": float(store["pooled"]["bits"].sum(axis=0).mean()),
            "deltaL_bits_per_example": C.clustered_bootstrap_mean(dL_ex),
            "delta_acc": C.clustered_bootstrap_mean(dacc_ex),
            "projected_gain": C.clustered_bootstrap_projection(ell_pool, ell_arm),
        }

    results = {"n_train": len(ytr_all), "n_val": n_val,
               "arms_present": True, "per_arm": {}, "chosen_lambda_mean": {}}
    arm_names = ["C-pos", "U-1"] + \
        ["{}@{}".format(a, k) for k in C.TOPK_SWEEP for a in ["SAV", "C-sparse", "U-2", "oracle"]]
    for a in arm_names:
        results["per_arm"][a] = _compare(a)
    for a, lam in lambdas.items():
        results["chosen_lambda_mean"][a] = float(np.mean(lam))
    results["sav_majority_vote_acc"] = {str(k): float(np.mean(v)) for k, v in mv_acc.items()}

    # head-set stability across the 5 seed draws (diagnostic only, R1a)
    stab = {}
    for k in C.TOPK_SWEEP:
        sets = topk_sets[k]
        inter = set.intersection(*sets) if sets else set()
        jac = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = len(sets[i] | sets[j]); jac.append(len(sets[i] & sets[j]) / u if u else 0.0)
        stab[str(k)] = {"intersection_size": len(inter), "mean_pairwise_jaccard": float(np.mean(jac)) if jac else 1.0}
    results["head_set_stability"] = stab
    return results


# --------------------------------------------------------------------------- #
# Decision (fail-closed)                                                       #
# --------------------------------------------------------------------------- #
def _k_pass_mhc(cmp_k):
    dL = cmp_k["deltaL_bits_per_example"]; pg = cmp_k["projected_gain"]
    pass_dL = bool(dL["mean"] > 0.0 and dL["ci_low"] > 0.0)
    pass_pg = bool(pg["mean"] > C.PROJECTED_GAIN_BAR and pg["ci_low"] > 0.0)
    return pass_dL and pass_pg, {"pass_deltaL": pass_dL, "pass_projected_gain": pass_pg,
                                 "projected_gain_bar": C.PROJECTED_GAIN_BAR}


def _noharm_hatemm(cmp_k):
    dL = cmp_k["deltaL_bits_per_example"]; da = cmp_k["delta_acc"]
    ok_dL = bool(dL["ci_high"] >= 0.0)                       # not significantly MORE bits
    ok_acc = bool(da["ci_low"] >= C.HATEMM_NOHARM_DACC)      # Δacc CI not below -0.010
    return ok_dL and ok_acc, {"noharm_deltaL": ok_dL, "noharm_delta_acc": ok_acc,
                              "noharm_dacc_floor": C.HATEMM_NOHARM_DACC}


def decide(ds_results):
    mhc = ds_results.get(C.CARRYING_DATASET)
    ham = ds_results.get(C.NOHARM_DATASET)
    required_arms = ["SAV@{}".format(k) for k in C.TOPK_SWEEP] + ["pooled(implicit)"]
    if mhc is None or ham is None:
        return {"verdict": "NO_VERDICT_MISSING_ARM", "reason": "carrying/no-harm dataset absent"}
    for k in C.TOPK_SWEEP:
        if "SAV@{}".format(k) not in mhc["per_arm"] or "SAV@{}".format(k) not in ham["per_arm"]:
            return {"verdict": "NO_VERDICT_MISSING_ARM", "reason": "SAV@{} missing".format(k)}
    per_k = {}
    proceed_k = []
    for k in C.TOPK_SWEEP:
        mp, mp_d = _k_pass_mhc(mhc["per_arm"]["SAV@{}".format(k)])
        nh, nh_d = _noharm_hatemm(ham["per_arm"]["SAV@{}".format(k)])
        per_k[str(k)] = {"mhc_pass": mp, "mhc_detail": mp_d, "hatemm_noharm": nh, "hatemm_detail": nh_d,
                         "proceed": bool(mp and nh)}
        if mp and nh:
            proceed_k.append(k)
    # oracle sanity (cheapest pre-check): best-k oracle projected gain on MHC
    oracle_pg = max(mhc["per_arm"]["oracle@{}".format(k)]["projected_gain"]["mean"] for k in C.TOPK_SWEEP)
    verdict = "PROCEED_TO_FG2" if proceed_k else "KILL"
    return {
        "verdict": verdict,
        "proceed_k": proceed_k,
        "per_k": per_k,
        "oracle_max_projected_gain_mhc": float(oracle_pg),
        "oracle_below_bar": bool(oracle_pg < C.PROJECTED_GAIN_BASE),
        "required_arms": required_arms,
    }


def main():
    ap = argparse.ArgumentParser(description="SAV F-G1 statistics engine.")
    ap.add_argument("--datasets", type=str, default=",".join(C.DATASETS))
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    ds_results = {}
    for ds in datasets:
        ds_results[ds] = run_dataset(ds)

    decision = decide(ds_results)
    payload = {
        "schema": "sav_f0_probe_verdict_v1",
        "status": "COMPLETE",
        "authority": "research-wiki/experiments/exp-sav-f0.md (Rev-2a) F-G1",
        "config": {
            "seeds": C.SEEDS, "selection_per_class": C.SELECTION_PER_CLASS,
            "topk_sweep": C.TOPK_SWEEP, "probe_train_frac": C.PROBE_TRAIN_FRAC,
            "cv_folds": C.CV_FOLDS, "lambdas": C.LAMBDAS.tolist(),
            "bootstrap_draws": C.BOOTSTRAP_DRAWS, "prob_clip": C.PROB_CLIP,
            "projected_gain_base": C.PROJECTED_GAIN_BASE, "noise_band_acc": C.NOISE_BAND_ACC,
            "projected_gain_bar": C.PROJECTED_GAIN_BAR, "hatemm_noharm_dacc": C.HATEMM_NOHARM_DACC,
            "probe_stream": C.PROBE_STREAM, "probe": "StandardScaler+L2-LogisticRegressionCV",
            "carrying_dataset": C.CARRYING_DATASET, "noharm_dataset": C.NOHARM_DATASET,
            "secondary_dataset": C.SECONDARY_DATASET,
        },
        "datasets": ds_results,
        "decision": decision,
        "verdict": decision["verdict"],
    }
    C.atomic_write_json(C.probe_verdict_path(), payload)
    print("[probe] VERDICT = {} -> {}".format(decision["verdict"], C.probe_verdict_path()), flush=True)


if __name__ == "__main__":
    main()
