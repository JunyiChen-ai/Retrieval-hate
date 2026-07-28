#!/usr/bin/env python
"""headspace_report.py -- REPORTING ONLY for refine-logs/HEADSPACE_TRANSFER_PREGATE.md.

Merges the per-seed arena JSONs and the fidelity JSON, re-reads every number at 4 dp and
emits the tables the record quotes.  Computes no arm, fits nothing, and is not part of
the freeze.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(REPO, "scripts/analysis"))

RAW = {  # the banked RAW-arena data this record is testing the transfer of
    "VSW_pow": 0.0255, "VSW_exp": 0.0255, "VSW_lin": 0.0188,
    "F95_mlp_max": -0.0040, "F95_mlp_mean3": 0.0054, "F95_cos_shape": -0.0417,
}
K_HST1_TRANSFER = 0.0128      # half of +0.0255
K_HST1_RAW = 0.0255


def m(xs):
    return round(float(np.mean(xs)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arena", nargs="+", required=True)
    ap.add_argument("--fidelity", required=True)
    ap.add_argument("--perm", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = [json.load(open(p)) for p in a.arena]
    F = json.load(open(a.fidelity))
    seeds = [x["meta"]["seed"] for x in A]

    # ---------------------------------------------------------------- ARENA-2
    posr = A[0]["result"]["posrate_bank"]
    maj = max(posr, 1 - posr)
    arena2 = {"pos_rate_bank": posr, "majority_rate": round(maj, 4),
              "lo": round(maj + 0.02, 4), "hi": 0.98,
              "head_deployed_acc_per_seed": [x["result"]["acc_deployed"] for x in A],
              "head_deployed_acc_3seedmean": m([x["result"]["acc_deployed"] for x in A]),
              "raw_deployed_acc": A[0]["membership"]["raw_deployed_acc"]}
    arena2["PASS"] = bool(all(maj + 0.02 <= x <= 0.98
                              for x in arena2["head_deployed_acc_per_seed"]))

    # ---------------------------------------------------------------- PARITY
    parity = {"n_gates_total": sum(x["parity_lambda0"]["n_gates"] for x in A),
              "n_pass_total": sum(x["parity_lambda0"]["n_pass"] for x in A)}
    parity["PASS"] = parity["n_gates_total"] == parity["n_pass_total"]

    # ---------------------------------------------------------------- K-HST-1
    arms = {}
    names = sorted(set().union(*[set(x["result"]["arms"]) for x in A])
                   | set().union(*[set(x["extra_arms"]) for x in A]))
    for nm in names:
        cells = [(x["result"]["arms"].get(nm) or x["extra_arms"].get(nm)) for x in A]
        if any(c is None for c in cells):
            continue
        row = {"dacc_per_seed": [c["dacc"] for c in cells],
               "dacc_3seedmean": m([c["dacc"] for c in cells]),
               "dmF1_3seedmean": m([c["mF1"] for c in cells])
               - m([x["result"]["mF1_deployed"] for x in A]),
               "acc_3seedmean": m([c["acc"] for c in cells]),
               "foldsigns_per_seed": [c["foldsigns"] for c in cells],
               "folddeltas_3seedmean": [round(float(np.mean(
                   [c["folddeltas"][f] for c in cells])), 4) for f in range(5)],
               "fixed_per_seed": [c["fixed"] for c in cells],
               "broke_per_seed": [c["broke"] for c in cells],
               "net_per_seed": [c["net"] for c in cells],
               "changed_per_seed": [c["changed"] for c in cells],
               "posrate_per_seed": [c["posrate"] for c in cells],
               "n_seeds_strictly_positive": int(sum(1 for c in cells if c["dacc"] > 0))}
        row["dmF1_3seedmean"] = round(row["dmF1_3seedmean"], 4)
        if "lambda_per_fold" in cells[0]:
            row["lambda_star_per_seed"] = [
                [(l["lam"] if not l["lam_is_inf"] else "inf")
                 for l in c["lambda_per_fold"]] for c in cells]
        if nm in RAW:
            row["raw_arena_dacc"] = RAW[nm]
            row["transfer_ratio"] = (round(row["dacc_3seedmean"] / RAW[nm], 4)
                                     if RAW[nm] else None)
        arms[nm] = row

    prim = arms["VSW_pow"]
    sm = prim["dacc_3seedmean"]
    nonneg = sum(1 for v in prim["folddeltas_3seedmean"] if v >= 0)
    khst1 = {"delta_head_3seedmean": sm, "raw_effect": K_HST1_RAW,
             "transfer_ratio": round(sm / K_HST1_RAW, 4),
             "bar_TRANSFERS": K_HST1_TRANSFER,
             "n_folds_seedmean_nonneg": nonneg,
             "n_seeds_strictly_positive": prim["n_seeds_strictly_positive"],
             "B_fid": F["gate"]["B_fid_abs_3seedmean"],
             "inside_instrument_band": bool(abs(sm) < F["gate"]["B_fid_abs_3seedmean"])}
    if sm >= K_HST1_TRANSFER and nonneg >= 3 and prim["n_seeds_strictly_positive"] >= 2:
        khst1["VERDICT"] = "TRANSFERS"
    elif sm > 0.0:
        khst1["VERDICT"] = "PARTIAL"
    else:
        khst1["VERDICT"] = "DOES NOT TRANSFER"

    # ---------------------------------------------------------------- K-HST-2 / 3
    mem_keys = list(A[0]["membership"])
    khst2 = {k: (m([x["membership"][k] for x in A])
                 if isinstance(A[0]["membership"][k], float)
                 else [x["membership"][k] for x in A]) for k in mem_keys}
    khst2["reading_rule_threshold_overlap"] = 10.0
    khst2["SUBSTANTIAL_MEMBERSHIP_CHANGE"] = bool(
        khst2["mean_top20_overlap"] < 10.0)

    c1keys = list(A[0]["control1"][0])
    khst3 = {k: m([c[k] for x in A for c in x["control1"]]) for k in c1keys}
    khst3["d_auc_per_seed"] = [m([c["d_auc_vs_cos"] for c in x["control1"]]) for x in A]
    khst3["raw_arena_d_auc_hatemm"] = 0.1572
    khst3["VERIFIER_UNINFORMATIVE"] = bool(khst3["d_auc_vs_cos"] <= 0)

    # ---------------------------------------------------------------- degeneracy
    deg = {"A_agree_threshold_shift": [x["result"]["degeneracy"]["A_agree_threshold_shift"] for x in A],
           "A_FIRES": [x["result"]["degeneracy"]["A_FIRES"] for x in A],
           "B_agree_fixk_max": [x["result"]["degeneracy"]["B_agree_fixk_max"] for x in A],
           "B_argmax_k": [x["result"]["degeneracy"]["B_argmax_k"] for x in A],
           "B_FIRES": [x["result"]["degeneracy"]["B_FIRES"] for x in A],
           "D_agree_ctrl_cos": [x["result"]["degeneracy"]["D_agree_ctrl_cos"] for x in A],
           "agree_deployed": [x["result"]["degeneracy"]["agree_deployed"] for x in A],
           "class_balance_primary_deviation": [
               x["result"]["class_balance"]["primary_deviation"] for x in A],
           "class_balance_PASS": [x["result"]["class_balance"]["PRIMARY_PASS"] for x in A]}

    # ---------------------------------------------------------------- lambda curve
    lam = A[0]["result"]["curve"]["pow"]
    curve = []
    for i, r0 in enumerate(lam):
        cs = [x["result"]["curve"]["pow"][i] for x in A]
        curve.append({"lam": ("inf" if r0["lam_is_inf"] else r0["lam"]),
                      "dacc_3seedmean": m([c["dacc"] for c in cs]),
                      "changed_3seedmean": m([c["changed"] for c in cs]),
                      "fixed_3seedmean": m([c["fixed"] for c in cs]),
                      "broke_3seedmean": m([c["broke"] for c in cs]),
                      "net_3seedmean": m([c["net"] for c in cs])})

    OUT = {"seeds": seeds, "arena_files": a.arena, "fidelity_file": a.fidelity,
           "GATE_FID": F["gate"], "GATE_ARENA2": arena2, "PARITY_lambda0": parity,
           "K_HST_1": khst1, "K_HST_2_membership": khst2,
           "K_HST_3_verifier": khst3, "degeneracy": deg,
           "deployed_floor": {
               "head_acc_per_seed": [x["result"]["acc_deployed"] for x in A],
               "head_mF1_per_seed": [x["result"]["mF1_deployed"] for x in A],
               "head_fold_acc_per_seed": [x["result"]["fold_acc_deployed"] for x in A],
               "raw_acc": A[0]["membership"]["raw_deployed_acc"]},
           "arms": arms, "lambda_curve_pow_3seedmean": curve}

    if a.perm and os.path.exists(a.perm):
        Pm = json.load(open(a.perm))
        obs = A[0]["result"]["arms"]["VSW_pow"]["dacc"]
        nulls = [d["arms"]["VSW_pow"]["dacc"] for d in Pm["draws"]]
        OUT["permutation_null_seed0"] = {
            "n_draws": len(nulls), "observed_dacc_seed0": obs,
            "null_mean": round(float(np.mean(nulls)), 4),
            "null_sd": round(float(np.std(nulls)), 4),
            "null_max": round(float(np.max(nulls)), 4),
            "frac_null_ge_zero": round(float(np.mean(np.array(nulls) >= 0)), 4),
            "n_null_ge_observed": int(np.sum(np.array(nulls) >= obs)),
            "p": round((1 + int(np.sum(np.array(nulls) >= obs))) / (len(nulls) + 1), 4)}

    tmp = a.out + ".tmp"
    json.dump(OUT, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(json.dumps({"GATE_FID": OUT["GATE_FID"], "GATE_ARENA2": arena2,
                      "PARITY": parity, "K_HST_1": khst1,
                      "K_HST_2": khst2, "K_HST_3": khst3}, indent=1))


if __name__ == "__main__":
    main()
