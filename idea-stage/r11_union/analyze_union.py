#!/usr/bin/env python
"""R11-UNION -- read-out for the union-purchase pilot.

Frozen design: idea-stage/R11_UNION_FREEZE.md sections 4 and 5.

Every arm's macro-F1 -- trained arms and derived (decision-level) arms alike --
is recomputed from the per-item head logits dumped by --dump_head_scores, so the
trained and derived arms go through exactly the same metric code.  A belt
requires the recomputed number to match the trainlog for every trained arm.

Epoch selection is the project protocol, computed from the trainlog dev curve:
  P1  epoch = argmax_{e>=5} dev macro-F1 (ties -> earliest)
  P2  epoch = 29
Each head selects its OWN epoch; a derived arm combines its parents at the
parents' own selected epochs.

Nothing in this file selects anything on test.  The weight of WAVG, the
reliability table of SEL and the lambda of the anchor arms are all fitted on
dev, pooled over seeds, per dataset and per protocol.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
from analyze_audit import parse, select_epoch, macro_f1_from_cm  # noqa: E402

BAR = 0.005
HARM = -0.002
N_BOOT = 20000
BOOT_SEED = 20260817
PROTOCOLS = ["P1", "P2"]
W_GRID = [round(0.05 * k, 2) for k in range(21)]      # 0.00 .. 1.00
N_BUCKETS = 3
A_CLIP = (0.51, 0.99)

TRAINED = ["A0", "LL", "CAT", "MC",
           "ANCA_l01", "ANCA_l03", "ANCA_l10",
           "ANCL_l01", "ANCL_l03", "ANCL_l10",
           "LBL_l01", "LBL_l03", "LBL_l10"]
LAMBDA_TAGS = ["l01", "l03", "l10"]
LAMBDA_VAL = {"l01": 0.1, "l03": 0.3, "l10": 1.0}


def macro_f1_from_pred(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return macro_f1_from_cm(tp, fp, fn, tn)


def load_scores(path):
    """-> {(split, epoch): (ids, labels, logits)}"""
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[(d["split"], int(d["epoch"]))] = (
                d["ids"], np.asarray(d["labels"], dtype=int),
                np.asarray(d["logits"], dtype=np.float64))
    return out


def paired_bootstrap_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(means.std(ddof=1))


def fit_reliability(z_dev, y_dev):
    """Pooled-over-seeds confidence-bucket reliability of one head on dev.

    Returns (edges, weights) where edges are the two interior tercile cuts of the
    pooled |z| and weights[b] = log(a_b / (1 - a_b)) with a_b the Laplace-smoothed
    accuracy of that bucket.
    """
    z = np.concatenate(z_dev)
    y = np.concatenate(y_dev)
    c = np.abs(z)
    edges = np.percentile(c, [100.0 / N_BUCKETS, 200.0 / N_BUCKETS])
    b = np.digitize(c, edges)
    correct = ((z > 0).astype(int) == y)
    w = []
    acc = []
    for k in range(N_BUCKETS):
        m = (b == k)
        a = (int(correct[m].sum()) + 1) / (int(m.sum()) + 2)
        a = float(np.clip(a, *A_CLIP))
        acc.append(a)
        w.append(float(np.log(a / (1.0 - a))))
    return edges, np.asarray(w), acc


def sel_predict(z_cat, z_ll, edges_c, w_c, edges_l, w_l):
    bc = np.digitize(np.abs(z_cat), edges_c)
    bl = np.digitize(np.abs(z_ll), edges_l)
    s = np.sign(z_cat) * w_c[bc] + np.sign(z_ll) * w_l[bl]
    pred = (s > 0).astype(int)
    tie = (s == 0)
    pred[tie] = (z_cat[tie] >= 0).astype(int)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--seeds_b", required=True, help="paired seeds of the CATB arm")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ds = a.dataset
    seeds = [int(s) for s in a.seeds.split(",")]
    seeds_b = [int(s) for s in a.seeds_b.split(",")]
    assert len(seeds) == len(seeds_b)
    rng = np.random.default_rng(BOOT_SEED)

    # ---------------- load every trained run ----------------
    Z = {}          # (arm, seed_index, protocol, split) -> (y, z)
    belt = []
    ep_sel = {}
    ids_ref = {}    # split -> id order; every run must agree or the pairing is invalid
    for arm, slist in [(t, seeds) for t in TRAINED] + [("CATB", seeds_b)]:
        for si, s in enumerate(slist):
            tl = os.path.join(a.logdir, "%s_%s_s%d.trainlog" % (ds, arm, s))
            sc = os.path.join(a.logdir, "%s_%s_s%d.scores.jsonl" % (ds, arm, s))
            r, err = parse(tl, ds)
            if r is None:
                raise SystemExit("HALT: %s -> %s" % (tl, err))
            sd = load_scores(sc)
            for prot in PROTOCOLS:
                e = select_epoch(r["dev"], prot)
                ep_sel[(arm, si, prot)] = e
                for split in ("dev", "test"):
                    ids, y, z = sd[(split, e)]
                    if split not in ids_ref:
                        ids_ref[split] = ids
                    elif ids != ids_ref[split]:
                        raise SystemExit(
                            "HALT: %s id order differs (%s s%d) -- per-item pairing "
                            "across arms would be wrong" % (split, arm, s))
                    Z[(arm, si, prot, split)] = (y, z)
                belt.append(abs(macro_f1_from_pred(Z[(arm, si, prot, "test")][0],
                                                   (Z[(arm, si, prot, "test")][1] >= 0).astype(int))
                                - r["test"][e]["macro_f1"]))
    belt_max = float(max(belt))
    print("BELT max |macroF1(dumped logits) - macroF1(trainlog)| = %.2e" % belt_max)
    if belt_max > 1e-4:
        raise SystemExit("HALT: dumped logits do not reproduce the logged metric")

    res = {"what": "R11-UNION read-out", "freeze": "idea-stage/R11_UNION_FREEZE.md",
           "dataset": ds, "seeds": seeds, "seeds_b": seeds_b,
           "n_boot": N_BOOT, "bootstrap_rng_seed": BOOT_SEED,
           "bar": BAR, "harmless_bar": HARM,
           "belt_max_abs_macro_f1_diff": belt_max,
           "dev_fits": {}, "arm_means": {}, "contrasts": {}, "verdict": {}}

    # ---------------- derived arms ----------------
    # per (protocol, split) a dict arm -> list over seeds of predictions
    PRED = {}
    for prot in PROTOCOLS:
        for split in ("dev", "test"):
            for arm in TRAINED + ["CATB"]:
                PRED[(arm, prot, split)] = [
                    (Z[(arm, si, prot, split)][1] >= 0).astype(int)
                    for si in range(len(seeds))]

    def zs(arm, prot, split):
        return [Z[(arm, si, prot, split)][1] for si in range(len(seeds))]

    def ys(prot, split):
        return [Z[("CAT", si, prot, split)][0] for si in range(len(seeds))]

    for prot in PROTOCOLS:
        # --- AVG and ECTL: fixed 0.5/0.5, nothing fitted ---
        for name, (p, q) in [("AVG", ("CAT", "LL")), ("ECTL", ("CAT", "CATB"))]:
            for split in ("dev", "test"):
                zp, zq = zs(p, prot, split), zs(q, prot, split)
                PRED[(name, prot, split)] = [
                    ((0.5 * zp[i] + 0.5 * zq[i]) >= 0).astype(int)
                    for i in range(len(seeds))]

        # --- WAVG: convex weight chosen on pooled dev ---
        zc_d, zl_d, y_d = zs("CAT", prot, "dev"), zs("LL", prot, "dev"), ys(prot, "dev")
        best_w, best_v = None, -1.0
        curve = []
        for w in W_GRID:
            v = float(np.mean([macro_f1_from_pred(
                y_d[i], ((w * zc_d[i] + (1 - w) * zl_d[i]) >= 0).astype(int))
                for i in range(len(seeds))]))
            curve.append([w, v])
            if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and best_w is not None
                                      and abs(w - 0.5) < abs(best_w - 0.5)):
                best_w, best_v = w, v
        res["dev_fits"]["%s/WAVG_w" % prot] = {"w": best_w, "dev_mean_macro_f1": best_v,
                                               "grid": curve}
        for split in ("dev", "test"):
            zc, zl = zs("CAT", prot, split), zs("LL", prot, split)
            PRED[("WAVG", prot, split)] = [
                ((best_w * zc[i] + (1 - best_w) * zl[i]) >= 0).astype(int)
                for i in range(len(seeds))]

        # --- SEL: reliability table fitted on pooled dev ---
        ec, wc, ac = fit_reliability(zs("CAT", prot, "dev"), ys(prot, "dev"))
        el, wl, al = fit_reliability(zs("LL", prot, "dev"), ys(prot, "dev"))
        res["dev_fits"]["%s/SEL_table" % prot] = {
            "CAT": {"edges": ec.tolist(), "bucket_acc": ac, "logodds": wc.tolist()},
            "LL": {"edges": el.tolist(), "bucket_acc": al, "logodds": wl.tolist()}}
        for split in ("dev", "test"):
            zc, zl = zs("CAT", prot, split), zs("LL", prot, split)
            PRED[("SEL", prot, split)] = [
                sel_predict(zc[i], zl[i], ec, wc, el, wl) for i in range(len(seeds))]

        # --- anchor families: lambda chosen on pooled dev ---
        for fam, pre in [("ANCA", "ANCA"), ("ANCL", "ANCL"), ("LBL", "LBL")]:
            best_t, best_v = None, -1.0
            curve = []
            for t in LAMBDA_TAGS:
                v = float(np.mean([macro_f1_from_pred(
                    Z[("%s_%s" % (pre, t), i, prot, "dev")][0],
                    PRED[("%s_%s" % (pre, t), prot, "dev")][i])
                    for i in range(len(seeds))]))
                curve.append([LAMBDA_VAL[t], v])
                if v > best_v:
                    best_t, best_v = t, v
            res["dev_fits"]["%s/%s_lambda" % (prot, fam)] = {
                "tag": best_t, "lambda": LAMBDA_VAL[best_t],
                "dev_mean_macro_f1": best_v, "grid": curve}
            for split in ("dev", "test"):
                PRED[(fam, prot, split)] = PRED[("%s_%s" % (pre, best_t), prot, split)]

    ALL_ARMS = TRAINED + ["CATB", "AVG", "WAVG", "SEL", "ECTL", "ANCA", "ANCL", "LBL"]

    # ---------------- macro-F1 per arm ----------------
    F = {}
    for prot in PROTOCOLS:
        for split in ("dev", "test"):
            for arm in ALL_ARMS:
                y = ys(prot, split)
                v = np.asarray([macro_f1_from_pred(y[i], PRED[(arm, prot, split)][i])
                                for i in range(len(seeds))], dtype=np.float64)
                F[(prot, split, arm)] = v
                res["arm_means"]["%s/%s/%s" % (prot, split, arm)] = {
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "per_seed": [float(x) for x in v]}
        for arm in TRAINED + ["CATB"]:
            res["arm_means"]["%s/test/%s" % (prot, arm)]["mean_selected_epoch"] = float(
                np.mean([ep_sel[(arm, si, prot)] for si in range(len(seeds))]))

    # ---------------- contrasts ----------------
    PAIRS = [("AVG", "CAT"), ("WAVG", "CAT"), ("SEL", "CAT"),
             ("ANCA", "CAT"), ("ANCL", "CAT"),
             ("AVG", "ECTL"), ("WAVG", "ECTL"), ("SEL", "ECTL"),
             ("ANCA", "LBL"), ("ANCL", "LBL"),
             ("ECTL", "CAT"), ("LBL", "CAT"),
             ("CAT", "A0"), ("LL", "A0"), ("CAT", "LL"),
             ("AVG", "A0"), ("MC", "A0"), ("MC", "CAT")]
    for prot in PROTOCOLS:
        for split in ("dev", "test"):
            for (L, R) in PAIRS:
                d = F[(prot, split, L)] - F[(prot, split, R)]
                lo, hi, se = paired_bootstrap_ci(d, rng)
                res["contrasts"]["%s/%s/%s-%s" % (prot, split, L, R)] = {
                    "mean": float(d.mean()), "std": float(d.std(ddof=1)),
                    "se_boot": se, "ci95": [lo, hi],
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "n_positive": int((d > 0).sum()), "n_seeds": len(d),
                    "passes_bar_and_ci": bool(d.mean() >= BAR and lo > 0),
                    "per_seed": [float(x) for x in d]}

    # ---------------- union accounting (no verdict power) ----------------
    ua = {}
    for arm in ["CAT", "LL", "AVG", "WAVG", "SEL", "ECTL", "ANCA", "ANCL", "LBL", "MC"]:
        keep, newbad, net, uni = [], [], [], []
        for i in range(len(seeds)):
            y = Z[("CAT", i, "P1", "test")][0]
            ok = {k: (PRED[(k, "P1", "test")][i] == y)
                  for k in ("A0", "CAT", "LL", arm)}
            e0 = ~ok["A0"]
            fix_union = e0 & (ok["CAT"] | ok["LL"])
            n_u = int(fix_union.sum())
            uni.append(n_u)
            keep.append(int((fix_union & ok[arm]).sum()) / n_u if n_u else np.nan)
            newbad.append(int((ok["A0"] & ~ok[arm]).sum()))
            net.append(int(e0.sum()) - int((~ok[arm]).sum()))
        ua[arm] = {"union_fix_pool_mean": float(np.mean(uni)),
                   "union_fix_retained_frac_mean": float(np.nanmean(keep)),
                   "new_errors_vs_A0_mean": float(np.mean(newbad)),
                   "net_errors_saved_vs_A0_mean": float(np.mean(net))}
        print("UNION %-6s pool=%.2f retained=%.3f new_err=%.2f net_saved=%.2f"
              % (arm, ua[arm]["union_fix_pool_mean"],
                 ua[arm]["union_fix_retained_frac_mean"],
                 ua[arm]["new_errors_vs_A0_mean"],
                 ua[arm]["net_errors_saved_vs_A0_mean"]))
    res["union_accounting_P1_test"] = ua

    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
