#!/usr/bin/env python
"""R6-AUDIT -- measurement-protocol certification analysis.

DIAGNOSTIC, not a pilot.  Measures whether the project's 3-seed / +0.005
macro-F1 pilot protocol can resolve the effects it claims to kill.

Test labels are read here under the user's 2026-08-09 test-set protocol ruling.
Nothing -- no threshold, no epoch rule, no arm definition -- is tuned on them.
Both read-out protocols were fixed before any number below was computed, and the
arms audited (R6RO A0 / CAT / RANDCAT) are already-killed arms.

Read-out protocols, all computed from the SAME 180 runs (no retraining):
  P1  epoch = argmax_{e>=5} dev macro-F1 (ties -> earliest); test macro-F1 @0.5
      -- this is the rule stated in idea-stage/R6_PILOT_FREEZE_2026-08-17.md.
  P2  epoch = last epoch (index 29 of 30); test macro-F1 @0.5.
  P1b epoch = argmax_{e>=5} (dev acc, dev roc) (ties -> earliest); test macro-F1
      @0.5 -- this is the rule the code in idea-stage/r6_readout/analyze.py
      actually executed (it imports scripts/rgcl_ablation_analyze.py::parse_run).
      Reported as the third variant because P3 (val-selected THRESHOLD) is not
      recoverable from trainlogs: they only ever print threshold-0.5 metrics.

Per-epoch test confusion matrices are reconstructed exactly from the logged
(acc, recall) pair plus the known split size N and positive count P:
    TP = round(recall*P);  TP+TN = round(acc*N);  TN = (TP+TN) - TP
    FN = P - TP;           FP = (N-P) - TN;       pred_pos = TP + FP
Logged values carry 4 decimals, so 0.00005*N < 0.011 for both splits and the
rounding is provably exact.  Every reconstruction is cross-checked against the
logged macroF1 (tolerance 1e-4) and the run is dropped if it disagrees.
The test-item bootstrap resamples the four confusion cells multinomially, which
is identical to resampling test items with replacement.
"""
import argparse
import itertools
import json
import os
import re
from collections import Counter

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
DATASETS = ["HateMM", "MHC_zh"]
ARMS = ["A0", "CAT", "RANDCAT"]
SEEDS = list(range(30))
PROTOCOLS = ["P1", "P2", "P1b"]
WARMUP = 5
LAST_EPOCH = 29
GO_BAR = 0.005
TARGET_SE = 0.0025
COLLAPSE = 0.45
PAIRS = [("CAT", "A0"), ("CAT", "RANDCAT"), ("RANDCAT", "A0")]
# test split sizes, from data/CLIP_Embedding/<ds>/test_seen_R6RO-A0.pt
SPLIT = {"HateMM": dict(N=215, P=86), "MHC_zh": dict(N=149, P=45)}

RE_DEV = re.compile(
    r"^dev\s+Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: [\d.]+ "
    r"f1: [\d.]+ loss: [\d.]+ \| macroF1: ([\d.]+)", re.M)
RE_TEST = re.compile(
    r"^test Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: ([\d.]+) "
    r"f1: [\d.]+ \| macroF1: ([\d.]+)", re.M)


def macro_f1_from_cm(tp, fp, fn, tn):
    f1s = []
    for a, b, c in ((tp, fp, fn), (tn, fn, fp)):   # (tp,fp,fn) for class1, class0
        pr = a / (a + b) if a + b else 0.0
        rc = a / (a + c) if a + c else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def parse(path, ds):
    """-> {epoch: {...}} or None."""
    N, P = SPLIT[ds]["N"], SPLIT[ds]["P"]
    txt = open(path, errors="replace").read()
    dev = {}
    for m in RE_DEV.finditer(txt):
        dev[int(m.group(1))] = dict(acc=float(m.group(2)), roc=float(m.group(3)),
                                    macro_f1=float(m.group(4)))
    test = {}
    for m in RE_TEST.finditer(txt):
        e = int(m.group(1))
        acc, roc, rec, mf1 = (float(m.group(2)), float(m.group(3)),
                              float(m.group(4)), float(m.group(5)))
        tp = int(round(rec * P))
        correct = int(round(acc * N))
        tn = correct - tp
        fn = P - tp
        fp = (N - P) - tn
        if min(tp, fp, fn, tn) < 0:
            return None, "negative cm cell at epoch %d" % e
        rec_mf1 = macro_f1_from_cm(tp, fp, fn, tn)
        if abs(rec_mf1 - mf1) > 1e-4:
            return None, ("cm mismatch at epoch %d: logged %.4f recon %.4f"
                          % (e, mf1, rec_mf1))
        test[e] = dict(acc=acc, roc=roc, macro_f1=rec_mf1, logged_macro_f1=mf1,
                       tp=tp, fp=fp, fn=fn, tn=tn, pred_pos=tp + fp)
    if len(dev) != 30 or len(test) != 30:
        return None, "incomplete: %d dev / %d test epochs" % (len(dev), len(test))
    return {"dev": dev, "test": test}, None


def select_epoch(dev, protocol):
    if protocol == "P2":
        return LAST_EPOCH
    cand = sorted(e for e in dev if e >= WARMUP)
    if protocol == "P1":
        key = lambda e: dev[e]["macro_f1"]
    else:                                        # P1b
        key = lambda e: (dev[e]["acc"], dev[e]["roc"])
    best = cand[0]
    for e in cand:                               # strict > keeps the EARLIEST tie
        if key(e) > key(best):
            best = e
    return best


def boot_item_std(cm, n_boot=2000, rng=None):
    """std of macro-F1 under resampling test items with replacement."""
    rng = rng or np.random.default_rng(20260817)
    cells = np.array([cm["tp"], cm["fp"], cm["fn"], cm["tn"]], dtype=float)
    n = int(cells.sum())
    draws = rng.multinomial(n, cells / n, size=n_boot)
    f1 = np.array([macro_f1_from_cm(*d) for d in draws])
    return float(f1.std(ddof=1)), float(f1.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/r6_audit/logs")
    ap.add_argument("--out", default="idea-stage/r6_audit/results.json")
    ap.add_argument("--n_mc", type=int, default=20000)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(20260817)

    runs, bad = {}, []
    for ds in DATASETS:
        for arm in ARMS:
            for s in SEEDS:
                p = os.path.join(ROOT, a.logdir, "%s_%s_s%d.trainlog" % (ds, arm, s))
                if not os.path.exists(p):
                    bad.append(("%s_%s_s%d" % (ds, arm, s), "missing"))
                    continue
                r, err = parse(p, ds)
                if r is None:
                    bad.append(("%s_%s_s%d" % (ds, arm, s), err))
                    continue
                runs[(ds, arm, s)] = r

    out = {
        "what": "R6-AUDIT: measurement-protocol certification of the 3-seed/+0.005 "
                "pilot rule. Diagnostic only; no candidate is being judged.",
        "grid": "2 datasets x 3 arms x 30 seeds = 180 head runs, hyperparameters "
                "identical to idea-stage/r6_readout/run_arms.sh",
        "test_label_use": "Test labels read under the user's 2026-08-09 protocol "
                          "ruling. No threshold, epoch rule or design was tuned on "
                          "them; this is a variance measurement of already-killed arms.",
        "protocols": {
            "P1": "epoch = argmax_{e>=5} dev macro-F1 (ties->earliest); test macro-F1 @0.5",
            "P2": "epoch = 29 (last of 30); test macro-F1 @0.5",
            "P1b": "epoch = argmax_{e>=5} (dev acc, dev roc) (ties->earliest); test "
                   "macro-F1 @0.5 -- the rule r6_readout/analyze.py actually ran",
            "P3_not_available": "val-selected THRESHOLD is not recoverable from "
                                "trainlogs (threshold-0.5 metrics only); would need "
                                "per-item logit dumps, so P1b is reported instead",
        },
        "splits": SPLIT,
        "failures": [{"run": r, "why": w} for r, w in bad],
        "n_runs_parsed": len(runs),
        "arms": {}, "deltas": {}, "item_bootstrap": {},
    }

    # ------------------------------------------------------------------ per arm
    for ds in DATASETS:
        for arm in ARMS:
            for pr in PROTOCOLS:
                seeds = [s for s in SEEDS if (ds, arm, s) in runs]
                eps, f1s, pps = [], [], []
                for s in seeds:
                    r = runs[(ds, arm, s)]
                    e = select_epoch(r["dev"], pr)
                    t = r["test"][e]
                    eps.append(e)
                    f1s.append(t["macro_f1"])
                    pps.append(t["pred_pos"])
                v = np.array(f1s)
                out["arms"]["%s|%s|%s" % (ds, arm, pr)] = {
                    "n_seeds": len(seeds),
                    "per_seed_test_macro_f1": [round(x, 6) for x in f1s],
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()),
                    "selected_epochs": eps,
                    "selected_epoch_hist": dict(sorted(Counter(eps).items())),
                    "pred_pos": pps,
                    "pred_pos_hist": dict(sorted(Counter(pps).items())),
                    "pred_pos_min": int(min(pps)), "pred_pos_max": int(max(pps)),
                    "n_collapsed_lt_0.45": int((v < COLLAPSE).sum()),
                    "collapsed_seeds": [seeds[i] for i in np.where(v < COLLAPSE)[0]],
                }

    # ------------------------------------------------------------ paired deltas
    combos = list(itertools.combinations(range(30), 3))
    for ds in DATASETS:
        for pr in PROTOCOLS:
            for a1, a2 in PAIRS:
                seeds = [s for s in SEEDS
                         if (ds, a1, s) in runs and (ds, a2, s) in runs]
                d = []
                for s in seeds:
                    r1, r2 = runs[(ds, a1, s)], runs[(ds, a2, s)]
                    d.append(r1["test"][select_epoch(r1["dev"], pr)]["macro_f1"]
                             - r2["test"][select_epoch(r2["dev"], pr)]["macro_f1"])
                d = np.array(d)
                sd = float(d.std(ddof=1))
                n = len(d)
                # exact enumeration of all C(n,3) 3-seed subsets
                cmb = combos if n == 30 else list(itertools.combinations(range(n), 3))
                sub = d[np.array(cmb)]
                go_exact = float(((sub.mean(1) >= GO_BAR) & (sub > 0).all(1)).mean())
                # MC replicate of the same quantity (>= n_mc draws), as a check
                idx = np.array([rng.choice(n, 3, replace=False) for _ in range(a.n_mc)])
                subm = d[idx]
                go_mc = float(((subm.mean(1) >= GO_BAR) & (subm > 0).all(1)).mean())
                out["deltas"]["%s|%s|%s-%s" % (ds, pr, a1, a2)] = {
                    "n_seeds": n,
                    "per_seed": [round(float(x), 6) for x in d],
                    "mean": float(d.mean()), "std": sd,
                    "mc_se_of_mean": sd / np.sqrt(n),
                    "n_star_for_se_le_0.0025": float((sd / TARGET_SE) ** 2),
                    "n_positive_of_30": int((d > 0).sum()),
                    "p_3seed_GO_exact_all_C30_3": go_exact,
                    "p_3seed_GO_mc": go_mc, "n_mc": a.n_mc,
                    "n_3seed_subsets": len(cmb),
                    "go_rule": "mean(3 seeds) >= +0.005 AND 3/3 seeds positive",
                }

    # ------------------------------------------------------- test-item bootstrap
    for ds in DATASETS:
        for arm in ARMS:
            for pr in PROTOCOLS:
                if (ds, arm, 0) not in runs:
                    continue
                r = runs[(ds, arm, 0)]
                e = select_epoch(r["dev"], pr)
                cm = r["test"][e]
                sd, mu = boot_item_std(cm, a.n_boot, rng)
                out["item_bootstrap"]["%s|%s|%s" % (ds, arm, pr)] = {
                    "seed_fixed": 0, "epoch": e, "n_boot": a.n_boot,
                    "point_macro_f1": cm["macro_f1"],
                    "bootstrap_mean": mu, "bootstrap_std": sd,
                    "cm": {k: cm[k] for k in ("tp", "fp", "fn", "tn")},
                }

    # -------------------------------------------------- variance decomposition
    dec = {}
    for ds in DATASETS:
        for arm in ARMS:
            seeds = [s for s in SEEDS if (ds, arm, s) in runs]
            p1 = np.array([runs[(ds, arm, s)]["test"]
                           [select_epoch(runs[(ds, arm, s)]["dev"], "P1")]["macro_f1"]
                           for s in seeds])
            p2 = np.array([runs[(ds, arm, s)]["test"][LAST_EPOCH]["macro_f1"]
                           for s in seeds])
            item = out["item_bootstrap"].get("%s|%s|P1" % (ds, arm), {}).get(
                "bootstrap_std")
            dec["%s|%s" % (ds, arm)] = {
                "var_P1": float(p1.var(ddof=1)), "std_P1": float(p1.std(ddof=1)),
                "var_P2": float(p2.var(ddof=1)), "std_P2": float(p2.std(ddof=1)),
                "var_of_P1_minus_P2": float((p1 - p2).var(ddof=1)),
                "std_of_P1_minus_P2": float((p1 - p2).std(ddof=1)),
                "cov_P1_P2": float(np.cov(p1, p2, ddof=1)[0, 1]),
                "n_seeds_where_P1_epoch_ne_29": int(sum(
                    1 for s in seeds
                    if select_epoch(runs[(ds, arm, s)]["dev"], "P1") != LAST_EPOCH)),
                "item_bootstrap_std_seed0_P1": item,
                "note": "var(P1) = var(P2) + var(P1-P2) + 2*cov(P2, P1-P2). "
                        "P2 isolates optimisation-seed variance at a fixed epoch; "
                        "var(P1-P2) is the variance injected by moving the epoch. "
                        "Item-sampling std is a common uncertainty on the mean over "
                        "the SAME fixed test set, so it does not add to seed spread.",
            }
    out["variance_decomposition"] = dec

    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1)

    # --------------------------------------------------------------- printout
    for ds in DATASETS:
        print("\n=== %s (test N=%d, P=%d) ===" % (ds, SPLIT[ds]["N"], SPLIT[ds]["P"]))
        print("%-9s %-4s %-7s %-7s %-7s %-7s %-5s %-22s %s"
              % ("arm", "prot", "mean", "std", "min", "max", "coll", "sel-epoch hist",
                 "pred-pos range"))
        for arm in ARMS:
            for pr in PROTOCOLS:
                r = out["arms"]["%s|%s|%s" % (ds, arm, pr)]
                print("%-9s %-4s %-7.4f %-7.4f %-7.4f %-7.4f %-5d %-22s %d-%d"
                      % (arm, pr, r["mean"], r["std"], r["min"], r["max"],
                         r["n_collapsed_lt_0.45"],
                         ",".join("%s:%s" % kv for kv in
                                  list(r["selected_epoch_hist"].items())[:5]),
                         r["pred_pos_min"], r["pred_pos_max"]))
        print("\n%-16s %-4s %-9s %-8s %-9s %-9s %-7s %s"
              % ("pair", "prot", "mean", "std", "MC-SE", "n*", "pos/30", "P(3-seed GO)"))
        for pr in PROTOCOLS:
            for a1, a2 in PAIRS:
                d = out["deltas"]["%s|%s|%s-%s" % (ds, pr, a1, a2)]
                print("%-16s %-4s %+9.4f %-8.4f %-9.4f %-9.1f %-7d %.4f"
                      % ("%s-%s" % (a1, a2), pr, d["mean"], d["std"],
                         d["mc_se_of_mean"], d["n_star_for_se_le_0.0025"],
                         d["n_positive_of_30"], d["p_3seed_GO_exact_all_C30_3"]))
        print("\nvariance decomposition (test macro-F1 over 30 seeds)")
        print("%-9s %-9s %-9s %-11s %-8s %s"
              % ("arm", "std_P1", "std_P2", "std(P1-P2)", "item_sd", "seeds w/ ep!=29"))
        for arm in ARMS:
            v = dec["%s|%s" % (ds, arm)]
            print("%-9s %-9.4f %-9.4f %-11.4f %-8.4f %d"
                  % (arm, v["std_P1"], v["std_P2"], v["std_of_P1_minus_P2"],
                     v["item_bootstrap_std_seed0_P1"] or float("nan"),
                     v["n_seeds_where_P1_epoch_ne_29"]))
    if bad:
        print("\nFAILED/UNPARSEABLE RUNS (%d):" % len(bad))
        for r, w in bad:
            print("  %s  %s" % (r, w))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
