#!/usr/bin/env python
"""R6-1C -- powered confirmation read-out and frozen decision rule.

Frozen design: idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md.

Parsing, confusion-matrix reconstruction and epoch selection are taken verbatim
from idea-stage/r6_audit/analyze_audit.py (single parser, no divergent copy).

Grid: 2 datasets x 4 arms {A0, CAT, RANDA, RANDB} x 60 seeds (30..89) = 480 runs.

Read-out protocols, both computed from the SAME runs:
  P1 (primary)      epoch = argmax_{e>=5} dev macro-F1 (ties -> earliest);
                    test macro-F1 @ threshold 0.5
  P2 (corroboration) epoch = 29 (last of 30); test macro-F1 @ threshold 0.5

Test labels are read only for the final metric.  Nothing -- no threshold, no
epoch rule, no arm definition -- is selected on them; epoch selection is on val.

Frozen decision rule (applied verbatim, see freeze doc):
  A dataset PASSES under P1 iff mean(CAT-A0) >= +0.005 with paired-bootstrap
  95% CI excluding 0, AND mean(CAT-RAND) >= +0.005 with its CI excluding 0.
  CONFIRMED-2DS   both datasets pass under P1 and P2 agrees in sign on both.
  CONFIRMED-1DS   exactly one passes under P1, P2 agrees in sign on it, and the
                  other has mean(CAT-A0) >= -0.002 under P1.
  NOT CONFIRMED   otherwise.
  VOID (overrides) if |mean(RANDA-RANDB)| >= 0.005 on either dataset under P1.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
# reuse the audit's parser / CM reconstruction / epoch selector verbatim
from analyze_audit import parse, select_epoch, SPLIT, LAST_EPOCH, WARMUP  # noqa: E402

DATASETS = ["HateMM", "MHC_zh"]
ARMS = ["A0", "CAT", "RANDA", "RANDB"]
SEEDS = list(range(30, 90))
PROTOCOLS = ["P1", "P2"]
GO_BAR = 0.005
NO_HARM_BAR = -0.002
VOID_BAR = 0.005
N_BOOT = 20000
BOOT_SEED = 20260817
COLLAPSE = 0.45

# derived-arm pairs: (name, left_spec, right_spec); a spec is a list of arms
# whose per-seed macro-F1 is averaged (RAND = mean(RANDA, RANDB)).
PAIRS = [
    ("CAT-A0", ["CAT"], ["A0"]),
    ("CAT-RAND", ["CAT"], ["RANDA", "RANDB"]),
    ("RANDA-A0", ["RANDA"], ["A0"]),
    ("RANDB-A0", ["RANDB"], ["A0"]),
    ("RANDA-RANDB", ["RANDA"], ["RANDB"]),
]


def paired_bootstrap_ci(d, n_boot, rng):
    """Paired bootstrap over seeds: resample the seed-wise deltas w/ replacement."""
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float(means.mean()), float(means.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/r6_confirm/logs")
    ap.add_argument("--out", default="idea-stage/r6_confirm/results.json")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)

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
    if bad:
        print("FAILED/UNPARSEABLE RUNS (%d):" % len(bad))
        for r, w in bad:
            print("  %s  %s" % (r, w))
        raise SystemExit(
            "HALT: %d of %d runs unusable. The frozen rule is defined over 60 "
            "complete paired seeds; refusing to compute a verdict on a subset."
            % (len(bad), len(DATASETS) * len(ARMS) * len(SEEDS)))

    out = {
        "what": "R6-1C powered confirmation. Frozen: "
                "idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md",
        "grid": "2 datasets x 4 arms {A0,CAT,RANDA,RANDB} x 60 seeds (30..89) "
                "= 480 head runs; hyperparameters identical to "
                "idea-stage/r6_readout/run_arms.sh",
        "seed_independence": "audit consumed seeds 0-29; this run uses 30-89 only",
        "protocols": {
            "P1": "epoch = argmax_{e>=%d} dev macro-F1 (ties->earliest); "
                  "test macro-F1 @0.5" % WARMUP,
            "P2": "epoch = %d (last of 30); test macro-F1 @0.5" % LAST_EPOCH,
        },
        "splits": SPLIT,
        "n_runs_parsed": len(runs),
        "n_boot": a.n_boot,
        "bootstrap_rng_seed": BOOT_SEED,
        "seeds": SEEDS,
        "arms": {},
        "deltas": {},
    }

    # -------------------------------------------------------------- per arm
    f1tab = {}          # (ds, arm, protocol) -> np.array over SEEDS
    for ds in DATASETS:
        for arm in ARMS:
            for pr in PROTOCOLS:
                eps, f1s, pps = [], [], []
                for s in SEEDS:
                    r = runs[(ds, arm, s)]
                    e = select_epoch(r["dev"], pr)
                    t = r["test"][e]
                    eps.append(e)
                    f1s.append(t["macro_f1"])
                    pps.append(t["pred_pos"])
                v = np.array(f1s)
                f1tab[(ds, arm, pr)] = v
                out["arms"]["%s|%s|%s" % (ds, arm, pr)] = {
                    "n_seeds": len(SEEDS),
                    "per_seed_test_macro_f1": [round(float(x), 6) for x in f1s],
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "mc_se_of_mean": float(v.std(ddof=1) / np.sqrt(len(v))),
                    "min": float(v.min()), "max": float(v.max()),
                    "selected_epochs": eps,
                    "selected_epoch_hist": dict(sorted(Counter(eps).items())),
                    "pred_pos_min": int(min(pps)), "pred_pos_max": int(max(pps)),
                    "n_collapsed_lt_0.45": int((v < COLLAPSE).sum()),
                    "collapsed_seeds": [SEEDS[i] for i in np.where(v < COLLAPSE)[0]],
                }

    # -------------------------------------------------------- paired deltas
    for ds in DATASETS:
        for pr in PROTOCOLS:
            for name, left, right in PAIRS:
                L = np.mean([f1tab[(ds, x, pr)] for x in left], axis=0)
                R = np.mean([f1tab[(ds, x, pr)] for x in right], axis=0)
                d = L - R
                n = len(d)
                sd = float(d.std(ddof=1))
                lo, hi, bmean, bsd = paired_bootstrap_ci(d, a.n_boot, rng)
                out["deltas"]["%s|%s|%s" % (ds, pr, name)] = {
                    "pair": name,
                    "left_arms": left, "right_arms": right,
                    "n_seeds": n,
                    "per_seed": [round(float(x), 6) for x in d],
                    "mean": float(d.mean()), "std": sd,
                    "mc_se_of_mean": sd / np.sqrt(n),
                    "boot_ci95_lo": lo, "boot_ci95_hi": hi,
                    "boot_ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "boot_mean": bmean, "boot_std": bsd,
                    "n_positive_of_%d" % n: int((d > 0).sum()),
                }

    # -------------------------------------------------- frozen decision rule
    def g(ds, pr, name):
        return out["deltas"]["%s|%s|%s" % (ds, pr, name)]

    dec = {"rule": "see idea-stage/R6_CONFIRM_FREEZE_2026-08-17.md, applied verbatim",
           "per_dataset": {}}
    void_reasons = []
    for ds in DATASETS:
        rr = g(ds, "P1", "RANDA-RANDB")
        if abs(rr["mean"]) >= VOID_BAR:
            void_reasons.append("%s: |mean(RANDA-RANDB)| under P1 = %.6f >= %.3f"
                                % (ds, abs(rr["mean"]), VOID_BAR))
    passes = {}
    for ds in DATASETS:
        ca = g(ds, "P1", "CAT-A0")
        cr = g(ds, "P1", "CAT-RAND")
        c1 = bool(ca["mean"] >= GO_BAR and ca["boot_ci_excludes_zero"]
                  and ca["boot_ci95_lo"] > 0)
        c2 = bool(cr["mean"] >= GO_BAR and cr["boot_ci_excludes_zero"]
                  and cr["boot_ci95_lo"] > 0)
        passes[ds] = c1 and c2
        p2ca, p2cr = g(ds, "P2", "CAT-A0"), g(ds, "P2", "CAT-RAND")
        dec["per_dataset"][ds] = {
            "P1_CAT-A0_mean": ca["mean"], "P1_CAT-A0_ci": [ca["boot_ci95_lo"],
                                                           ca["boot_ci95_hi"]],
            "P1_CAT-A0_meets_bar_and_ci": c1,
            "P1_CAT-RAND_mean": cr["mean"], "P1_CAT-RAND_ci": [cr["boot_ci95_lo"],
                                                               cr["boot_ci95_hi"]],
            "P1_CAT-RAND_meets_bar_and_ci": c2,
            "passes_P1": passes[ds],
            "P2_CAT-A0_mean": p2ca["mean"], "P2_CAT-RAND_mean": p2cr["mean"],
            "P2_agrees_in_sign": bool(p2ca["mean"] > 0 and p2cr["mean"] > 0),
            "P1_RANDA-RANDB_mean": g(ds, "P1", "RANDA-RANDB")["mean"],
        }

    npass = sum(passes.values())
    if void_reasons:
        verdict = "VOID"
        why = "; ".join(void_reasons)
    elif npass == 2 and all(dec["per_dataset"][d]["P2_agrees_in_sign"]
                            for d in DATASETS):
        verdict = "CONFIRMED-2DS"
        why = "both datasets pass under P1 and P2 agrees in sign on both"
    elif npass == 1:
        pds = [d for d in DATASETS if passes[d]][0]
        ods = [d for d in DATASETS if not passes[d]][0]
        if (dec["per_dataset"][pds]["P2_agrees_in_sign"]
                and g(ods, "P1", "CAT-A0")["mean"] >= NO_HARM_BAR):
            verdict = "CONFIRMED-1DS"
            why = ("%s passes under P1, P2 agrees in sign on it, and %s has "
                   "mean(CAT-A0) = %.6f >= %.3f under P1"
                   % (pds, ods, g(ods, "P1", "CAT-A0")["mean"], NO_HARM_BAR))
        else:
            verdict = "NOT CONFIRMED"
            why = ("%s passes under P1 but the CONFIRMED-1DS side conditions fail "
                   "(P2 sign agreement=%s; %s mean(CAT-A0)=%.6f vs bar %.3f)"
                   % (pds, dec["per_dataset"][pds]["P2_agrees_in_sign"], ods,
                      g(ods, "P1", "CAT-A0")["mean"], NO_HARM_BAR))
    else:
        verdict = "NOT CONFIRMED"
        why = "%d of 2 datasets pass under P1" % npass
    dec["n_datasets_passing_P1"] = npass
    dec["void_reasons"] = void_reasons
    dec["verdict"] = verdict
    dec["verdict_reason"] = why
    out["decision"] = dec

    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------------------------- printout
    for ds in DATASETS:
        print("\n=== %s (test N=%d, P=%d) ==="
              % (ds, SPLIT[ds]["N"], SPLIT[ds]["P"]))
        print("%-7s %-4s %-8s %-8s %-8s %-8s %-5s %s"
              % ("arm", "prot", "mean", "std", "min", "max", "coll", "sel-epoch hist"))
        for arm in ARMS:
            for pr in PROTOCOLS:
                r = out["arms"]["%s|%s|%s" % (ds, arm, pr)]
                print("%-7s %-4s %-8.4f %-8.4f %-8.4f %-8.4f %-5d %s"
                      % (arm, pr, r["mean"], r["std"], r["min"], r["max"],
                         r["n_collapsed_lt_0.45"],
                         ",".join("%s:%s" % kv for kv in
                                  list(r["selected_epoch_hist"].items())[:6])))
        print("\n%-13s %-4s %-10s %-8s %-8s %-22s %-5s %s"
              % ("pair", "prot", "mean", "std", "MC-SE", "boot 95% CI", "pos", "CI!=0"))
        for pr in PROTOCOLS:
            for name, _, _ in PAIRS:
                d = g(ds, pr, name)
                print("%-13s %-4s %+10.5f %-8.5f %-8.5f [%+8.5f,%+8.5f] %2d/60 %s"
                      % (name, pr, d["mean"], d["std"], d["mc_se_of_mean"],
                         d["boot_ci95_lo"], d["boot_ci95_hi"],
                         d["n_positive_of_60"], d["boot_ci_excludes_zero"]))
    print("\n--- frozen decision rule ---")
    for ds in DATASETS:
        v = dec["per_dataset"][ds]
        print("%s: P1 CAT-A0=%+.5f CI=[%+.5f,%+.5f] ok=%s | P1 CAT-RAND=%+.5f "
              "CI=[%+.5f,%+.5f] ok=%s | PASS=%s | P2 signs ok=%s | "
              "P1 RANDA-RANDB=%+.5f"
              % (ds, v["P1_CAT-A0_mean"], v["P1_CAT-A0_ci"][0], v["P1_CAT-A0_ci"][1],
                 v["P1_CAT-A0_meets_bar_and_ci"], v["P1_CAT-RAND_mean"],
                 v["P1_CAT-RAND_ci"][0], v["P1_CAT-RAND_ci"][1],
                 v["P1_CAT-RAND_meets_bar_and_ci"], v["passes_P1"],
                 v["P2_agrees_in_sign"], v["P1_RANDA-RANDB_mean"]))
    print("VOID check: %s" % (void_reasons if void_reasons else "clean (no dataset "
                              "reaches |mean(RANDA-RANDB)| >= 0.005 under P1)"))
    print("VERDICT: %s  --  %s" % (verdict, why))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
