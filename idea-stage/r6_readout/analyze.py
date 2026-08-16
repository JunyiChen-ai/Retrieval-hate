"""R6-1 analysis -- frozen readout and frozen decision rule.

Freeze: idea-stage/R6_PILOT_FREEZE_2026-08-17.md (pilot R6-1).

Epoch selection and metric readout are imported verbatim from
scripts/rgcl_ablation_analyze.py::parse_run (the "I1" head rung), which is the
same function idea-stage/text_merge/analyze_arms.py used for the 2026-08-13/14
pilot series: epoch = argmax over epochs >= warmup(5) of (dev head acc, dev head
roc); the reported number is the TEST macro-F1 at that epoch. Test labels are
never used for selection.

Frozen decision rule, applied verbatim:
  GO        iff on BOTH datasets: mean(CAT-A0) >= +0.005 AND 3/3 seeds of
            (CAT-A0) positive AND mean(CAT-RANDCAT) >= +0.005
  AMBIGUOUS iff those conditions hold on exactly one dataset
  KILL      otherwise
L24-A0 is recorded but is NOT part of the rule.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rgcl_ablation_analyze import parse_run  # noqa: E402

DATASETS = ["HateMM", "MHC_zh"]
ARMS = ["A0", "L24", "CAT", "RANDCAT"]
SEEDS = [0, 1, 2]
THRESH = 0.005

RE_TEST_HEAD = re.compile(
    r"^test Epoch (\d+) acc: ([\d.]+) roc: ([\d.]+) pre: [\d.]+ recall: [\d.]+ "
    r"f1: [\d.]+ \| macroF1: ([\d.]+)", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/r6_readout/logs")
    ap.add_argument("--out", default="idea-stage/r6_readout/results.json")
    a = ap.parse_args()

    res = {}
    for ds in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                p = os.path.join(ROOT, a.logdir, "%s_%s_s%d.trainlog" % (ds, arm, seed))
                parsed = parse_run(p) if os.path.exists(p) else None
                if parsed is None:
                    print("MISSING/UNPARSEABLE", p)
                    continue
                r = dict(parsed["I1"])
                txt = open(p, errors="replace").read()
                for m in RE_TEST_HEAD.finditer(txt):
                    if int(m.group(1)) == r["epoch"]:
                        r["test_acc"] = float(m.group(2))
                        r["test_roc"] = float(m.group(3))
                res[(ds, arm, seed)] = r

    out = {"freeze": "idea-stage/R6_PILOT_FREEZE_2026-08-17.md (pilot R6-1)",
           "selection": "epoch = argmax_{e>=5}(dev head acc, dev head roc); "
                        "reported = test macro-F1 at that epoch (parse_run I1)",
           "per_run": {"%s_%s_s%d" % k: v for k, v in res.items()},
           "table": {}, "deltas": {}}

    def vals(ds, arm, k="test"):
        return [res[(ds, arm, s)][k] for s in SEEDS if (ds, arm, s) in res]

    def paired(ds, a1, a2, k="test"):
        d = [res[(ds, a1, s)][k] - res[(ds, a2, s)][k] for s in SEEDS
             if (ds, a1, s) in res and (ds, a2, s) in res]
        return float(np.mean(d)) if d else float("nan"), d

    for ds in DATASETS:
        print("\n=== %s ===" % ds)
        print("%-9s %-8s %-8s  %-26s %s" % ("arm", "mean", "std", "per-seed test macroF1",
                                            "epochs"))
        out["table"][ds] = {}
        for arm in ARMS:
            v = vals(ds, arm)
            if not v:
                continue
            rc = vals(ds, arm, "test_roc")
            out["table"][ds][arm] = {
                "test_macro_f1": v,
                "test_macro_f1_mean": float(np.mean(v)),
                "test_macro_f1_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "test_roc": rc,
                "test_roc_mean": float(np.mean(rc)) if rc else None,
                "val_macro_f1": vals(ds, arm, "val"),
                "epochs": vals(ds, arm, "epoch"),
            }
            print("%-9s %-8.4f %-8.4f  %-26s %s"
                  % (arm, np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0,
                     " ".join("%.4f" % x for x in v),
                     " ".join(str(e) for e in vals(ds, arm, "epoch"))))

        print("paired deltas (test macro-F1)")
        out["deltas"][ds] = {}
        for a1, a2 in [("CAT", "A0"), ("CAT", "RANDCAT"), ("L24", "A0"),
                       ("RANDCAT", "A0")]:
            m, d = paired(ds, a1, a2)
            npos = sum(1 for x in d if x > 0)
            out["deltas"][ds]["%s-%s" % (a1, a2)] = {
                "mean": m, "per_seed": d, "n_positive": npos, "n_seeds": len(d)}
            print("  %-14s %+.4f   per-seed %s   positive %d/%d"
                  % ("%s-%s" % (a1, a2), m, " ".join("%+.4f" % x for x in d),
                     npos, len(d)))

    # ------------------------------------------------------- frozen decision rule
    per_ds = {}
    for ds in DATASETS:
        g = out["deltas"].get(ds, {}).get("CAT-A0")
        c = out["deltas"].get(ds, {}).get("CAT-RANDCAT")
        if not (g and c):
            per_ds[ds] = {"pass": False, "reason": "missing runs"}
            continue
        c1 = bool(g["mean"] >= THRESH)
        c2 = bool(g["n_seeds"] == 3 and g["n_positive"] == 3)
        c3 = bool(c["mean"] >= THRESH)
        per_ds[ds] = {
            "clause1_mean_CAT_minus_A0_ge_0.005": c1,
            "clause2_3of3_seeds_CAT_minus_A0_positive": c2,
            "clause3_mean_CAT_minus_RANDCAT_ge_0.005": c3,
            "pass": bool(c1 and c2 and c3),
        }
    npass = sum(1 for v in per_ds.values() if v.get("pass"))
    verdict = "GO" if npass == 2 else ("AMBIGUOUS" if npass == 1 else "KILL")
    out["verdict"] = {"per_dataset": per_ds, "n_datasets_passing": npass,
                      "verdict": verdict,
                      "rule": "GO iff both datasets pass; AMBIGUOUS iff exactly one; "
                              "KILL otherwise. L24-A0 recorded, not in the rule."}
    print("\nFROZEN VERDICT: %s" % verdict)
    print(json.dumps(per_ds, indent=1))

    with open(os.path.join(ROOT, a.out), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
