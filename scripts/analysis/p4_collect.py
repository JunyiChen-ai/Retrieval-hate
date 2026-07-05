#!/usr/bin/env python
"""Parse the P4 trainlogs into the results table (CPU, read-only).

For each run (dataset x lambda x seed) reads slurm/logs/p4aux_<ds>_l<lam>_s<seed>.trainlog
and extracts, per epoch, the Val/Test retrieval metrics. Reports two protocols:
  val-selected : epoch (>= warmup) with best Val_Retrieval acc (tie-break Val ROC) ->
                 that epoch's TEST macro-F1 / acc.
  final-epoch  : the last epoch's TEST macro-F1 / acc.

Then floor (lambda=0) vs aux (lambda=0.1) paired per-seed deltas, mean over seeds, both
protocols, both datasets, plus the bit-for-bit floor check and the pre-registered verdict.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

WARM = 5
KNOWN_FLOOR = {  # val-selected TEST (acc, macro_f1) from RAC_video_CLIP seed 0
    "MHC": (0.7826, 0.7113),
    "MHC_zh": (0.8054, 0.7706),
}
VAL_RE = re.compile(
    r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: [\d.]+ macroR: [\d.]+ "
    r"acc: ([\d.]+) roc: ([\d.]+)")
TEST_RE = re.compile(
    r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: [\d.]+ macroR: [\d.]+ "
    r"acc: ([\d.]+) roc: ([\d.]+)")


def parse_log(path):
    ep = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = VAL_RE.search(line)
            if m:
                e = int(m.group(1))
                ep.setdefault(e, {})
                ep[e].update(val_maf1=float(m.group(2)), val_acc=float(m.group(3)),
                             val_roc=float(m.group(4)))
            m = TEST_RE.search(line)
            if m:
                e = int(m.group(1))
                ep.setdefault(e, {})
                ep[e].update(test_maf1=float(m.group(2)), test_acc=float(m.group(3)),
                             test_roc=float(m.group(4)))
    if not ep:
        return None
    complete = {e: d for e, d in ep.items()
                if all(k in d for k in ("val_acc", "val_roc", "test_acc", "test_maf1"))}
    if not complete:
        return None
    elig = [e for e in complete if e >= WARM] or list(complete)
    best = max(elig, key=lambda e: (complete[e]["val_acc"], complete[e]["val_roc"]))
    final = max(complete)
    return {
        "n_epochs": len(complete),
        "val_sel": {"epoch": best, "test_acc": complete[best]["test_acc"],
                    "test_maf1": complete[best]["test_maf1"],
                    "val_acc": complete[best]["val_acc"]},
        "final": {"epoch": final, "test_acc": complete[final]["test_acc"],
                  "test_maf1": complete[final]["test_maf1"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="slurm/logs")
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--lambdas", default="0,0.1")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="scripts/analysis/p4_out/p4_results.json")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    lambdas = args.lambdas.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = {}
    for ds in datasets:
        for lam in lambdas:
            for seed in seeds:
                tag = "p4aux_{}_l{}_s{}".format(ds, lam, seed)
                hits = glob.glob(os.path.join(args.logdir, tag + ".trainlog"))
                runs[(ds, lam, seed)] = parse_log(hits[0]) if hits else None

    result = {"runs": {}, "bit_for_bit": {}, "compare": {}, "verdict": {}}
    for k, v in runs.items():
        result["runs"]["{}|l{}|s{}".format(*k)] = v

    # bit-for-bit: lambda=0 seed 0 val-sel vs known floor
    print("===== P4 results =====")
    print("\n[bit-for-bit] lambda=0 seed0 val-selected TEST vs known RAC_video_CLIP floor:")
    for ds in datasets:
        r = runs.get((ds, "0", 0))
        if r is None:
            print("  {}: MISSING".format(ds)); continue
        got = (round(r["val_sel"]["test_acc"], 4), round(r["val_sel"]["test_maf1"], 4))
        exp = KNOWN_FLOOR.get(ds)
        ok = exp is not None and abs(got[0] - exp[0]) < 1e-4 and abs(got[1] - exp[1]) < 1e-4
        result["bit_for_bit"][ds] = {"got_acc_maf1": got, "expected": exp, "exact": bool(ok)}
        print("  {}: got acc/maF1={} expected={} -> {}".format(
            ds, got, exp, "BIT-FOR-BIT OK" if ok else "MISMATCH"))

    # floor vs aux, paired per-seed deltas, both protocols
    print("\n[floor vs aux] macro-F1 / acc, paired per-seed deltas:")
    for ds in datasets:
        result["compare"][ds] = {}
        for proto in ("val_sel", "final"):
            fl, ax, dmaf1, dacc = [], [], [], []
            for seed in seeds:
                rf = runs.get((ds, "0", seed))
                ra = runs.get((ds, "0.1", seed))
                if rf is None or ra is None:
                    continue
                fl.append((rf[proto]["test_maf1"], rf[proto]["test_acc"]))
                ax.append((ra[proto]["test_maf1"], ra[proto]["test_acc"]))
                dmaf1.append(ra[proto]["test_maf1"] - rf[proto]["test_maf1"])
                dacc.append(ra[proto]["test_acc"] - rf[proto]["test_acc"])
            if not dmaf1:
                continue
            row = {
                "floor_maf1_mean": round(float(np.mean([x[0] for x in fl])), 4),
                "aux_maf1_mean": round(float(np.mean([x[0] for x in ax])), 4),
                "floor_acc_mean": round(float(np.mean([x[1] for x in fl])), 4),
                "aux_acc_mean": round(float(np.mean([x[1] for x in ax])), 4),
                "delta_maf1_mean": round(float(np.mean(dmaf1)), 4),
                "delta_maf1_per_seed": [round(x, 4) for x in dmaf1],
                "delta_maf1_pos_seeds": int(sum(x > 0 for x in dmaf1)),
                "delta_acc_mean": round(float(np.mean(dacc)), 4),
                "delta_acc_per_seed": [round(x, 4) for x in dacc],
                "delta_acc_pos_seeds": int(sum(x > 0 for x in dacc)),
                "n_seeds": len(dmaf1),
            }
            result["compare"][ds][proto] = row
            print("  {} [{:8s}] floor maF1={:.4f} acc={:.4f} | aux maF1={:.4f} acc={:.4f} "
                  "| dMaF1={:+.4f} ({} /{} seeds+) dAcc={:+.4f} ({}/{}+)".format(
                      ds, proto, row["floor_maf1_mean"], row["floor_acc_mean"],
                      row["aux_maf1_mean"], row["aux_acc_mean"],
                      row["delta_maf1_mean"], row["delta_maf1_pos_seeds"], row["n_seeds"],
                      row["delta_acc_mean"], row["delta_acc_pos_seeds"], row["n_seeds"]))

    # pre-registered verdict:
    # (3) aux beats floor mean dMaF1 > 0.01 with >=2/3 seeds positive on >=1 dataset
    #     under BOTH protocols, and no >0.01 harm (dMaF1 < -0.01) elsewhere.
    win_datasets = []
    harm = []
    for ds in datasets:
        comp = result["compare"].get(ds, {})
        both = all(
            (comp.get(p, {}).get("delta_maf1_mean", -9) > 0.01 and
             comp.get(p, {}).get("delta_maf1_pos_seeds", 0) >= 2)
            for p in ("val_sel", "final")) and len(comp) == 2
        if both:
            win_datasets.append(ds)
        for p in ("val_sel", "final"):
            d = comp.get(p, {}).get("delta_maf1_mean", 0.0)
            if d < -0.01:
                harm.append((ds, p, d))
    passed = len(win_datasets) >= 1 and len(harm) == 0
    result["verdict"] = {"win_datasets": win_datasets, "harm": harm,
                         "criterion3_pass": bool(passed)}
    print("\n[verdict] win datasets (dMaF1>0.01, >=2/3 seeds+, both protocols): {}".format(
        win_datasets))
    print("[verdict] >1pt harm cells: {}".format(harm))
    print("[verdict] criterion (3) {}".format("PASS" if passed else "FAIL/within-noise"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n[out] wrote", args.out)


if __name__ == "__main__":
    main()
