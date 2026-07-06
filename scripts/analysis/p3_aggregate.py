#!/usr/bin/env python
"""P3 training aggregator (EXP_p3_evidence_pooling §0.5/0.6).

Parses the per-run trainlogs (slurm/logs/p3_<DS>_<pool>_seed<seed>_<job>.trainlog)
and emits the pre-registered tables: per-seed val-selected + final-epoch
(macro-F1 / acc), 3-seed mean +/- std per pool, paired per-seed deltas
(weighted - floor), and the success-criteria verdict.

Usage: python scripts/analysis/p3_aggregate.py --dataset MHC
"""
import argparse
import glob
import os
import re

import numpy as np

VRE = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
TRE = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
WARMUP = 5


def parse_log(path):
    """Parse the raw per-epoch Val/Test macro lines and reproduce the two
    pre-registered lenses: val-selected (warmup>=5, by val acc then roc) and
    final-epoch. Returns (val, fin) each = (test_macroF1, test_acc, test_roc)."""
    val_by_e, test_by_e = {}, {}
    with open(path) as f:
        for line in f:
            m = VRE.search(line)
            if m:
                e = int(m.group(1))
                val_by_e[e] = (float(m.group(2)), float(m.group(5)), float(m.group(6)))  # f1,acc,roc
            m = TRE.search(line)
            if m:
                e = int(m.group(1))
                test_by_e[e] = (float(m.group(2)), float(m.group(5)), float(m.group(6)))
    if not val_by_e or not test_by_e:
        return None, None
    warm = [e for e in val_by_e if e >= WARMUP]
    cand = warm or list(val_by_e.keys())
    best = max(cand, key=lambda e: (val_by_e[e][1], val_by_e[e][2]))  # val acc, roc
    fe = max(val_by_e.keys())
    val = test_by_e.get(best)
    fin = test_by_e.get(fe)
    return val, fin


def collect(dataset, pools, seeds, logdir):
    data = {}  # (pool,seed) -> {"val":(f1,acc,roc), "fin":...}
    for pool in pools:
        for seed in seeds:
            pat = os.path.join(logdir, "p3_{}_{}_seed{}_*.trainlog".format(dataset, pool, seed))
            files = sorted(glob.glob(pat), key=os.path.getmtime)
            if not files:
                continue
            val, fin = parse_log(files[-1])
            data[(pool, seed)] = {"val": val, "fin": fin, "log": files[-1]}
    return data


def fmt(x):
    return "  --  " if x is None else "{:.4f}".format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC")
    ap.add_argument("--pools", default="mean,wsoftT1,wmild")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--logdir", default="slurm/logs")
    ap.add_argument("--noise_pts", type=float, default=0.01, help="1pt noise floor.")
    args = ap.parse_args()
    pools = args.pools.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    data = collect(args.dataset, pools, seeds, args.logdir)

    print("=" * 78)
    print("P3 TRAINING RESULTS  dataset={}".format(args.dataset))
    print("=" * 78)
    for lens, key in (("VAL-SELECTED (warmup>=5)", "val"), ("FINAL-EPOCH", "fin")):
        print("\n### {}  (Test macro-F1 / acc)".format(lens))
        print("{:>10} | {}".format("pool", " | ".join("seed{}: F1/acc".format(s) for s in seeds)))
        for pool in pools:
            cells = []
            for s in seeds:
                d = data.get((pool, s))
                if d and d[key]:
                    cells.append("{:.4f}/{:.4f}".format(d[key][0], d[key][1]))
                else:
                    cells.append("  --  ")
            print("{:>10} | {}".format(pool, " | ".join(cells)))
        # mean+/-std
        print("  -- 3-seed mean+/-std --")
        agg = {}
        for pool in pools:
            f1 = [data[(pool, s)][key][0] for s in seeds if data.get((pool, s)) and data[(pool, s)][key]]
            ac = [data[(pool, s)][key][1] for s in seeds if data.get((pool, s)) and data[(pool, s)][key]]
            if f1:
                agg[pool] = (np.array(f1), np.array(ac))
                print("{:>10} | F1 {:.4f}+/-{:.4f}  acc {:.4f}+/-{:.4f}".format(
                    pool, np.mean(f1), np.std(f1), np.mean(ac), np.std(ac)))
        # paired deltas vs floor(mean)
        if "mean" in agg:
            base_f1 = {s: data[("mean", s)][key][0] for s in seeds
                       if data.get(("mean", s)) and data[("mean", s)][key]}
            base_ac = {s: data[("mean", s)][key][1] for s in seeds
                       if data.get(("mean", s)) and data[("mean", s)][key]}
            for pool in pools:
                if pool == "mean":
                    continue
                df1 = [data[(pool, s)][key][0] - base_f1[s] for s in seeds
                       if s in base_f1 and data.get((pool, s)) and data[(pool, s)][key]]
                dac = [data[(pool, s)][key][1] - base_ac[s] for s in seeds
                       if s in base_ac and data.get((pool, s)) and data[(pool, s)][key]]
                if df1:
                    npos = sum(1 for x in df1 if x > 0)
                    print("  {:>8} vs mean: paired ΔF1 {:+.4f}+/-{:.4f} ({}/{} seeds +) | "
                          "Δacc {:+.4f}+/-{:.4f} | per-seed ΔF1 {}".format(
                              pool, np.mean(df1), np.std(df1), npos, len(df1),
                              np.mean(dac), np.std(dac),
                              ["{:+.3f}".format(x) for x in df1]))

    # verdict for PRIMARY (wsoftT1) under both lenses
    print("\n" + "=" * 78)
    print("VERDICT (PRIMARY wsoftT1 vs floor; criterion: >=2/3 seeds + AND mean ΔF1 > {:.2f}, "
          "BOTH protocols)".format(args.noise_pts))
    ok_both = True
    for lens, key in (("val-selected", "val"), ("final-epoch", "fin")):
        df1 = []
        for s in seeds:
            a = data.get(("wsoftT1", s))
            b = data.get(("mean", s))
            if a and b and a[key] and b[key]:
                df1.append(a[key][0] - b[key][0])
        if not df1:
            print("  {}: no data".format(lens)); ok_both = False; continue
        npos = sum(1 for x in df1 if x > 0)
        passed = (npos >= 2) and (np.mean(df1) > args.noise_pts)
        ok_both = ok_both and passed
        print("  {:>13}: mean ΔF1 {:+.4f}, {}/{} seeds +  -> {}".format(
            lens, np.mean(df1), npos, len(df1), "PASS" if passed else "within-noise/FAIL"))
    print("  OVERALL (both protocols): {}".format(
        "METHOD ROLE EARNED" if ok_both else "WITHIN-NOISE / NO CLAIM"))
    print("=" * 78)


if __name__ == "__main__":
    main()
