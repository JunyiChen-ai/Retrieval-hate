#!/usr/bin/env python
"""Parse P5 trainlogs -> floor vs cf vs cfrand table (CPU, read-only).

Trainlogs: slurm/logs/p5cf_<ds>_<cond>_s<seed>.trainlog, cond in {floor, cf, cfrand}.
Reports val-selected (best Val acc epoch >= warmup) and final-epoch TEST macro-F1/acc, the
bit-for-bit floor check, and paired per-seed deltas cf-vs-floor and cfrand-vs-floor.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

WARM = 5
KNOWN_FLOOR = {"MHC": (0.7826, 0.7113), "MHC_zh": (0.8054, 0.7706)}
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
                e = int(m.group(1)); ep.setdefault(e, {})
                ep[e].update(val_acc=float(m.group(3)), val_roc=float(m.group(4)))
            m = TEST_RE.search(line)
            if m:
                e = int(m.group(1)); ep.setdefault(e, {})
                ep[e].update(test_maf1=float(m.group(2)), test_acc=float(m.group(3)))
    complete = {e: d for e, d in ep.items()
                if all(k in d for k in ("val_acc", "val_roc", "test_acc", "test_maf1"))}
    if not complete:
        return None
    elig = [e for e in complete if e >= WARM] or list(complete)
    best = max(elig, key=lambda e: (complete[e]["val_acc"], complete[e]["val_roc"]))
    final = max(complete)
    return {"val_sel": {"epoch": best, "test_acc": complete[best]["test_acc"],
                        "test_maf1": complete[best]["test_maf1"]},
            "final": {"epoch": final, "test_acc": complete[final]["test_acc"],
                      "test_maf1": complete[final]["test_maf1"]}}


def paired(runs, ds, cond, seeds, proto):
    fl, tr, d = [], [], []
    for s in seeds:
        rf = runs.get((ds, "floor", s)); rt = runs.get((ds, cond, s))
        if rf is None or rt is None:
            continue
        fl.append(rf[proto]["test_maf1"]); tr.append(rt[proto]["test_maf1"])
        d.append(rt[proto]["test_maf1"] - rf[proto]["test_maf1"])
    if not d:
        return None
    return {"floor_maf1": round(float(np.mean(fl)), 4),
            "cond_maf1": round(float(np.mean(tr)), 4),
            "delta_maf1_mean": round(float(np.mean(d)), 4),
            "delta_per_seed": [round(x, 4) for x in d],
            "pos_seeds": int(sum(x > 0 for x in d)), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="slurm/logs")
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--conds", default="floor,cf,cfrand")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="scripts/analysis/p5_out/p5_results.json")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    conds = args.conds.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = {}
    for ds in datasets:
        for c in conds:
            for s in seeds:
                hits = glob.glob(os.path.join(args.logdir, "p5cf_{}_{}_s{}.trainlog".format(ds, c, s)))
                runs[(ds, c, s)] = parse_log(hits[0]) if hits else None

    result = {"runs": {"{}|{}|s{}".format(*k): v for k, v in runs.items()},
              "bit_for_bit": {}, "compare": {}, "verdict": {}}

    print("===== P5 results =====")
    print("\n[bit-for-bit] floor seed0 val-sel TEST vs known floor:")
    for ds in datasets:
        r = runs.get((ds, "floor", 0))
        if r is None:
            print("  {}: MISSING".format(ds)); continue
        got = (round(r["val_sel"]["test_acc"], 4), round(r["val_sel"]["test_maf1"], 4))
        exp = KNOWN_FLOOR.get(ds)
        ok = exp and abs(got[0] - exp[0]) < 1e-4 and abs(got[1] - exp[1]) < 1e-4
        result["bit_for_bit"][ds] = {"got": got, "expected": exp, "exact": bool(ok)}
        print("  {}: got={} expected={} -> {}".format(ds, got, exp, "OK" if ok else "MISMATCH"))

    print("\n[floor vs cf / cfrand] macro-F1 paired deltas:")
    for ds in datasets:
        result["compare"][ds] = {}
        for cond in ("cf", "cfrand"):
            result["compare"][ds][cond] = {}
            for proto in ("val_sel", "final"):
                row = paired(runs, ds, cond, seeds, proto)
                if row is None:
                    continue
                result["compare"][ds][cond][proto] = row
                print("  {} {:6s} [{:8s}] floor={:.4f} {}={:.4f} dMaF1={:+.4f} {} ({}/{}+)".format(
                    ds, cond, proto, row["floor_maf1"], cond, row["cond_maf1"],
                    row["delta_maf1_mean"], row["delta_per_seed"], row["pos_seeds"], row["n"]))

    # verdict: cf beats floor >0.01, >=2/3 seeds, both protocols, >=1 dataset, no >0.01 harm;
    # and cf should beat cfrand (pairing matters).
    win = []
    harm = []
    for ds in datasets:
        cf = result["compare"].get(ds, {}).get("cf", {})
        both = len(cf) == 2 and all(
            cf.get(p, {}).get("delta_maf1_mean", -9) > 0.01 and cf.get(p, {}).get("pos_seeds", 0) >= 2
            for p in ("val_sel", "final"))
        if both:
            win.append(ds)
        for p in ("val_sel", "final"):
            d = cf.get(p, {}).get("delta_maf1_mean", 0.0)
            if d < -0.01:
                harm.append((ds, p, d))
    passed = len(win) >= 1 and len(harm) == 0
    result["verdict"] = {"win_datasets": win, "harm": harm, "criterion_pass": bool(passed)}
    print("\n[verdict] cf win datasets: {} | harm: {} | criterion {}".format(
        win, harm, "PASS" if passed else "FAIL/within-noise"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n[out] wrote", args.out)


if __name__ == "__main__":
    main()
