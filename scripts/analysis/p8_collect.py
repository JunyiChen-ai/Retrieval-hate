#!/usr/bin/env python
"""Parse P8 trainlogs -> A(floor)/B(summary)/C(trunc)/D(concat) table (CPU, read-only).

Trainlogs slurm/logs/p8sum_<ds>_<cond>_s<seed>.trainlog. Reports val-selected (best Val acc
epoch>=warmup) + final-epoch TEST macro-F1/acc, the bit-for-bit A-floor check, paired per-seed
deltas B/C/D vs A, and the pre-registered verdict (B>A >1pt both protocols >=2/3 seeds AND B>C).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

WARM = 5
KNOWN_FLOOR = {"MHC": (0.7826, 0.7113), "MHC_zh": (0.8054, 0.7706)}  # HateMM ~0.828 acc (report)
VAL_RE = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: [\d.]+ macroR: [\d.]+ "
                    r"acc: ([\d.]+) roc: ([\d.]+)")
TEST_RE = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: [\d.]+ macroR: [\d.]+ "
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
    comp = {e: d for e, d in ep.items() if all(k in d for k in ("val_acc", "val_roc", "test_acc", "test_maf1"))}
    if not comp:
        return None
    elig = [e for e in comp if e >= WARM] or list(comp)
    best = max(elig, key=lambda e: (comp[e]["val_acc"], comp[e]["val_roc"]))
    final = max(comp)
    return {"val_sel": {"epoch": best, "test_acc": comp[best]["test_acc"], "test_maf1": comp[best]["test_maf1"]},
            "final": {"epoch": final, "test_acc": comp[final]["test_acc"], "test_maf1": comp[final]["test_maf1"]}}


def paired(runs, ds, cond, seeds, proto):
    fl, tr, d = [], [], []
    for s in seeds:
        ra = runs.get((ds, "A", s)); rc = runs.get((ds, cond, s))
        if ra is None or rc is None:
            continue
        fl.append(ra[proto]["test_maf1"]); tr.append(rc[proto]["test_maf1"])
        d.append(rc[proto]["test_maf1"] - ra[proto]["test_maf1"])
    if not d:
        return None
    return {"A_maf1": round(float(np.mean(fl)), 4), "cond_maf1": round(float(np.mean(tr)), 4),
            "delta_mean": round(float(np.mean(d)), 4), "delta_per_seed": [round(x, 4) for x in d],
            "pos_seeds": int(sum(x > 0 for x in d)), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="slurm/logs")
    ap.add_argument("--datasets", default="MHC,MHC_zh,HateMM")
    ap.add_argument("--conds", default="A,B,C,D")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="scripts/analysis/p8_out/p8_results.json")
    args = ap.parse_args()
    datasets = args.datasets.split(","); conds = args.conds.split(","); seeds = [int(s) for s in args.seeds.split(",")]

    runs = {}
    for ds in datasets:
        for c in conds:
            for s in seeds:
                hits = glob.glob(os.path.join(args.logdir, "p8sum_{}_{}_s{}.trainlog".format(ds, c, s)))
                runs[(ds, c, s)] = parse_log(hits[0]) if hits else None

    result = {"runs": {"{}|{}|s{}".format(*k): v for k, v in runs.items()},
              "bit_for_bit": {}, "compare": {}, "verdict": {}}
    print("===== P8 results =====")
    print("\n[bit-for-bit] A(floor) seed0 val-sel TEST vs known floor:")
    for ds in datasets:
        r = runs.get((ds, "A", 0))
        if r is None:
            print("  {}: MISSING".format(ds)); continue
        got = (round(r["val_sel"]["test_acc"], 4), round(r["val_sel"]["test_maf1"], 4))
        exp = KNOWN_FLOOR.get(ds)
        ok = (exp is not None) and abs(got[0] - exp[0]) < 1e-4 and abs(got[1] - exp[1]) < 1e-4
        result["bit_for_bit"][ds] = {"got": got, "expected": exp, "exact": bool(ok) if exp else None}
        print("  {}: got acc/maF1={} expected={} -> {}".format(
            ds, got, exp, ("OK" if ok else "MISMATCH") if exp else "(report only, HateMM ~0.828)"))

    print("\n[A vs B/C/D] macro-F1 paired deltas:")
    for ds in datasets:
        result["compare"][ds] = {}
        for cond in ("B", "C", "D"):
            result["compare"][ds][cond] = {}
            for proto in ("val_sel", "final"):
                row = paired(runs, ds, cond, seeds, proto)
                if row is None:
                    continue
                result["compare"][ds][cond][proto] = row
                print("  {} {} [{:8s}] A={:.4f} {}={:.4f} dMaF1={:+.4f} {} ({}/{}+)".format(
                    ds, cond, proto, row["A_maf1"], cond, row["cond_maf1"],
                    row["delta_mean"], row["delta_per_seed"], row["pos_seeds"], row["n"]))

    # verdict: B beats A >0.01, >=2/3 seeds, both protocols, >=1 ds, AND B>C (rent test), no >0.01 harm
    win = []; harm = []
    for ds in datasets:
        cmp = result["compare"].get(ds, {})
        B = cmp.get("B", {}); C = cmp.get("C", {})
        b_ok = len(B) == 2 and all(B.get(p, {}).get("delta_mean", -9) > 0.01 and B.get(p, {}).get("pos_seeds", 0) >= 2
                                   for p in ("val_sel", "final"))
        # rent test: B final-epoch delta > C final-epoch delta (B beats trunc)
        rent = B.get("final", {}).get("delta_mean", -9) > C.get("final", {}).get("delta_mean", 9)
        if b_ok and rent:
            win.append(ds)
        for p in ("val_sel", "final"):
            if B.get(p, {}).get("delta_mean", 0.0) < -0.01:
                harm.append((ds, p, B[p]["delta_mean"]))
    passed = len(win) >= 1 and len(harm) == 0
    result["verdict"] = {"win_datasets": win, "harm": harm, "criterion_pass": bool(passed)}
    print("\n[verdict] B win (>0.01 both protocols, >=2/3 seeds, AND B>C rent): {} | harm: {} | {}".format(
        win, harm, "PASS" if passed else "FAIL/within-noise"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("\n[out] wrote", args.out)


if __name__ == "__main__":
    main()
