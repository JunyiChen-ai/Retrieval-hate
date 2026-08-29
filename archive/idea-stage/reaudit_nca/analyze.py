#!/usr/bin/env python3
"""RE-AUDIT NCA readout + paired-bootstrap analysis.

Parses the Val_Retrieval / Test_Retrieval macroF1/acc lines of a trainlog and
produces the frozen read-outs:
  P1  = epoch (>= warmup 5) maximising Val_Retrieval macroF1 -> Test_Retrieval macroF1
  P2  = final epoch Test_Retrieval macroF1
  ORIGSEL (descriptive, non-gating) = epoch (>= 5) maximising (Val acc, Val roc)
          -> Test_Retrieval acc and macroF1, i.e. the 2026-07-25 selection rule.
"""
import argparse, glob, json, os, re, sys
import numpy as np

VRE = re.compile(r"Val_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
TRE = re.compile(r"Test_Retrieval Epoch\s+(\d+) macroF1: ([\d.]+) macroP: ([\d.]+) macroR: ([\d.]+) acc: ([\d.]+) roc: ([\d.]+)")
WARMUP = 5
KEYS = ("p1_f1", "p2_f1", "orig_f1", "orig_acc", "p1_acc",
        "dev_p1_f1", "dev_orig_acc", "dev_best_f1", "dev_best_acc")


def read_log(path):
    txt = open(path).read()
    val, test = {}, {}
    for m in VRE.finditer(txt):
        e = int(m.group(1))
        val[e] = dict(f1=float(m.group(2)), acc=float(m.group(5)), roc=float(m.group(6)))
    for m in TRE.finditer(txt):
        e = int(m.group(1))
        test[e] = dict(f1=float(m.group(2)), acc=float(m.group(5)), roc=float(m.group(6)))
    if not val or not test:
        raise RuntimeError("NO_PARSE " + path)
    warm = [e for e in sorted(val) if e >= WARMUP] or sorted(val)
    e_p1 = max(warm, key=lambda e: (val[e]["f1"], e * -1e-9))
    e_orig = max(warm, key=lambda e: (val[e]["acc"], val[e]["roc"]))
    e_fin = max(test)
    return dict(
        n_val=len(val), n_test=len(test),
        p1_epoch=e_p1, p1_f1=test[e_p1]["f1"], p1_acc=test[e_p1]["acc"],
        p2_epoch=e_fin, p2_f1=test[e_fin]["f1"], p2_acc=test[e_fin]["acc"],
        orig_epoch=e_orig, orig_f1=test[e_orig]["f1"], orig_acc=test[e_orig]["acc"],
        dev_p1_f1=val[e_p1]["f1"], dev_orig_acc=val[e_orig]["acc"],
        dev_best_f1=max(val[e]["f1"] for e in warm),
        dev_best_acc=max(val[e]["acc"] for e in warm),
    )


def boot(delta, n=20000, seed=12345):
    rng = np.random.default_rng(seed)
    k = len(delta)
    idx = rng.integers(0, k, size=(n, k))
    means = delta[idx].mean(axis=1)
    return float(means.std(ddof=1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", required=True, help="comma list")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = ["floor", "nca01"]
    per = {arm: {} for arm in arms}
    missing = []
    for arm in arms:
        for s in seeds:
            p = os.path.join(a.logdir, f"{a.dataset}_{arm}_s{s}.trainlog")
            if not os.path.exists(p):
                missing.append(p); continue
            per[arm][s] = read_log(p)
    if missing:
        print("MISSING:", *missing, sep="\n"); sys.exit(1)

    res = {"dataset": a.dataset, "seeds": seeds, "arms": {}, "contrasts": {}}
    for arm in arms:
        for key in KEYS:
            v = np.array([per[arm][s][key] for s in seeds])
            res["arms"].setdefault(arm, {})[key] = dict(
                mean=float(v.mean()), std=float(v.std(ddof=1)))
        res["arms"][arm]["mean_p1_epoch"] = float(np.mean([per[arm][s]["p1_epoch"] for s in seeds]))
    for key in KEYS:
        d = np.array([per["nca01"][s][key] - per["floor"][s][key] for s in seeds])
        se, lo, hi = boot(d)
        res["contrasts"][key] = dict(mean=float(d.mean()), std=float(d.std(ddof=1)),
                                     boot_se=se, ci_lo=lo, ci_hi=hi,
                                     n_pos=int((d > 0).sum()), n=len(d),
                                     per_seed={str(s): float(x) for s, x in zip(seeds, d)})
    res["per_run"] = {arm: {str(s): per[arm][s] for s in seeds} for arm in arms}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("dataset", "arms", "contrasts")}, indent=2))


if __name__ == "__main__":
    main()
