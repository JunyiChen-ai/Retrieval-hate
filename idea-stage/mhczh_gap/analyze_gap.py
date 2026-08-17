#!/usr/bin/env python
"""MHC-ZH baseline gap diagnostic.

Pairs, seed-for-seed over seeds 30..89:
  CURRIC = head trained on data/CLIP_Embedding/MHC_zh/*_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF.pt
           (the cache that produced the 0.7821 contrast line)
  PLAIN  = head trained on *_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt
           (bit-identical to *-ro_L28 and to R6RO-A0; the r6_confirm A0 arm)
PLAIN runs are read from logging/runs/r6_confirm/logs (verified bit-identical for s30).

Three read-out protocols computed from the SAME runs:
  P1  epoch = argmax_{e>=5} dev macro-F1 (ties -> earliest)      [r6 primary]
  P2  epoch = 29 (last)                                          [r6 corroboration]
  P1b epoch = argmax_{e>=5} (dev acc, dev roc) (ties -> earliest) [the deployed
      rgcl_ablation_analyze.py key that produced 0.7821]
"""
import os, sys, json
import numpy as np
ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
from analyze_audit import parse, select_epoch  # noqa

SEEDS = list(range(30, 90))
PROTS = ["P1", "P2", "P1b"]
CUR = os.path.join(ROOT, "logging/runs/mhczh_gap/logs/MHC_zh_CURRIC_s%d.trainlog")
PLA = os.path.join(ROOT, "logging/runs/r6_confirm/logs/MHC_zh_A0_s%d.trainlog")
CAT = os.path.join(ROOT, "logging/runs/r6_confirm/logs/MHC_zh_CAT_s%d.trainlog")

def series(pathfmt):
    out = {p: [] for p in PROTS}
    for s in SEEDS:
        r, err = parse(pathfmt % s, "MHC_zh")
        assert r is not None, (pathfmt % s, err)
        for p in PROTS:
            out[p].append(r["test"][select_epoch(r["dev"], p)]["macro_f1"])
    return {p: np.array(v) for p, v in out.items()}

def boot(d, n=20000, seed=20260817):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

cur, pla, cat = series(CUR), series(PLA), series(CAT)
res = {"seeds": SEEDS, "abs": {}, "deltas": {}}
print("absolute test macro-F1, 60 seeds (30..89)")
print("%-4s %-8s %-8s %-8s %-8s" % ("prot", "CURRIC", "PLAIN", "CAT", "std_cur/std_pla"))
for p in PROTS:
    res["abs"][p] = {"CURRIC": float(cur[p].mean()), "PLAIN": float(pla[p].mean()),
                     "CAT": float(cat[p].mean()),
                     "CURRIC_std": float(cur[p].std(ddof=1)),
                     "PLAIN_std": float(pla[p].std(ddof=1)),
                     "CAT_std": float(cat[p].std(ddof=1))}
    print("%-4s %-8.4f %-8.4f %-8.4f %.4f/%.4f" % (p, cur[p].mean(), pla[p].mean(),
          cat[p].mean(), cur[p].std(ddof=1), pla[p].std(ddof=1)))
print()
print("paired deltas over the same 60 seeds")
for name, a, b in [("PLAIN-CURRIC", pla, cur), ("CAT-CURRIC", cat, cur), ("CAT-PLAIN", cat, pla)]:
    res["deltas"][name] = {}
    for p in PROTS:
        d = a[p] - b[p]
        lo, hi = boot(d)
        res["deltas"][name][p] = {"mean": float(d.mean()), "std": float(d.std(ddof=1)),
                                  "mc_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                                  "ci95": [lo, hi], "pos": int((d > 0).sum())}
        print("%-13s %-4s mean %+.4f  MC SE %.4f  CI [%+.4f, %+.4f]  pos %d/60"
              % (name, p, d.mean(), d.std(ddof=1) / np.sqrt(len(d)), lo, hi, int((d > 0).sum())))
print()
print("3-seed sub-reads (what a 3-seed protocol would have reported), seeds 30,31,32")
for p in PROTS:
    print("  %-4s CURRIC %.4f  PLAIN %.4f" % (p, cur[p][:3].mean(), pla[p][:3].mean()))
with open(os.path.join(ROOT, "idea-stage/mhczh_gap/results.json"), "w") as f:
    json.dump(res, f, indent=1)
print("wrote idea-stage/mhczh_gap/results.json")
