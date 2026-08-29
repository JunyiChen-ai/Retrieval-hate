#!/usr/bin/env python
"""R8-1 BLR read-out and frozen decision rule.

Applies idea-stage/R8_BLR_FREEZE.md §4 verbatim. Run exactly once on the complete grid.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]
ARMS = ["A0", "BALBCE", "PAIRG", "PAIRL", "RANDL"]
BAR, N_BOOT, BOOT_SEED = 0.005, 20000, 20260817
PRIMARY = [("PAIRL", "A0"), ("PAIRL", "PAIRG"), ("PAIRL", "BALBCE"), ("PAIRL", "RANDL")]
SECONDARY = [("PAIRG", "A0"), ("BALBCE", "A0"), ("PAIRG", "BALBCE"), ("RANDL", "PAIRG")]


def boot_ci(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    raw = json.load(open(os.path.join(HERE, "results.json")))
    rng = np.random.default_rng(BOOT_SEED)
    nseed = len(raw["meta"]["seeds"])
    out = {"meta": raw["meta"], "arm_means": {}, "contrasts": {}, "verdict": {}}

    # completeness / VOID guard
    for ds in DATASETS:
        for arm in ARMS:
            rows = raw["runs"][ds][arm]
            assert len(rows) == nseed, f"{ds}/{arm}: {len(rows)} rows, expected {nseed}"
            for r in rows:
                for k in ("P1", "P2", "P1_roc", "P2_roc"):
                    if not np.isfinite(r[k]):
                        print(f"VOID: non-finite {k} in {ds}/{arm}")
                        sys.exit(1)

    V = {ds: {a: {k: np.array([r[k] for r in raw["runs"][ds][a]])
                  for k in ("P1", "P2", "P1_roc", "P2_roc")} for a in ARMS}
         for ds in DATASETS}

    for ds in DATASETS:
        out["arm_means"][ds] = {
            a: {k: {"mean": float(V[ds][a][k].mean()), "sd": float(V[ds][a][k].std(ddof=1))}
                for k in ("P1", "P2", "P1_roc", "P2_roc")} for a in ARMS}
        out["contrasts"][ds] = {}
        for L, R in PRIMARY + SECONDARY:
            cell = {}
            for k in ("P1", "P2", "P1_roc", "P2_roc"):
                d = V[ds][L][k] - V[ds][R][k]
                lo, hi = boot_ci(d, rng)
                cell[k] = {"mean": float(d.mean()), "ci": [lo, hi],
                           "pos": int((d > 0).sum()), "n": len(d)}
            out["contrasts"][ds][f"{L}-{R}"] = cell

        c = out["contrasts"][ds]
        cond = {
            "1_vs_A0": c["PAIRL-A0"]["P1"]["mean"] >= BAR and c["PAIRL-A0"]["P1"]["ci"][0] > 0,
            "2_vs_PAIRG": (c["PAIRL-PAIRG"]["P1"]["mean"] >= BAR
                           and c["PAIRL-PAIRG"]["P1"]["ci"][0] > 0),
            "3_vs_BALBCE": (c["PAIRL-BALBCE"]["P1"]["mean"] >= BAR
                            and c["PAIRL-BALBCE"]["P1"]["ci"][0] > 0),
            "4_vs_RANDL": c["PAIRL-RANDL"]["P1"]["mean"] > 0,
        }
        out["verdict"][ds] = {"conditions": cond, "passes": all(cond.values()),
                              "P2_sign_agrees": c["PAIRL-A0"]["P2"]["mean"] > 0}

    npass = sum(v["passes"] for v in out["verdict"].values())
    p2ok = all(out["verdict"][d]["P2_sign_agrees"]
               for d in DATASETS if out["verdict"][d]["passes"])
    out["VERDICT"] = ("GO" if (npass >= 2 and p2ok) else
                      "WEAK" if npass == 1 else "KILL")
    out["n_datasets_passing"] = npass
    json.dump(out, open(os.path.join(HERE, "verdict.json"), "w"), indent=2)

    print(f"seeds={nseed}  VERDICT = {out['VERDICT']}  ({npass}/4 datasets pass)\n")
    for ds in DATASETS:
        m = out["arm_means"][ds]
        print(f"== {ds}")
        print("   arm      P1 mean   sd      P2 mean   P1 roc")
        for a in ARMS:
            print(f"   {a:7s} {m[a]['P1']['mean']:.4f}  {m[a]['P1']['sd']:.4f}  "
                  f"{m[a]['P2']['mean']:.4f}  {m[a]['P1_roc']['mean']:.4f}")
        for L, R in PRIMARY + SECONDARY:
            k = f"{L}-{R}"
            cc = out["contrasts"][ds][k]
            print(f"   {k:16s} P1 {cc['P1']['mean']:+.4f} "
                  f"[{cc['P1']['ci'][0]:+.4f},{cc['P1']['ci'][1]:+.4f}] {cc['P1']['pos']:2d}/{nseed}"
                  f" | P2 {cc['P2']['mean']:+.4f} | dROC {cc['P1_roc']['mean']:+.4f} "
                  f"[{cc['P1_roc']['ci'][0]:+.4f},{cc['P1_roc']['ci'][1]:+.4f}]")
        print(f"   conditions: {out['verdict'][ds]['conditions']} -> "
              f"{'PASS' if out['verdict'][ds]['passes'] else 'fail'}\n")
    print("WROTE", os.path.join(HERE, "verdict.json"))


if __name__ == "__main__":
    main()
