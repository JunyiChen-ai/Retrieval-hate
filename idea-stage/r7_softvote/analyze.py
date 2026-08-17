#!/usr/bin/env python
"""R7-1 -- annotator-vote soft-label training: read-out and frozen decision rule.

Frozen design: idea-stage/R7_SOFTVOTE_FREEZE.md.
Parser / CM reconstruction / epoch selector imported verbatim from
idea-stage/r6_audit/analyze_audit.py (single parser, no divergent copy).

Grid: 2 datasets x 5 arms {A0,SOFT10,SOFT05,LS10,LS05} x 30 seeds (100..129).

P1 (primary)      epoch = argmax_{e>=5} dev macro-F1 (ties -> earliest); test macro-F1 @0.5
P2 (corroboration) epoch = 29; test macro-F1 @0.5
"""
import json
import os
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
import analyze_audit as AA  # noqa: E402

# MHC (EN) is not in the audit's SPLIT table; add it (from
# data/CLIP_Embedding/MHC/test_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt: N=161, P=49)
AA.SPLIT["MHC"] = dict(N=161, P=49)
SPLIT = AA.SPLIT

DATASETS = ["MHC_zh", "MHC"]
ARMS = ["A0", "SOFT10", "SOFT05", "LS10", "LS05"]
SEEDS = list(range(100, 130))
PROTOCOLS = ["P1", "P2"]
GO_BAR = 0.005
NO_HARM_BAR = -0.002
N_BOOT = 20000
BOOT_SEED = 20260817
SOFT_TO_LS = {"SOFT10": "LS10", "SOFT05": "LS05"}

PAIRS = [("SOFT10", "A0"), ("SOFT05", "A0"),
         ("LS10", "A0"), ("LS05", "A0"),
         ("SOFT10", "LS10"), ("SOFT05", "LS05")]


def boot_ci(d, rng, n_boot, level=95.0):
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo_p = (100.0 - level) / 2.0
    lo, hi = np.percentile(means, [lo_p, 100.0 - lo_p])
    return float(lo), float(hi)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="logging/runs/r7_softvote/logs")
    ap.add_argument("--out", default="idea-stage/r7_softvote/results.json")
    a = ap.parse_args()
    rng = np.random.default_rng(BOOT_SEED)

    runs, bad = {}, []
    for ds in DATASETS:
        for arm in ARMS:
            for s in SEEDS:
                p = os.path.join(ROOT, a.logdir, "%s_%s_s%d.trainlog" % (ds, arm, s))
                if not os.path.exists(p):
                    bad.append(("%s_%s_s%d" % (ds, arm, s), "missing")); continue
                r, err = AA.parse(p, ds)
                if r is None:
                    bad.append(("%s_%s_s%d" % (ds, arm, s), err)); continue
                runs[(ds, arm, s)] = r
    if bad:
        for r, w in bad:
            print("  BAD %s  %s" % (r, w))
        raise SystemExit("HALT: %d of %d runs unusable; the frozen rule is defined "
                         "over 30 complete paired seeds."
                         % (len(bad), len(DATASETS) * len(ARMS) * len(SEEDS)))

    out = {"what": "R7-1 annotator-vote soft-label training. Frozen: "
                   "idea-stage/R7_SOFTVOTE_FREEZE.md",
           "grid": "2 datasets x 5 arms x 30 seeds (100..129) = 300 head runs",
           "seed_independence": "R6 audit used 0-29, R6 confirmation 30-89; this uses 100-129",
           "protocols": {"P1": "epoch=argmax_{e>=5} dev macroF1 (earliest tie); test macroF1@0.5",
                         "P2": "epoch=29; test macroF1@0.5"},
           "splits": {d: SPLIT[d] for d in DATASETS},
           "n_boot": N_BOOT, "bootstrap_rng_seed": BOOT_SEED,
           "arms": {}, "deltas": {}, "verdict": {}}

    per = {}   # (ds, proto, arm) -> np.array over seeds
    for ds in DATASETS:
        for proto in PROTOCOLS:
            for arm in ARMS:
                v = []
                for s in SEEDS:
                    r = runs[(ds, arm, s)]
                    e = AA.select_epoch(r["dev"], proto)
                    v.append(r["test"][e]["macro_f1"])
                v = np.array(v, float)
                per[(ds, proto, arm)] = v
                out["arms"]["%s/%s/%s" % (ds, proto, arm)] = {
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                    "min": float(v.min()), "max": float(v.max()),
                    "per_seed": [float(x) for x in v]}

    for ds in DATASETS:
        for proto in PROTOCOLS:
            for L, R in PAIRS:
                d = per[(ds, proto, L)] - per[(ds, proto, R)]
                lo95, hi95 = boot_ci(d, rng, N_BOOT, 95.0)
                lo975, hi975 = boot_ci(d, rng, N_BOOT, 97.5)
                out["deltas"]["%s/%s/%s-%s" % (ds, proto, L, R)] = {
                    "mean": float(d.mean()), "std": float(d.std(ddof=1)),
                    "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                    "ci95": [lo95, hi95], "ci_bonf975": [lo975, hi975],
                    "n_pos": int((d > 0).sum()), "n": int(len(d))}

    # ---------------- frozen decision rule ----------------
    ds_pass, detail = {}, {}
    for ds in DATASETS:
        cands = {w: out["deltas"]["%s/P1/%s-A0" % (ds, w)]["mean"] for w in SOFT_TO_LS}
        wstar = max(cands, key=cands.get)
        dstar = out["deltas"]["%s/P1/%s-A0" % (ds, wstar)]
        dls = out["deltas"]["%s/P1/%s-%s" % (ds, wstar, SOFT_TO_LS[wstar])]
        c1 = dstar["mean"] >= GO_BAR
        c2 = dstar["ci_bonf975"][0] > 0 or dstar["ci_bonf975"][1] < 0
        c3 = dls["mean"] > 0
        p2 = out["deltas"]["%s/P2/%s-A0" % (ds, wstar)]["mean"]
        ds_pass[ds] = bool(c1 and c2 and c3)
        detail[ds] = {"w_star": wstar,
                      "P1_mean_SOFT_minus_A0": dstar["mean"],
                      "P1_ci_bonf975": dstar["ci_bonf975"],
                      "P1_ci95": dstar["ci95"],
                      "P1_mean_SOFT_minus_LS": dls["mean"],
                      "P1_ci95_SOFT_minus_LS": dls["ci95"],
                      "P2_mean_SOFT_minus_A0": p2,
                      "c1_bar": bool(c1), "c2_ci_excludes_0": bool(c2),
                      "c3_beats_LS": bool(c3),
                      "P2_agrees_in_sign": bool(np.sign(p2) == np.sign(dstar["mean"])
                                                and dstar["mean"] != 0),
                      "PASS": ds_pass[ds]}

    npass = sum(ds_pass.values())
    if npass == 2 and all(detail[d]["P2_agrees_in_sign"] for d in DATASETS):
        verdict = "GO-2DS"
    elif npass == 1:
        p = [d for d in DATASETS if ds_pass[d]][0]
        o = [d for d in DATASETS if not ds_pass[d]][0]
        other = out["deltas"]["%s/P1/%s-A0" % (o, detail[o]["w_star"])]["mean"]
        verdict = ("GO-1DS" if detail[p]["P2_agrees_in_sign"] and other >= NO_HARM_BAR
                   else "KILL")
    else:
        verdict = "KILL"

    # TRICK check: bar+CI met somewhere but the LS control is not beaten there
    trick = []
    for ds in DATASETS:
        w = detail[ds]["w_star"]
        d = out["deltas"]["%s/P1/%s-A0" % (ds, w)]
        if d["mean"] >= GO_BAR and (d["ci_bonf975"][0] > 0 or d["ci_bonf975"][1] < 0) \
                and not detail[ds]["c3_beats_LS"]:
            trick.append(ds)
    if trick and verdict == "KILL":
        verdict = "TRICK"

    out["verdict"] = {"verdict": verdict, "datasets_passing": ds_pass,
                      "trick_datasets": trick, "detail": detail}

    op = os.path.join(ROOT, a.out)
    with open(op, "w") as f:
        json.dump(out, f, indent=1)

    print("\n=== arm means (P1 primary) ===")
    for ds in DATASETS:
        print(" %s" % ds)
        for arm in ARMS:
            k = "%s/P1/%s" % (ds, arm)
            print("   %-7s mean %.4f  std %.4f  se %.4f"
                  % (arm, out["arms"][k]["mean"], out["arms"][k]["std"],
                     out["arms"][k]["se"]))
    print("\n=== paired deltas (P1) ===")
    for ds in DATASETS:
        for L, R in PAIRS:
            d = out["deltas"]["%s/P1/%s-%s" % (ds, L, R)]
            print(" %-7s %-14s mean %+.4f  ci95 [%+.4f, %+.4f]  %d/30 pos"
                  % (ds, "%s-%s" % (L, R), d["mean"], d["ci95"][0], d["ci95"][1],
                     d["n_pos"]))
    print("\n=== P2 corroboration ===")
    for ds in DATASETS:
        for L, R in PAIRS:
            d = out["deltas"]["%s/P2/%s-%s" % (ds, L, R)]
            print(" %-7s %-14s mean %+.4f  ci95 [%+.4f, %+.4f]"
                  % (ds, "%s-%s" % (L, R), d["mean"], d["ci95"][0], d["ci95"][1]))
    print("\n=== VERDICT: %s ===" % verdict)
    print(json.dumps(out["verdict"], indent=1))
    print("wrote", op)


if __name__ == "__main__":
    main()
