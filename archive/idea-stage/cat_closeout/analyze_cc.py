#!/usr/bin/env python
"""CAT close-out -- generic read-out from dumped per-item head logits.

Frozen design: idea-stage/CAT_CLOSEOUT_FREEZE.md section 7.

Unlike idea-stage/reaudit/analyze_grid.py this analyzer reconstructs NOTHING: dev and
test macro-F1 are computed EXACTLY from the per-item logits dumped by
--dump_head_scores, so it needs no hard-coded split sizes and works unchanged for
MHC-EN and for the CV cells.

  P1 (judging)    epoch = argmax_{e>=5} dev macro-F1 (earliest tie); test macro-F1 @0.5
  P2 (sign only)  epoch = 29

BELT E1 (gating): for every run, |macroF1(dumped logits) - macroF1(trainlog)| <= 1e-4
on BOTH splits at every epoch; otherwise HALT.

Test labels are read only for the final metric.  No threshold, epoch rule or arm
definition is selected on them.
"""
import argparse
import json
import os
import re

import numpy as np

BAR = 0.005
N_BOOT = 20000
BOOT_SEED = 20260817
WARMUP = 5
LAST_EPOCH = 29
BELT_TOL = 1e-4

RE_DEV = re.compile(r"^dev\s+Epoch (\d+) .*\| macroF1: ([\d.]+)", re.M)
RE_TEST = re.compile(r"^test Epoch (\d+) .*\| macroF1: ([\d.]+)", re.M)


def macro_f1(y, pred):
    f1s = []
    for pos in (1, 0):
        tp = int(((pred == pos) & (y == pos)).sum())
        fp = int(((pred == pos) & (y != pos)).sum())
        fn = int(((pred != pos) & (y == pos)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def read_scores(path):
    """-> {split: {epoch: (ids, y, logits)}}"""
    out = {"dev": {}, "test": {}}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[d["split"]][int(d["epoch"])] = (
                d["ids"], np.asarray(d["labels"]), np.asarray(d["logits"], dtype=float))
    return out


def curves(path_scores, path_trainlog):
    sc = read_scores(path_scores)
    txt = open(path_trainlog, errors="replace").read()
    logged = {"dev": {int(m.group(1)): float(m.group(2)) for m in RE_DEV.finditer(txt)},
              "test": {int(m.group(1)): float(m.group(2)) for m in RE_TEST.finditer(txt)}}
    res, worst = {"dev": {}, "test": {}}, 0.0
    for sp in ("dev", "test"):
        if len(sc[sp]) != 30 or len(logged[sp]) != 30:
            raise SystemExit("HALT: incomplete run %s (%d dumped / %d logged %s epochs)"
                             % (path_trainlog, len(sc[sp]), len(logged[sp]), sp))
        for e, (_ids, y, z) in sc[sp].items():
            m = macro_f1(y, (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int))
            res[sp][e] = m
            worst = max(worst, abs(m - logged[sp][e]))
    return res, sc, worst


def select_p1(dev):
    cand = sorted(e for e in dev if e >= WARMUP)
    best = cand[0]
    for e in cand:
        if dev[e] > dev[best]:
            best = e
    return best


def paired_boot(d, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--contrasts", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    pairs = [tuple(c.split("-", 1)) for c in a.contrasts.split(",")]
    rng = np.random.default_rng(BOOT_SEED)

    f1 = {p: {} for p in ("P1", "P2")}
    epochs, belt_worst = {}, 0.0
    for arm in arms:
        for s in seeds:
            tl = os.path.join(a.logdir, "%s_%s_s%d.trainlog" % (a.dataset, arm, s))
            sj = os.path.join(a.logdir, "%s_%s_s%d.scores.jsonl" % (a.dataset, arm, s))
            for p in (tl, sj):
                if not os.path.exists(p):
                    raise SystemExit("HALT: missing %s" % p)
            cur, _sc, worst = curves(sj, tl)
            belt_worst = max(belt_worst, worst)
            e1 = select_p1(cur["dev"])
            epochs[(arm, s)] = e1
            f1["P1"][(arm, s)] = cur["test"][e1]
            f1["P2"][(arm, s)] = cur["test"][LAST_EPOCH]

    print("BELT E1 max |macroF1(dumped) - macroF1(trainlog)| = %.2e" % belt_worst)
    if belt_worst > BELT_TOL:
        raise SystemExit("HALT: BELT E1 failed (%.3e > %.3e)" % (belt_worst, BELT_TOL))

    res = {"what": "CAT close-out read-out", "label": a.label, "dataset": a.dataset,
           "arms": arms, "seeds": seeds, "n_units": len(seeds), "bar": BAR,
           "belt_E1_max_abs_diff": belt_worst, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "arm_levels": {}, "contrasts": {}}

    for p in ("P1", "P2"):
        res["arm_levels"][p] = {}
        for arm in arms:
            v = np.array([f1[p][(arm, s)] for s in seeds])
            res["arm_levels"][p][arm] = {"mean": float(v.mean()),
                                         "std": float(v.std(ddof=1)),
                                         "per_unit": [float(x) for x in v]}
            print("%-4s %-6s mean=%.4f +- %.4f" % (p, arm, v.mean(), v.std(ddof=1)))

    for L, Rn in pairs:
        key = "%s-%s" % (L, Rn)
        res["contrasts"][key] = {}
        for p in ("P1", "P2"):
            d = np.array([f1[p][(L, s)] - f1[p][(Rn, s)] for s in seeds])
            lo, hi = paired_boot(d, rng)
            res["contrasts"][key][p] = {
                "mean": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                "n_pos": int((d > 0).sum()), "std": float(d.std(ddof=1)),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "clears_bar_and_ci": bool(d.mean() >= BAR and lo > 0),
                "per_unit": [float(x) for x in d]}
        c = res["contrasts"][key]["P1"]
        print("%-12s P1 mean=%+.4f [%+.4f,%+.4f] %d/%d>0   P2 mean=%+.4f"
              % (key, c["mean"], c["ci_lo"], c["ci_hi"], c["n_pos"], len(seeds),
                 res["contrasts"][key]["P2"]["mean"]))

    res["mean_p1_epoch"] = {arm: float(np.mean([epochs[(arm, s)] for s in seeds]))
                            for arm in arms}
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
