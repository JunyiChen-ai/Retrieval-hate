#!/usr/bin/env python
"""R10-COMBO diagnostic 1 -- test error-set overlap between arms.

DIAGNOSTIC ONLY, no verdict power (idea-stage/R10_COMBO_FREEZE.md 4.1).

For each seed, each arm's P1 epoch is selected from its own dev curve (never
test).  The per-item head logits dumped by --dump_head_scores at that epoch give
the exact prediction set behind the reported macro-F1; a belt recomputes macro-F1
from the dumped logits and requires agreement with the trainlog to 1e-4.

Reported per arm pair: mean over seeds of Jaccard(E_A, E_B) where E_X is the set
of misclassified test ids, together with an independence null obtained by drawing
two random subsets of the same observed sizes from the test split (2000 draws per
seed).  Ratio observed / null is the redundancy read-out.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r6_audit"))
from analyze_audit import parse, select_epoch, macro_f1_from_cm  # noqa: E402

NULL_DRAWS = 2000
RNG_SEED = 20260817


def load_epoch_scores(path, epoch, split="test"):
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d["epoch"] == epoch and d["split"] == split:
                return d["ids"], np.asarray(d["labels"]), np.asarray(d["logits"])
    raise SystemExit("HALT: epoch %d/%s not in %s" % (epoch, split, path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    rng = np.random.default_rng(RNG_SEED)

    err = {}       # (arm, seed) -> boolean mask over test items, in id order
    ids_ref = None
    belt = []
    for arm in arms:
        for s in seeds:
            tl = os.path.join(a.logdir, "%s_%s_s%d.trainlog" % (a.dataset, arm, s))
            sc = os.path.join(a.logdir, "%s_%s_s%d.scores.jsonl" % (a.dataset, arm, s))
            r, e_ = parse(tl, a.dataset)
            if r is None:
                raise SystemExit("HALT: %s -> %s" % (tl, e_))
            ep = select_epoch(r["dev"], "P1")
            ids, y, z = load_epoch_scores(sc, ep, "test")
            if ids_ref is None:
                ids_ref = ids
            assert ids == ids_ref, "test id order differs (%s s%d)" % (arm, s)
            pred = (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())
            mf1 = macro_f1_from_cm(tp, fp, fn, tn)
            belt.append(abs(mf1 - r["test"][ep]["macro_f1"]))
            err[(arm, s)] = (pred != y)

    belt_max = float(max(belt))
    print("BELT max |macroF1(dumped logits) - macroF1(trainlog)| = %.2e" % belt_max)
    if belt_max > 1e-4:
        raise SystemExit("HALT: dumped logits do not reproduce the logged metric")

    n_items = len(ids_ref)
    res = {"what": "R10-COMBO error-set overlap", "dataset": a.dataset,
           "arms": arms, "seeds": seeds, "n_test": n_items,
           "belt_max_abs_macro_f1_diff": belt_max,
           "mean_n_errors": {arm: float(np.mean([err[(arm, s)].sum() for s in seeds]))
                             for arm in arms},
           "pairs": {}}

    for i, A in enumerate(arms):
        for B in arms[i + 1:]:
            js, nulls = [], []
            for s in seeds:
                ea, eb = err[(A, s)], err[(B, s)]
                inter = int((ea & eb).sum())
                union = int((ea | eb).sum())
                js.append(inter / union if union else 1.0)
                na, nb = int(ea.sum()), int(eb.sum())
                sa = np.argsort(rng.random((NULL_DRAWS, n_items)), axis=1)[:, :na]
                sb = np.argsort(rng.random((NULL_DRAWS, n_items)), axis=1)[:, :nb]
                MA = np.zeros((NULL_DRAWS, n_items), bool)
                MB = np.zeros((NULL_DRAWS, n_items), bool)
                np.put_along_axis(MA, sa, True, axis=1)
                np.put_along_axis(MB, sb, True, axis=1)
                inter0 = (MA & MB).sum(axis=1)
                union0 = (MA | MB).sum(axis=1)
                nulls.append(float(np.mean(inter0 / np.maximum(union0, 1))))
            js, nulls = np.asarray(js), np.asarray(nulls)
            res["pairs"]["%s|%s" % (A, B)] = {
                "jaccard_mean": float(js.mean()), "jaccard_std": float(js.std(ddof=1)),
                "null_mean": float(nulls.mean()),
                "ratio_obs_over_null": float(js.mean() / max(nulls.mean(), 1e-12)),
                "per_seed_jaccard": [float(x) for x in js]}
            print("%-10s Jaccard=%.3f  null=%.3f  ratio=%.2f"
                  % ("%s|%s" % (A, B), js.mean(), nulls.mean(),
                     js.mean() / max(nulls.mean(), 1e-12)))

    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
