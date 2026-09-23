"""README section 14: random_train (8-trial search, trials 0-7 = the hparams of trials 0-7 of the iteration-5
study) against the default's first 8 trials. Main comparison: best of 8 by the search objective (AP + ROC +
within) / 3, three-seed means; auxiliary: mean of the paired per-trial differences (default - random).
usage: python randtrain_compare.py  ->  runs/20260910_online_query_within_it5_randtrain/compare.json"""
import json, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEF = os.path.join(ROOT, "runs", "20260910_online_query_within_it5")
RND = os.path.join(ROOT, "runs", "20260910_online_query_within_it5_randtrain")
K = ("pooled_ap", "pooled_roc", "within_roc")


def trials(root, corpus, seed):
    out = {}
    for t in range(8):
        p = os.path.join(root, corpus, "seed%d" % seed, "trial%d" % t)
        if os.path.exists(os.path.join(p, "summary.json")):
            s = json.load(open(os.path.join(p, "summary.json")))
            out[t] = {"hp": json.load(open(os.path.join(p, "hparams.json"))), "test": [s["test"][k] for k in K]}
    return out


res = {}
for corpus in ("hatemm", "hateclipseg"):
    best_d, best_r, pairs = [], [], []
    per_seed = {}
    for seed in (234, 2025, 3407):
        d, r = trials(DEF, corpus, seed), trials(RND, corpus, seed)
        for t in r:                                # pairing check: identical searched hparams
            hd = {k: v for k, v in d[t]["hp"].items() if k != "policies"}
            hr = {k: v for k, v in r[t]["hp"].items() if k != "policies"}
            assert hd == hr, (corpus, seed, t, hd, hr)
        obj = lambda x: sum(x) / 3.0              # noqa: E731
        bd = max(d, key=lambda t: obj(d[t]["test"]))
        br = max(r, key=lambda t: obj(r[t]["test"]))
        best_d.append(d[bd]["test"]); best_r.append(r[br]["test"])
        common = sorted(set(d) & set(r))
        pairs += [np.array(d[t]["test"]) - np.array(r[t]["test"]) for t in common]
        per_seed[seed] = {"default_best_trial": bd, "default_best": d[bd]["test"], "random_best_trial": br,
                          "random_best": r[br]["test"], "n_random_trials": len(r)}
    md, mr, mp = np.mean(best_d, 0), np.mean(best_r, 0), np.mean(pairs, 0)
    res[corpus] = {"default_best8_mean": dict(zip(K, md.tolist())), "random_best8_mean": dict(zip(K, mr.tolist())),
                   "default_minus_random": dict(zip(K, (md - mr).tolist())),
                   "paired_mean_diff": dict(zip(K, mp.tolist())), "n_pairs": len(pairs),
                   "paired_ap_diff_positive": int(sum(p[0] > 0 for p in pairs)), "per_seed": per_seed}
    print("== %s  default best-of-8 %.4f/%.4f/%.4f | random best-of-8 %.4f/%.4f/%.4f | default-random %+.4f/%+.4f/%+.4f"
          % (corpus, *md, *mr, *(md - mr)))
    print("   paired (n=%d) mean default-random %+.4f/%+.4f/%+.4f ; AP diff > 0 in %d pairs"
          % (len(pairs), *mp, res[corpus]["paired_ap_diff_positive"]))
    for s, v in per_seed.items():
        print("   seed %d: default trial %d %s | random trial %d %s (%d trials)" % (
            s, v["default_best_trial"], "/".join("%.4f" % x for x in v["default_best"]), v["random_best_trial"],
            "/".join("%.4f" % x for x in v["random_best"]), v["n_random_trials"]))
json.dump(res, open(os.path.join(RND, "compare.json"), "w"), indent=2)
