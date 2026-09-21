"""Module-1 iteration 5 arm table: full (best trial per seed) minus each diagnostic arm, per corpus,
three-seed means (rule 14(g): a component is claimable when AP or ROC drops >= .01 on both corpora).
usage: python arms_summary.py [_it5]"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
suffix = sys.argv[1] if len(sys.argv) > 1 else "_it5"
R = os.path.join(ROOT, "runs", "20260910_online_query_within" + suffix)
ARMS = ("temper_icc", "text_prior_off", "single_prior", "eoc_model_raw", "global_alloc",
        "text_feat_hate", "text_feat_hate_prior_off", "bert_utterance")   # iteration 6 (README section 12)
KEYS = ("pooled_ap", "pooled_roc", "within_roc")
out = {}
for corpus in ("hatemm", "hateclipseg"):
    rows = {"full": []}; per = {}
    for seed in (234, 2025, 3407):
        ss = os.path.join(R, corpus, "seed%d" % seed, "study_summary.json")
        if not os.path.exists(ss):
            continue
        b = json.load(open(ss))["best"]["number"]
        full = json.load(open(os.path.join(R, corpus, "seed%d" % seed, "trial%d" % b, "summary.json")))
        rows["full"].append([full["test"][k] for k in KEYS])
        for arm in ARMS:
            p = os.path.join(R, "diag", corpus, "seed%d" % seed, arm, "summary.json")
            if os.path.exists(p):
                d = json.load(open(p))
                rows.setdefault(arm, []).append([d["test"][k] for k in KEYS])
                per.setdefault(arm, {})[seed] = {"test": {k: round(d["test"][k], 4) for k in KEYS},
                                                 "fixed34": {k: round(d["results"]["fixed34"][k], 4) for k in KEYS},
                                                 "rho": d["results"].get("iteration5", {}).get("rho"),
                                                 "prior_w": d["results"].get("iteration5", {}).get("prior_w_fine_text_video")}
    if not rows["full"]:
        continue
    fm = np.mean(rows["full"], 0)
    print("== %s  full (n=%d) AP/ROC/within %.4f/%.4f/%.4f" % (corpus, len(rows["full"]), *fm))
    out[corpus] = {"full_mean": dict(zip(KEYS, fm.tolist())), "n_full": len(rows["full"]), "arms": {}}
    for arm in ARMS:
        if arm not in rows:
            continue
        am = np.mean(rows[arm], 0); d = fm[:len(am)] - am
        n = len(rows[arm])
        print("   %-15s n=%d arm %.4f/%.4f/%.4f | full-arm AP %+.4f ROC %+.4f within %+.4f%s"
              % (arm, n, *am, *d, "" if n == len(rows["full"]) else "  (seed counts differ: full uses all seeds)"))
        out[corpus]["arms"][arm] = {"n": n, "arm_mean": dict(zip(KEYS, am.tolist())), "full_minus_arm": dict(zip(KEYS, d.tolist())), "per_seed": per.get(arm, {})}
json.dump(out, open(os.path.join(R, "arms_summary.json"), "w"), indent=2, default=float)
