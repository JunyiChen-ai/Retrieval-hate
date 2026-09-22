"""Module-1 iteration 5 arm table: full (best trial per seed) minus each diagnostic arm, per corpus,
three-seed means (rule 14(g): a component is claimable when AP or ROC drops >= .01 on both corpora).
Section 13 (2026-09-23) adds the full ablation table, a same-machine rerun of the default
configuration (`full_rerun`, the noise reference: arm differences are also given against it) and
the evaluation-level rows read from each seed's best-trial summary (no training).
usage: python arms_summary.py [_it5]"""
import json, os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
suffix = sys.argv[1] if len(sys.argv) > 1 else "_it5"
R = os.path.join(ROOT, "runs", "20260910_online_query_within" + suffix)
ARMS = ("temper_icc", "text_prior_off", "single_prior", "eoc_model_raw", "global_alloc",
        "text_feat_hate", "text_feat_hate_prior_off", "bert_utterance",   # iteration 6 (README section 12)
        "unified_chunks", "xt_chunks",                                      # iteration 6b (README section 12b)
        # README section 13: ablation table of the default method
        "full_rerun",
        "no_verdict", "coarse_only", "fixed_uniform_train", "no_dropout", "no_missing_state",
        "no_hmm", "seconds_time", "no_constraint", "no_decomp", "no_video_term", "no_text_term",
        "avce", "no_qk_enc", "no_cell", "no_bias", "no_context",
        "no_cmal", "no_block", "no_window_loss", "no_prior")
KEYS = ("pooled_ap", "pooled_roc", "within_roc")


def pick(m):
    return [m[k] for k in KEYS]


out = {}
for corpus in ("hatemm", "hateclipseg"):
    rows = {"full": []}; per = {}; seeds_of = {}
    evals = {"uniform_8calls": [], "coarse4": [], "fixed34": [], "stop_rule": [], "stop_calls": []}
    rerun = {}
    for seed in (234, 2025, 3407):
        ss = os.path.join(R, corpus, "seed%d" % seed, "study_summary.json")
        if not os.path.exists(ss):
            continue
        b = json.load(open(ss))["best"]["number"]
        full = json.load(open(os.path.join(R, corpus, "seed%d" % seed, "trial%d" % b, "summary.json")))
        rows["full"].append(pick(full["test"]))
        res = full["results"]
        evals["uniform_8calls"].append(pick(res["curves"]["uniform"]["4"]))
        evals["coarse4"].append(pick(res["coarse4"]))
        evals["fixed34"].append(pick(res["fixed34"]))
        evals["stop_rule"].append(pick(full["stop_rule"]["test"]))
        evals["stop_calls"].append(full["stop_rule"]["test"]["mean_calls"])
        for arm in ARMS:
            p = os.path.join(R, "diag", corpus, "seed%d" % seed, arm, "summary.json")
            if os.path.exists(p):
                d = json.load(open(p))
                rows.setdefault(arm, []).append(pick(d["test"]))
                seeds_of.setdefault(arm, []).append(seed)
                if arm == "full_rerun":
                    rerun[seed] = pick(d["test"])
                per.setdefault(arm, {})[seed] = {"test": {k: round(d["test"][k], 4) for k in KEYS},
                                                 "fixed34": {k: round(d["results"]["fixed34"][k], 4) for k in KEYS},
                                                 "rho": d["results"].get("iteration5", {}).get("rho"),
                                                 "prior_w": d["results"].get("iteration5", {}).get("prior_w_fine_text_video")}
    if not rows["full"]:
        continue
    fm = np.mean(rows["full"], 0)
    print("== %s  full (n=%d) AP/ROC/within %.4f/%.4f/%.4f" % (corpus, len(rows["full"]), *fm))
    out[corpus] = {"full_mean": dict(zip(KEYS, fm.tolist())), "n_full": len(rows["full"]), "arms": {},
                   "eval_level": {k: (dict(zip(KEYS, np.mean(v, 0).tolist())) if k != "stop_calls" else float(np.mean(v)))
                                  for k, v in evals.items()}}
    for k in ("uniform_8calls", "coarse4", "fixed34", "stop_rule"):
        em = np.mean(evals[k], 0)
        print("   eval %-15s %.4f/%.4f/%.4f | full-this AP %+.4f ROC %+.4f within %+.4f" % (k, *em, *(fm - em)))
    for arm in ARMS:
        if arm not in rows:
            continue
        am = np.mean(rows[arm], 0); d = fm[:len(am)] - am
        n = len(rows[arm])
        entry = {"n": n, "arm_mean": dict(zip(KEYS, am.tolist())), "full_minus_arm": dict(zip(KEYS, d.tolist())),
                 "per_seed": per.get(arm, {})}
        extra = ""
        both = [s for s in seeds_of[arm] if s in rerun]
        if arm != "full_rerun" and both:
            arm_by_seed = dict(zip(seeds_of[arm], rows[arm]))
            dr = np.mean([np.array(rerun[s]) - np.array(arm_by_seed[s]) for s in both], 0)
            entry["rerun_minus_arm"] = dict(zip(KEYS, dr.tolist()))
            entry["n_rerun_pairs"] = len(both)
            extra = " | rerun-arm AP %+.4f ROC %+.4f within %+.4f (n=%d)" % (*dr, len(both))
        print("   %-24s n=%d arm %.4f/%.4f/%.4f | full-arm AP %+.4f ROC %+.4f within %+.4f%s%s"
              % (arm, n, *am, *d, extra, "" if n == len(rows["full"]) else "  (seed counts differ: full uses all seeds)"))
        out[corpus]["arms"][arm] = entry
json.dump(out, open(os.path.join(R, "arms_summary.json"), "w"), indent=2, default=float)
