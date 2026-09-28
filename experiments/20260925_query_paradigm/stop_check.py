"""Revision 5 step 4, offline check (README section 17.4; development evidence on test under rule 10; the stopping
thresholds are calibrated on validation WITHOUT labels, by the mean number of calls): adaptive stopping rules on a
finished trial's network (revision 4 decoded-answer trials as the reference, revision 5 soft-answer trials).

Rules, all "ask while value >= c", c calibrated on the validation runs so that the validation mean number of calls
equals the target budget (policy.calibrate), then applied to test:
  eig    the EIG of the next question (revision 3 rule)
  voi    the expected squared-error risk reduction of the next question (stopping.py, revision 3 rule)
  stab1  1 - Spearman(per-second scores after the last call, after the call before): stop when the output stopped
         moving (needs no calibrated posterior; README 17.4 "output-stability stop")
  stab2  the larger of the last two such changes (two consecutive stable steps)
  variants of README 17.4: voi_norm (VOI / T), <rule>_f4 (floor of 4 calls), <rule>_strat (a threshold per half of
         the videos split at the median prior P(G = 1), each half calibrated to the mean budget)
Each trial is re-run on validation and test with 32 calls (record_voi), optionally with the copy likelihood of
README 17.2 (--copy-pi neg|<x>). Reported per rule and mean budget {4, 8, 12, 16}: test AP / ROC / within, mean calls
(positive / negative videos), against the fixed budget with the same mean calls.

    python experiments/20260925_query_paradigm/stop_check.py --corpus hatemm --trials <trial dirs> [--device cpu]
Writes runs/20260929_query_paradigm_r5/stop_check/<corpus><suffix>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train as TR                              # noqa: E402  (import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import cpolicy                                  # noqa: E402
import policy                                   # noqa: E402
import concern_diagnostics as cd                # noqa: E402
import copy_check as cc                         # noqa: E402

OUT = os.path.join(ROOT, "runs", "20260929_query_paradigm_r5", "stop_check")
BUDGETS = [4, 8, 12, 16]
FIXED = [0, 2, 4, 8, 12, 16, 24, 32]


def instability(scores):
    """Per call k (1..n): 1 - Spearman(scores[k], scores[k-1]); index 0 (before any call) is +inf."""
    out = [np.inf]
    for k in range(1, len(scores)):
        a, b = np.asarray(scores[k]), np.asarray(scores[k - 1])
        if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
            out.append(0.0 if np.allclose(a, b) else 1.0)
        else:
            out.append(float(1.0 - spearmanr(a, b).correlation))
    return out


def rule_values(run, T):
    """Per rule, the value read before call k+1 (k = 0..n-1) so that policy.stop_calls applies unchanged.
    README 17.4 variants: voi_norm = VOI / T (length-free: the expected squared change per second, variant (b));
    <rule>_f<k> = the rule with a floor of k calls, k in FLOORS (variant (c)); the prior-stratified thresholds (variant (a))
    are handled in main (a threshold per half of the videos by the prior P(G = 1))."""
    n = len(run["eig"])
    inst = instability(run["scores"])                 # length n + 1: inst[k] = change made by call k
    vals = {"eig": list(run["eig"]), "voi": list(run["voi"]),
            "voi_norm": [v / float(T) for v in run["voi"]],
            "stab1": [inst[k] for k in range(n)],
            "stab2": [max(inst[k], inst[k - 1] if k >= 1 else np.inf) for k in range(n)]}
    for r in ("eig", "voi", "stab1"):
        for f in FLOORS:
            vals["%s_f%d" % (r, f)] = [np.inf if k < f else vals[r][k] for k in range(n)]
    return vals


FLOORS = (2, 4, 6)                                     # floor sensitivity (README 17.4 variant (c))
RULES = ("eig", "voi", "voi_norm", "stab1", "stab2") + tuple("%s_f%d" % (r, f) for r in ("eig", "voi", "stab1") for f in FLOORS)
STRAT = ("eig", "voi", "stab1")                        # variant (a): thresholds per prior half


def calls_strat(vals, rule, pg0, B):
    """Variant (a): the videos are split at the median of their prior P(G = 1) (no labels); the rule's threshold is
    calibrated separately in each half on validation so that each half's mean number of calls is B; returns the
    two thresholds and, for a split dict, the calls per video."""
    med = float(np.median([pg0["val"][v] for v in pg0["val"]]))
    cs = {}
    for half in (0, 1):
        vv = [v for v in vals["val"] if (pg0["val"][v] >= med) == bool(half)]
        cs[half] = policy.calibrate([vals["val"][v][rule] for v in vv], float(B))[0] if vv else 0.0
    return med, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--copy-pi", default=None, help="neg (from the negative training videos) or a constant")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    cfg0 = json.load(open(os.path.join(a.trials[0], "config.json")))
    qdata.configure_source(cfg0["answer_source"], int(cfg0.get("soft_levels", qdata.SOFT_LEVELS)))
    answers, T = qdata.load_answers(a.corpus, cfg0["answer_source"])
    vids = {sp: [v for v in ids[sp] if v in answers] for sp in ("val", "test")}
    hate_val = {v for v in ids["val"] if labels[v] == 1}
    os.makedirs(OUT, exist_ok=True)
    res = {"corpus": a.corpus, "copy_pi": a.copy_pi, "trials": {}}
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids["val"] + vids["test"], cfg.get("text_sources", ["bert"]))
        cats = list(cfg["categories"])
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        fn = None
        if a.copy_pi == "neg":
            pi_neg = cc.estimate_pi_neg(answers, labels, ids["train"], T, am, cats)[0]
            fn = (lambda L, _pi=pi_neg: float(_pi[cc.bucket_of(L)]))
        elif a.copy_pi is not None:
            fn = (lambda L, _x=float(a.copy_pi): _x)
        runs = {}
        for sp in ("val", "test"):
            order = sorted(vids[sp], key=lambda v: store.T[v])
            k = int(cfg["eval_chunk"])
            runs[sp] = {}
            for i in range(0, len(order), k):
                runs[sp].update(cpolicy.run_batch(model, store, order[i:i + k], am, chain, answers, cats, 32,
                                                  a.device, record_voi=True, copy_pi=fn))
        vals = {sp: {v: rule_values(r, store.T[v]) for v, r in runs[sp].items()} for sp in runs}
        pg0 = {sp: {v: float(r["p_G"][0]) for v, r in runs[sp].items()} for sp in runs}
        od = os.path.join(OUT, a.corpus, tag + a.out_suffix)
        os.makedirs(od, exist_ok=True)

        def test_eval(name, scores):
            sp = os.path.join(od, "scores_test_%s.jsonl" % name)
            hc.write_scores(sp, scores)
            m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_%s.json" % name))
            m = m["results"]["score_av"]
            return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}

        rt = {"fixed": {}, "rules": {}}
        for B in FIXED:
            st = {v: r["scores"][min(B, len(r["eig"]))] for v, r in runs["test"].items()}
            rt["fixed"][str(B)] = {"test": test_eval("fixed%d" % B, st),
                                   "test_mean_calls": float(np.mean([min(B, len(r["eig"])) for r in runs["test"].values()]))}
            x = rt["fixed"][str(B)]
            print("== %s fixed %2d | calls %5.2f | %.4f / %.4f / %.4f" % (tag, B, x["test_mean_calls"], x["test"]["pooled_ap"],
                                                                        x["test"]["pooled_roc"], x["test"]["within_roc"]), flush=True)
        for rule in RULES + tuple(r + "_strat" for r in STRAT):
            rt["rules"][rule] = {}
            for B in BUDGETS:
                if rule.endswith("_strat"):
                    base = rule[:-6]
                    med, cs = calls_strat(vals, base, pg0, B)
                    thr = (lambda sp, v: cs[int(pg0[sp][v] >= med)])
                    calls = {v: policy.stop_calls(vals["test"][v][base], thr("test", v)) for v in vals["test"]}
                    calls_v = {v: policy.stop_calls(vals["val"][v][base], thr("val", v)) for v in vals["val"]}
                    c, val_mean = float(cs[1]), float(np.mean(list(calls_v.values())))
                else:
                    c, val_mean = policy.calibrate([vals["val"][v][rule] for v in vals["val"]], float(B))
                    calls = {v: policy.stop_calls(vals["test"][v][rule], c) for v in vals["test"]}
                    calls_v = {v: policy.stop_calls(vals["val"][v][rule], c) for v in vals["val"]}
                st = {v: runs["test"][v]["scores"][calls[v]] for v in calls}
                sv = {v: runs["val"][v]["scores"][calls_v[v]] for v in vals["val"]}
                cv = np.array(list(calls.values()))
                rt["rules"][rule][str(B)] = {
                    "c": float(c), "val_mean_calls": val_mean, "val": hc.frame_metrics(sv, gt["val"], hate_val),
                    "test": test_eval("%s%d" % (rule, B), st), "test_mean_calls": float(cv.mean()),
                    "test_mean_calls_pos": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
                    "test_mean_calls_neg": float(np.mean([calls[v] for v in calls if labels[v] == 0])),
                    "test_calls_quantiles": [float(x) for x in np.percentile(cv, [0, 25, 50, 75, 100])]}
                x = rt["rules"][rule][str(B)]
                print("== %s %-5s mean %2d (c %.4g) | calls %5.2f (pos %5.2f neg %5.2f) | %.4f / %.4f / %.4f" % (
                    tag, rule, B, c, x["test_mean_calls"], x["test_mean_calls_pos"], x["test_mean_calls_neg"],
                    x["test"]["pooled_ap"], x["test"]["pooled_roc"], x["test"]["within_roc"]), flush=True)
        res["trials"][trial] = rt
        json.dump(res, open(os.path.join(OUT, "%s%s.json" % (a.corpus, a.out_suffix)), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
