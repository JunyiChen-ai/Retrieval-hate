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


from qtl.stop_rules import (instability, _hb, rule_values, _logit, QMIX,
                            add_quantile_rules, RELAX, add_relaxed_rules,
                            FLOORS, STATE, CHANGE, RULES, STRAT)


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
    ap.add_argument("--dump", action="store_true", help="save the recorded runs (scores / p_G / eig / voi / asked per "
                    "call, validation and test) as runs.pkl in the trial's output directory, so that further rules can "
                    "be evaluated without re-running the policy")
    ap.add_argument("--rules", nargs="*", default=None, help="restrict to these rules (default: all)")
    ap.add_argument("--erm", type=int, default=0, help="samples for the expected remaining movement (cpolicy record_erm)")
    ap.add_argument("--from-dump", default=None, help="suffix of an earlier --dump run: read its runs.pkl per trial "
                    "instead of re-running the policy (the rules are then evaluated on the saved per-call posteriors)")
    ap.add_argument("--select-val", action="store_true", help="README 17.4 item 11: the threshold c is chosen on the "
                    "validation LABELS instead of a mean budget: the rule is calibrated to mean budgets 1..24 (step .5) on "
                    "validation, the validation AP / ROC / within of each is computed (hc.frame_metrics), and c is the "
                    "one with the best validation value (exact maximum, and the smallest budget within .005 of it)")
    ap.add_argument("--select-rules", nargs="*", default=None, help="rules for --select-val (default: --rules)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    if a.from_dump is None:
        cfg0 = json.load(open(os.path.join(a.trials[0], "config.json")))
        qdata.configure_source(cfg0["answer_source"], int(cfg0.get("soft_levels", qdata.SOFT_LEVELS)))
        answers, T = qdata.load_answers(a.corpus, cfg0["answer_source"])
        vids = {sp: [v for v in ids[sp] if v in answers] for sp in ("val", "test")}
    hate_val = {v for v in ids["val"] if labels[v] == 1}
    os.makedirs(OUT, exist_ok=True)
    res = {"corpus": a.corpus, "copy_pi": a.copy_pi, "from_dump": a.from_dump, "trials": {}}
    store = None
    for trial in a.trials:
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        od = os.path.join(OUT, a.corpus, tag + a.out_suffix)
        os.makedirs(od, exist_ok=True)
        if a.from_dump is not None:
            import pickle
            D = pickle.load(open(os.path.join(OUT, a.corpus, tag + a.from_dump, "runs.pkl"), "rb"))
            runs, Tmap = D["runs"], D["T"]
        else:
            summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
            if store is None:
                store = qdata.Store(a.corpus, vids["val"] + vids["test"], cfg.get("text_sources", ["bert"]))
            cats = list(cfg["categories"])
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
                                                      a.device, record_voi=True, copy_pi=fn, record_erm=a.erm))
            Tmap = {v: int(store.T[v]) for sp in runs for v in runs[sp]}
            if a.dump:
                import pickle
                with open(os.path.join(od, "runs.pkl"), "wb") as f:
                    pickle.dump({"runs": runs, "T": Tmap}, f)
        vals = {sp: {v: rule_values(r, Tmap[v]) for v, r in runs[sp].items()} for sp in runs}
        add_quantile_rules(vals)
        add_relaxed_rules(vals)
        pg0 = {sp: {v: float(r["p_G"][0]) for v, r in runs[sp].items()} for sp in runs}

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
            if a.rules is not None and rule not in a.rules:
                continue
            if any(rule.replace("_strat", "") not in vals[sp][v] for sp in vals for v in vals[sp]):
                continue
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
        if a.select_val:
            rt["select"] = {}
            for rule in (a.select_rules or a.rules or list(rt["rules"])):
                if any(rule not in vals[sp][v] for sp in vals for v in vals[sp]):
                    continue
                vr = [vals["val"][v][rule] for v in vals["val"]]
                curve = []
                for B in np.arange(1.0, 24.01, 0.5):
                    c, vm = policy.calibrate(vr, float(B))
                    calls_v = {v: policy.stop_calls(vals["val"][v][rule], c) for v in vals["val"]}
                    sv = {v: runs["val"][v]["scores"][calls_v[v]] for v in vals["val"]}
                    m = hc.frame_metrics(sv, gt["val"], hate_val)
                    m["sum"] = m["pooled_ap"] + m["pooled_roc"] + m["within_roc"]
                    curve.append({"B": float(B), "c": float(c), "val_mean_calls": float(vm), "val": m})
                sel = {}
                for obj in ("pooled_ap", "pooled_roc", "within_roc", "sum"):
                    top = max(e["val"][obj] for e in curve)
                    tol = 0.005 * (3 if obj == "sum" else 1)
                    for mode, pick in (("max", next(e for e in curve if e["val"][obj] >= top - 1e-12)),
                                       ("tol", next(e for e in curve if e["val"][obj] >= top - tol))):
                        c = pick["c"]
                        calls = {v: policy.stop_calls(vals["test"][v][rule], c) for v in vals["test"]}
                        st = {v: runs["test"][v]["scores"][calls[v]] for v in calls}
                        cv = np.array(list(calls.values()))
                        name = "%s_%s" % (obj, mode)
                        sel[name] = {"B": pick["B"], "c": float(c), "val_mean_calls": pick["val_mean_calls"], "val": pick["val"],
                                     "test": test_eval("sel_%s_%s" % (rule, name), st), "test_mean_calls": float(cv.mean()),
                                     "test_mean_calls_pos": float(np.mean([calls[v] for v in calls if labels[v] == 1])),
                                     "test_mean_calls_neg": float(np.mean([calls[v] for v in calls if labels[v] == 0])),
                                     "test_calls_quantiles": [float(x) for x in np.percentile(cv, [0, 25, 50, 75, 100])]}
                        x = sel[name]
                        print("== %s %s select %-15s val B %4.1f (c %.4g) | calls %5.2f (pos %5.2f neg %5.2f) | %.4f / %.4f / %.4f" % (
                            tag, rule, name, x["B"], c, x["test_mean_calls"], x["test_mean_calls_pos"], x["test_mean_calls_neg"],
                            x["test"]["pooled_ap"], x["test"]["pooled_roc"], x["test"]["within_roc"]), flush=True)
                rt["select"][rule] = {"curve": curve, "selected": sel}
        res["trials"][trial] = rt
        json.dump(res, open(os.path.join(OUT, "%s%s.json" % (a.corpus, a.out_suffix)), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
