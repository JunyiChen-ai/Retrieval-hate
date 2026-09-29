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


def _hb(p):
    """Binary entropy in bits of a probability array."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def rule_values(run, T):
    """Per rule, the value read before call k+1 (k = 0..n-1) so that policy.stop_calls applies unchanged.
    Next-question rules (README 17.4 baseline): eig, voi, voi_norm = VOI / T (variant (b)); last-change rules: stab1,
    stab2; <rule>_f<k> = a floor of k calls (variant (c), kept for reference only: the user's 2026-09-29 ruling forbids
    floors); the prior-stratified thresholds (variant (a)) are handled in main.
    State-confidence rules (README 17.4 "no floor", the rule family of the stopping literature: the posterior after
    k calls, not the value of the next question; Wald's SPRT, Naghshvar & Javidi 2013, FrameExit / AdaFrame gates,
    VideoAgent's sufficiency score, EcoFrame's output-entropy gate):
      hG     binary entropy of the video posterior P(G = 1 | answers)
      hT     mean over seconds of the binary entropy of the per-second posterior p_t
      hmax   max(hG, hT): ask while either the video-level or the per-second map is uncertain
      vsum   sum over seconds of p_t (1 - p_t): the expected squared error of the per-second map, weighted as the
             pooled metrics weight seconds (a long uncertain video counts more)
      vmean  the same per second (length-free)"""
    n = len(run["eig"])
    inst = instability(run["scores"])                 # length n + 1: inst[k] = change made by call k
    vals = {"eig": list(run["eig"]), "voi": list(run["voi"]),
            "voi_norm": [v / float(T) for v in run["voi"]],
            "stab1": [inst[k] for k in range(n)],
            "stab2": [max(inst[k], inst[k - 1] if k >= 1 else np.inf) for k in range(n)]}
    for r in ("eig", "voi", "stab1"):
        for f in FLOORS:
            vals["%s_f%d" % (r, f)] = [np.inf if k < f else vals[r][k] for k in range(n)]
    hG = _hb(run["p_G"][:n])
    ps = [np.asarray(run["scores"][k], dtype=np.float64) for k in range(n)]
    hT = np.array([float(np.mean(_hb(p))) for p in ps]) if n else np.zeros(0)
    vals["hG"] = [float(x) for x in hG]
    vals["hT"] = [float(x) for x in hT]
    vals["hmax"] = [float(max(a, b)) for a, b in zip(hG, hT)]
    vals["vsum"] = [float(np.sum(p * (1.0 - p))) for p in ps]
    vals["vmean"] = [float(np.mean(p * (1.0 - p))) for p in ps]
    # last-change rules on the log-odds scale (README 17.4 "no floor", second family: the change the last answer made
    # to the scores on the scale the pooled ranking sees; +inf before the first call, so the first call is always made)
    lp = [_logit(run["scores"][k]) for k in range(n + 1)]
    lg = _logit(run["p_G"][:n + 1])
    d = [np.inf] + [float(np.mean(np.abs(lp[k] - lp[k - 1]))) for k in range(1, n + 1)]
    dg = [np.inf] + [float(abs(lg[k] - lg[k - 1])) for k in range(1, n + 1)]
    vals["dlogit"] = d[:n]                                                    # mean over seconds, last call
    vals["dlogit2"] = [max(d[k], d[k - 1] if k >= 1 else np.inf) for k in range(n)]   # larger of the last two
    vals["dlogit3"] = [float(np.mean(d[max(1, k - 2):k + 1])) if k >= 1 else np.inf for k in range(n)]  # mean of last 3
    ds = [np.inf] + [float(np.sum(np.abs(lp[k] - lp[k - 1]))) for k in range(1, n + 1)]
    vals["dlsum"] = ds[:n]                                                    # sum over seconds (pool-weighted)
    vals["dlsum2"] = [max(ds[k], ds[k - 1] if k >= 1 else np.inf) for k in range(n)]
    vals["dG"] = dg[:n]                                                       # video-level log-odds change
    vals["dG2"] = [max(dg[k], dg[k - 1] if k >= 1 else np.inf) for k in range(n)]
    # remaining-information rules (README 17.4, the nonmyopic direction): the EIG summed over ALL candidate questions
    # still askable (first-order approximation of the information left in the video; a slowly moving video with many
    # small-EIG questions keeps a large sum, a converged one does not), the sum of the three largest, and the sum per
    # candidate (recorded by cpolicy.run_batch; absent in older dumps)
    if "eig_sum" in run and len(run["eig_sum"]) >= n:
        vals["eigsum"] = [float(x) for x in run["eig_sum"][:n]]
        vals["eigtop3"] = [float(x) for x in run["eig_top3"][:n]]
        vals["eigmean"] = [float(x) / max(1, m) for x, m in zip(run["eig_sum"][:n], run["n_cand"][:n])]
    # expected remaining movement (README 17.4, nonmyopic; cpolicy.run_batch record_erm; absent in older dumps)
    if "erm" in run and len(run["erm"]) >= n:
        vals["erm"] = [float(x) for x in run["erm"][:n]]
        vals["ermsum"] = [float(x) for x in run["ermsum"][:n]]
    # one-sided undecidedness (README 17.4 third round): the band applies only to videos on the negative side of the
    # decision boundary (P(G) < .5); a likely-positive video is left to the change rule (its calls go to localisation)
    pg = np.asarray(run["p_G"][:n], dtype=np.float64)
    vals["hGn"] = [float(h) if g < 0.5 else 0.0 for h, g in zip(hG, pg)]
    for u, name in ((0.5, "n50"), (0.2, "n20")):
        for base in ("dlogit", "dlogit2", "dlsum"):
            vals["%s_%s" % (base, name)] = [np.inf if (hG[k] >= u and pg[k] < 0.5) else vals[base][k] for k in range(n)]
    # composites: an undecided video (H(P(G)) >= u bits; u = .5: P(G) in [.11, .89]; u = .2: [.03, .97], the decision
    # band of a sequential test) keeps asking whatever the last change was; a decided one stops by the change rule
    for u, name in ((0.5, "u50"), (0.2, "u20")):
        for base in ("dlogit", "dlogit2", "dlogit3"):
            vals["%s_%s" % (base, name)] = [np.inf if hG[k] >= u else vals[base][k] for k in range(n)]
    return vals


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


QMIX = {"qmix": ("dlogit", "stab1"), "qmix2": ("dlogit2", "stab2"), "qmixG": ("dlogit", "stab1", "hG"),
        "qmix2G": ("dlogit2", "stab2", "hG"),
        # second round (README 17.4): the pool-weighted change (dlsum) in place of the per-second mean, the video-level
        # undecidedness alone as the state component, and the per-second entropy (hT) as a localisation state
        "qmixD": ("dlogit", "hG"), "qmixD2": ("dlogit2", "hG"), "qmixS": ("dlsum", "hG"), "qmixS2": ("dlsum2", "hG"),
        "qmixSG": ("dlsum", "stab1", "hG"), "qmixS2G": ("dlsum2", "stab2", "hG"),
        "qmixT": ("dlogit", "stab1", "hG", "hT"), "qmixST": ("dlsum", "stab1", "hG", "hT"), "qmixDT": ("dlogit", "hG", "hT"),
        "qmixS_T": ("dlsum", "hT"),
        # third round: the remaining-information sum as the state component
        "qmixE": ("dlsum", "eigsum"), "qmixEG": ("dlsum", "eigsum", "hG"), "qmixE3": ("dlsum", "eigtop3"),
        "qmixDE": ("dlogit", "eigsum"), "qmixSE": ("dlsum", "stab1", "eigsum"),
        # one-sided undecidedness as the state component
        "qmixN": ("dlogit", "stab1", "hGn"), "qmix2N": ("dlogit2", "stab2", "hGn"), "qmixSN": ("dlsum", "stab1", "hGn"),
        "qmixDN": ("dlogit2", "hGn"), "qmixSGn": ("dlsum", "hGn"),
        # expected remaining movement as the state component
        "qmixRS": ("dlsum", "ermsum"), "qmixRN": ("ermsum", "hGn"), "qmixRSN": ("dlsum", "ermsum", "hGn"),
        "qmixRm": ("dlogit", "erm"), "qmixRmN": ("erm", "hGn"),
        # "qavg": the MEAN of the component quantiles instead of the largest (a softer combination: a video keeps
        # asking when its components are jointly high, not when any one of them is)
        "qavgRS": ("dlsum", "ermsum"), "qavgRm": ("dlogit", "erm"), "qavgRSN": ("dlsum", "ermsum", "hGn"),
        "qavgRmN": ("dlogit", "erm", "hGn"), "qavgR2N": ("dlogit2", "ermsum", "hGn"), "qavgSN": ("dlsum", "hGn"),
        "qavg2N": ("dlogit2", "hGn"), "qavgR2": ("dlogit2", "ermsum")}


def add_quantile_rules(vals):
    """Constant-free combinations (README 17.4): each component value is replaced by its rank among all finite
    validation (video, call) values of that component (its validation quantile, no labels), and the rule is the
    largest quantile: ask while any component (cross-video log-odds change, within-video rank change, video-level
    undecidedness) is still larger than a fraction c of what validation runs show. One threshold, no floor."""
    have = set(next(iter(next(iter(vals.values())).values())).keys())
    for name, comps in QMIX.items():
        if any(c not in have for c in comps):
            continue
        ref = {}
        for c in comps:
            x = np.concatenate([np.asarray(vals["val"][v][c], dtype=np.float64) for v in vals["val"]])
            ref[c] = np.sort(x[np.isfinite(x)])
        avg = name.startswith("qavg")
        for sp in vals:
            for v in vals[sp]:
                n = len(vals[sp][v]["eig"])
                q = np.zeros(n)
                for c in comps:
                    x = np.asarray(vals[sp][v][c], dtype=np.float64)
                    qc = np.where(np.isfinite(x), np.searchsorted(ref[c], x, side="right") / max(1, len(ref[c])), np.inf)
                    q = q + qc / len(comps) if avg else np.maximum(q, qc)
                vals[sp][v][name] = [float(t) for t in q]


FLOORS = (2, 4, 6)                                     # floor sensitivity (README 17.4 variant (c))
STATE = ("hG", "hT", "hmax", "vsum", "vmean")          # README 17.4 "no floor": state-confidence rules
CHANGE = (("dlogit", "dlogit2", "dlogit3", "dlsum", "dlsum2", "dG", "dG2", "eigsum", "eigtop3", "eigmean", "erm", "ermsum")
          + tuple("%s_%s" % (b, u) for u in ("u50", "u20") for b in ("dlogit", "dlogit2", "dlogit3"))
          + tuple("%s_%s" % (b, u) for u in ("n50", "n20") for b in ("dlogit", "dlogit2", "dlsum")))   # log-odds change rules
RULES = (("eig", "voi", "voi_norm", "stab1", "stab2") + tuple("%s_f%d" % (r, f) for r in ("eig", "voi", "stab1") for f in FLOORS)
         + STATE + CHANGE + tuple(QMIX))
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
    ap.add_argument("--dump", action="store_true", help="save the recorded runs (scores / p_G / eig / voi / asked per "
                    "call, validation and test) as runs.pkl in the trial's output directory, so that further rules can "
                    "be evaluated without re-running the policy")
    ap.add_argument("--rules", nargs="*", default=None, help="restrict to these rules (default: all)")
    ap.add_argument("--erm", type=int, default=0, help="samples for the expected remaining movement (cpolicy record_erm)")
    ap.add_argument("--from-dump", default=None, help="suffix of an earlier --dump run: read its runs.pkl per trial "
                    "instead of re-running the policy (the rules are then evaluated on the saved per-call posteriors)")
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
        res["trials"][trial] = rt
        json.dump(res, open(os.path.join(OUT, "%s%s.json" % (a.corpus, a.out_suffix)), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
