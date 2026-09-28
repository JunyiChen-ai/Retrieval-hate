"""Revision 5 step 2, offline check (README section 17.2; development evidence on test under rule 10; nothing here
trains or selects): copy-type persistent noise for nested questions (cpolicy.run_batch copy_pi). Each revision-4
trial (network, chain, node potentials, anchored answer model unchanged) is re-run on test with
  tree        copy_pi None (the method; must reproduce the trial's summary)
  copy_neg    pi per child-length bucket estimated WITHOUT labels on the negative training videos: every node of a
              negative video is state 0, so P(o_C = o_P) = pi + (1 - pi) sum_o P(o | 0)^2 over parent-child pairs,
              pi = (r - q) / (1 - q) with r the observed equality rate and q from the trial's answer model
  copy_<x>    constant pi = x (sensitivity)
Pooled AP / ROC / within at fixed budgets through the shared evaluator, mean calls, and the share of the first 8
questions nested with an earlier one.

    python experiments/20260925_query_paradigm/copy_check.py --corpus hatemm --trials <trial dirs> [--device cpu]
Writes runs/20260929_query_paradigm_r5/copy_check/<corpus>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import concern_diagnostics as cd                # noqa: E402  (sets up the import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import policy                                   # noqa: E402
import cpolicy                                  # noqa: E402
from tendency_check import nested_share         # noqa: E402

BUDGETS = (0, 2, 4, 8, 16, 32)
BUCKETS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
OUT = os.path.join(ROOT, "runs", "20260929_query_paradigm_r5", "copy_check")


def bucket_of(length):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= length < hi:
            return i
    return len(BUCKETS) - 1


def estimate_pi_neg(answers, labels, train_ids, T, am, cats):
    """Label-free copy probability per child-length bucket from parent-child pairs of negative training videos."""
    r_num, r_den, q_sum = np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS))
    for v in train_ids:
        if labels[v] != 0 or v not in answers:
            continue
        vt = cpolicy._vt(T[v])
        asker = policy.Asker(vt, am, cats)
        tr = vt.tr
        for n in asker.q:
            par = int(tr["parent"][n]) if "parent" in tr else -1
            if par < 0:
                continue
            oc = answers[v].get((int(tr["a"][n]), int(tr["b"][n])))
            op = answers[v].get((int(tr["a"][par]), int(tr["b"][par])))
            if oc is None or op is None:
                continue
            k = bucket_of(tr["b"][n] - tr["a"][n])
            ic = qtree.answer_index(np.asarray(oc)[cats]); ip = qtree.answer_index(np.asarray(op)[cats])
            r_num[k] += ic == ip
            r_den[k] += 1
            q_sum[k] += float(np.sum(asker.po_s[asker.qpos[int(n)], 0] ** 2))
    r = r_num / np.maximum(r_den, 1)
    q = q_sum / np.maximum(r_den, 1)
    pi = np.clip((r - q) / np.maximum(1 - q, 1e-9), 0.0, 0.95)
    pi[r_den == 0] = 0.0
    return pi, r, q, r_den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--variants", default="tree,copy_neg,copy_0.25,copy_0.5,copy_0.75")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    cfg0 = json.load(open(os.path.join(a.trials[0], "config.json")))
    qdata.configure_source(cfg0.get("answer_source", "k30"), int(cfg0.get("soft_levels", 8)))
    answers, T = qdata.load_answers(a.corpus, cfg0.get("answer_source", "k30"))
    vids = list(ids["test"])
    os.makedirs(OUT, exist_ok=True)
    res = {"corpus": a.corpus, "buckets": [list(b) for b in BUCKETS], "trials": {}}
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
        cats = list(cfg["categories"])
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        pi_neg, r, q, n = estimate_pi_neg(answers, labels, ids["train"], T, am, cats)
        print("== %s: pi from negatives per bucket %s (equality rate %s, chance %s, n %s)" % (
            tag, np.round(pi_neg, 3).tolist(), np.round(r, 3).tolist(), np.round(q, 3).tolist(), n.astype(int).tolist()),
            flush=True)
        rt = {"pi_neg": pi_neg.tolist(), "eq_rate_neg": r.tolist(), "chance_eq": q.tolist(), "n_pairs_neg": n.tolist(),
              "method": {str(B): {m: summ["fixed"][str(B)]["test"][m] for m in ("pooled_ap", "pooled_roc", "within_roc")}
                         for B in BUDGETS}}
        for variant in a.variants.split(","):
            if variant == "tree":
                fn = None
            elif variant == "copy_neg":
                fn = (lambda L, _pi=pi_neg: float(_pi[bucket_of(L)]))
            else:
                x = float(variant.split("_")[1])
                fn = (lambda L, _x=x: _x)
            order = sorted(vids, key=lambda v: store.T[v])
            k = int(cfg["eval_chunk"])
            runs = {}
            for i in range(0, len(order), k):
                chunk = order[i:i + k]
                runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cats, max(BUDGETS), a.device,
                                              copy_pi=fn))
            od = os.path.join(OUT, a.corpus, variant, tag)
            os.makedirs(od, exist_ok=True)
            rr = {}
            for B in BUDGETS:
                sc = {v: runs[v]["scores"][min(B, len(runs[v]["eig"]))] for v in vids}
                sp = os.path.join(od, "scores_test_fixed%d.jsonl" % B)
                hc.write_scores(sp, sc)
                m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_fixed%d.json" % B))
                m = m["results"]["score_av"]
                rr[str(B)] = {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                              "within_roc": m["per_video"]["macro_auc"],
                              "mean_calls": float(np.mean([min(B, len(runs[v]["eig"])) for v in vids]))}
            n8 = [nested_share(runs[v]["asked"][:8], qtree.tree(store.T[v])) for v in vids]
            rr["nested_share_first8"] = float(np.mean([x for x in n8 if x is not None]))
            rt[variant] = rr
            print("== %s %s: nested share of first 8 %.3f" % (tag, variant, rr["nested_share_first8"]), flush=True)
            for B in map(str, BUDGETS):
                x = rr[B]
                print("  B=%-2s %.4f / %.4f / %.4f | calls %.2f" % (B, x["pooled_ap"], x["pooled_roc"], x["within_roc"],
                                                                   x["mean_calls"]), flush=True)
            res["trials"][trial] = rt
            json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
