"""Monotonicity diagnosis (README section 2): does the r5 method get better with more calls when the answers follow
its own answer model?

For each r5 test-selected trial, runs the batched EIG policy (cpolicy.run_batch) up to --max-calls on validation and
test under four answer sources:
  real_copy   the cached VLM answers, copy likelihood in EIG and posterior (the current method)
  real_indep  the cached VLM answers, answers treated as independent (r5 without the copy likelihood)
  sim_indep   simulated answers drawn independently from the checkpoint's anchored answer table given each node's
              true state; inference treats answers as independent (the model is exactly right about the answers)
  sim_copy    simulated answers drawn top-down: a node copies its parent's answer with probability pi (the copy
              probability estimated from negative training videos, by child length) and is otherwise drawn from the
              table; inference uses the copy likelihood
Second round (README section 2.2, which deviation of the real answers matters):
  real_val3       cached answers, inference with the three-state length answer model fitted on validation GT
                  (qtl.answer_calib, no gradient), answers independent
  real_val3_copy  the same with the copy likelihood
  shuf_anch       every node's cached answer replaced by the cached answer of a random node of another video of the
                  same split with the same true state and length bucket (copy_noise.bucket_of): the real answer
                  distribution per state and length is kept, the dependence between answers of one video is removed;
                  inference with the checkpoint's anchored model, independent
  shuf_val3       the shuffled answers, inference with the validation-fitted model
  sim_val3        answers drawn independently from the validation-fitted model, inference with it
Third round (README section 3.2): real_len / real_len_copy = cached answers, inference with the anchored two-state
  model refitted on the training answers WITH the log-length term (qtree.fit_anchored length_term, label-free)
True node state (simulation only): 0 on negative videos; on positive videos 2 if the node holds a GT hateful second,
else 1 (the anchored table gives state 1 the state-0 table). Validation answers use validation GT the same way; no
gradient, nothing selected for the method (rule 10 developmental evidence; validation labels without gradient,
user ruling 2026-09-29).

Reported per source (simulated sources: mean over --reps draws): fixed k = 0..max-calls and qmixE1 at budgets
(threshold calibrated on validation mean calls, no labels) with the per-video cap 8 and with the cap max-calls; AP /
ROC / within by the shared core evaluator (hc.frame_metrics), and per group: mean score of negative-video seconds,
of hateful and of clean seconds of positive videos, and the Brier score of negative and of positive videos.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy, qtree, stop_rules
from qtl.backbones import make_model
from qtl.checkpoint import load_trial
from qtl.answer_calib import fit_validation_answer_model
from qtl.copy_noise import bucket_of, estimate_pi_neg
import hier_evidence_common as hc

RULE = "qmixE1"
CAP = 8
SOURCES = ("real_copy", "real_indep", "sim_indep", "sim_copy")
EXTRA = ("real_val3", "real_val3_copy", "shuf_anch", "shuf_val3", "sim_val3", "real_len", "real_len_copy")


def node_states(T, label, gt_v):
    """(n_nodes,) true state of every node of the tree of [0, T): 0 / 1 / 2 as in the docstring."""
    tr = qtree.tree(T)
    if label == 0:
        return np.zeros(len(tr["a"]), dtype=np.int64)
    y = np.zeros(T)
    g = np.asarray(gt_v, dtype=np.float64)[:T]
    y[:len(g)] = g
    c = np.concatenate([[0.0], np.cumsum(y > 0.5)])
    return np.where(c[tr["b"]] - c[tr["a"]] > 0, 2, 1).astype(np.int64)


def simulate(vids, store, am, cats, labels, gt, pi, copy, rng):
    """Simulated answers {video: {(a, b): answer vector}} for every queryable node."""
    out = {}
    for v in vids:
        T = store.T[v]
        vt = cpolicy._vt(T)
        asker = policy.Asker(vt, am, cats)
        tr = vt.tr
        st = node_states(T, labels[v], gt.get(v, np.zeros(T)))
        o = {}
        for i, n in enumerate(asker.q):                  # breadth-first: a parent precedes its children
            n = int(n)
            par = int(tr["parent"][n])
            length = float(tr["b"][n] - tr["a"][n])
            if copy and par >= 0 and par in o and rng.random() < pi[bucket_of(length)]:
                o[n] = o[par]
            else:
                p = asker.po_s[i, st[n]]
                o[n] = int(min(np.searchsorted(np.cumsum(p), rng.random() * p.sum(), side="right"), len(p) - 1))
        out[v] = {(int(tr["a"][n]), int(tr["b"][n])): np.array([k], dtype=np.int64) for n, k in o.items()}
    return out


def shuffle(vids, store, answers, labels, gt, rng):
    """Shuffled cached answers (shuf_*): per (state, length bucket) pool over the videos of one split."""
    st_of, pools = {}, {}
    for v in vids:
        T = store.T[v]
        tr = qtree.tree(T)
        st = node_states(T, labels[v], gt.get(v, np.zeros(T)))
        for n in np.where(tr["queryable"])[0]:
            key = (int(tr["a"][n]), int(tr["b"][n]))
            k = (int(st[n]), bucket_of(float(key[1] - key[0])))
            st_of[(v, key)] = k
            o = answers[v].get(key)
            if o is not None:
                pools.setdefault(k, []).append((v, o))
    out = {}
    for (v, key), k in st_of.items():
        pool = pools.get(k)
        if not pool:
            continue
        for _ in range(20):                              # from another video when the pool allows
            w, o = pool[int(rng.integers(len(pool)))]
            if w != v:
                break
        out.setdefault(v, {})[key] = o
    return out


def group_stats(scores, vids, labels, gt):
    neg, pos_h, pos_c, b_neg, b_pos = [], [], [], [], []
    for v in vids:
        s = np.asarray(scores[v], dtype=np.float64)
        y = np.asarray(gt[v], dtype=np.float64)[:len(s)] > 0.5
        s = s[:len(y)]
        if labels[v] == 0:
            neg.append(s)
            b_neg.append(np.mean(s ** 2))
        else:
            pos_h.append(s[y])
            pos_c.append(s[~y])
            b_pos.append(np.mean((s - y) ** 2))
    cat = lambda xs: float(np.mean(np.concatenate(xs))) if xs and sum(len(x) for x in xs) else float("nan")  # noqa
    return {"neg_mean": cat(neg), "pos_hate_mean": cat(pos_h), "pos_clean_mean": cat(pos_c),
            "brier_neg": float(np.mean(b_neg)), "brier_pos": float(np.mean(b_pos))}


def run_source(model, store, ids, am, chain, answers, cfg, max_calls, device, copy_fn):
    runs = {}
    for split in ("val", "test"):
        vids = sorted(ids[split], key=lambda v: store.T[v])
        runs[split] = {}
        for i in range(0, len(vids), int(cfg["eval_chunk"])):
            with torch.no_grad():
                runs[split].update(cpolicy.run_batch(
                    model, store, vids[i:i + int(cfg["eval_chunk"])], am, chain, answers, cfg["categories"],
                    max_calls, device, order="eig", copy_pi=copy_fn, copy_mode="both"))
    for split in runs:                                   # rule_values reads "voi"; qmixE1 does not use it
        for r in runs[split].values():
            r["voi"] = [0.0] * len(r["eig"])
    return runs


def truncate(run, cap):
    n = min(cap, len(run["eig"]))
    return {k: (list(v[:n + 1]) if k in ("scores", "p_G") else list(v[:n])) for k, v in run.items()}


def evaluate(runs, ids, labels, gt, hate, max_calls, budgets_cap, budgets_max):
    test, val = list(ids["test"]), list(ids["val"])
    pos = np.array([labels[v] == 1 for v in test])

    def summ(calls):
        sc = {v: runs["test"][v]["scores"][calls[v]] for v in test}
        m = hc.frame_metrics(sc, gt["test"], hate)
        c = np.array([calls[v] for v in test])
        r = {"pooled_ap": m["pooled_ap"], "pooled_roc": m["pooled_roc"], "within_roc": m["within_roc"],
             "mean_calls": float(c.mean()), "mean_calls_positive": float(c[pos].mean()),
             "mean_calls_negative": float(c[~pos].mean())}
        r.update(group_stats(sc, test, labels, gt["test"]))
        return r
    row = {"fixed": {}, "rule_cap8": {}, "rule_cap%d" % max_calls: {}}
    for k in range(max_calls + 1):
        row["fixed"][str(k)] = summ({v: min(k, len(runs["test"][v]["eig"])) for v in test})
    for cap, budgets, key in ((CAP, budgets_cap, "rule_cap8"), (max_calls, budgets_max, "rule_cap%d" % max_calls)):
        tr = {s: {v: truncate(r, cap) for v, r in runs[s].items()} for s in runs}
        vals = {s: {v: stop_rules.rule_values(r, len(r["scores"][0])) for v, r in tr[s].items()} for s in tr}
        stop_rules.add_quantile_rules(vals)
        for b in budgets:
            thr, vm = policy.calibrate([vals["val"][v][RULE] for v in val], b)
            calls = {v: policy.stop_calls(vals["test"][v][RULE], thr) for v in test}
            r_ = summ(calls)
            r_.update(validation_mean_calls=vm)
            row[key]["%g" % b] = r_
    return row


def mean_rows(rows):
    out = {}
    for key in rows[0]:
        out[key] = {}
        for k in rows[0][key]:
            out[key][k] = {m: float(np.mean([r[key][k][m] for r in rows])) for m in rows[0][key][k]}
            if len(rows) > 1:
                for m in ("pooled_ap", "pooled_roc", "within_roc"):
                    out[key][k][m + "_sd"] = float(np.std([r[key][k][m] for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--trial-dir", required=True)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES + EXTRA)
    ap.add_argument("--max-calls", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budgets-cap", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--budgets-max", nargs="+", type=float, default=[1, 2, 4, 6, 8, 12, 16, 24])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", default="runs/20261101_r5_completion/sim_monotone")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {v for v in ids["test"] if labels[v] == 1}
    gt_all = {**gt["val"], **gt["test"]}
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, ids["val"] + ids["test"], ["bert"])
    trial = Path(a.trial_dir).resolve()
    summary, cfg, model, am, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
    assert cfg["answer_source"] == "soft_both", cfg["answer_source"]
    pi, *_ = estimate_pi_neg(answers, labels, ids["train"], answer_T, am, cfg["categories"])
    copy_fn = lambda length, p=pi: float(p[bucket_of(length)])  # noqa: E731
    name = f'seed{summary["seed"]}_{trial.name}'
    out = (ROOT / a.out_root / a.corpus / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    am_val3, am_len = None, None
    if any("len" in s_ for s_ in a.sources):
        loglen = np.log([b - a_ for v in ids["train"] for (a_, b) in answers[v]])
        ta = [(labels[v], [(x, y, answer_T[v], o) for (x, y), o in answers[v].items()]) for v in ids["train"]]
        th_l, om_l, n_l = qtree.fit_anchored(ta, loglen.mean(), loglen.std(), True)
        am_len = qtree.AnswerModel(th_l, loglen.mean(), loglen.std(), True, 2, cfg["categories"], om_l).to(a.device)
        am_len.requires_grad_(False)
        print("length-term anchored model fitted on", n_l, flush=True)
    if any("val3" in s_ for s_ in a.sources):
        am_val3, counts = fit_validation_answer_model(answers, labels, gt["val"], ids["val"], store.T, am.len_mu,
                                                      am.len_sd, cfg["categories"])
        am_val3 = am_val3.to(a.device)
        print("val3 state counts", counts, flush=True)
    (out / "run.pid").write_text(str(__import__("os").getpid()))
    prev = out / "summary.json"
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "trial": str(trial), "pi_neg": [float(x) for x in pi], "max_calls": a.max_calls, "reps": a.reps,
              "sources": {}}
    if prev.exists():                                    # a later round adds sources to the same summary
        report["sources"] = json.loads(prev.read_text())["sources"]
    print("host", socket.gethostname(), name, "pi", np.round(pi, 3).tolist(), flush=True)
    for src in a.sources:
        rows = []
        n_rep = 1 if src.startswith("real") else a.reps
        am_s = am_val3 if "val3" in src else (am_len if "_len" in src else am)
        for rep in range(n_rep):
            rng = np.random.default_rng(1000 * int(summary["seed"]) + rep)
            if src.startswith("real"):
                ans = answers
            elif src.startswith("shuf"):
                ans = {}
                for split in ("val", "test"):
                    ans.update(shuffle(ids[split], store, answers, labels, gt_all, rng))
            else:
                ans = simulate(ids["val"] + ids["test"], store, am_s, cfg["categories"], labels, gt_all, pi,
                               src == "sim_copy", rng)
            runs = run_source(model, store, ids, am_s, chain, ans, cfg, a.max_calls, a.device,
                              copy_fn if src.endswith("copy") else None)
            if rep == 0:
                with (out / f"runs_{src}.pkl").open("wb") as fh:
                    pickle.dump({"runs": runs}, fh)
            rows.append(evaluate(runs, ids, labels, gt, hate, a.max_calls, a.budgets_cap, a.budgets_max))
            f = rows[-1]["fixed"]
            print(name, src, rep, " ".join("k%s %.3f/%.3f/%.3f" % (k, f[k]["pooled_ap"], f[k]["pooled_roc"],
                                                                  f[k]["within_roc"]) for k in ("0", "4", "8", "16", "32")
                                           if k in f), flush=True)
        report["sources"][src] = {"mean": mean_rows(rows), "per_rep": rows if len(rows) > 1 else None}
        (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", name, datetime.now().astimezone().isoformat(), flush=True)


if __name__ == "__main__":
    main()
