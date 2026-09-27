"""Revision-5 option 1 premise check (README section 16.4; development evidence, rule 10; the network is not
retrained and nothing is selected here): an answer model fitted on VALIDATION span labels.

README 16.4: with video labels only, "much hate and weak answers" and "little hate and reliable answers" explain the
training answers equally well, so the anchored answer model over-trusts short-node answers. Validation span labels
are already used by the method (checkpoint selection: pooled AP / ROC at 8 calls on validation, train.py). Here they
fit the three-state answer model with a length term (s = 0 negative video, 1 positive video and no GT harm in the
node, 2 GT harm in the node; per category and state a multinomial logistic regression of the level on the
standardised log length, one pseudo-answer per level at the mean length), on every answered node of the validation
videos. Each revision-4 trial (network, chain, node potentials unchanged) is re-run on test with it:
    val_len    fitted on validation (the option)
    test_len   cross-fitted on test over 5 folds of videos (the ceiling of README 14.2 with the same features)
Pooled AP / ROC / within at fixed budgets through the shared evaluator, next to the trial's own numbers.

    python experiments/20260925_query_paradigm/val_calibration_check.py --corpus hatemm --trials <trial dirs>
Writes runs/20260928_query_paradigm_r5_analysis/val_calibration/<corpus>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import concern_diagnostics as cd                # noqa: E402  (sets up the import paths)
import train as TR                              # noqa: E402
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import cpolicy                                  # noqa: E402
from answer_model_ceiling import NodeAsker, fit_predict   # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32)
OUT = os.path.join(ROOT, "runs", "20260928_query_paradigm_r5_analysis", "val_calibration")


def nodes(vids, T, answers, gt, labels):
    """Every queryable node of the videos: video, node id, state, answer (or None), log length."""
    meta = []
    for gi, v in enumerate(vids):
        y = np.asarray(gt[v])
        tr = qtree.tree(T[v])
        for n in np.where(tr["queryable"])[0]:
            a, b = int(tr["a"][n]), int(tr["b"][n])
            s = 0 if labels[v] == 0 else (2 if y[a:b].any() else 1)
            meta.append({"v": v, "g": gi, "node": int(n), "state": s, "o": answers[v].get((a, b)),
                         "x": [np.log(b - a)]})
    return meta


def fit_on(meta_fit, meta_pred):
    """log P(o_k = l | s, length) for meta_pred nodes (N, N_CAT, 3, N_LEV), fitted on meta_fit."""
    xf = np.array([m["x"][0] for m in meta_fit])
    mu, sd = xf.mean(), xf.std()
    Xf = ((xf - mu) / sd)[:, None]
    Xp = ((np.array([m["x"][0] for m in meta_pred]) - mu) / sd)[:, None]
    state = np.array([m["state"] for m in meta_fit])
    has = np.array([m["o"] is not None for m in meta_fit])
    O = np.array([m["o"] if m["o"] is not None else np.zeros(qtree.N_CAT, dtype=int) for m in meta_fit])
    out = np.zeros((len(meta_pred), qtree.N_CAT, 3, qtree.N_LEV))
    for s in range(3):
        rows = (state == s) & has
        for k in range(qtree.N_CAT):
            X = np.concatenate([Xf[rows], np.zeros((qtree.N_LEV, 1))])
            y = np.concatenate([O[rows, k], np.arange(qtree.N_LEV)])
            clf = LogisticRegression(max_iter=3000).fit(X, y)
            lp = np.full((len(meta_pred), qtree.N_LEV), -30.0)
            lp[:, clf.classes_] = clf.predict_log_proba(Xp)
            out[:, k, s] = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    return out


def per_video(meta, lp):
    rows = {}
    for i, m in enumerate(meta):
        rows.setdefault(m["v"], []).append((m["node"], lp[i]))
    return {v: np.stack([x for _, x in sorted(r, key=lambda t: t[0])]) for v, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, T = qdata.load_answers(a.corpus, "words")
    vids = list(ids["test"])
    meta_val = nodes(ids["val"], T, answers, gt["val"], labels)
    meta_test = nodes(vids, T, answers, gt["test"], labels)
    tables = {"val_len": per_video(meta_test, fit_on(meta_val, meta_test)),
              "test_len": per_video(meta_test, fit_predict(meta_test, np.array([m["x"] for m in meta_test])))}
    rate = {}
    for name, mt in (("val", meta_val), ("test", meta_test)):
        r = {}
        for s in range(3):
            o = [max(m["o"]) >= 2 for m in mt if m["state"] == s and m["o"] is not None]
            r[s] = [len(o), float(np.mean(o))]
        rate[name] = r
    print("%s: %d validation nodes; 'some category >= 2' rate per state (n, rate): %s" % (
        a.corpus, len(meta_val), json.dumps(rate)), flush=True)
    res = {"corpus": a.corpus, "yes_rate_by_state": rate, "n_val_videos": len(ids["val"]), "trials": {}}
    os.makedirs(OUT, exist_ok=True)
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        r = {"method": {str(B): {k: summ["fixed"][str(B)]["test"][k] for k in ("pooled_ap", "pooled_roc",
                                                                                 "within_roc")} for B in BUDGETS}}
        for name, tab in tables.items():
            runs = {}
            order = sorted(vids, key=lambda v: store.T[v])
            k = int(cfg["eval_chunk"])
            for i in range(0, len(order), k):
                chunk = order[i:i + k]

                def mk(b, vt, chunk=chunk, tab=tab):
                    return NodeAsker(vt, tab[chunk[b]] if len(vt.q_ids) else np.zeros((0, 5, 3, 4)),
                                     cfg["categories"])

                runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"],
                                              max(BUDGETS), a.device, make_asker=mk))
            od = os.path.join(OUT, a.corpus, name, tag)
            os.makedirs(od, exist_ok=True)
            r[name] = {}
            for B in BUDGETS:
                sp = os.path.join(od, "scores_test_fixed%d.jsonl" % B)
                hc.write_scores(sp, TR.at_budget(runs, B))
                m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_fixed%d.json" % B))
                m = m["results"]["score_av"]
                r[name][str(B)] = {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                                   "within_roc": m["per_video"]["macro_auc"]}
        res["trials"][trial] = r
        print("== %s" % tag, flush=True)
        for B in map(str, BUDGETS):
            print("  B=%-2s " % B + " | ".join("%s %.4f/%.4f/%.4f" % (n, r[n][B]["pooled_ap"], r[n][B]["pooled_roc"],
                                                                     r[n][B]["within_roc"])
                                               for n in ("method", "val_len", "test_len")), flush=True)
        json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
