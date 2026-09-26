"""How much could a better answer model give? (README section 14; development evidence on test, rule 10: the answer
models here are fitted on test GT and are ceilings, never a method.)

For every test node with a parsed answer the true state is s = 0 (negative video), 2 (the node contains GT harm)
or 1 (positive video, node without harm). Per category and state, a multinomial logistic regression of the answer
level on node features, cross-fitted over 5 folds of test videos (each video's nodes are predicted by a model that
never saw that video), one pseudo-answer per level at the feature mean:
    gt_len   features: standardised log length (the three-state length model of README 10.4)
    gt_full  + share of the video, transcript words, words per second, and 8 PCA components each of the per-video
             z-scored node means of VGGish, BERT rows and five-crop-mean I3D (what a content-aware answer head
             could see)
Each trained revision-3 trial is then re-run with its own prior and chain, the real cached answers, and the
per-node outcome tables of the fitted model (questions chosen by EIG under that model). Pooled AP / ROC / within
at fixed budgets through the shared evaluator; the anchored model of the method is the trial's own summary.

    python experiments/20260925_query_paradigm/answer_model_ceiling.py --corpus hatemm --trials <trial dirs>
Writes <out>/<corpus>_ceiling.json and score / metric files under <out>/<corpus>_ceiling/<variant>/<trial tag>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import concern_diagnostics as cd                # noqa: E402  (sets up the import paths)
import train as TR                              # noqa: E402
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import policy                                   # noqa: E402
import cpolicy                                  # noqa: E402
from macilsd import align                       # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32)


class NodeAsker(policy.Asker):
    """policy.Asker with per-node outcome tables: lp (nq, N_CAT, 3, N_LEV) for vt.q_ids in order."""

    def __init__(self, vt, lp, categories):
        self.vt = vt
        self.q = vt.q_ids
        self.qpos = {int(n): i for i, n in enumerate(self.q)}
        self.cats = list(categories)
        self.table = qtree.outcome_table(lp[:, self.cats])
        self.po_s = np.exp(self.table)
        self.h_s = -np.sum(self.po_s * self.table, axis=2)


def node_table(corpus, store, vids, answers, gt, labels):
    """Per queryable node of every video: features, state, answer (or None)."""
    words = {}
    for line in open(os.path.join(ROOT, "data", "vlm_tree", qdata.CORPUS_DIR[corpus], "manifest.jsonl")):
        r = json.loads(line)
        if r["id"] in vids:
            words[r["id"]] = {(int(a), int(b)): len(text.split()) for a, b, _idx, text in r["nodes"]}
    A = align.A_DIM
    meta, blocks = [], {"vgg": [], "bert": [], "i3d": []}
    for gi, v in enumerate(vids):
        T = store.T[v]
        y = np.asarray(gt[v])[:T]
        at = store.at[v]
        vis = np.mean([store.visual(v, c) for c in range(5)], axis=0)
        feats = {"vgg": cd.zs(at[:, :A]), "bert": cd.zs(at[:, A:A + hc.TEXT_DIM]), "i3d": cd.zs(vis)}
        tr = qtree.tree(T)
        for n in np.where(tr["queryable"])[0]:
            a, b = int(tr["a"][n]), int(tr["b"][n])
            w = words.get(v, {}).get((a, b), 0)
            s = 0 if labels[v] == 0 else (2 if y[a:b].any() else 1)
            meta.append({"v": v, "g": gi, "node": int(n), "state": s, "o": answers[v].get((a, b)),
                         "x": [np.log(b - a), (b - a) / T, np.log1p(w), w / (b - a)]})
            for k in blocks:
                blocks[k].append(feats[k][a:b].mean(0))
    pcs = [PCA(8, random_state=0).fit_transform(np.asarray(blocks[k])) for k in ("vgg", "bert", "i3d")]
    X_full = np.concatenate([np.array([m["x"] for m in meta])] + pcs, axis=1)
    X_len = X_full[:, :1]
    return meta, {"gt_len": X_len, "gt_full": X_full}


def fit_predict(meta, X, n_folds=5):
    """Cross-fitted log P(o_k = l | s, x) for every node: (N, N_CAT, 3, N_LEV)."""
    X = StandardScaler().fit_transform(X)
    N = len(meta)
    groups = np.array([m["g"] for m in meta])
    state = np.array([m["state"] for m in meta])
    has = np.array([m["o"] is not None for m in meta])
    O = np.array([m["o"] if m["o"] is not None else np.zeros(qtree.N_CAT, dtype=int) for m in meta])
    out = np.zeros((N, qtree.N_CAT, 3, qtree.N_LEV))
    pseudo_X = np.zeros((qtree.N_LEV, X.shape[1]))
    pseudo_y = np.arange(qtree.N_LEV)
    for trn, tst in GroupKFold(n_folds).split(X, state, groups):
        for s in range(3):
            fit_rows = trn[(state[trn] == s) & has[trn]]
            for k in range(qtree.N_CAT):
                Xf = np.concatenate([X[fit_rows], pseudo_X])
                yf = np.concatenate([O[fit_rows, k], pseudo_y])
                clf = LogisticRegression(max_iter=3000).fit(Xf, yf)
                lp = np.full((len(tst), qtree.N_LEV), -30.0)
                lp[:, clf.classes_] = clf.predict_log_proba(X[tst])
                out[tst, k, s] = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "20260927_query_paradigm_r4", "diagnostics"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, _ = qdata.load_answers(a.corpus)
    vids = ids["test"]
    res = {"corpus": a.corpus, "variants": {}, "trials": {}}
    store, tables = None, {}
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
            meta, Xs = node_table(a.corpus, store, vids, answers, gt["test"], labels)
            for name, X in Xs.items():
                lp = fit_predict(meta, X)
                per_v = {}
                for i, m in enumerate(meta):
                    per_v.setdefault(m["v"], []).append((m["node"], lp[i]))
                tables[name] = {v: np.stack([x for _, x in sorted(r, key=lambda t: t[0])]) for v, r in per_v.items()}
                # held-out log-likelihood of the observed answers under the true state (per answered node)
                st = np.array([m["state"] for m in meta])
                ok = np.array([m["o"] is not None for m in meta])
                O = np.array([m["o"] if m["o"] is not None else np.zeros(5, dtype=int) for m in meta])
                ll = lp[np.arange(len(meta))[:, None], np.arange(5)[None], st[:, None], O].sum(1)
                res["variants"][name] = {"n_nodes": int(ok.sum()), "heldout_ll_per_node": float(ll[ok].mean())}
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        r = {"anchored": {B: {k: summ["fixed"][B]["test"][k] for k in ("pooled_ap", "pooled_roc", "within_roc")}
                          for B in map(str, BUDGETS)}}
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
                                              max(BUDGETS), a.device, "eig", "tree", False, None, None, mk))
            od = os.path.join(a.out, "%s_ceiling" % a.corpus, name, tag)
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
        print("== %s" % trial, flush=True)
        for B in map(str, BUDGETS):
            print("  B=%-2s " % B + " | ".join("%s %.4f/%.4f/%.4f" % (n, r[n][B]["pooled_ap"], r[n][B]["pooled_roc"],
                                                                     r[n][B]["within_roc"])
                                               for n in ("anchored", "gt_len", "gt_full")), flush=True)
        json.dump(res, open(os.path.join(a.out, "%s_ceiling.json" % a.corpus), "w"), indent=1, default=float)
    print("held-out log-likelihood per answered node:", json.dumps(res["variants"]))


if __name__ == "__main__":
    main()
