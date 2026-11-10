"""Diagnostic (README section 2): how well calibrated against each other are the two sources of the video decision,
the backbone's video logit g and the evidence of the root's answer? Revision-6 test-selected checkpoints, the
revision-6 answer table, splits train / val / test (train = the videos the backbone was fitted on).

Per trial and split, over the videos (root-only replay, 1 call):
  g           the backbone's video logit, logit P(G = 1) before any call;
  llr         log P(o_root | hateful) - log P(o_root | no hate) of the revision-6 table
              (= logit P(G = 1 | o_root) - g);
  x           logit of the raw soft answer P(Yes) of the root.
Reported: video-level ROC-AUC of g, llr, x, g + llr (the model's video posterior), g + x; the coefficients of an
unregularised logistic regression of the video label on (g, llr) and on (g, x) (a coefficient of 1 on both = the
model's sum is the calibrated one; below 1 = that source is over-confident on this split); and the ROC-AUC of the
regression's fitted combination (in-sample, an upper bound for a linear re-weighting).
Output runs/20261110_answer_likelihood/video_evidence/<corpus>.json.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata                                  # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261110_answer_likelihood/video_evidence")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, _gt, _ = hc.load_fixed_cohort(a.corpus)
    qdata.configure_source("soft_both", 8)
    answers, T_ans = qdata.load_answers(a.corpus, "soft_both")
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    vids = {s: list(ids[s]) for s in a.splits}
    store = qdata.Store(a.corpus, sum(vids.values(), []), ["bert"])
    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        name = f'seed{summary["seed"]}_{trial.name}'
        row = {}
        for split, vs in vids.items():
            runs = {}
            order = sorted(vs, key=lambda v: store.T[v])
            for i in range(0, len(order), int(cfg["eval_chunk"])):
                chunk = order[i:i + int(cfg["eval_chunk"])]
                with torch.no_grad():
                    runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], 1,
                                                  "cpu", allowed=[{0} for _ in chunk]))
            y = np.array([labels[v] for v in vs])
            g = np.array([logit(runs[v]["p_G"][0]) for v in vs])
            llr = np.array([logit(runs[v]["p_G"][-1]) for v in vs]) - g
            x = np.array([logit(P[v][(0, T_ans[v])]) if P[v].get((0, T_ans[v])) is not None else 0.0 for v in vs])
            res = {"n": int(len(y)), "positive_rate": float(y.mean()),
                   "auc": {k: float(roc_auc_score(y, s)) for k, s in
                           (("g", g), ("llr", llr), ("x", x), ("g+llr", g + llr), ("g+x", g + x))},
                   "sd": {"g": float(g.std()), "llr": float(llr.std()), "x": float(x.std())}}
            for tag, ev in (("llr", llr), ("x", x)):
                lr = LogisticRegression(C=1e6, max_iter=2000).fit(np.stack([g, ev], 1), y)
                res[f"logreg_g_{tag}"] = {"coef_g": float(lr.coef_[0, 0]), f"coef_{tag}": float(lr.coef_[0, 1]),
                                          "intercept": float(lr.intercept_[0]),
                                          "auc_fitted": float(roc_auc_score(y, lr.decision_function(
                                              np.stack([g, ev], 1))))}
            row[split] = res
            print(a.corpus, name, split, json.dumps(res), flush=True)
        rows[name] = row

    def avg(path):
        vals = []
        for r in rows.values():
            d = r
            for k in path:
                d = d[k]
            vals.append(d)
        return float(np.mean(vals))

    mean = {}
    for split in vids:
        mean[split] = {"auc": {k: avg((split, "auc", k)) for k in ("g", "llr", "x", "g+llr", "g+x")},
                       "sd": {k: avg((split, "sd", k)) for k in ("g", "llr", "x")},
                       "logreg_g_llr": {k: avg((split, "logreg_g_llr", k))
                                        for k in ("coef_g", "coef_llr", "intercept", "auc_fitted")},
                       "logreg_g_x": {k: avg((split, "logreg_g_x", k))
                                      for k in ("coef_g", "coef_x", "intercept", "auc_fitted")}}
        print("MEAN", a.corpus, split, json.dumps({k: {kk: round(vv, 3) for kk, vv in d.items()}
                                                   for k, d in mean[split].items()}), flush=True)
    out = (ROOT / a.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
