"""Stage A (developmental, README section 3): label-free prior weights on the revision-6 test-selected checkpoints.

Nothing is retrained. Per trial the weights (a, b, c) of src/qtl/prior_weight.py are estimated on the VALIDATION
videos' cached answers without labels, then the test videos are replayed with the weighted prior (posterior and
question choice). Variants:
  r6         no weights (a, b, c) = (1, 0, 1): revision 6;
  root       (a, b) from the root answers, c = 1;
  root_all   (a, b) from the root answers, c from all cached answers (the proposed estimator);
  joint      (a, b, c) from all cached answers at once.
Reported per variant, test, AP / ROC / within at 0 / 1 / 2 / 4 / 8 calls of the EIG tree and after the 8 equal
windows (u8); the video-level ROC-AUC of P(G = 1 | answers) after 1 and 8 calls; the pooled AP when every second
gets its video's mean score (how much of the pooled AP is whole-video ordering). For reference only (not used by any
variant): the labelled logistic regression of the video label on g, on validation and on test.
Scores at 6 decimals as the evaluator reads them; unified core evaluator (hc.frame_metrics).
Output runs/20261110_prior_weight/stage_a/<corpus>/summary.json.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy, prior_weight, qtree      # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
BUDGETS = (0, 1, 2, 4, 8)


def replay(model, store, vids, am, chain, answers, cfg, max_calls, allowed, w):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], max_calls, "cpu",
                                          allowed=None if allowed is None else [allowed(store.T[v]) for v in chunk],
                                          prior_weight=None if w is None else prior_weight.as_tuple(w)))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--modes", nargs="+", default=["root", "root_all", "joint"])
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--out-root", default="runs/20261110_prior_weight/stage_a")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    od = (ROOT / a.out_root / a.corpus).resolve()
    od.mkdir(parents=True, exist_ok=True)
    (od / "run.pid").write_text(str(os.getpid()))
    print("host", socket.gethostname(), datetime.now().astimezone().isoformat(), flush=True)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    val, test = list(ids["val"]), list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    qdata.configure_source("soft_both", 8)
    answers, _T = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, val + test, ["bert"])

    def mets(sc):
        sc = {v: np.round(np.asarray(s, dtype=np.float64), 6) for v, s in sc.items()}
        m = hc.frame_metrics(sc, gt["test"], hate)
        return {k: float(m[k]) for k in METS}

    pri = {}
    plain_prior = policy.video_prior_nodes
    policy.video_prior_nodes = lambda model, store_, v, device: pri[v]   # the backbone runs once per video and trial
    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        name = f'seed{summary["seed"]}_{trial.name}'
        pri.clear()
        with torch.no_grad():
            for v in val + test:
                pri[v] = plain_prior(model, store, v, "cpu")
        weights = {"r6": None}
        for mode in a.modes:
            t0 = time.time()
            weights[mode] = prior_weight.fit(pri, store, val, am, chain, answers, "cpu", mode=mode)
            weights[mode]["seconds"] = time.time() - t0
            print(a.corpus, name, "weights", mode, json.dumps({k: (round(x, 4) if isinstance(x, float) else x)
                                                               for k, x in weights[mode].items()}), flush=True)
        ref = {s: prior_weight.label_reference(pri, vs, labels) for s, vs in (("val", val), ("test", test))}
        print(a.corpus, name, "labelled reference (not used)", json.dumps(ref), flush=True)
        row = {"weights": weights, "label_reference": ref, "variants": {}}
        y_video = np.array([labels[v] for v in test])
        for vname, w in weights.items():
            runs = replay(model, store, test, am, chain, answers, cfg, 8, None, w)
            res = {}
            for B in BUDGETS:
                res["eig%d" % B] = mets({v: runs[v]["scores"][min(B, len(runs[v]["eig"]))] for v in test})
            sc8 = {v: runs[v]["scores"][min(8, len(runs[v]["eig"]))] for v in test}
            res["eig8_video_mean"] = mets({v: np.full(len(s), float(np.mean(s))) for v, s in sc8.items()})
            for B in (0, 1, 8):
                pg = np.array([runs[v]["p_G"][min(B, len(runs[v]["eig"]))] for v in test])
                res["video_auc_pG_%d" % B] = float(roc_auc_score(y_video, pg))
            ru = replay(model, store, test, am, chain, answers, cfg, 8,
                        lambda T: set(qtree.depth_nodes(T, 3).tolist()), w)
            res["u8"] = mets({v: ru[v]["scores"][-1] for v in test})
            row["variants"][vname] = res
            print(a.corpus, name, vname, json.dumps({k: ([round(r[m], 4) for m in METS] if isinstance(r, dict)
                                                         else round(r, 4)) for k, r in res.items()}), flush=True)
        rows[name] = row
    names = list(next(iter(rows.values()))["variants"])
    keys = list(next(iter(rows.values()))["variants"][names[0]])
    mean = {}
    for c in names:
        mean[c] = {}
        for k in keys:
            vals = [rows[t]["variants"][c][k] for t in rows]
            mean[c][k] = ({m: float(np.mean([x[m] for x in vals])) for m in METS} if isinstance(vals[0], dict)
                          else float(np.mean(vals)))
    mean_w = {c: {p: float(np.mean([rows[t]["weights"][c][p] for t in rows])) for p in ("a", "b", "c")}
              for c in names if c != "r6"}
    (od / "summary.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean, "mean_weights": mean_w}, indent=1) + "\n")
    for c in names:
        print("MEAN", a.corpus, "%-9s" % c, " | ".join(
            "%s %s" % (k, "/".join("%.3f" % mean[c][k][m] for m in METS) if isinstance(mean[c][k], dict)
                       else "%.3f" % mean[c][k]) for k in keys), flush=True)
    print("MEAN WEIGHTS", a.corpus, json.dumps({c: {p: round(x, 3) for p, x in d.items()}
                                                for c, d in mean_w.items()}), flush=True)
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
