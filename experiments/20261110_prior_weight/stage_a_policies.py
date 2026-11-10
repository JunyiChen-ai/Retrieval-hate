"""Stage A (developmental, README section 4): which question policy under the label-free prior weights?

Revision-6 test-selected checkpoints, nothing retrained. Per trial the prior weights are estimated on the validation
answers without labels (src/qtl/prior_weight.py, --weights modes; "none" = revision 6's prior), then the test videos
are replayed at a fixed number of calls per video (--budgets) with each policy (all already in cpolicy.run_batch):
  eig         greedy expected information gain on the full tree (revision 6);
  eer         greedy expected reduction of the per-second squared error;
  spread      greedy expected ranking gain;
  eig_loc     greedy information gain about the seconds given G = 1;
  cover_eig   the first questions are disjoint nodes that together cover the video (as many as the budget), chosen
              by EIG; cover_eer the same chosen by eer;
  rootcover_eig  the root first, then a cover with the remaining calls;
  level       one tree depth: the 2^d equal windows with 2^d <= budget (the single-level control; fewer calls than
              the budget when the budget is not a power of two);
  plan_eig / plan_eer  budget-aware plan (cpolicy._plan_disjoint): before every call the remaining calls are planned
              as the pairwise disjoint unasked nodes with the largest total EIG / eer, the best planned node is asked,
              the plan is rebuilt after the answer.
AP / ROC / within; the mean depth and mean length (seconds) of the asked nodes. Scores at 6 decimals (hc.frame_metrics).
Output runs/20261110_prior_weight/stage_a_policies/<corpus>/summary_<tag>.json.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy, prior_weight, qtree      # noqa: E402
from qtl.answer_calib import fit_validation_answer_model                # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
POLICIES = {
    "eig": dict(order="eig"),
    "eer": dict(order="eer"),
    "spread": dict(order="spread"),
    "eig_loc": dict(order="eig_loc"),
    "cover_eig": dict(order="eig", cover="plain"),
    "cover_eer": dict(order="eer", cover="plain"),
    "rootcover_eig": dict(order="eig", cover="root"),
    "level": dict(order="eig", level=True),
    "plan_eig": dict(order="eig", plan="disjoint"),
    "plan_eer": dict(order="eer", plan="disjoint"),
}


def replay(model, store, vids, am, chain, answers, cfg, budget, w, kw):
    runs = {}
    kw = dict(kw)
    level = kw.pop("level", False)
    depth = int(np.floor(np.log2(budget)))
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        allowed = [set(qtree.depth_nodes(store.T[v], depth).tolist()) for v in chunk] if level else None
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], budget, "cpu",
                                          allowed=allowed, prior_weight=None if w is None else
                                          prior_weight.as_tuple(w), **kw))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--weights", nargs="+", default=["none", "joint", "loc"])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--budgets", nargs="+", type=int, default=[8])
    ap.add_argument("--tag", default="b8")
    ap.add_argument("--table", choices=("r6", "val3"), default="r6",
                    help="r6: the checkpoint's anchored table; val3: the three-state table with a length slope per "
                         "state fitted on the validation per-second labels (answer_calib.py; no gradient)")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--out-root", default="runs/20261110_prior_weight/stage_a_policies")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    od = (ROOT / a.out_root / a.corpus).resolve()
    od.mkdir(parents=True, exist_ok=True)
    (od / f"run_{a.tag}.pid").write_text(str(os.getpid()))
    print("host", socket.gethostname(), datetime.now().astimezone().isoformat(), flush=True)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    val, test = list(ids["val"]), list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    qdata.configure_source("soft_both", 8)
    answers, T_all = qdata.load_answers(a.corpus, "soft_both")
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
        if a.table == "val3":
            am, counts = fit_validation_answer_model(answers, labels, gt["val"], val, T_all, am.len_mu, am.len_sd,
                                                     cfg["categories"])
            print(a.corpus, name, "val3 table, answers per state", counts, flush=True)
        pri.clear()
        with torch.no_grad():
            for v in val + test:
                pri[v] = plain_prior(model, store, v, "cpu")
        row = {"weights": {}, "results": {}}
        for mode in a.weights:
            w = None if mode == "none" else prior_weight.fit(pri, store, val, am, chain, answers, "cpu", mode=mode)
            row["weights"][mode] = w
            for B in a.budgets:
                for pol in a.policies:
                    runs = replay(model, store, test, am, chain, answers, cfg, B, w, POLICIES[pol])
                    res = mets({v: runs[v]["scores"][-1] for v in test})
                    dep, ln = [], []
                    for v in test:
                        tr = qtree.tree(store.T[v])
                        nodes = np.asarray(runs[v]["asked"], dtype=np.int64)
                        if len(nodes):
                            dep.append(float(tr["depth"][nodes].mean()))
                            ln.append(float((tr["b"][nodes] - tr["a"][nodes]).mean()))
                    res.update(mean_calls=float(np.mean([len(runs[v]["asked"]) for v in test])),
                               mean_depth=float(np.mean(dep)), mean_length=float(np.mean(ln)))
                    row["results"]["%s|%s|%d" % (mode, pol, B)] = res
                    print(a.corpus, name, mode, pol, B, json.dumps({k: round(x, 4) for k, x in res.items()}),
                          flush=True)
        rows[name] = row
    keys = list(next(iter(rows.values()))["results"])
    mean = {k: {m: float(np.mean([rows[t]["results"][k][m] for t in rows]))
                for m in METS + ("mean_calls", "mean_depth", "mean_length")} for k in keys}
    (od / f"summary_{a.tag}.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean}, indent=1) + "\n")
    for k in keys:
        print("MEAN", a.corpus, "%-26s" % k, "/".join("%.3f" % mean[k][m] for m in METS),
              "calls %.2f depth %.2f length %.1f" % (mean[k]["mean_calls"], mean[k]["mean_depth"],
                                                     mean[k]["mean_length"]), flush=True)
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
