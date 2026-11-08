"""Diagnostic (README section 6, with naive_questions.py): our model and tree inference, but the questions are the 8
equal windows of tree depth 3 (the deepest depth with all nodes queryable when the video is shorter), asked in EIG
order, all of them (8 calls or fewer); no other node can be asked. Revision-6 test-selected checkpoints, cached
answers, test split; against the method at fixed 8 calls (replay runs_L0.pkl). Unified evaluator.
Output runs/20261109_tree_within/uniform8/<corpus>/ (summary.json).
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
from qtl import cpolicy, data as qdata, qtree                           # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
REPLAY = ROOT / "runs/20261108_astra_review/replay"


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def depth3(T):
    tr = qtree.tree(T)
    d = 3
    while d > 0 and not tr["queryable"][tr["depth"] == d].all():
        d -= 1
    return set(np.where(tr["depth"] == d)[0].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/uniform8")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    _labels, ids, _gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, test, ["bert"])
    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        name = f'seed{summary["seed"]}_{trial.name}'
        od = (ROOT / a.out_root / a.corpus / name).resolve()
        od.mkdir(parents=True, exist_ok=True)
        full = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        runs = {}
        vids = sorted(test, key=lambda v: store.T[v])
        for i in range(0, len(vids), int(cfg["eval_chunk"])):
            chunk = vids[i:i + int(cfg["eval_chunk"])]
            with torch.no_grad():
                runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], 8, "cpu",
                                              allowed=[depth3(store.T[v]) for v in chunk]))
        with (od / "runs_uniform8.pkl").open("wb") as fh:
            pickle.dump({"runs": runs, "trial": str(trial), "host": socket.gethostname(),
                         "created": datetime.now().astimezone().isoformat()}, fh)
        mu = score(a.corpus, od, "uniform8", {v: runs[v]["scores"][-1] for v in test})
        mm = score(a.corpus, od, "method_fixed8", {v: full[v]["scores"][min(8, len(full[v]["scores"]) - 1)]
                                                   for v in test})
        rows[name] = {"uniform8": mu, "method": mm, "uniform8_minus_method": {m: mu[m] - mm[m] for m in METS},
                      "mean_calls": float(np.mean([len(runs[v]["asked"]) for v in test]))}
        print(a.corpus, name, json.dumps(rows[name]), flush=True)
    mean = {k: {m: float(np.mean([rows[t][k][m] for t in rows])) for m in METS}
            for k in ("uniform8", "method", "uniform8_minus_method")}
    mean["mean_calls"] = float(np.mean([rows[t]["mean_calls"] for t in rows]))
    (ROOT / a.out_root / a.corpus / "summary.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps(mean), flush=True)


if __name__ == "__main__":
    main()
