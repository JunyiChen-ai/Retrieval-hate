"""Stage A (README section 4): the hidden Markov tree answer model at test time on the revision-6 test-selected
checkpoints. Backbone, chain and node prior unchanged; the answer model refitted with hmt.fit_hmt on the same training
answers (label-only, before any test data is seen); policy hpolicy.run_batch_hmt on the test split, up to 8 calls.

Per trial: the full tree and each single level L (4 / 8 / 16: only the nodes of one depth, qtree.level_nodes), both
under the HMT. E3 comparison: per video k_v = min(8, the level's nodes), full tree and level read after k_v calls;
and fixed 8 calls against revision 6 (the method's replay runs_L0.pkl of experiments/20261108_astra_review). AP / ROC /
within by the unified evaluator. Output runs/20261109_tree_within/stage_a/<corpus>/ (hmt_fit.json, per trial pickles
runs_hmt_L<level>.pkl, scores and metrics, summary.json).
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, hmt, hpolicy, qtree                      # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
R6 = ROOT / "runs/20261108_astra_review"


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def read_metrics(path):
    m = json.load(open(path))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def replay(model, store, vids, ham, chain, answers, cfg, level):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        allowed = None if not level else [set(qtree.level_nodes(store.T[v], level).tolist()) for v in chunk]
        with torch.no_grad():
            res = hpolicy.run_batch_hmt(model, store, chunk, ham, chain, answers, 8, "cpu", allowed=allowed)
        runs.update(dict(zip(chunk, res)))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/stage_a")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    root = ROOT / a.out_root / a.corpus
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.pid").write_text(str(os.getpid()))
    print("host", socket.gethostname(), datetime.now().astimezone().isoformat(), flush=True)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    fit_path = root / "hmt_fit.json"
    if fit_path.exists():
        fit = json.load(open(fit_path))
        ham = hmt.HMTAnswerModel.from_dict(fit["model"])
    else:
        ta = [(labels[v], max(b for (_, b) in answers[v]), answers[v]) for v in ids["train"]
              if answers[v]]                        # videos shorter than 4 s have no question
        loglen = np.log([b - a_ for v in ids["train"] for (a_, b) in answers[v]])
        t0 = datetime.now()
        ham, info = hmt.fit_hmt(ta, float(loglen.mean()), float(loglen.std()))
        info["seconds"] = (datetime.now() - t0).total_seconds()
        fit_path.write_text(json.dumps({"model": ham.to_dict(), "info": info,
                                        "created": datetime.now().astimezone().isoformat(),
                                        "host": socket.gethostname()}, indent=1) + "\n")
        print("fit", json.dumps({k: info[k] for k in ("rho", "sigma_v", "sigma_h", "omega", "n_negative_videos",
                                                       "n_roots", "seconds")}), flush=True)
    store = qdata.Store(a.corpus, test, ["bert"])
    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, _am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        assert cfg["answer_source"] == "soft_both" and int(cfg["soft_levels"]) == 8
        assert cfg["answer_model"] == "anchored_shared" and int(cfg["n_state"]) == 2
        ham.cats = list(cfg["categories"])
        name = f'seed{summary["seed"]}_{trial.name}'
        od = root / name
        od.mkdir(parents=True, exist_ok=True)
        runs = {}
        for L in [0] + a.levels:
            path = od / f"runs_hmt_L{L}.pkl"
            if path.exists():
                runs[L] = pickle.load(path.open("rb"))["runs"]
                continue
            t0 = datetime.now()
            runs[L] = replay(model, store, test, ham, chain, answers, cfg, L)
            with path.open("wb") as fh:
                pickle.dump({"runs": runs[L], "trial": str(trial), "level": L, "host": socket.gethostname(),
                             "created": datetime.now().astimezone().isoformat()}, fh)
            print(datetime.now().astimezone().isoformat(), a.corpus, name, "L%d" % L,
                  "%.0f s" % (datetime.now() - t0).total_seconds(), flush=True)
        full = runs[0]
        r6_full = pickle.load(open(R6 / "replay" / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        row = {}
        for L in a.levels:
            k = {v: min(8, len(runs[L][v]["eig"])) for v in test}
            mf = score(a.corpus, od, f"hmt_full_matchL{L}",
                       {v: full[v]["scores"][min(k[v], len(full[v]["scores"]) - 1)] for v in test})
            ml = score(a.corpus, od, f"hmt_level{L}", {v: runs[L][v]["scores"][k[v]] for v in test})
            r6f = read_metrics(R6 / "e3" / a.corpus / name / f"metrics_test_full_matchL{L}.json")
            r6l = read_metrics(R6 / "e3" / a.corpus / name / f"metrics_test_level{L}.json")
            row[str(L)] = {"hmt_full": mf, "hmt_level": ml, "r6_full": r6f, "r6_level": r6l,
                           "hmt_full_minus_hmt_level": {m: mf[m] - ml[m] for m in METS},
                           "hmt_full_minus_r6_full": {m: mf[m] - r6f[m] for m in METS},
                           "mean_calls": float(np.mean(list(k.values())))}
        m8 = score(a.corpus, od, "hmt_fixed8", {v: full[v]["scores"][-1] for v in test})
        r8 = score(a.corpus, od, "r6_fixed8", {v: r6_full[v]["scores"][min(8, len(r6_full[v]["scores"]) - 1)]
                                               for v in test})
        row["fixed8"] = {"hmt": m8, "r6": r8, "hmt_minus_r6": {m: m8[m] - r8[m] for m in METS}}
        nested = []
        for v in test:
            tr = qtree.tree(store.T[v])
            asked = full[v]["asked"]
            nested.append(sum(any((tr["a"][m] <= tr["a"][n] and tr["b"][n] <= tr["b"][m]) or
                                  (tr["a"][n] <= tr["a"][m] and tr["b"][m] <= tr["b"][n]) for m in asked[:i])
                              for i, n in enumerate(asked)) / max(len(asked), 1))
        row["nested_share"] = float(np.mean(nested))
        rows[name] = row
        print(a.corpus, name, json.dumps({L: [round(x, 4) for x in row[L]["hmt_full_minus_hmt_level"].values()]
                                          for L in map(str, a.levels)}),
              "fixed8 hmt-r6", [round(x, 4) for x in row["fixed8"]["hmt_minus_r6"].values()],
              "nested share %.2f" % row["nested_share"], flush=True)
    mean = {}
    for L in map(str, a.levels):
        mean[L] = {kk: {m: float(np.mean([rows[t][L][kk][m] for t in rows])) for m in METS}
                   for kk in ("hmt_full", "hmt_level", "r6_full", "r6_level", "hmt_full_minus_hmt_level",
                              "hmt_full_minus_r6_full")}
        mean[L]["mean_calls"] = float(np.mean([rows[t][L]["mean_calls"] for t in rows]))
    mean["fixed8"] = {kk: {m: float(np.mean([rows[t]["fixed8"][kk][m] for t in rows])) for m in METS}
                      for kk in ("hmt", "r6", "hmt_minus_r6")}
    mean["nested_share"] = float(np.mean([rows[t]["nested_share"] for t in rows]))
    (root / "summary.json").write_text(json.dumps({"corpus": a.corpus, "host": socket.gethostname(),
                                                   "created": datetime.now().astimezone().isoformat(),
                                                   "trials": rows, "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps({L: mean[L]["hmt_full_minus_hmt_level"] for L in map(str, a.levels)}),
          json.dumps(mean["fixed8"]["hmt_minus_r6"]), flush=True)


if __name__ == "__main__":
    main()
