"""Revision 5 step 3, offline check (README section 17.3; development evidence on test under rule 10; nothing here
trains or selects): the finished soft-answer trials re-run at test time with the THREE-STATE answer model of
train.py "anchored_ext" (states 0 / 2 = the trial's anchored tables, state 1 = an external table for the nodes of
positive videos without hate), the network, chain and node potentials unchanged. Tables:
  mv    multiview_check.py estimate without per-second labels (README 17.3 variant, backbone third rater)
  val   the fallback: the validation per-second labels (multiview_check.py <corpus>_val_state1_table.json)
Reported per table and fixed budget: test AP / ROC / within through the shared evaluator, against the trial's
two-state model ("tree", must reproduce the summary).

    python experiments/20260925_query_paradigm/ext_table_check.py --corpus hatemm --trials <trial dirs> \\
        --tables mv=runs/.../multiview/hatemm_backbone_smax_state1_table.json val=runs/.../hatemm_val_state1_table.json
Writes runs/20260929_query_paradigm_r5/ext_table/<corpus><suffix>.json.
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
import train as TR                              # noqa: E402  (import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import cpolicy                                  # noqa: E402
import concern_diagnostics as cd                # noqa: E402
import copy_check as cc                         # noqa: E402

OUT = os.path.join(ROOT, "runs", "20260929_query_paradigm_r5", "ext_table")
BUDGETS = [0, 2, 4, 8, 16, 32]


def ext_answer_model(am, table, cats):
    """A three-state AnswerModel: states 0 / 2 from the two-state anchored model `am`, state 1 from `table`."""
    th2 = am.theta.detach().cpu().numpy()                      # N_CAT, 2, N_LEV
    assert int(table["levels"]) == th2.shape[2] and len(cats) == 1
    theta = np.zeros((th2.shape[0], 3, th2.shape[2]))
    theta[:, 0], theta[:, 2] = th2[:, 0], th2[:, 1]
    theta[:, 1] = np.log(np.clip(np.asarray(table["p_state1"], float), 1e-4, None))[None]
    m = qtree.AnswerModel(theta, am.len_mu, am.len_sd, False, 3, cats).to(am.theta.device)
    m.requires_grad_(False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--tables", nargs="+", required=True, help="name=path of state-1 tables")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--copy-pi", default=None, help="neg or a constant: also apply the copy likelihood (17.2)")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    cfg0 = json.load(open(os.path.join(a.trials[0], "config.json")))
    qdata.configure_source(cfg0["answer_source"], int(cfg0.get("soft_levels", qdata.SOFT_LEVELS)))
    answers, T = qdata.load_answers(a.corpus, cfg0["answer_source"])
    vids = [v for v in ids["test"] if v in answers]
    tables = {}
    for spec in a.tables:
        name, path = spec.split("=", 1)
        tables[name] = json.load(open(path))
    os.makedirs(OUT, exist_ok=True)
    res = {"corpus": a.corpus, "tables": {k: {"path": [s for s in a.tables if s.startswith(k + "=")][0],
                                              "source": t.get("source"), "p_state1": t["p_state1"]} for k, t in tables.items()},
           "copy_pi": a.copy_pi, "trials": {}}
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
        cats = list(cfg["categories"])
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        fn = None
        if a.copy_pi == "neg":
            pi_neg = cc.estimate_pi_neg(answers, labels, ids["train"], T, am, cats)[0]
            fn = (lambda L, _pi=pi_neg: float(_pi[cc.bucket_of(L)]))
        elif a.copy_pi is not None:
            fn = (lambda L, _x=float(a.copy_pi): _x)
        rt = {"method": {str(B): {m: summ["fixed"][str(B)]["test"][m] for m in ("pooled_ap", "pooled_roc", "within_roc")}
                         for B in BUDGETS}}
        for name in ["tree"] + list(tables):
            am_use = am if name == "tree" else ext_answer_model(am, tables[name], cats)
            order = sorted(vids, key=lambda v: store.T[v])
            k = int(cfg["eval_chunk"])
            runs = {}
            for i in range(0, len(order), k):
                runs.update(cpolicy.run_batch(model, store, order[i:i + k], am_use, chain, answers, cats, max(BUDGETS),
                                              a.device, copy_pi=fn))
            od = os.path.join(OUT, a.corpus, name + a.out_suffix, tag)
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
                print("== %s %-5s B=%-2d %.4f / %.4f / %.4f | calls %.2f" % (
                    tag, name, B, rr[str(B)]["pooled_ap"], rr[str(B)]["pooled_roc"], rr[str(B)]["within_roc"],
                    rr[str(B)]["mean_calls"]), flush=True)
            rt[name] = rr
        res["trials"][trial] = rt
        json.dump(res, open(os.path.join(OUT, "%s%s.json" % (a.corpus, a.out_suffix)), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
