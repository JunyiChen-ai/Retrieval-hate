"""Revision-5 premise check (README section 16.4; development evidence on test under rule 10; nothing here trains or
selects): is the tree's weakness against single-level windows (concern C6) and the drop after 8 calls (C3) a matter
of WHERE the questions go?

Each revision-4 trial (network, chain, node potentials, anchored answer model unchanged) is re-run on test with the
questions restricted at test time only:
  tree        every queryable node (the method; must reproduce the trial's summary)
  level<L>    only the nodes of one tree depth (equal windows of L to 2L seconds, qtree.level_nodes), EIG order
  disjoint    every queryable node that neither contains nor lies inside an already asked node (EIG order), i.e.
              no nested questions
Pooled AP / ROC / within at fixed budgets through the shared evaluator, Spearman of each positive video's mean
score with its GT hate fraction, and the median length of the first 8 asked nodes.

    python experiments/20260925_query_paradigm/asking_check.py --corpus hatemm --trials <trial dirs>
Writes runs/20260928_query_paradigm_r5_analysis/asking/<corpus>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import concern_diagnostics as cd                # noqa: E402  (sets up the import paths)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402
import cpolicy                                  # noqa: E402

BUDGETS = (0, 1, 2, 4, 8, 16, 32)
OUT = os.path.join(ROOT, "runs", "20260928_query_paradigm_r5_analysis", "asking")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trials", nargs="+", required=True)
    ap.add_argument("--variants", default="tree,disjoint,level4,level8,level16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    answers, _ = qdata.load_answers(a.corpus, "words")
    vids = list(ids["test"])
    frac = {v: float(np.mean(gt["test"][v])) for v in vids}
    pos = [v for v in vids if labels[v] == 1 and frac[v] > 0]
    os.makedirs(OUT, exist_ok=True)
    res = {"corpus": a.corpus, "trials": {}}
    store = None
    for trial in a.trials:
        summ, cfg, model, am, chain = cd.load_trial(trial, a.device, answers, ids)
        if store is None:
            store = qdata.Store(a.corpus, vids, cfg.get("text_sources", ["bert"]))
        tag = "_".join(trial.rstrip("/").split("/")[-2:])
        rt = {"method": {str(B): {m: summ["fixed"][str(B)]["test"][m] for m in ("pooled_ap", "pooled_roc",
                                                                                 "within_roc")} for B in BUDGETS}}
        for variant in a.variants.split(","):
            order = sorted(vids, key=lambda v: store.T[v])
            k = int(cfg["eval_chunk"])
            runs = {}
            for i in range(0, len(order), k):
                chunk = order[i:i + k]
                allowed = None
                if variant.startswith("level"):
                    L = int(variant[5:])
                    allowed = [set(qtree.level_nodes(store.T[v], L).tolist()) & set(np.where(
                        qtree.tree(store.T[v])["queryable"])[0].tolist()) for v in chunk]
                runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"],
                                              max(BUDGETS), a.device, allowed=allowed,
                                              no_nested=variant == "disjoint"))
            od = os.path.join(OUT, a.corpus, variant, tag)
            os.makedirs(od, exist_ok=True)
            r = {}
            for B in BUDGETS:
                sc = {v: runs[v]["scores"][min(B, len(runs[v]["eig"]))] for v in vids}
                sp = os.path.join(od, "scores_test_fixed%d.jsonl" % B)
                hc.write_scores(sp, sc)
                m = hc.run_evaluator(a.corpus, "test", sp, os.path.join(od, "metrics_test_fixed%d.json" % B))
                m = m["results"]["score_av"]
                r[str(B)] = {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                             "within_roc": m["per_video"]["macro_auc"],
                             "mean_calls": float(np.mean([min(B, len(runs[v]["eig"])) for v in vids])),
                             "fraction_spearman": float(spearmanr([frac[v] for v in pos],
                                                                  [float(sc[v].mean()) for v in pos]).correlation)}
            lens = [int(qtree.tree(store.T[v])["b"][n] - qtree.tree(store.T[v])["a"][n])
                    for v in vids for n in runs[v]["asked"][:8]]
            r["median_asked_len_first8"] = float(np.median(lens))
            rt[variant] = r
            print("== %s %s: median asked length %.0f s" % (tag, variant, r["median_asked_len_first8"]), flush=True)
            for B in map(str, BUDGETS):
                x = r[B]
                print("  B=%-2s %.4f / %.4f / %.4f | calls %.2f | fraction Spearman %.3f" % (
                    B, x["pooled_ap"], x["pooled_roc"], x["within_roc"], x["mean_calls"], x["fraction_spearman"]),
                    flush=True)
            res["trials"][trial] = rt
            json.dump(res, open(os.path.join(OUT, "%s.json" % a.corpus), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
