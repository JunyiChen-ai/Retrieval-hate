"""Diagnostic (README section 6): a budget-aware tree policy. E3 reads one anytime tree run after k_v calls, while
the single level at k_v calls asks its complete partition; no single question order can contain every level's
partition as a prefix. Here the tree is told the budget B_v = k_v and its first min(B_v, ...) questions must form an
exact cover of the video by disjoint tree nodes of any sizes (cpolicy.run_batch cover=), chosen adaptively by the
order's objective; the single level's partition is one of the allowed covers.
Variants: cover (EIG), cover_root (the root first, then a cover with B_v - 1 nodes), cover_flatpol (EIG under the
flat location prior, scores from the full prior). Revision-6 test-selected checkpoints, cached answers, test split.
Per trial and level L (4 / 8 / 16): tree run with B_v = k_v(L) = min(8, nodes at depth L) against the single level
at the same k_v (experiments/20261108_astra_review e3 metrics) and the method at k_v; and B_v = 8 against the method
at fixed 8 calls. Also the share of videos whose asked set equals the level's partition.
Output runs/20261109_tree_within/budget_cover/<corpus>/<trial>/ and summary.json per corpus.
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
from qtl import cpolicy, data as qdata                                  # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
REPLAY = ROOT / "runs/20261108_astra_review/replay"
E3 = ROOT / "runs/20261108_astra_review/e3"
VARIANTS = {"cover": {"cover": "plain"}, "cover_root": {"cover": "root"},
            "cover_flatpol": {"cover": "plain", "policy_prior_scale": 0.0}}


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def conv(m):
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def replay(variant, model, store, vids, am, chain, answers, cfg, budget):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], 8, "cpu",
                                          budgets=[budget[v] for v in chunk], **VARIANTS[variant]))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/budget_cover")
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
        (od / f"run_{a.variant}.pid").write_text(str(os.getpid()))
        full = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        row = {}
        for L in a.levels + [0]:
            if L:
                lev = pickle.load(open(REPLAY / a.corpus / name / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
                k = {v: min(8, len(lev[v]["eig"])) for v in test}
            else:
                k = {v: 8 for v in test}
            path = od / f"runs_{a.variant}_{'B' + str(L) if L else 'fixed8'}.pkl"
            if path.exists():
                runs = pickle.load(path.open("rb"))["runs"]
            else:
                runs = replay(a.variant, model, store, test, am, chain, answers, cfg, k)
                with path.open("wb") as fh:
                    pickle.dump({"runs": runs, "trial": str(trial), "variant": a.variant, "level": L,
                                 "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat()}, fh)
            label = f"{a.variant}_B{'L' + str(L) if L else '8'}"
            mv = score(a.corpus, od, label, {v: runs[v]["scores"][-1] for v in test})
            lens = [float(np.mean(cpolicy.qtree.tree(store.T[v])["b"][runs[v]["asked"]] -
                                  cpolicy.qtree.tree(store.T[v])["a"][runs[v]["asked"]]))
                    for v in test if runs[v]["asked"]]
            if L:
                mm = conv(json.load(open(E3 / a.corpus / name / f"metrics_test_full_matchL{L}.json"))["results"]["score_av"])
                ml = conv(json.load(open(E3 / a.corpus / name / f"metrics_test_level{L}.json"))["results"]["score_av"])
                same = float(np.mean([sorted(runs[v]["asked"]) == sorted(lev[v]["asked"][:k[v]]) for v in test]))
                row[str(L)] = {"variant": mv, "method": mm, "single_level": ml, "same_as_level": same,
                               "mean_asked_length": float(np.mean(lens)), "mean_calls": float(np.mean(list(k.values()))),
                               "variant_minus_level": {m: mv[m] - ml[m] for m in METS},
                               "variant_minus_method": {m: mv[m] - mm[m] for m in METS}}
            else:
                mm = score(a.corpus, od, "method_fixed8", {v: full[v]["scores"][min(8, len(full[v]["scores"]) - 1)]
                                                           for v in test})
                row["fixed8"] = {"variant": mv, "method": mm, "mean_asked_length": float(np.mean(lens)),
                                 "variant_minus_method": {m: mv[m] - mm[m] for m in METS}}
        rows[name] = row
        print(a.corpus, name, a.variant, json.dumps({L: [round(x, 4) for x in row[L]["variant_minus_level"].values()]
                                                     for L in map(str, a.levels)}),
              "fixed8", json.dumps({m: round(x, 4) for m, x in row["fixed8"]["variant_minus_method"].items()}),
              "same_as_level", [round(row[str(L)]["same_as_level"], 2) for L in a.levels], flush=True)
    mean = {}
    for L in [str(x) for x in a.levels] + ["fixed8"]:
        keys = ["variant", "variant_minus_level", "variant_minus_method"] if L != "fixed8" else \
            ["variant", "variant_minus_method"]
        mean[L] = {kk: {m: float(np.mean([rows[t][L][kk][m] for t in rows])) for m in METS} for kk in keys}
        for kk in ("same_as_level", "mean_asked_length"):
            if kk in rows[next(iter(rows))][L]:
                mean[L][kk] = float(np.mean([rows[t][L][kk] for t in rows]))
    od = ROOT / a.out_root / a.corpus
    (od / f"summary_{a.variant}.json").write_text(json.dumps(
        {"corpus": a.corpus, "variant": a.variant, "created": datetime.now().astimezone().isoformat(),
         "host": socket.gethostname(), "trials": rows, "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, a.variant, json.dumps({L: {kk: [round(x, 4) for x in mean[L][kk].values()]
                                                       for kk in mean[L] if isinstance(mean[L][kk], dict)}
                                                   for L in mean}), flush=True)


if __name__ == "__main__":
    main()
