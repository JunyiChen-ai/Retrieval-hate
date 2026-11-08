"""Diagnostic (README section 1): does the full tree lose within to one level because of WHICH nodes its EIG policy
asks? Same revision-6 test-selected checkpoints, same cached answers, same answer table, test split, nothing retrained.

Variants of cpolicy.run_batch (the method = order "eig", every node askable):
  nonest   EIG order, but a node nested with (containing or inside) an already asked node can no longer be asked;
  eer      expected per-second squared-error reduction (cpolicy._expected_error_reduction) instead of EIG;
  noadj    EIG order, but after a node is asked its parent and its two children can no longer be asked;
  noadj2   the same for every node within tree distance 2 (also grandparent, grandchildren, sibling);
  eigloc   information about the seconds given that the video is hateful, I(o_n; y | G = 1), instead of EIG
           I(o_n; G, y) (which adds the information about G itself);
  distw    EIG and the posterior with each answer weighted by w(d) = 1 - r(d)^2, d = tree distance to the nearest
           already asked node, r(d) = correlation of the answer levels of two nodes at distance d in the negative
           training videos (d >= 5 pooled); the first answer has weight 1;
  halfprior  EIG, with the prior's per-second logits s_t and node potentials phi_n halved (video logit g unchanged):
           tests whether the over-confident prior shape keeps the policy from covering the video. Not a method.
  flatprior  the same with s_t and phi_n set to 0 (no location prior at all; video logit g kept). Not a method.
  flatpol  questions chosen with EIG under the flat location prior (as flatprior), scores from the full prior: tests
           whether following the prior's idea of where the hate is keeps the policy from covering the video.
  spread   expected gain in ranking accuracy (cpolicy._expected_spread_gain: between videos on P(G = 1) and between
           seconds within the video), full prior; spread_flatpol the same under the flat location prior.
  val3     EIG and the posterior under the three-state answer table fitted on the validation GT
           (answer_calib.fit_validation_answer_model: negative video / clean node of a hateful video / hateful node,
           each with its own length slope) instead of the anchored two-state table, whose "no hate" row comes from
           negative videos only and so over-trusts a Yes inside hateful videos (README section 11).
The single-level comparison of halfprior / flatprior / flatpol / spread / spread_flatpol / val3 re-runs the single
levels the same way (runs_<variant>_L<L>).
Per trial and level L (4 / 8 / 16): each variant is read after the E3 matched calls k_v = min(8, nodes at depth L)
(from experiments/20261108_astra_review replays runs_L<L>.pkl) and compared with the method (runs_L0.pkl) and the
single level at the same k_v; also at fixed 8 calls. AP / ROC / within by the unified evaluator.
Output runs/20261109_tree_within/policy_variants/<corpus>/<trial>/ and summary.json per corpus.
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
from qtl.answer_calib import fit_validation_answer_model                # noqa: E402
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


def corr_by_distance(answers, labels, train_ids, dmax=5, seed=0):
    """r(d), d = 1 .. dmax (dmax pools >= dmax): Pearson correlation of the answer levels of node pairs at tree
    distance d in the same negative training video (up to 60 random pairs per video)."""
    rng = np.random.default_rng(seed)
    acc = {d: [] for d in range(1, dmax + 1)}
    for v in train_ids:
        if labels[v] != 0 or not answers[v]:
            continue
        T = max(b for (_, b) in answers[v])
        tr = cpolicy.qtree.tree(T)
        q = [int(n) for n in np.where(tr["queryable"])[0]
             if answers[v].get((int(tr["a"][n]), int(tr["b"][n]))) is not None]
        if len(q) < 2:
            continue
        anc = cpolicy._ancestors(tr)
        lev = {n: int(answers[v][(int(tr["a"][n]), int(tr["b"][n]))][0]) for n in q}
        for _ in range(min(60, len(q) * (len(q) - 1) // 2)):
            i, j = rng.choice(len(q), 2, replace=False)
            d = int(cpolicy._min_distance(tr, anc, np.array([q[i]]), [q[j]])[0])
            acc[min(d, dmax)].append((lev[q[i]], lev[q[j]]))
    return {d: float(np.corrcoef(np.array(x).T)[0, 1]) for d, x in acc.items()}


def replay(variant, model, store, vids, am, chain, answers, cfg, dist_weight=None, level=0):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], 8, "cpu",
                                          order={"eer": "eer", "eigloc": "eig_loc", "spread": "spread",
                                                 "spread_flatpol": "spread"}.get(variant, "eig"),
                                          no_nested=(variant == "nonest"), no_adjacent={"noadj": 1, "noadj2": 2}.get(variant, 0),
                                          dist_weight=dist_weight if variant == "distw" else None,
                                          prior_scale={"halfprior": 0.5, "flatprior": 0.0}.get(variant, 1.0),
                                          policy_prior_scale=0.0 if variant in ("flatpol", "spread_flatpol") else None,
                                          allowed=None if not level else
                                          [set(cpolicy.qtree.level_nodes(store.T[v], level).tolist()) for v in chunk]))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--variants", nargs="+", default=["nonest", "eer"])
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/policy_variants")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    qdata.configure_source("soft_both", 8)
    answers, T_all = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, test, ["bert"])
    dist_weight = None
    if "distw" in a.variants:
        r = corr_by_distance(answers, labels, ids["train"])
        dist_weight = np.array([1.0] + [1.0 - max(r[d], 0.0) ** 2 for d in sorted(r)])
        print("answer correlation by tree distance", json.dumps(r), "weights", np.round(dist_weight, 3).tolist(),
              flush=True)
    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        assert cfg["answer_source"] == "soft_both" and int(cfg["soft_levels"]) == 8
        if "val3" in a.variants:
            am_val3, counts = fit_validation_answer_model(answers, labels, gt["val"], ids["val"], T_all, am.len_mu,
                                                          am.len_sd, cfg["categories"])
            print("val3 state counts", counts, flush=True)
        name = f'seed{summary["seed"]}_{trial.name}'
        od = (ROOT / a.out_root / a.corpus / name).resolve()
        od.mkdir(parents=True, exist_ok=True)
        (od / "run.pid").write_text(str(os.getpid()))
        full = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        lev = {L: pickle.load(open(REPLAY / a.corpus / name / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
               for L in a.levels}
        row = {}
        for var in a.variants:
            am_var = am_val3 if var == "val3" else am
            path = od / f"runs_{var}.pkl"
            if path.exists():
                runs = pickle.load(path.open("rb"))["runs"]
            else:
                runs = replay(var, model, store, test, am_var, chain, answers, cfg, dist_weight)
                with path.open("wb") as fh:
                    pickle.dump({"runs": runs, "trial": str(trial), "variant": var, "host": socket.gethostname(),
                                 "created": datetime.now().astimezone().isoformat()}, fh)
            res = {}
            for L in a.levels:
                k = {v: min(8, len(lev[L][v]["eig"])) for v in test}
                pick = lambda r, v: r[v]["scores"][min(k[v], len(r[v]["scores"]) - 1)]
                mv = score(a.corpus, od, f"{var}_matchL{L}", {v: pick(runs, v) for v in test})
                mm = json.load(open(ROOT / "runs/20261108_astra_review/e3" / a.corpus / name /
                                    f"metrics_test_full_matchL{L}.json"))["results"]["score_av"]
                ml = json.load(open(ROOT / "runs/20261108_astra_review/e3" / a.corpus / name /
                                    f"metrics_test_level{L}.json"))["results"]["score_av"]
                conv = lambda m: {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"],
                                  "within_roc": m["per_video"]["macro_auc"]}
                mm, ml = conv(mm), conv(ml)
                if var in ("halfprior", "flatprior", "flatpol", "spread", "spread_flatpol", "val3"):   # single level, same prior / policy / table
                    lpath = od / f"runs_{var}_L{L}.pkl"
                    if lpath.exists():
                        lruns = pickle.load(lpath.open("rb"))["runs"]
                    else:
                        lruns = replay(var, model, store, test, am_var, chain, answers, cfg, dist_weight, level=L)
                        with lpath.open("wb") as fh:
                            pickle.dump({"runs": lruns, "trial": str(trial), "variant": var, "level": L,
                                         "host": socket.gethostname()}, fh)
                    ml = score(a.corpus, od, f"{var}_level{L}", {v: lruns[v]["scores"][k[v]] for v in test})
                res[str(L)] = {"variant": mv, "method": mm, "single_level": ml,
                               "variant_minus_level": {m: mv[m] - ml[m] for m in METS},
                               "variant_minus_method": {m: mv[m] - mm[m] for m in METS}}
            m8v = score(a.corpus, od, f"{var}_fixed8", {v: runs[v]["scores"][-1] for v in test})
            m8m = score(a.corpus, od, "method_fixed8", {v: full[v]["scores"][min(8, len(full[v]["scores"]) - 1)]
                                                       for v in test})
            res["fixed8"] = {"variant": m8v, "method": m8m, "variant_minus_method": {m: m8v[m] - m8m[m] for m in METS}}
            nest = []
            for v in test:
                tr = cpolicy.qtree.tree(store.T[v])
                asked = runs[v]["asked"]
                nest.append(np.mean([float(np.mean(tr["b"][asked] - tr["a"][asked]))]) if len(asked) else np.nan)
            res["mean_asked_length"] = float(np.nanmean(nest))
            row[var] = res
            print(a.corpus, name, var, json.dumps({L: {kk: [round(x, 4) for x in res[L][kk].values()]
                                                       for kk in ("variant_minus_level", "variant_minus_method")}
                                                   for L in map(str, a.levels)}),
                  "fixed8", json.dumps(res["fixed8"]["variant_minus_method"]), flush=True)
        rows[name] = row
    mean = {}
    for var in a.variants:
        mean[var] = {}
        for L in [str(x) for x in a.levels] + ["fixed8"]:
            keys = ["variant_minus_level", "variant_minus_method"] if L != "fixed8" else ["variant_minus_method"]
            mean[var][L] = {kk: {m: float(np.mean([rows[t][var][L][kk][m] for t in rows])) for m in METS}
                            for kk in keys + ["variant"]}
    out = {"corpus": a.corpus, "created": datetime.now().astimezone().isoformat(), "host": socket.gethostname(),
           "trials": rows, "mean": mean}
    od = ROOT / a.out_root / a.corpus
    (od / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps(mean), flush=True)


if __name__ == "__main__":
    main()
