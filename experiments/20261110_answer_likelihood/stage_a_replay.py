"""Stage A (developmental, README section 3): candidate answer likelihoods swapped in at test time on the revision-6
test-selected checkpoints. Backbone, chain and training are unchanged; each candidate is fitted on the same training
answers with video labels only (as the anchored model), the cached soft answers are re-binned to its levels.

Candidates (name: levels, grid, fit):
  q8 / q16 / q32          train quantile levels, free table per state (q8 = revision 6);
  u16_free / u32_free     uniform grid on logit P(Yes), free table per state;
  u16_tilt1 / u32_tilt1   uniform grid, hateful state = no-hate table tilted by beta x (answer_density.fit_tilted);
  u32_cal                 the same with beta fixed to 1 per logit (the VLM's probability taken as calibrated);
  u32_tilt2               tilt beta_1 x + beta_2 x^2;
  u32_gauss / u32_smooth4 both states smooth, degree 2 / 4 (answer_density.fit_smooth).
Per candidate and trial, test split, AP / ROC / within:
  flat_root, flat_u8   sigmoid(logit p0 + llr): the 0-call score plus the log-likelihood ratio of the root's answer /
                       of the answer of the second's window among the 8 equal windows (no tree);
  root                 tree inference after the root's answer only (1 call);
  u8                   tree inference after the 8 equal windows (qtree.depth_nodes(T, 3));
  eig8                 the method's policy: EIG on the full tree, fixed 8 calls, questions chosen under the candidate.
Scores at 6 decimals as the evaluator reads them; unified core evaluator (hc.frame_metrics).
Output runs/20261110_answer_likelihood/stage_a/<corpus>/summary.json (per trial runs_<candidate>_<policy>.pkl).
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
from qtl import answer_density, cpolicy, data as qdata, policy, qtree   # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

METS = ("pooled_ap", "pooled_roc", "within_roc")
CANDS = {
    "q8": dict(levels=8, grid="quantile", fit="free"),
    "q16": dict(levels=16, grid="quantile", fit="free"),
    "q32": dict(levels=32, grid="quantile", fit="free"),
    "u16_free": dict(levels=16, grid="logit", fit="free"),
    "u32_free": dict(levels=32, grid="logit", fit="free"),
    "u16_tilt1": dict(levels=16, grid="logit", fit="tilt", degree=1),
    "u32_tilt1": dict(levels=32, grid="logit", fit="tilt", degree=1),
    "u32_cal": dict(levels=32, grid="logit", fit="tilt", degree=1, fixed=[answer_density.X_SCALE]),
    "u32_tilt2": dict(levels=32, grid="logit", fit="tilt", degree=2),
    "u32_gauss": dict(levels=32, grid="logit", fit="smooth", degree=2),
    "u32_smooth4": dict(levels=32, grid="logit", fit="smooth", degree=4),
}
POLICIES = {"root": (1, lambda T: {0}), "u8": (8, lambda T: set(qtree.depth_nodes(T, 3).tolist())), "eig8": (8, None)}


def build(corpus, spec, labels, train_ids):
    L = int(spec["levels"])
    centers = None
    if spec["grid"] == "quantile":
        qdata.configure_source("soft_both", L)
    else:
        edges, centers = answer_density.logit_grid(L)
        qdata.configure_source("soft_both", L, soft_fixed=edges.tolist())
    answers, T = qdata.load_answers(corpus, "soft_both")
    ta = [(labels[v], [(a, b, T[v], o) for (a, b), o in answers[v].items()]) for v in train_ids]
    loglen = np.log([b - a for v in train_ids for (a, b) in answers[v]])
    mu, sd = float(loglen.mean()), float(loglen.std())
    if spec["fit"] == "free":
        theta, omega, n = qtree.fit_anchored_shared(ta, mu, sd)
        info = {"n": n}
    elif spec["fit"] == "tilt":
        theta, omega, info = answer_density.fit_tilted(ta, mu, sd, centers, spec["degree"], spec.get("fixed"))
    else:
        theta, omega, info = answer_density.fit_smooth(ta, mu, sd, centers, spec["degree"])
    am = qtree.AnswerModel(theta, mu, sd, True, 2, [0], omega)
    am.requires_grad_(False)
    am.eval()
    info = dict(info, levels=L, edges=np.asarray(qdata.SOFT_EDGES[(corpus, "soft_both")]).round(6).tolist())
    with torch.no_grad():                        # log-likelihood ratio per level at the median node length
        lp = am.level_logp(torch.as_tensor([float(np.exp(np.median(loglen)))])).double().numpy()[0, 0]
    info["llr_median_length"] = (lp[1] - lp[0]).round(3).tolist()
    return answers, am, info


def replay(model, store, vids, am, chain, answers, cfg, max_calls, allowed):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], max_calls, "cpu",
                                          allowed=None if allowed is None else [allowed(store.T[v]) for v in chunk]))
    return runs


def flat(store, test, am, answers, cats, p0, nodes_of):
    """sigmoid(logit p0 + llr of the shortest given node covering the second); no answer: llr 0."""
    out = {}
    for v in test:
        T = store.T[v]
        vt = cpolicy._vt(T)
        asker = policy.Asker(vt, am, cats)
        llr, ln = np.zeros(T), np.full(T, np.inf)
        for n in nodes_of(T):
            a, b = int(vt.tr["a"][n]), int(vt.tr["b"][n])
            o = answers[v].get((a, b))
            if o is None or int(n) not in asker.qpos:
                continue
            ll = asker.loglik(n, o)
            sel = (b - a) < ln[a:b]
            llr[a:b][sel] = float(ll[2] - ll[0])
            ln[a:b][sel] = b - a
        out[v] = 1.0 / (1.0 + np.exp(-(hc.logit(p0[v]) + llr)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--cands", nargs="+", default=list(CANDS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261110_answer_likelihood/stage_a")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    od0 = (ROOT / a.out_root / a.corpus).resolve()
    od0.mkdir(parents=True, exist_ok=True)
    (od0 / "run.pid").write_text(str(os.getpid()))
    print("host", socket.gethostname(), datetime.now().astimezone().isoformat(), flush=True)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    built = {}
    for c in a.cands:
        built[c] = build(a.corpus, CANDS[c], labels, ids["train"])
        print(a.corpus, "candidate", c, json.dumps({k: v for k, v in built[c][2].items() if k != "edges"}), flush=True)
    store = qdata.Store(a.corpus, test, ["bert"])

    def mets(sc):
        sc = {v: np.round(np.asarray(s, dtype=np.float64), 6) for v, s in sc.items()}
        m = hc.frame_metrics(sc, gt["test"], hate)
        return {k: float(m[k]) for k in METS}

    prior_cache = {}
    plain_prior = policy.video_prior_nodes

    def cached_prior(model, store_, v, device):   # the backbone's outputs do not depend on the answer model
        key = (id(model), v)
        if key not in prior_cache:
            prior_cache[key] = plain_prior(model, store_, v, device)
        return prior_cache[key]

    policy.video_prior_nodes = cached_prior
    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        prior_cache.clear()
        summary, cfg, model, am6, chain = load_trial(trial, "cpu", built[a.cands[0]][0], ids, model_factory=make_model)
        assert cfg["answer_source"] == "soft_both" and int(cfg["soft_levels"]) == 8 and cfg["categories"] == [0]
        name = f'seed{summary["seed"]}_{trial.name}'
        od = od0 / name
        od.mkdir(parents=True, exist_ok=True)
        row = {}
        for c in a.cands:
            answers, am, _info = built[c]
            if c == "q8":                        # the refit must be the checkpoint's table
                with torch.no_grad():
                    ln = torch.as_tensor([4.0, 16.0, 64.0])
                    diff = float((am.level_logp(ln) - am6.level_logp(ln)).abs().max())
                print(a.corpus, name, "q8 refit vs checkpoint table, max |log p| difference %.2e" % diff, flush=True)
            res, p0 = {}, None
            for pol in a.policies:
                max_calls, allowed = POLICIES[pol]
                path = od / f"runs_{c}_{pol}.pkl"
                if path.exists():
                    runs = pickle.load(path.open("rb"))["runs"]
                else:
                    runs = replay(model, store, test, am, chain, answers, cfg, max_calls, allowed)
                    with path.open("wb") as fh:
                        pickle.dump({"runs": {v: {"scores": [r["scores"][0], r["scores"][-1]], "asked": r["asked"],
                                                  "p_G": [r["p_G"][0], r["p_G"][-1]]} for v, r in runs.items()},
                                     "trial": str(trial), "candidate": c, "policy": pol,
                                     "host": socket.gethostname(),
                                     "created": datetime.now().astimezone().isoformat()}, fh)
                res[pol] = mets({v: runs[v]["scores"][-1] for v in test})
                res[pol]["mean_calls"] = float(np.mean([len(runs[v]["asked"]) for v in test]))
                p0 = {v: runs[v]["scores"][0] for v in test}
            res["flat_root"] = mets(flat(store, test, am, answers, cfg["categories"], p0, lambda T: [0]))
            res["flat_u8"] = mets(flat(store, test, am, answers, cfg["categories"], p0,
                                       lambda T: qtree.depth_nodes(T, 3)))
            row[c] = res
            print(a.corpus, name, c, json.dumps({k: [round(r[m], 4) for m in METS] for k, r in res.items()}),
                  flush=True)
        rows[name] = row
    keys = list(next(iter(next(iter(rows.values())).values())))
    mean = {c: {k: {m: float(np.mean([rows[t][c][k][m] for t in rows])) for m in METS} for k in keys}
            for c in a.cands}
    (od0 / "summary.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "candidates": {c: built[c][2] for c in a.cands}, "trials": rows, "mean": mean}, indent=1) + "\n")
    for c in a.cands:
        print("MEAN", a.corpus, "%-12s" % c, " | ".join(
            "%s %s" % (k, "/".join("%.3f" % mean[c][k][m] for m in METS)) for k in keys), flush=True)
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
