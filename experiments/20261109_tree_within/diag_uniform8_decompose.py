"""Diagnostic (README section 12): on the SAME 8 equal windows (tree depth 3, as diag_uniform8.py), where does our
inference lose pooled AP against painting the raw VLM answers on the seconds (section 7 item 3)? Revision-6
test-selected checkpoints, cached answers, test split, nothing retrained. Every variant uses the same 8 answers.

Answers only (no network):
  paint_raw      per second the soft answer P(Yes) of its window (= naive_questions.py alpha 0);
  paint_level    the answer's level 0..7 (the 8 training quantile levels our answer model reads);
  paint_llr      log P(o | hateful node, length) - log P(o | no hate, length) of the revision-6 answer table;
  tree_noprior   our exact tree inference with the answer table and the chain, per-second logits s_t and node
                 potentials phi_n set to 0 and video logit g = 0 (no backbone).
Backbone prior p0 (scores before any call) plus the answers:
  prior          p0 alone;
  prior_paint_a  sigmoid(alpha logit p0 + (1 - alpha) logit P(Yes)), the fusion of the baseline control, alpha grid;
  prior_llr      sigmoid(logit p0 + llr): the answer table without the tree (flat fusion);
  tree_flatloc   tree inference, s_t and phi_n set to 0, the backbone's video logit g kept;
  tree_full      tree inference with the full prior (= diag_uniform8.py).
Parts of tree_full, p_t = P(G = 1 | answers) r_t with r_t = P(y_t = 1 | G = 1, answers):
  r_only         r_t;
  pG_x_paint     P(G = 1 | answers) x P(Yes) of the window (our video decision, painted answers inside the video);
  paintmean_x_r  mean P(Yes) of the video's windows x r_t (painted video decision, our ranking inside the video).
Reference: <baseline>_paint_a0.50 (the section-7 control) and method_fixed8 (revision 6, EIG tree, 8 calls).
Per variant AP / ROC / within and the pooled AP / ROC of the video-mean score (every second gets its video's mean:
how much of the pooled metric is the ordering of whole videos). Unified core evaluator (hc.frame_metrics).
Output runs/20261109_tree_within/uniform8_decompose/<corpus>/ (summary.json, runs_<prior>.pkl per trial).
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
from qtl import cpolicy, data as qdata, policy, qtree                   # noqa: E402
from qtl.backbones import make_model                                    # noqa: E402
from qtl.checkpoint import load_trial                                   # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402
from diag_uniform8 import depth3                                        # noqa: E402
from naive_questions import evidence                                    # noqa: E402

REPLAY = ROOT / "runs/20261108_astra_review/replay"
ALPHAS = np.round(np.linspace(0.1, 0.9, 9), 2)
KEYS = ("pooled_ap", "pooled_roc", "within_roc", "video_mean_ap", "video_mean_roc")


class NoVideoPrior:
    """The backbone with its video logit g set to 0 (P(G = 1) = .5 before any answer)."""

    def __init__(self, model):
        self.model = model
        self.node_prior = getattr(model, "node_prior", False)

    def __call__(self, *args):
        out = list(self.model(*args))
        out[1] = torch.zeros_like(out[1])
        return tuple(out)

    def node_logits(self, *args):
        return self.model.node_logits(*args)


def replay(model, store, vids, am, chain, answers, cfg, prior_scale):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], 8, "cpu",
                                          prior_scale=prior_scale, allowed=[depth3(store.T[v]) for v in chunk]))
    return runs


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/uniform8_decompose")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, test, ["bert"])

    def mets(sc):
        # 6 decimals, as the evaluator reads the scores from the score files (hc.write_scores): saturated posteriors
        # tie there, which moves within by up to .01 in single trials
        sc = {v: np.round(np.asarray(s, dtype=np.float64), 6) for v, s in sc.items()}
        m = hc.frame_metrics(sc, gt["test"], hate)
        vm = hc.frame_metrics({v: np.full(len(s), round(float(np.mean(s)), 6)) for v, s in sc.items()}, gt["test"],
                              hate)
        return {"pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
                "within_roc": float(m["within_roc"]), "video_mean_ap": float(vm["pooled_ap"]),
                "video_mean_roc": float(vm["pooled_roc"])}

    rows = {}
    for td in trials:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, "cpu", answers, ids, model_factory=make_model)
        assert cfg["answer_source"] == "soft_both" and int(cfg["soft_levels"]) == 8
        name = f'seed{summary["seed"]}_{trial.name}'
        od = (ROOT / a.out_root / a.corpus / name).resolve()
        od.mkdir(parents=True, exist_ok=True)
        (od / "run.pid").write_text(str(os.getpid()))
        runs = {}
        for tag, mdl, scale in (("full", model, 1.0), ("flatloc", model, 0.0), ("noprior", NoVideoPrior(model), 0.0)):
            path = od / f"runs_{tag}.pkl"
            if path.exists():
                runs[tag] = pickle.load(path.open("rb"))["runs"]
            else:
                runs[tag] = replay(mdl, store, test, am, chain, answers, cfg, scale)
                with path.open("wb") as fh:
                    pickle.dump({"runs": runs[tag], "trial": str(trial), "prior": tag, "host": socket.gethostname(),
                                 "created": datetime.now().astimezone().isoformat()}, fh)
        full = pickle.load(open(REPLAY / a.corpus / name / "runs_L0.pkl", "rb"))["runs"]["test"]
        sc = {k: {} for k in ("paint_raw", "paint_level", "paint_llr", "tree_noprior", "prior", "prior_llr",
                              "tree_flatloc", "tree_full", "r_only", "pG_x_paint", "paintmean_x_r", "method_fixed8")}
        sc.update({"prior_paint_a%.2f" % al: {} for al in ALPHAS})
        base = {m: hc.baseline_scores(a.corpus, m, int(summary["seed"])) for m in hc.STRONGEST_BASELINE[a.corpus]}
        sc.update({f"{m}_paint_a0.50": {} for m in base})
        for v in test:
            T = store.T[v]
            vt = cpolicy._vt(T)
            tr, nodes = vt.tr, sorted(depth3(T))
            asker = policy.Asker(vt, am, cfg["categories"])
            lev, llr = {}, {}
            for n in nodes:
                key = (int(tr["a"][n]), int(tr["b"][n]))
                o = answers[v].get(key)
                if o is None or int(n) not in asker.qpos:
                    continue
                ll = asker.loglik(n, o)
                lev[key], llr[key] = float(o[0]), float(ll[2] - ll[0])
            e_raw, _ = evidence(tr, T, nodes, P[v])
            e_lev, _ = evidence(tr, T, nodes, lev)
            e_llr, _ = evidence(tr, T, nodes, llr)
            if not llr:                                  # no answered window: no evidence
                e_llr = np.zeros(T)
            r = runs["full"][v]
            p0, pf, pG = r["scores"][0], r["scores"][-1], float(r["p_G"][-1])
            sc["paint_raw"][v] = e_raw
            sc["paint_level"][v] = e_lev
            sc["paint_llr"][v] = sig(e_llr)
            sc["tree_noprior"][v] = runs["noprior"][v]["scores"][-1]
            sc["prior"][v] = p0
            sc["prior_llr"][v] = sig(hc.logit(p0) + e_llr)
            sc["tree_flatloc"][v] = runs["flatloc"][v]["scores"][-1]
            sc["tree_full"][v] = pf
            sc["r_only"][v] = pf / max(pG, 1e-12)
            sc["pG_x_paint"][v] = pG * e_raw
            sc["paintmean_x_r"][v] = float(np.mean(e_raw)) * pf / max(pG, 1e-12)
            sc["method_fixed8"][v] = full[v]["scores"][min(8, len(full[v]["scores"]) - 1)]
            for al in ALPHAS:
                sc["prior_paint_a%.2f" % al][v] = sig(al * hc.logit(p0) + (1 - al) * hc.logit(e_raw))
            for m, b in base.items():
                bb = b.get(v, np.full(T, 0.5))
                n_ = min(len(bb), T)
                bb = np.concatenate([bb[:n_], np.full(T - n_, bb[n_ - 1] if n_ else 0.5)])
                sc[f"{m}_paint_a0.50"][v] = sig(0.5 * hc.logit(bb) + 0.5 * hc.logit(e_raw))
        rows[name] = {k: mets(s) for k, s in sc.items()}
        rows[name]["mean_calls"] = float(np.mean([len(runs["full"][v]["asked"]) for v in test]))
        print(a.corpus, name, json.dumps({k: [round(x, 4) for x in rows[name][k].values()]
                                          for k in rows[name] if k != "mean_calls"}), flush=True)
    names = [k for k in next(iter(rows.values())) if k != "mean_calls"]
    mean = {k: {m: float(np.mean([rows[t][k][m] for t in rows])) for m in KEYS} for k in names}
    grid = ["prior_paint_a%.2f" % al for al in ALPHAS]
    best = max(grid, key=lambda k: mean[k]["pooled_ap"] + mean[k]["pooled_roc"])
    (ROOT / a.out_root / a.corpus / "summary.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean, "prior_paint_best_alpha_on_test": best,
         "mean_calls": float(np.mean([rows[t]["mean_calls"] for t in rows]))}, indent=1) + "\n")
    for k in names:
        if k in grid and k not in (best, "prior_paint_a0.50"):
            continue
        print("MEAN", a.corpus, "%-28s" % k, " ".join("%.4f" % mean[k][m] for m in KEYS), flush=True)
    print("DONE", a.corpus, "best", best, flush=True)


if __name__ == "__main__":
    main()
