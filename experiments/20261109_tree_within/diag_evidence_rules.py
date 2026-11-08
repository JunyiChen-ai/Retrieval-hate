"""Diagnostic (README section 6): with the tree's questions fixed, does the way nested answers are combined cost
within? Revision-6 test-selected trials; the method's asked nodes after the E3 matched calls k_v (replays
runs_L0.pkl / runs_L<L>.pkl of experiments/20261108_astra_review); same model, answer table and cached answers; the
posterior recomputed (ctree.marginals) from a subset of those answers:
  all      every asked answer (= the method's scores at k_v, a check);
  finest   only asked nodes with no asked node inside them;
  coarsest only asked nodes with no asked node containing them.
Also the single level's answers (all, a check). Within / AP / ROC by the unified evaluator; per corpus and level the
3-trial means of each rule minus the single level. Output runs/20261109_tree_within/evidence_rules/<corpus>/.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, ctree, data as qdata, policy                   # noqa: E402
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


def nested(tr, n, m):
    return bool((tr["a"][m] <= tr["a"][n] and tr["b"][n] <= tr["b"][m]) or
                (tr["a"][n] <= tr["a"][m] and tr["b"][m] <= tr["b"][n]))


def subset(tr, asked, rule):
    if rule == "all":
        return list(asked)
    keep = []
    for n in asked:
        others = [m for m in asked if m != n and nested(tr, n, m)]
        if rule == "finest" and not any(tr["b"][m] - tr["a"][m] < tr["b"][n] - tr["a"][n] for m in others):
            keep.append(n)
        if rule == "coarsest" and not any(tr["b"][m] - tr["a"][m] > tr["b"][n] - tr["a"][n] for m in others):
            keep.append(n)
    return keep


def infer(model, store, vids, am, chain, answers, cats, use):
    """Posterior second scores per video from the answers of the nodes use[v]."""
    out = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), 64):
        chunk = vids[i:i + 64]
        Ts = [store.T[v] for v in chunk]
        fo = ctree.Forest(Ts, max(Ts))
        S = torch.zeros(len(chunk), max(Ts), dtype=torch.float64)
        G = torch.zeros(len(chunk), dtype=torch.float64)
        PHI = None
        A3 = torch.zeros(fo.N, 3, dtype=torch.float64)
        for b, v in enumerate(chunk):
            s, g, phi = policy.video_prior_nodes(model, store, v, "cpu")
            S[b, :len(s)] = torch.from_numpy(s)
            G[b] = g
            if phi is not None:
                if PHI is None:
                    PHI = torch.zeros(fo.N, dtype=torch.float64)
                PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
            vt = cpolicy._vt(Ts[b])
            asker = policy.Asker(vt, am, cats)
            for n in use[v]:
                o = answers[v].get((int(vt.tr["a"][n]), int(vt.tr["b"][n])))
                if o is not None:
                    A3[fo.offs[b] + n] = torch.as_tensor(asker.loglik(n, o), dtype=torch.float64)
        _, _, p = ctree.marginals(fo, S, G, A3, chain, PHI)
        p = p.cpu().numpy()
        for b, v in enumerate(chunk):
            out[v] = p[b, :Ts[b]].copy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/evidence_rules")
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
        row = {}
        for L in a.levels:
            lev = pickle.load(open(REPLAY / a.corpus / name / f"runs_L{L}.pkl", "rb"))["runs"]["test"]
            k = {v: min(8, len(lev[v]["eig"])) for v in test}
            res = {"level": score(a.corpus, od, f"level{L}", infer(model, store, test, am, chain, answers,
                                                                    cfg["categories"],
                                                                    {v: lev[v]["asked"][:k[v]] for v in test}))}
            for rule in ("all", "finest", "coarsest"):
                use = {}
                for v in test:
                    tr = cpolicy._vt(store.T[v]).tr
                    use[v] = subset(tr, [int(n) for n in full[v]["asked"][:k[v]]], rule)
                res[rule] = score(a.corpus, od, f"{rule}_L{L}", infer(model, store, test, am, chain, answers,
                                                                       cfg["categories"], use))
                res[rule + "_minus_level"] = {m: res[rule][m] - res["level"][m] for m in METS}
            row[str(L)] = res
            print(a.corpus, name, L, json.dumps({r: [round(res[r + "_minus_level"][m], 4) for m in METS]
                                                 for r in ("all", "finest", "coarsest")}), flush=True)
        rows[name] = row
    mean = {L: {r: {m: float(np.mean([rows[t][L][r][m] for t in rows])) for m in METS}
                for r in ("level", "all", "finest", "coarsest", "all_minus_level", "finest_minus_level",
                          "coarsest_minus_level")} for L in map(str, a.levels)}
    (ROOT / a.out_root / a.corpus / "summary.json").write_text(json.dumps({"trials": rows, "mean": mean}, indent=1))
    print("DONE", a.corpus, json.dumps({L: {r: [round(mean[L][r + "_minus_level"][m], 4) for m in METS]
                                            for r in ("all", "finest", "coarsest")} for L in mean}), flush=True)


if __name__ == "__main__":
    main()
