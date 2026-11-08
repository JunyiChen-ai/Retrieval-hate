"""Diagnostic (README section 6): if the answers followed the revision-6 answer model exactly, would the full tree
beat the single levels on within? Revision-6 test-selected trials, test split; real backbone prior and chain; every
cached answer replaced by a draw from the trial's anchored table given the node's TRUE state (G = video label,
z_n = any GT-hateful second in the node), independently across nodes; one draw per (video, node), shared by every
policy. Positive videos without GT spans keep their cached answers (they are not in the within metric). Policy = the
method's EIG (cpolicy.run_batch); E3 comparison at the matched calls (full tree vs each single level L4 / L8 / L16,
both on the simulated answers) and fixed 8. If the tree still loses within here, EIG is not aligned with within even
when the model is right; if it wins, the real answers' departures from the model cause the loss.
--variant (policy, tree and levels alike): eig (the method), flatpol (EIG under the flat location prior, scores from
the full prior; cpolicy policy_prior_scale 0), eigloc_flatpol (I(o; y | G = 1) under the flat location prior), noadj
(no parent / child of an asked node).
Output runs/20261109_tree_within/simulated[_<variant>]/<corpus>/ (summary.json).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
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

METS = ("pooled_ap", "pooled_roc", "within_roc")


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def simulate(test, store, labels, gt, am, cats, seed):
    """Per video: {node: (3,) log-likelihood of the drawn answer}, or None (keep the cached answers)."""
    sim = {}
    for i, v in enumerate(sorted(test)):
        if labels[v] == 1 and v not in gt:
            sim[v] = None
            continue
        rng = np.random.default_rng(seed * 100003 + i)
        vt = cpolicy._vt(store.T[v])
        asker = policy.Asker(vt, am, cats)
        y = np.asarray(gt[v]).astype(int) if labels[v] == 1 else np.zeros(store.T[v], dtype=int)
        d = {}
        for pos, n in enumerate(asker.q):
            a_, b_ = int(vt.tr["a"][n]), int(vt.tr["b"][n])
            s = 0 if labels[v] == 0 else (2 if y[a_:b_].any() else 1)
            o = int(rng.choice(asker.po_s.shape[2], p=asker.po_s[pos, s] / asker.po_s[pos, s].sum()))
            d[int(n)] = asker.table[pos, :, o].copy()
        sim[v] = d
    return sim


VARIANTS = {"eig": {}, "flatpol": {"policy_prior_scale": 0.0},
            "eigloc_flatpol": {"order": "eig_loc", "policy_prior_scale": 0.0}, "noadj": {"no_adjacent": 1}}


def replay(model, store, vids, am, chain, answers, cfg, sim, level, variant="eig"):
    runs = {}
    vids = sorted(vids, key=lambda v: store.T[v])
    for i in range(0, len(vids), int(cfg["eval_chunk"])):
        chunk = vids[i:i + int(cfg["eval_chunk"])]
        asks = [policy.Asker(cpolicy._vt(store.T[v]), am, cfg["categories"]) for v in chunk]

        def answer_ll(b, node, chunk=chunk, asks=asks):
            v = chunk[b]
            if sim[v] is not None:
                return sim[v][int(node)]
            tr = asks[b].vt.tr
            o = answers[v].get((int(tr["a"][node]), int(tr["b"][node])))
            return None if o is None else asks[b].loglik(node, o)
        with torch.no_grad():
            runs.update(cpolicy.run_batch(model, store, chunk, am, chain, answers, cfg["categories"], 8, "cpu",
                                          answer_ll=answer_ll, **VARIANTS[variant],
                                          allowed=None if not level else
                                          [set(qtree.level_nodes(store.T[v], level).tolist()) for v in chunk]))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--levels", nargs="+", type=int, default=[4, 8, 16])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="eig")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    if a.out_root is None:
        a.out_root = "runs/20261109_tree_within/simulated" + ("" if a.variant == "eig" else "_" + a.variant)
    torch.set_num_threads(a.threads)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = [r["trial"] for r in summ["trials"].values()]
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    gt = {v: np.asarray(y) for v, y in gt["test"].items()}
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
        sim = simulate(test, store, labels, gt, am, cfg["categories"], int(summary["seed"]))
        runs = {L: replay(model, store, test, am, chain, answers, cfg, sim, L, a.variant) for L in [0] + a.levels}
        full = runs[0]
        row = {}
        for L in a.levels:
            k = {v: min(8, len(runs[L][v]["eig"])) for v in test}
            mf = score(a.corpus, od, f"full_matchL{L}",
                       {v: full[v]["scores"][min(k[v], len(full[v]["scores"]) - 1)] for v in test})
            ml = score(a.corpus, od, f"level{L}", {v: runs[L][v]["scores"][k[v]] for v in test})
            row[str(L)] = {"full": mf, "level": ml, "full_minus_level": {m: mf[m] - ml[m] for m in METS}}
        row["fixed8"] = score(a.corpus, od, "full_fixed8", {v: full[v]["scores"][-1] for v in test})
        nested = []
        for v in test:
            tr = qtree.tree(store.T[v])
            asked = full[v]["asked"]
            nested.append(sum(any((tr["a"][m] <= tr["a"][n] and tr["b"][n] <= tr["b"][m]) or
                                  (tr["a"][n] <= tr["a"][m] and tr["b"][m] <= tr["b"][n]) for m in asked[:i])
                              for i, n in enumerate(asked)) / max(len(asked), 1))
        row["nested_share"] = float(np.mean(nested))
        rows[name] = row
        print(a.corpus, name, json.dumps({L: [round(x, 4) for x in row[L]["full_minus_level"].values()]
                                          for L in map(str, a.levels)}),
              "fixed8", [round(x, 4) for x in row["fixed8"].values()], "nested %.2f" % row["nested_share"], flush=True)
    mean = {L: {kk: {m: float(np.mean([rows[t][L][kk][m] for t in rows])) for m in METS}
                for kk in ("full", "level", "full_minus_level")} for L in map(str, a.levels)}
    mean["fixed8"] = {m: float(np.mean([rows[t]["fixed8"][m] for t in rows])) for m in METS}
    mean["nested_share"] = float(np.mean([rows[t]["nested_share"] for t in rows]))
    (ROOT / a.out_root / a.corpus / "summary.json").write_text(json.dumps(
        {"corpus": a.corpus, "host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
         "trials": rows, "mean": mean}, indent=1) + "\n")
    print("DONE", a.corpus, json.dumps({L: mean[L]["full_minus_level"] for L in map(str, a.levels)}),
          json.dumps(mean["fixed8"]), "nested %.2f" % mean["nested_share"], flush=True)


if __name__ == "__main__":
    main()
