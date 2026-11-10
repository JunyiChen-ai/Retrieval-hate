"""Diagnostic (README section 12): the naive-question control (section 7) at every call budget, with two per-second
priors under the SAME fusion: the strongest baseline's scores, and our own backbone's scores before any call.
Windows: the 2^d equal windows of tree depth d = 0..5 (1, 2, 4, 8, 16, 32 calls; for a video too short, the deepest
depth whose nodes are all queryable). Fusion as naive_questions.py (rule 14(f)): per second the soft answer p of its
window, sigmoid(alpha logit(prior_t) + (1 - alpha) logit(p_t)); alpha 0 (VLM answers only), .5 (no tuning) and the
alpha best on test ((AP + ROC) / 2 of the 3-seed mean, favours the control).
  prior "ours0": the revision-6 backbone before any call (scores_test_fixed0.jsonl of the test-selected trials);
  prior <baseline>: its own 3-seed final runs (seeds paired).
Against revision 6 at the same fixed number of calls (final_curves/test_selected summary). Nothing is trained;
unified core evaluator (hc.frame_metrics). Output runs/20261109_tree_within/control_curve/<corpus>.json.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree                                    # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402
from naive_questions import evidence                                    # noqa: E402

DEPTHS = (0, 1, 2, 3, 4, 5)
ALPHAS = np.round(np.linspace(0, 1, 11), 2)
METS = ("pooled_ap", "pooled_roc", "within_roc")
SEEDS = (234, 2025, 3407)


def windows(T, d):
    tr = qtree.tree(T)
    while d > 0 and not tr["queryable"][tr["depth"] == d].all():
        d -= 1
    return tr, np.where(tr["depth"] == d)[0]


def read_scores(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def fit_length(b, T):
    n = min(len(b), T)
    return np.concatenate([b[:n], np.full(T - n, b[n - 1] if n else 0.5)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/control_curve")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = {int(r["seed"]): Path(r["trial"]) for r in summ["trials"].values()}
    priors = {"ours0": {s: read_scores(trials[s] / "scores_test_fixed0.jsonl") for s in SEEDS}}
    for method in hc.STRONGEST_BASELINE[a.corpus]:
        priors[method] = {s: hc.baseline_scores(a.corpus, method, s) for s in SEEDS}
    T = {v: len(priors["ours0"][SEEDS[0]][v]) for v in test}
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "trials": {s: str(p) for s, p in trials.items()}, "priors": {k: {} for k in priors}, "revision6": {}}
    for d in DEPTHS:
        K = str(2 ** d)
        ev, calls = {}, []
        for v in test:
            tr, nodes = windows(T[v], d)
            ev[v], k = evidence(tr, T[v], nodes, P[v])
            calls.append(k)
        r6 = summ["mean"]["fixed"][K]
        report["revision6"][K] = {k: float(r6[k]) for k in METS + ("mean_calls",)}
        for name, by_seed in priors.items():
            per_seed = []
            for s in SEEDS:
                pr = {v: fit_length(by_seed[s].get(v, np.full(T[v], 0.5)), T[v]) for v in test}
                row = {}
                for al in ALPHAS:
                    sc = {v: 1.0 / (1.0 + np.exp(-(al * hc.logit(pr[v]) + (1 - al) * hc.logit(ev[v])))) for v in test}
                    m = hc.frame_metrics(sc, gt["test"], hate)
                    row["%.2f" % al] = {k: float(m[k]) for k in METS}
                per_seed.append(row)
            mean = {al: {k: float(np.mean([r[al][k] for r in per_seed])) for k in METS} for al in per_seed[0]}
            best = max(mean, key=lambda al: mean[al]["pooled_ap"] + mean[al]["pooled_roc"])
            report["priors"][name][K] = {"mean": mean, "best_alpha_on_test": best, "mean_calls": float(np.mean(calls))}
            print(a.corpus, "K", K, "calls %.2f" % np.mean(calls), name,
                  "| vlm only", [round(mean["0.00"][k], 4) for k in METS],
                  "| alpha .5", [round(mean["0.50"][k], 4) for k in METS],
                  "| best", best, [round(mean[best][k], 4) for k in METS],
                  "| prior only", [round(mean["1.00"][k], 4) for k in METS],
                  "| r6 fixed", [round(report["revision6"][K][k], 4) for k in METS],
                  "calls %.2f" % report["revision6"][K]["mean_calls"], flush=True)
    out = (ROOT / a.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
