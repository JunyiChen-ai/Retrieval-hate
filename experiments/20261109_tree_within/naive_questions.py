"""Control (minor issue after the GPT 6 Astra loop): the strongest baseline given VLM answers to windows chosen WITHOUT
our policy. The rule-14(f) / E1 / E1b controls gave the baseline the 8 nodes our policy asked, so they already contain
part of our method (which windows to ask). Here, per test video, node sets that need no policy:
  uniform8   the 8 equal windows of tree depth 3 (fewer, equal windows when the video is too short: the deepest
             depth whose nodes are all queryable);
  top8       the 8 windows of the finest level (qtree.level_nodes(T, 4), 4-8 s) with the highest mean baseline score;
  finest     every window of the finest level (T / 4 to T / 8 calls; "VLM only" = alpha 0);
  ours8      the first 8 nodes our method asked (revision-6 replay runs_L0.pkl, experiments/20261108_astra_review),
             i.e. the rule-14(f) control, for reference.
Fusion as rule 14(f) (experiments/20261101_r5_completion/baseline_plus_vlm.py): per second the soft answer p of the
shortest asked node containing it (seconds no asked node covers: the mean p of the video's answers), then
sigmoid(alpha logit(baseline_t) + (1 - alpha) logit(p_t)) on the alpha grid; reported alpha 0 (VLM only), .5, 1
(baseline only) and the alpha best on test ((AP + ROC) / 2, favours the control). Baseline: its own 3-seed final
runs; seeds paired with our revision-6 test-selected trials. Nothing is trained; unified core evaluator
(hc.frame_metrics). Output runs/20261109_tree_within/naive_questions/<corpus>.json.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, qtree                                    # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

REPLAY = ROOT / "runs/20261108_astra_review/replay"
ALPHAS = np.round(np.linspace(0, 1, 11), 2)
SETS = ("uniform8", "top8", "finest", "ours8")


def node_sets(T, base, asked):
    tr = qtree.tree(T)
    d = 3
    while d > 0 and not tr["queryable"][tr["depth"] == d].all():
        d -= 1
    fine = qtree.level_nodes(T, 4)
    fine = fine[tr["queryable"][fine]]
    mean_b = np.array([base[tr["a"][n]:tr["b"][n]].mean() for n in fine])
    return tr, {"uniform8": np.where(tr["depth"] == d)[0], "top8": fine[np.argsort(-mean_b, kind="stable")[:8]],
                "finest": fine, "ours8": np.asarray(asked[:8], dtype=np.int64)}


def evidence(tr, T, nodes, P_v):
    e = np.full(T, np.nan)
    ln = np.full(T, np.inf)
    ps = []
    for n in nodes:
        x, y = int(tr["a"][n]), int(tr["b"][n])
        p = P_v.get((x, y))
        if p is None:
            continue
        ps.append(p)
        sel = (y - x) < ln[x:y]
        e[x:y][sel] = p
        ln[x:y][sel] = y - x
    e[np.isnan(e)] = np.mean(ps) if ps else 0.5
    return e, len(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--out-root", default="runs/20261109_tree_within/naive_questions")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    # the revision-6 test-selected trials; the replay directory also holds the validation-selected ones, and until
    # 2026-11-10 this took whichever directory of a seed was listed last (only the ours8 row depends on the trial)
    summ = json.load(open(ROOT / f"runs/20261101_r6_length_table/final_curves/test_selected/{a.corpus}/summary.json"))
    trials = {int(r["seed"]): REPLAY / a.corpus / f'seed{r["seed"]}_{Path(r["trial"]).name}'
              for r in summ["trials"].values()}
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "trials": {s: str(p) for s, p in trials.items()}, "methods": {}}
    for method in hc.STRONGEST_BASELINE[a.corpus]:
        per_seed, calls = [], {s: [] for s in SETS}
        for seed in (234, 2025, 3407):
            full = pickle.load(open(trials[seed] / "runs_L0.pkl", "rb"))["runs"]["test"]
            base = hc.baseline_scores(a.corpus, method, seed)
            rows = {s: {} for s in SETS}
            ev, bs = {s: {} for s in SETS}, {}
            for v in test:
                T = len(full[v]["scores"][0])
                b = base.get(v, np.full(T, 0.5))
                n = min(len(b), T)
                bs[v] = np.concatenate([b[:n], np.full(T - n, b[n - 1] if n else 0.5)])
                tr, sets = node_sets(T, bs[v], full[v]["asked"])
                for s in SETS:
                    ev[s][v], k = evidence(tr, T, sets[s], P[v])
                    calls[s].append(k)
            for s in SETS:
                for al in ALPHAS:
                    sc = {v: 1.0 / (1.0 + np.exp(-(al * hc.logit(bs[v]) + (1 - al) * hc.logit(ev[s][v]))))
                          for v in test}
                    m = hc.frame_metrics(sc, gt["test"], hate)
                    rows[s]["%.2f" % al] = {k: float(m[k]) for k in ("pooled_ap", "pooled_roc", "within_roc")}
            per_seed.append(rows)
        res = {}
        for s in SETS:
            mean = {al: {k: float(np.mean([r[s][al][k] for r in per_seed])) for k in per_seed[0][s][al]}
                    for al in per_seed[0][s]}
            best = max(mean, key=lambda al: mean[al]["pooled_ap"] + mean[al]["pooled_roc"])
            res[s] = {"mean": mean, "best_alpha_on_test": best, "mean_calls": float(np.mean(calls[s])),
                      "per_seed": [r[s] for r in per_seed]}
            print(a.corpus, method, s, "calls %.2f" % res[s]["mean_calls"],
                  "| vlm only", [round(x, 4) for x in mean["0.00"].values()],
                  "| alpha .5", [round(x, 4) for x in mean["0.50"].values()],
                  "| best", best, [round(x, 4) for x in mean[best].values()],
                  "| baseline", [round(x, 4) for x in mean["1.00"].values()], flush=True)
        report["methods"][method] = res
    out = (ROOT / a.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", a.corpus, flush=True)


if __name__ == "__main__":
    main()
