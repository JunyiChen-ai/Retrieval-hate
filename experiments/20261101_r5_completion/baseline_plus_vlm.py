"""Rule 14(f) control (README section 5): the strongest trained baseline of each corpus given exactly the VLM answers
our method used.

For every test video: the baseline's per-second scores (its own 3-seed final runs) and the soft answers p (first-token
P(Yes)) of the nodes our method asked at a fixed budget k (stored policy runs of the same seed). VLM evidence per
second e_t = p of the shortest asked node containing t (no asked node contains t: the mean p of the video's asked
nodes). Fusion per second: sigmoid(alpha logit(baseline_t) + (1 - alpha) logit(e_t)), alpha on a grid from 0 (VLM
only) to 1 (baseline only). Reported: alpha = .5 and the alpha best on test (test (AP + ROC) / 2, an upper bound that
favours the control), 3-seed means; AP / ROC / within by the shared core evaluator (hc.frame_metrics).
Rule-10 developmental evidence; nothing is trained.
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
from qtl import data as qdata, qtree
import hier_evidence_common as hc

# copies of the official final runs in Hate-follow-up/runs/legacy_lab2_results_pre20261004/reproduction/official_val/final
LEGACY = ROOT / "runs/20261101_r5_completion/baseline_scores"
DEHATE = ROOT / "runs/20260926_dehate_external/baselines/final"
STRONGEST = {"hatemm": ["macilsd"], "hateclipseg": ["fed_wsvad_3client", "dsanet"], "dehate": ["fed_wsvad_3client"]}
ALPHAS = np.round(np.linspace(0, 1, 11), 2)


def baseline_scores(corpus, method, seed):
    root = DEHATE if corpus == "dehate" else LEGACY
    out = {}
    for line in open(root / method / corpus / f"seed_{seed}" / "scores.jsonl"):
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r.get("score_av", r.get("score_align")), dtype=np.float64)
    return out


def logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(x) - np.log1p(-x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--runs", nargs=3, required=True, help="runs pickles of seeds 234, 2025, 3407 (in that order)")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default="runs/20261101_r5_completion/baseline_plus_vlm")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test = list(ids["test"])
    hate = {v for v in test if labels[v] == 1}
    P, _, _ = qdata.load_soft_p(a.corpus, "soft_both")
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "budget": a.budget, "runs": a.runs, "methods": {}}
    for method in STRONGEST[a.corpus]:
        per_seed = []
        for seed, rp in zip((234, 2025, 3407), a.runs):
            with open(rp, "rb") as fh:
                runs = pickle.load(fh)["runs"]["test"]
            base = baseline_scores(a.corpus, method, seed)
            ev = {}
            for v in test:
                T = len(runs[v]["scores"][0])
                tr = qtree.tree(T)
                e = np.full(T, np.nan)
                ln = np.full(T, np.inf)
                ps = []
                for node in runs[v]["asked"][:a.budget]:
                    x, y = int(tr["a"][node]), int(tr["b"][node])
                    p = P[v].get((x, y))
                    if p is None:
                        continue
                    ps.append(p)
                    sel = (y - x) < ln[x:y]
                    e[x:y][sel] = p
                    ln[x:y][sel] = y - x
                e[np.isnan(e)] = np.mean(ps) if ps else 0.5
                ev[v] = e
            rows = {}
            for al in ALPHAS:
                sc = {}
                for v in test:
                    b = base.get(v, np.full(len(ev[v]), 0.5))
                    n = min(len(b), len(ev[v]))
                    b = np.concatenate([b[:n], np.full(len(ev[v]) - n, b[n - 1] if n else 0.5)])
                    sc[v] = 1.0 / (1.0 + np.exp(-(al * logit(b) + (1 - al) * logit(ev[v]))))
                m = hc.frame_metrics(sc, gt["test"], hate)
                rows["%.2f" % al] = {k: float(m[k]) for k in ("pooled_ap", "pooled_roc", "within_roc")}
            per_seed.append(rows)
        mean = {al: {k: float(np.mean([r[al][k] for r in per_seed])) for k in per_seed[0][al]} for al in per_seed[0]}
        best = max(mean, key=lambda al: mean[al]["pooled_ap"] + mean[al]["pooled_roc"])
        report["methods"][method] = {"mean": mean, "per_seed": per_seed, "best_alpha_on_test": best}
        print(a.corpus, method, "baseline", mean["1.00"], "| vlm only", mean["0.00"], "| alpha .5", mean["0.50"],
              "| best alpha", best, mean[best], flush=True)
    out = (ROOT / a.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}_{a.tag}_k{a.budget}.json").write_text(json.dumps(report, indent=1) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
