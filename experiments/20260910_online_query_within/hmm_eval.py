"""Training-free check of the regime-mixture interval HMM (module-1 iteration 2,
plan section D, gate D1): fit on TRAIN video labels only, then on val/test

  1. score the 34 verdicts alone (posterior log-odds per second) through the
     shared evaluator: pooled AP / ROC / within;
  2. predictive quality for windows NOT asked: per video, 8 fine windows are
     masked at random (fixed seed) and p(b_w = 1 | the other verdicts) is
     compared with the cached verdict (log-loss and Brier); also from the test
     start state (4 coarse blocks only, all 30 fine windows predicted).

Variants: regimes = 1 (revision-3 model, conditionally independent verdicts)
and regimes = 3 (video-level reliability mixture). Gate D1: the masked-window
log-loss improves on both corpora and pooled AP/ROC on test are not below the
R = 1 values - .005.

    python experiments/20260910_online_query_within/hmm_eval.py --corpus hatemm
Writes runs/20260910_online_query_within/hmm_only/<corpus>/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hate_common import data as hdata          # noqa: E402
from macilsd import align                      # noqa: E402
import hier_evidence_common as hc              # noqa: E402
import interval_evidence_hmm as ieh            # noqa: E402
import verdict_hmm                             # noqa: E402
import vlm_verdict                             # noqa: E402

EVALUATOR = os.path.join(ROOT, "scripts", "reproduction_baselines", "eval_baseline_scores.py")
K, J = vlm_verdict.GRANULARITIES
BASE = dict(positive_constraint=True, normalized_time=True)     # revision-4 fusion settings
VARIANTS = {"indep": dict(regimes=1), "regimes3": dict(regimes=3)}
N_MASK = 8


def load_binary(corpus):
    V = {K: vlm_verdict.load_verdicts(corpus, k=K, tag="qwen"),
         J: vlm_verdict.load_verdicts(corpus, k=J, tag="qwen")}
    return {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
            for v in V[K] if v in V[J]}


def predictive_check(hmm, B, vids, durations, seed=0):
    """Masked-window prediction: (log-loss, Brier, n) for 8 random masked fine
    windows per video and for the coarse-only start state (30 per video)."""
    rng = np.random.RandomState(seed)
    out = {}
    for name in ("masked8", "coarse_only"):
        ll, br, n = 0.0, 0.0, 0
        for vid in vids:
            bf, bc = B[vid]
            if name == "masked8":
                m = rng.choice(K, size=N_MASK, replace=False)
                mb = bf.copy()
                mb[m] = ieh.MISSING
            else:
                m = np.arange(K)
                mb = np.full(K, ieh.MISSING, dtype=int)
            pred = np.clip(hmm.infer(mb, bc, durations[vid])["pred_fine"], 1e-6, 1 - 1e-6)
            y = bf[m].astype(float)
            p = pred[m]
            ll += float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
            br += float(((p - y) ** 2).sum())
            n += len(m)
        out[name] = {"logloss": ll / n, "brier": br / n, "n": n}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "20260910_online_query_within", "hmm_only"))
    args = ap.parse_args(argv)
    corpus = args.corpus
    B = load_binary(corpus)
    labels = hdata.load_labels(corpus)
    train_ids = [v for v in hdata.load_split(corpus, "train") if v in B]
    out_dir = os.path.join(args.out_root, corpus)
    os.makedirs(out_dir, exist_ok=True)
    gts = {s: hdata.gt_arrays(corpus, s) for s in args.splits.split(",")}
    results = {}
    for name, opts in VARIANTS.items():
        t0 = time.time()
        hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, B, model="interval", **BASE, **opts)
        hmm.save(os.path.join(out_dir, name + "_params.json"))
        p = hmm.params()
        print("%s %s: fitted on %d pos / %d neg in %.0fs; train loglik %.1f -> %.1f (monotone %s); %s" % (
            corpus, name, n_pos, n_neg, time.time() - t0, p["fit_loglik"][0], p["fit_loglik"][-1],
            all(b >= a - 1e-6 for a, b in zip(p["fit_loglik"], p["fit_loglik"][1:])),
            json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 4))
                        for k, v in p.items() if k in ("lam01", "lam10", "pi", "q_fine", "r_fine", "q_coarse", "r_coarse",
                                                        "q_fine_z", "r_fine_z", "q_coarse_z", "r_coarse_z")})), flush=True)
        results[name] = {"params": p}
        for split, gt in gts.items():
            sd = os.path.join(out_dir, split)
            os.makedirs(sd, exist_ok=True)
            sp = os.path.join(sd, name + "_scores.jsonl")
            vids = [v for v in sorted(gt) if v in B]
            durations = {v: float(len(gt[v])) for v in vids}
            with open(sp, "w") as fh:
                for vid in sorted(gt):
                    n = len(gt[vid])
                    if vid not in B:
                        sc = [0.0] * n
                    else:
                        bf, bc = B[vid]
                        lo = hmm.posterior_log_odds(bf, bc, float(n))
                        sc = ieh.rows_from_segments(lo, hmm.grid, align.second_bounds(n), float(n))
                    fh.write(json.dumps({"video_id": vid, "n_frames": n,
                                         "score_av": [round(float(x), 6) for x in sc]}) + "\n")
            jo = os.path.join(sd, name + "_metrics.json")
            subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus, "--split", split,
                            "--scores", sp, "--json-out", jo], check=True, cwd=ROOT,
                           stdout=subprocess.DEVNULL)
            r = json.load(open(jo))["results"]["score_av"]
            pc = predictive_check(hmm, B, vids, durations)
            results[name][split] = dict(pooled_ap=r["pr_auc"], pooled_roc=r["roc_auc"],
                                        within_roc=r["per_video"]["macro_auc"], predictive=pc)
            print("  %s %-9s %s AP %.4f ROC %.4f within %.4f | masked8 logloss %.4f brier %.4f | coarse-only logloss %.4f brier %.4f" % (
                corpus, name, split, r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"],
                pc["masked8"]["logloss"], pc["masked8"]["brier"], pc["coarse_only"]["logloss"], pc["coarse_only"]["brier"]), flush=True)
    a, b = results["indep"]["test"], results["regimes3"]["test"]
    gate = {"logloss_masked8_improves": b["predictive"]["masked8"]["logloss"] < a["predictive"]["masked8"]["logloss"],
            "logloss_coarse_only_improves": b["predictive"]["coarse_only"]["logloss"] < a["predictive"]["coarse_only"]["logloss"],
            "pooled_not_below": b["pooled_ap"] >= a["pooled_ap"] - 0.005 and b["pooled_roc"] >= a["pooled_roc"] - 0.005}
    gate["D1_pass"] = gate["logloss_masked8_improves"] and gate["pooled_not_below"]
    print("%s gate D1: %s" % (corpus, json.dumps(gate)), flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"corpus": corpus, "variants": results, "gate_D1": gate}, fh, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
