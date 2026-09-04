"""CPU gate for the coarse-first conditional label model (module 3 candidate,
idea discovery 2026-09-05, README section 2). No network training.

For each context arm (global = candidate 1's HMM, b4, bprev, cond) the HMM is
fitted on TRAIN video labels only, then scored posterior-alone on val and test
through the shared evaluator (same path as candidate 1's verdict_hmm_eval.py),
and the per-video hate-proportion estimate q_v = mean_t P(s_t = 1) is compared
with the ground-truth hate fraction on the test split (MAE, bias, Pearson r).

    python experiments/20260905_verdict_conditioned_density/gate_label_model.py --corpus hatemm

Writes runs/20260905_verdict_conditioned_density/gate/<corpus>/<arm>/{hmm_params.json,
<split>/scores.jsonl, <split>/metrics.json} and gate_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hate_common import data as hdata          # noqa: E402
import vlm_verdict                              # noqa: E402
import verdict_hmm                              # noqa: E402
from verdict_hmm_cond import CondEvidenceHMM, CONTEXTS   # noqa: E402

EVALUATOR = os.path.join(ROOT, "scripts", "reproduction_baselines", "eval_baseline_scores.py")
K, J = vlm_verdict.GRANULARITIES


def load_binary(corpus):
    V = {K: vlm_verdict.load_verdicts(corpus, k=K, tag="qwen"),
         J: vlm_verdict.load_verdicts(corpus, k=J, tag="qwen")}
    return {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
            for v in V[K] if v in V[J]}


def fit(corpus, B, context, monotone=True):
    labels = hdata.load_labels(corpus)
    train = [v for v in hdata.load_split(corpus, "train") if v in B]
    pos = [B[v] for v in train if labels[v] == 1]
    neg = [B[v] for v in train if labels[v] == 0]
    return CondEvidenceHMM(K, J, context).fit(pos, neg, monotone=monotone), len(pos), len(neg)


def score_split(m, B, corpus, split, out_dir, variants):
    gt = hdata.gt_arrays(corpus, split)
    sd = os.path.join(out_dir, split)
    os.makedirs(sd, exist_ok=True)
    sp = os.path.join(sd, "scores.jsonl")
    prop = []
    with open(sp, "w") as fh:
        for vid in sorted(gt):
            n = len(gt[vid])
            rec = {"video_id": vid, "n_frames": n}
            if vid not in B:
                for name in variants:
                    rec[name] = [0.0] * n
            else:
                bf, bc = B[vid]
                for name, kw in variants.items():
                    lo = m.posterior_log_odds(bf, bc, **kw)
                    rec[name] = [round(float(x), 6) for x in verdict_hmm.rows_from_windows(lo, n, K)]
                p_s, _ = m.posterior(bf, bc)
                prop.append((vid, float(p_s.mean()), float(np.mean(gt[vid]))))
            fh.write(json.dumps(rec) + "\n")
    jo = os.path.join(sd, "metrics.json")
    subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus, "--split", split,
                    "--scores", sp, "--json-out", jo], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL)
    res = json.load(open(jo))["results"]
    out = {name: {"ap": res[name]["pr_auc"], "roc": res[name]["roc_auc"],
                  "within": res[name]["per_video"]["macro_auc"]} for name in variants}
    q = np.array([p[1] for p in prop])
    g = np.array([p[2] for p in prop])
    labels = hdata.load_labels(corpus)
    pos = np.array([labels[p[0]] == 1 for p in prop])
    out["proportion"] = {
        "n": int(len(q)), "mae_all": float(np.abs(q - g).mean()),
        "bias_all": float((q - g).mean()),
        "mae_pos": float(np.abs(q[pos] - g[pos]).mean()) if pos.any() else None,
        "bias_pos": float((q[pos] - g[pos]).mean()) if pos.any() else None,
        "pearson_pos": float(np.corrcoef(q[pos], g[pos])[0, 1]) if pos.sum() > 2 else None,
        "mean_q_neg": float(q[~pos].mean()) if (~pos).any() else None,
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "20260905_verdict_conditioned_density", "gate"))
    ap.add_argument("--contexts", default=",".join(CONTEXTS))
    ap.add_argument("--no-monotone", action="store_true", help="unconstrained EM (gate run 1)")
    args = ap.parse_args(argv)
    B = load_binary(args.corpus)
    root = os.path.join(args.out_root, args.corpus + ("_unconstrained" if args.no_monotone else ""))
    os.makedirs(root, exist_ok=True)
    summary = {"corpus": args.corpus, "host": socket.gethostname(), "arms": {}}
    variants = {"score_hmm_posterior": dict(), "score_hmm_fine_only": dict(w_coarse=0.0),
                "score_hmm_coarse_only": dict(w_fine=0.0)}
    for context in args.contexts.split(","):
        m, n_pos, n_neg = fit(args.corpus, B, context, monotone=not args.no_monotone)
        od = os.path.join(root, context)
        os.makedirs(od, exist_ok=True)
        m.save(os.path.join(od, "hmm_params.json"))
        arm = {"n_pos": n_pos, "n_neg": n_neg, "params": m.params()}
        print("%s [%s] p_fine s=0 %s | s=1 %s | q_c %.3f r_c %.3f | A %s" % (
            args.corpus, context, np.round(m.p_fine[0], 3).tolist(),
            np.round(m.p_fine[1], 3).tolist(), m.q_c, m.r_c, np.round(m.A, 3).tolist()))
        for split in args.splits.split(","):
            r = score_split(m, B, args.corpus, split, od, variants)
            arm[split] = r
            p = r["score_hmm_posterior"]
            pr = r["proportion"]
            print("  %-4s posterior AP %.4f ROC %.4f within %.4f | q_v MAE(pos) %.3f bias(pos) %+.3f r %.3f | mean q neg %.3f" % (
                split, p["ap"], p["roc"], p["within"], pr["mae_pos"], pr["bias_pos"],
                pr["pearson_pos"] if pr["pearson_pos"] is not None else float("nan"), pr["mean_q_neg"]))
        summary["arms"][context] = arm
    with open(os.path.join(root, "gate_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
