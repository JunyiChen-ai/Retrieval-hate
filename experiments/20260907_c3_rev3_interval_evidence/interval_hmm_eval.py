"""Training-free evaluation of the interval evidence HMM against the index
HMM: fit on TRAIN video labels, score the verdicts alone (no localizer) on
val/test through the shared evaluator. Gate A1 of the revision-3 plan.

    python experiments/20260907_c3_rev3_interval_evidence/interval_hmm_eval.py --corpus hatemm

Writes runs/20260907_c3_rev3_interval_evidence/hmm_only/<corpus>/
    <variant>_params.json, <split>/scores.jsonl, <split>/metrics.json
Variants: index (src/verdict_hmm, the current fusion), interval,
interval_constraint, interval_effect, interval_both.

Startup check: on a nested grid (k = 32, j = 4, equal segments) the interval
model with the index model's transition matrix gives the same posteriors as
the index model (they are the same chain there).
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
VARIANTS = {
    "index": None,
    "interval": dict(),
    "interval_constraint": dict(positive_constraint=True),
    "interval_effect": dict(video_effect=True),
    "interval_both": dict(positive_constraint=True, video_effect=True),
    "interval_norm": dict(normalized_time=True),
    "interval_norm_constraint": dict(normalized_time=True, positive_constraint=True),
}


def nested_equivalence_check():
    rng = np.random.RandomState(0)
    k, j = 32, 4
    idx = verdict_hmm.HierEvidenceHMM(k, j)
    itv = ieh.IntervalEvidenceHMM(k, j)
    itv.lam01, itv.lam10 = 0.05, 0.03
    dur = 96.0                       # every segment 3 s
    P = ieh._ctmc(itv.lam01, itv.lam10, np.array([3.0]))[0]
    idx.A = P
    idx.p0 = itv.p0.copy()
    idx.q_f, idx.r_f, idx.q_c, idx.r_c = itv.q_f, itv.r_f, itv.q_c, itv.r_c
    for _ in range(20):
        bf = rng.binomial(1, 0.3, k)
        bc = rng.binomial(1, 0.5, j)
        ps_i, ph_i = idx.posterior(bf, bc)
        ps_v, ph_v = itv.posterior(bf, bc, dur)
        assert np.allclose(ps_i, ps_v, atol=1e-9) and np.allclose(ph_i, ph_v, atol=1e-9), \
            (np.abs(ps_i - ps_v).max(), np.abs(ph_i - ph_v).max())
    return True


def load_binary(corpus):
    V = {K: vlm_verdict.load_verdicts(corpus, k=K, tag="qwen"),
         J: vlm_verdict.load_verdicts(corpus, k=J, tag="qwen")}
    return {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
            for v in V[K] if v in V[J]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "20260907_c3_rev3_interval_evidence", "hmm_only"))
    args = ap.parse_args(argv)
    assert nested_equivalence_check()
    print("nested-grid equivalence check passed", flush=True)
    B = load_binary(args.corpus)
    labels = hdata.load_labels(corpus := args.corpus)
    train_ids = [v for v in hdata.load_split(corpus, "train") if v in B]
    out_dir = os.path.join(args.out_root, corpus)
    os.makedirs(out_dir, exist_ok=True)
    gts = {s: hdata.gt_arrays(corpus, s) for s in args.splits.split(",")}
    results = {}
    for name in args.variants.split(","):
        opts = VARIANTS[name]
        t0 = time.time()
        if opts is None:
            hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, B, model="index")
        else:
            hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, B, model="interval", **opts)
        hmm.save(os.path.join(out_dir, name + "_params.json"))
        print("%s %s: fitted on %d pos / %d neg in %.0fs: %s" % (
            corpus, name, n_pos, n_neg, time.time() - t0,
            json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in hmm.params().items() if k not in ("k", "j", "A")})), flush=True)
        for split, gt in gts.items():
            sd = os.path.join(out_dir, split)
            os.makedirs(sd, exist_ok=True)
            sp = os.path.join(sd, name + "_scores.jsonl")
            with open(sp, "w") as fh:
                for vid in sorted(gt):
                    n = len(gt[vid])
                    if vid not in B:
                        sc = [0.0] * n
                    else:
                        bf, bc = B[vid]
                        if opts is None:
                            lo = hmm.posterior_log_odds(bf, bc)
                            sc = verdict_hmm.rows_from_windows(lo, n, K)
                        else:
                            lo = hmm.posterior_log_odds(bf, bc, float(n))
                            sc = ieh.rows_from_segments(lo, hmm.grid, align.second_bounds(n), float(n))
                    fh.write(json.dumps({"video_id": vid, "n_frames": n,
                                         "score_av": [round(float(x), 6) for x in sc]}) + "\n")
            jo = os.path.join(sd, name + "_metrics.json")
            subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus, "--split", split,
                            "--scores", sp, "--json-out", jo], check=True, cwd=ROOT,
                           stdout=subprocess.DEVNULL)
            r = json.load(open(jo))["results"]["score_av"]
            results.setdefault(name, {})[split] = dict(
                pooled_ap=r["pr_auc"], pooled_roc=r["roc_auc"], within_roc=r["per_video"]["macro_auc"])
            print("  %s %-20s %s AP %.4f ROC %.4f within %.4f" % (
                corpus, name, split, r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"]), flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"corpus": corpus, "variants": results,
                   "reference_index_rows": "runs/20260903_hier_evidence_mil/verdict_hmm_only"}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
