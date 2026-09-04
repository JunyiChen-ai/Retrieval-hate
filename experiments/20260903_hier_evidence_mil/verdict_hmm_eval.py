"""Fit the hierarchical evidence HMM on TRAIN video labels and score the verdicts alone
(no localizer) through the shared evaluator. Writes
runs/20260903_hier_evidence_mil/verdict_hmm_only/<corpus>/{hmm_params.json, <split>/scores.jsonl, metrics.json}.

    python experiments/20260903_hier_evidence_mil/verdict_hmm_eval.py --corpus hatemm
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines")); sys.path.insert(0, os.path.join(ROOT, "src"))
from hate_common import data as hdata   # noqa: E402
import vlm_verdict                       # noqa: E402
import verdict_hmm                       # noqa: E402
EVALUATOR = os.path.join(ROOT, "scripts", "reproduction_baselines", "eval_baseline_scores.py")
K, J = vlm_verdict.GRANULARITIES  # (30, 4)


def load_binary(corpus, fine_tag="qwen", coarse_tag="qwen"):
    V = {K: vlm_verdict.load_verdicts(corpus, k=K, tag=fine_tag),
         J: vlm_verdict.load_verdicts(corpus, k=J, tag=coarse_tag)}
    return {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
            for v in V[K] if v in V[J]}


def fit(corpus, B):
    labels = hdata.load_labels(corpus)
    train = [v for v in hdata.load_split(corpus, "train") if v in B]
    pos = [B[v] for v in train if labels[v] == 1]
    neg = [B[v] for v in train if labels[v] == 0]
    m = verdict_hmm.HierEvidenceHMM(K, J).fit(pos, neg)
    return m, len(pos), len(neg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "20260903_hier_evidence_mil", "verdict_hmm_only"))
    ap.add_argument("--fine-tag", default="qwen", help="K30 verdict cache tag (qwenctx = context-conditioned)")
    ap.add_argument("--w-fine", default="", help="comma list of extra K30 tempering exponents, e.g. 0.25,0.5,0.75 (branches score_hmm_wf<value>)")
    args = ap.parse_args(argv)
    if args.fine_tag != "qwen" and args.out_root == ap.get_default("out_root"):
        args.out_root = args.out_root + "_" + args.fine_tag   # never overwrite the qwen reference row
    B = load_binary(args.corpus, args.fine_tag)
    m, n_pos, n_neg = fit(args.corpus, B)
    out_dir = os.path.join(args.out_root, args.corpus)
    os.makedirs(out_dir, exist_ok=True)
    m.save(os.path.join(out_dir, "hmm_params.json"))
    print("%s: fitted on %d positive / %d negative train videos: %s" % (args.corpus, n_pos, n_neg, json.dumps(m.params())))
    variants = {
        "score_hmm_posterior": dict(),
        "score_hmm_fine_only": dict(w_coarse=0.0),
        "score_hmm_coarse_only": dict(w_fine=0.0),
        "score_hmm_independent": dict(independent=True),
        "score_hmm_flat_coarse": dict(flat_coarse=True),
    }
    for w in [float(x) for x in args.w_fine.split(",") if x]:
        variants["score_hmm_wf%03d" % int(round(w * 100))] = dict(w_fine=w)
    for split in args.splits.split(","):
        gt = hdata.gt_arrays(args.corpus, split)
        sd = os.path.join(out_dir, split); os.makedirs(sd, exist_ok=True)
        sp = os.path.join(sd, "scores.jsonl"); n_missing = 0
        with open(sp, "w") as fh:
            for vid in sorted(gt):
                n = len(gt[vid]); rec = {"video_id": vid, "n_frames": n}
                if vid not in B:
                    n_missing += 1
                    for name in list(variants) + ["score_mean_level"]:
                        rec[name] = [0.0] * n
                else:
                    bf, bc = B[vid]
                    for name, kw in variants.items():
                        lo = m.posterior_log_odds(bf, bc, **kw)
                        rec[name] = [round(float(x), 6) for x in verdict_hmm.rows_from_windows(lo, n, K)]
                    ml = (verdict_hmm.rows_from_windows(bf, n, K) + verdict_hmm.rows_from_windows(bc, n, J)) / 2.0
                    rec["score_mean_level"] = [round(float(x), 6) for x in ml]
                fh.write(json.dumps(rec) + "\n")
        jo = os.path.join(sd, "metrics.json")
        subprocess.run([sys.executable, EVALUATOR, "--corpus", args.corpus, "--split", split, "--scores", sp, "--json-out", jo], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        res = json.load(open(jo))["results"]
        for name in ["score_mean_level"] + list(variants):
            r = res[name]
            print("  %s %-16s AP %.4f ROC %.4f within %.4f (missing %d)" % (split, name, r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"], n_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
