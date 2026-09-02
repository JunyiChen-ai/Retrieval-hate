"""Score the frozen-VLM verdicts alone through the shared evaluator.

Reference number for README section 1.4 (input signal on its own, no
training). Writes runs/<exp>/verdict_only/<corpus>/{scores.jsonl,metrics.json}.
Default is K30 only (branch score_verdict); --k 30 4 also writes score_k30,
score_k4 and their mean score_k30_k4 (README section 9 analysis 2).

    python experiments/20260902_verdict_boundary_contrast_mil/verdict_only_eval.py --corpus hateclipseg
    python experiments/20260902_verdict_boundary_contrast_mil/verdict_only_eval.py --corpus hatemm --k 30 4 --out-root runs/<exp>/verdict_only_gran
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from hate_common import data as hdata      # noqa: E402
import vlm_verdict                         # noqa: E402

EVALUATOR = os.path.join(REPO_ROOT, "scripts", "reproduction_baselines",
                         "eval_baseline_scores.py")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-root", default=os.path.join(
        REPO_ROOT, "runs", "20260902_verdict_boundary_contrast_mil",
        "verdict_only"))
    ap.add_argument("--k", type=int, nargs="+", default=[30])
    args = ap.parse_args(argv)
    gt = hdata.gt_arrays(args.corpus, args.split)
    per_k = {k: vlm_verdict.load_verdicts(args.corpus, k=k, tag="qwen")
             for k in args.k}
    verdicts = per_k[args.k[0]]
    out_dir = os.path.join(args.out_root, args.corpus, args.split)
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")
    n_missing = 0
    with open(scores_path, "w") as fh:
        for vid in sorted(gt):
            n = int(len(gt[vid]))
            rb = np.stack([np.arange(n), np.arange(n) + 1], 1)
            rows = {}
            for k in args.k:
                sc = per_k[k].get(vid)
                if sc is None:
                    if k == args.k[0]:
                        n_missing += 1
                    rows[k] = np.zeros(n)
                else:
                    rows[k] = vlm_verdict.verdict_rows(sc, rb, n) / 3.0
            rec = {"video_id": vid, "n_frames": n,
                   "score_verdict": [round(float(x), 6) for x in rows[args.k[0]]]}
            if len(args.k) > 1:
                for k in args.k:
                    rec["score_k%d" % k] = [round(float(x), 6) for x in rows[k]]
                mean = np.mean([rows[k] for k in args.k], 0)
                rec["score_" + "_".join("k%d" % k for k in args.k)] = [
                    round(float(x), 6) for x in mean]
            fh.write(json.dumps(rec) + "\n")
    json_out = os.path.join(out_dir, "metrics.json")
    subprocess.run([sys.executable, EVALUATOR, "--corpus", args.corpus,
                    "--split", args.split, "--scores", scores_path,
                    "--json-out", json_out], check=True, cwd=REPO_ROOT)
    with open(json_out) as fh:
        results = json.load(fh)["results"]
    for name, r in results.items():
        print("%s/%s %s: AP %.4f ROC %.4f within %.4f (videos %d, "
              "missing verdict %d)" % (args.corpus, args.split, name,
                                       r["pr_auc"], r["roc_auc"],
                                       r["per_video"]["macro_auc"],
                                       r["n_videos"], n_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
