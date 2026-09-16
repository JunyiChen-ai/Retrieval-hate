"""Module-1 iteration 5 gate T5 (no training): decomposed evidence E_t = ell_fine + x_t + v
used directly as the per-second score on test, with the fine emissions tempered by
hc.fine_kappa (rho = ICC of the fine-verdict errors on the negative training videos)
versus untempered (rho = 0), at 4 coarse + {0, 4 uniform, 8 uniform, all 30} fine
verdicts. Output runs/20260910_online_query_within_it5/hmm_only/<corpus>/summary.json.
"""
import json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines")); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
from hate_common import data as hdata           # noqa: E402
import hier_evidence_common as hc               # noqa: E402
import interval_evidence_hmm as ieh             # noqa: E402
from macilsd import align                       # noqa: E402
from hmm_eval import load_binary, EVALUATOR, K  # noqa: E402
from acquire import bit_reversal_order          # noqa: E402


def main(corpus, out_root=os.path.join(ROOT, "runs", "20260910_online_query_within_it5", "hmm_only")):
    B = load_binary(corpus)
    labels = hdata.load_labels(corpus)
    train_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "train")) if v in B]
    test_gt = hdata.gt_arrays(corpus, "test")
    out_dir = os.path.join(out_root, corpus); os.makedirs(out_dir, exist_ok=True)
    hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, B, model="interval", positive_constraint=True, normalized_time=True)
    rho = hc.fine_verdict_icc(B, train_ids, labels)
    centre = hc.text_centre(corpus, train_ids)
    text_x = {v: hc.text_logit_seconds(hc.load_text_hate(corpus, v), centre) for v in test_gt if hc.load_text_hate(corpus, v) is not None}
    uni = bit_reversal_order(K)
    res = {"rho": rho, "n_neg_train": n_neg, "centre": centre, "hmm": hmm.params()}
    for tag, r in (("rho0", 0.0), ("icc", rho)):
        for nm, n_fine in (("coarse4", 0), ("uni4", 4), ("uni8", 8), ("all30", 30)):
            sp = os.path.join(out_dir, "%s_%s_scores.jsonl" % (tag, nm))
            with open(sp, "w") as fh:
                for vid in sorted(test_gt):
                    n = len(test_gt[vid])
                    if vid not in B:
                        sc = [0.0] * n
                    else:
                        bf, bc = B[vid]
                        mb = np.full(K, ieh.MISSING, dtype=int)
                        for w in (range(K) if n_fine == 30 else uni[:n_fine]):
                            mb[w] = int(bf[w])
                        ell, v = hc.decomposed_logodds(hmm, mb, bc, float(n), video_term=True, rho=r)
                        sc = ieh.rows_from_segments(np.asarray(ell, np.float32), hmm.grid, align.second_bounds(n), float(n))
                        if vid in text_x:
                            sc = sc + text_x[vid][:n]
                    fh.write(json.dumps({"video_id": vid, "n_frames": n, "score_av": [round(float(x), 6) for x in sc]}) + "\n")
            jo = os.path.join(out_dir, "%s_%s_metrics.json" % (tag, nm))
            subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus, "--split", "test", "--scores", sp, "--json-out", jo],
                           check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            m = json.load(open(jo))["results"]["score_av"]
            res["%s_%s" % (tag, nm)] = dict(pooled_ap=m["pr_auc"], pooled_roc=m["roc_auc"], within_roc=m["per_video"]["macro_auc"])
            print("%s rho=%s %-8s AP %.4f ROC %.4f within %.4f" % (corpus, tag, nm, m["pr_auc"], m["roc_auc"], m["per_video"]["macro_auc"]), flush=True)
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    for c in sys.argv[1:] or ("hatemm", "hateclipseg"):
        main(c)
