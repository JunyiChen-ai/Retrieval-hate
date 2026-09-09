"""Training-free check of the text-evidence interval HMM (module-1 iteration 2,
README section 7, gate T1): fit on TRAIN video labels only (34 verdicts + text),
then on test score the HMM posterior alone through the shared evaluator with the
test-time evidence states: 4 coarse blocks + m uniform fine windows (m = 0, 4, 8)
and all 34 verdicts; variants text off / text on. Also the masked-8 predictive
log-loss of unobserved fine verdicts (as hmm_eval.py).

Gate T1: with 4 coarse + 4 uniform fine windows (the 8-call state), within on test
HateMM >= .62 and HCS >= .56, and pooled AP / ROC not below the no-text variant - .005.

    python experiments/20260910_online_query_within/text_eval.py --corpus hatemm
Writes runs/20260910_online_query_within_it1/hmm_text/<corpus>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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
from acquire import bit_reversal_order         # noqa: E402

K, J = vlm_verdict.GRANULARITIES
BASE = dict(positive_constraint=True, normalized_time=True, regimes=1)
N_MASK = 8


def load_binary(corpus):
    V = {K: vlm_verdict.load_verdicts(corpus, k=K, tag="qwen"),
         J: vlm_verdict.load_verdicts(corpus, k=J, tag="qwen")}
    return {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
            for v in V[K] if v in V[J]}


def masked(bf, keep):
    out = np.full(len(bf), ieh.MISSING, dtype=int)
    for w in keep:
        out[w] = int(bf[w])
    return out


def evaluate(corpus, out_dir, name, scores):
    sp = os.path.join(out_dir, "scores_test_%s.jsonl" % name)
    hc.write_scores(sp, scores)
    r = hc.run_evaluator(corpus, "test", sp, os.path.join(out_dir, "metrics_test_%s.json" % name))["results"]["score_av"]
    return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"], "within_roc": r["per_video"]["macro_auc"], "n_videos": r["n_videos"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "20260910_online_query_within_it1", "hmm_text"))
    ap.add_argument("--text-weights", default="1.0")
    a = ap.parse_args(argv)
    corpus = a.corpus
    out_dir = os.path.join(a.out_root, corpus)
    os.makedirs(out_dir, exist_ok=True)
    labels = hdata.load_labels(corpus)
    B = load_binary(corpus)
    train_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "train")) if v in B]
    test_gt = hdata.gt_arrays(corpus, "test")
    test_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt and v in B]
    grid = ieh.make_grid(K, J)
    text = hc.text_observations(corpus, train_ids + test_ids, grid)
    dur = {v: float(hc.video_duration(corpus, v)) for v in test_ids}
    uni = bit_reversal_order(K)
    summary = {"corpus": corpus, "n_train": len(train_ids), "n_test": len(test_ids),
               "text_coverage_test": float(np.mean([v in text for v in test_ids])), "variants": {}}
    variants = [("notext", dict(text=False), None)] + [("text_w%g" % float(w), dict(text=True, text_weight=float(w)), text)
                                                       for w in a.text_weights.split(",")]
    for name, opts, tx in variants:
        hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, B, model="interval", text_obs=tx, **BASE, **opts)
        hmm.save(os.path.join(out_dir, name + "_params.json"))
        res = {"params": {k: v for k, v in hmm.params().items() if k not in ("fit_loglik",)},
               "fit_loglik_last": hmm.fit_history[-1] if hmm.fit_history else None,
               "loglik_monotone": all(b >= a_ - 1e-6 for a_, b in zip(hmm.fit_history, hmm.fit_history[1:]))}
        for m in (0, 4, 8, 30):
            sc = {}
            for v in test_ids:
                bf, bc = B[v]
                n = int(dur[v])
                lo = hmm.posterior_log_odds(masked(bf, uni[:m]), bc, float(n), xt=(tx or {}).get(v))
                sc[v] = ieh.rows_from_segments(lo, hmm.grid, align.second_bounds(n), float(n))
            res["coarse4_fine%d" % m] = evaluate(corpus, out_dir, "%s_fine%d" % (name, m), sc)
            print(corpus, name, "fine%d" % m, res["coarse4_fine%d" % m], flush=True)
        # masked-8 predictive check
        rng = np.random.RandomState(0)
        ll, br, n = 0.0, 0.0, 0
        for v in test_ids:
            bf, bc = B[v]
            hide = rng.choice(K, N_MASK, replace=False)
            mb = bf.copy()
            mb[hide] = ieh.MISSING
            pred = np.clip(hmm.infer(mb, bc, dur[v], xt=(tx or {}).get(v))["pred_fine"], 1e-6, 1 - 1e-6)
            y = bf[hide].astype(float)
            p = pred[hide]
            ll += float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
            br += float(np.sum((p - y) ** 2))
            n += len(hide)
        res["masked8"] = {"logloss": ll / n, "brier": br / n, "n": n}
        print(corpus, name, "masked8", res["masked8"], flush=True)
        summary["variants"][name] = res
    ref, tv = summary["variants"]["notext"], summary["variants"].get("text_w1")
    if tv is not None:
        c8r, c8t = ref["coarse4_fine4"], tv["coarse4_fine4"]
        floor = {"hatemm": 0.62, "hateclipseg": 0.56}[corpus]
        summary["gate_T1"] = {"within_floor": floor, "within_text": c8t["within_roc"], "within_notext": c8r["within_roc"],
                              "pooled_ok": bool(c8t["pooled_ap"] >= c8r["pooled_ap"] - 0.005 and c8t["pooled_roc"] >= c8r["pooled_roc"] - 0.005),
                              "within_ok": bool(c8t["within_roc"] >= floor)}
        summary["gate_T1"]["pass"] = summary["gate_T1"]["pooled_ok"] and summary["gate_T1"]["within_ok"]
        print(corpus, "gate T1", summary["gate_T1"])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
