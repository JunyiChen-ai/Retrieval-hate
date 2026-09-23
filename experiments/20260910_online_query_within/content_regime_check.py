"""README section 15 (2026-09-24): offline check of direction 1's premise -- the content model's per-second
score as an observation family of the interval HMM, with and without the video-level reliability mixture
(regimes = 3). No network training, 0 new VLM calls.

Step 1 (--score, GPU): rebuild each seed's `no_verdict` arm model (content + text x_t, no VLM information) and
score the train and test videos; the test scores are checked against the arm's scores_test_coarse4.jsonl.
Step 2 (--fit, CPU): per seed, fit the variants on TRAIN video labels (8-call state: 4 coarse + uniform fine
windows 0/15/7/22), score test with the posterior log-odds through the shared evaluator, and write the group
analysis (VLM-silent videos: fine-rate < .1 over the 30 cached fine verdicts).

    python experiments/20260910_online_query_within/content_regime_check.py --corpus hatemm --seed 234 --score
    python experiments/20260910_online_query_within/content_regime_check.py --corpus hatemm --seed 234 --fit
Output: runs/20260910_online_query_within_it5/content_regime/<corpus>/seed<seed>/
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
sys.path.insert(0, HERE)
from hate_common import data as hdata          # noqa: E402
from macilsd import align                      # noqa: E402
import hier_evidence_common as hc              # noqa: E402
import interval_evidence_hmm as ieh            # noqa: E402
from acquire import bit_reversal_order         # noqa: E402
from text_eval import load_binary, masked      # noqa: E402

K, J = 30, 4
RUNS = os.path.join(ROOT, "runs", "20260910_online_query_within_it5")
BASE = dict(positive_constraint=True, normalized_time=True)
UNI = bit_reversal_order(K)[:4]


def out_dir(corpus, seed):
    d = os.path.join(RUNS, "content_regime", corpus, "seed%d" % seed)
    os.makedirs(d, exist_ok=True)
    return d


def ids_of(corpus, split, B):
    ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, split)) if v in B]
    if split == "test":
        gt = hdata.gt_arrays(corpus, "test")
        ids = [v for v in ids if v in gt]
    return ids


# ------------------------------------------------------------------ step 1: content scores
def score(corpus, seed, device):
    import torch
    from torch.utils.data import DataLoader
    import train as T
    from model import ERCA
    arm = os.path.join(RUNS, "diag", corpus, "seed%d" % seed, "no_verdict")
    cfg = dict(T.DEFAULTS)
    cfg.update(json.load(open(os.path.join(arm, "config_in.json"))))
    a = T.Args(cfg)
    a["text_input"] = False
    a["text_in_ell"] = True
    labels = hdata.load_labels(corpus)
    B = load_binary(corpus)
    train_ids, test_ids = ids_of(corpus, "train", B), ids_of(corpus, "test", B)
    all_ids = train_ids + test_ids
    # the no_verdict model zeroes every evidence column and reads only COL_TEXT (x_t) from the scaffold; the HMM
    # below only fills the zeroed columns (fit on the coarse verdicts, as in the arm: no fine window revealed)
    bin_train = {v: (masked(B[v][0], []), B[v][1]) for v in train_ids}
    hmm, _, _ = hc.fit_hmm(corpus, train_ids, labels, bin_train, model="interval", **BASE)
    centre = hc.text_centre(corpus, train_ids, str(a.text_hate_source))
    text_x = {}
    for v in all_ids:
        arr = hc.load_text_hate(corpus, v, str(a.text_hate_source))
        if arr is not None:
            text_x[v] = hc.text_logit_seconds(arr, centre)
    kw = dict(text=None, text_llr=text_x, evidence="decomp", video_term=True, text_in_ell=True, rho=0.0)
    cache = hc.ScaffoldCache(corpus, all_ids, hc.make_scaffold_fn(hmm, B, "full", 1.0, **kw),
                             masked_fn=hc.make_masked_scaffold_fn(hmm, B, **kw), text_source=str(a.text_feat))
    model = ERCA(a, a.prior_scale, arm="full", no_verdict=True).to(device)
    model.load_state_dict(torch.load(os.path.join(arm, "model.pth"), map_location=device))
    res = {}
    for split, ids in (("train", train_ids), ("test", test_ids)):
        masks = {v: masked(B[v][0], []) for v in ids}
        loader = DataLoader(hc.EvalDataset(corpus, ids, cache, masks=masks), batch_size=1, shuffle=False, num_workers=4)
        sc = hc.score_split(model, loader, device)
        hc.write_scores(os.path.join(out_dir(corpus, seed), "content_scores_%s.jsonl" % split), sc)
        res[split] = sc
    # check against the arm's own test scores (coarse4 = its operating point)
    ref = {}
    for line in open(os.path.join(arm, "scores_test_coarse4.jsonl")):
        d = json.loads(line)
        ref[d["video_id"]] = np.asarray(d["score_av"])
    diff = max(float(np.max(np.abs(np.round(res["test"][v], 6) - ref[v]))) for v in ref)
    print("%s seed %d: test score reconstruction max |diff| = %.2e over %d videos" % (corpus, seed, diff, len(ref)), flush=True)
    assert diff < 1e-4, diff
    return diff


def load_scores(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        out[d["video_id"]] = np.asarray(d["score_av"], dtype=np.float64)
    return out


# ------------------------------------------------------------------ step 2: HMM variants
def content_obs(grid, p, W):
    """Content family as the HMM's text family "asr": 10-level bins of the per-second probability, weight W / T
    per second (the whole video counts W observations)."""
    T = len(p)
    return ieh.text_observation(grid, {"p_asr": p, "w_asr": np.full(T, W / max(T, 1))})


def group_ap(scores, gt, ids):
    from sklearn.metrics import average_precision_score
    s = np.concatenate([scores[v][:len(gt[v])] for v in ids])
    g = np.concatenate([np.asarray(gt[v])[:len(scores[v])] for v in ids])
    return float(average_precision_score(g, s)) if g.min() != g.max() else None


def evaluate(corpus, d, name, scores):
    sp = os.path.join(d, "scores_test_%s.jsonl" % name)
    hc.write_scores(sp, scores)
    r = hc.run_evaluator(corpus, "test", sp, os.path.join(d, "metrics_test_%s.json" % name))["results"]["score_av"]
    return {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"], "within_roc": r["per_video"]["macro_auc"]}


def fit(corpus, seed, variants):
    d = out_dir(corpus, seed)
    labels = hdata.load_labels(corpus)
    B = load_binary(corpus)
    train_ids, test_ids = ids_of(corpus, "train", B), ids_of(corpus, "test", B)
    gt = hdata.gt_arrays(corpus, "test")
    grid = ieh.make_grid(K, J)
    cs = {**load_scores(os.path.join(d, "content_scores_train.jsonl")), **load_scores(os.path.join(d, "content_scores_test.jsonl"))}
    B8 = {v: (masked(B[v][0], UNI), B[v][1]) for v in train_ids + test_ids}
    fine_rate = {v: float(np.mean(B[v][0])) for v in test_ids}
    silent = [v for v in test_ids if fine_rate[v] < 0.1]
    rest = [v for v in test_ids if fine_rate[v] >= 0.1]
    summary = {"corpus": corpus, "seed": seed, "n_train": len(train_ids), "n_test": len(test_ids),
               "n_silent": len(silent), "n_silent_pos": int(sum(labels[v] for v in silent)), "variants": {}}
    # reference: the no-VLM model's own scores
    ref = {v: cs[v] for v in test_ids}
    summary["variants"]["content_only"] = {"test": evaluate(corpus, d, "content_only", ref),
                                          "silent_ap": group_ap(ref, gt, silent), "rest_ap": group_ap(ref, gt, rest)}
    for name, R, W in variants:
        obs = {v: content_obs(grid, cs[v], W) for v in train_ids + test_ids} if W else None
        hmm, _, _ = hc.fit_hmm(corpus, train_ids, labels, B8, model="interval", text_obs=obs, regimes=R,
                               text=bool(W), text_weight=1.0, **BASE)
        hmm.save(os.path.join(d, "hmm_%s.json" % name))
        hist = hmm.fit_history
        sc, rho_silent_pos, rho_silent_neg = {}, [], []
        for v in test_ids:
            bf, bc = B8[v]
            n = len(cs[v])
            xt = (obs or {}).get(v)
            post = hmm._posterior_video(bf, bc, float(n), xt=xt)
            p_s = post["gamma"][:, ieh.S_OF == 1].sum(1)
            lo = np.log(p_s + 1e-6) - np.log(1.0 - p_s + 1e-6)
            sc[v] = ieh.rows_from_segments(lo, hmm.grid, align.second_bounds(n), float(n))
            if v in silent:
                (rho_silent_pos if labels[v] == 1 else rho_silent_neg).append(post["rho"].tolist())
        res = {"R": R, "W": W, "test": evaluate(corpus, d, name, sc),
               "silent_ap": group_ap(sc, gt, silent), "rest_ap": group_ap(sc, gt, rest),
               "loglik_monotone": all(b >= a_ - 1e-6 for a_, b in zip(hist, hist[1:])),
               "params": {k: v for k, v in hmm.params().items() if k in ("q_fine", "r_fine", "q_coarse", "r_coarse",
                                                                        "pi", "q_fine_z", "r_fine_z", "q_coarse_z",
                                                                        "r_coarse_z", "lam01", "lam10")}}
        if R > 1:
            res["rho_silent_pos_mean"] = np.mean(rho_silent_pos, 0).tolist() if rho_silent_pos else None
            res["rho_silent_neg_mean"] = np.mean(rho_silent_neg, 0).tolist() if rho_silent_neg else None
        summary["variants"][name] = res
        print("%s seed %d %-4s R=%d W=%s | AP %.4f ROC %.4f within %.4f | silent AP %s rest AP %s | monotone %s"
              % (corpus, seed, name, R, W, res["test"]["pooled_ap"], res["test"]["pooled_roc"], res["test"]["within_roc"],
                 None if res["silent_ap"] is None else round(res["silent_ap"], 4), round(res["rest_ap"], 4),
                 res["loglik_monotone"]), flush=True)
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=2, default=float)


VARIANTS = [("V0", 1, 0), ("V1", 3, 0), ("V2", 1, 30), ("V3", 3, 30), ("V2b", 1, 8), ("V3b", 3, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--variants", default=",".join(v[0] for v in VARIANTS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.score:
        score(a.corpus, a.seed, a.device)
    if a.fit:
        want = a.variants.split(",")
        fit(a.corpus, a.seed, [v for v in VARIANTS if v[0] in want])


if __name__ == "__main__":
    main()
