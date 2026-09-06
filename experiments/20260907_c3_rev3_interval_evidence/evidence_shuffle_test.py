"""Mechanism check (no training): does the evidence-routed attention use the
TIME CORRESPONDENCE of the evidence, or only its video-level summary?

Loads a trained trial (model.pth, config.json, hmm_params.json), rebuilds the
test scaffold exactly as train.py does, and scores the test split (a) as is,
(b) with the evidence codes e_t permuted in time inside each video before
they enter the attention (q/k encoding and routing term). The video-level
calibration c = Linear(mean_t e_t) is permutation-invariant and the prior
alpha * ell_t / L reads the unpermuted scaffold column, so only the
where-to-aggregate path is disturbed. If pooled AP/ROC do not fall, the
routing does not use the time correspondence of the evidence.

Also reports the between-video share of the score variance (baseline).

    python experiments/20260907_c3_rev3_interval_evidence/evidence_shuffle_test.py \
        --corpus hatemm --trial-dir runs/20260907_c3_rev3_interval_evidence/hatemm/seed234/trial<k> \
        --out runs/20260907_c3_rev3_interval_evidence/mechanism/hatemm/seed234
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from hate_common import data as hdata          # noqa: E402
import hier_evidence_common as hc              # noqa: E402
import interval_evidence_hmm as ieh            # noqa: E402
import verdict_hmm                             # noqa: E402
import vlm_verdict                             # noqa: E402
from model import ERCA                         # noqa: E402
from train import DEFAULTS, STRUCT_ARMS, Args  # noqa: E402

K_FINE, J_COARSE = hc.K_FINE, hc.J_COARSE


def load_hmm(path):
    d = json.load(open(path))
    if d.get("model") == "interval":
        return ieh.IntervalEvidenceHMM.from_params(d)
    return verdict_hmm.HierEvidenceHMM.from_params(d)


def variance_decomposition(scores):
    allv = np.concatenate(list(scores.values()))
    n = {v: len(s) for v, s in scores.items()}
    mu = allv.mean()
    between = sum(n[v] * (scores[v].mean() - mu) ** 2 for v in scores) / len(allv)
    total = allv.var()
    return {"between_video_share": float(between / max(total, 1e-12)), "total_var": float(total)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--trial-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    cfg_rec = json.load(open(os.path.join(args.trial_dir, "config.json")))
    cfg = dict(DEFAULTS)
    cfg.update(cfg_rec["hparams"])
    ablation = cfg_rec["ablation"]
    a = Args(cfg)
    corpus = args.corpus
    labels = hdata.load_labels(corpus)
    test_gt = hdata.gt_arrays(corpus, "test")
    test_ids = [v for v in hc.usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (K_FINE, J_COARSE)}
    binary = {v: (verdict_hmm.binarize(V[K_FINE][v]), verdict_hmm.binarize(V[J_COARSE][v]))
              for v in V[K_FINE] if v in V[J_COARSE]}
    hmm = load_hmm(os.path.join(args.trial_dir, "hmm_params.json"))
    cache = hc.ScaffoldCache(corpus, test_ids, hc.make_scaffold_fn(hmm, binary, ablation, a.w_fine))
    loader = DataLoader(hc.EvalDataset(corpus, test_ids, cache), batch_size=1,
                        shuffle=False, num_workers=args.num_workers)
    arm = ablation if ablation in STRUCT_ARMS else "full"
    prior_scale = 0.0 if ablation in ("no_prior", "no_verdict") else a.prior_scale
    model = ERCA(a, prior_scale, arm=arm, no_verdict=(ablation == "no_verdict")).to(args.device)
    model.load_state_dict(torch.load(os.path.join(args.trial_dir, "model.pth"), map_location=args.device))
    model.eval()
    assert model.enc is not None, "the avce arm has no evidence encoder to permute"

    results = {}

    def score_and_eval(tag):
        scores = hc.score_split(model, loader, args.device)
        sp = os.path.join(args.out, "scores_%s.jsonl" % tag)
        hc.write_scores(sp, scores)
        ev = hc.run_evaluator(corpus, "test", sp, os.path.join(args.out, "metrics_%s.json" % tag))
        r = ev["results"]["score_av"]
        results[tag] = {"pooled_ap": r["pr_auc"], "pooled_roc": r["roc_auc"],
                        "within_roc": r["per_video"]["macro_auc"]}
        print("%-10s AP %.4f ROC %.4f within %.4f" % (tag, r["pr_auc"], r["roc_auc"],
                                                     r["per_video"]["macro_auc"]), flush=True)
        return scores

    base_scores = score_and_eval("baseline")
    results["variance"] = variance_decomposition(base_scores)
    orig_forward = model.enc.forward
    for p in range(args.n_perm):
        gen = torch.Generator().manual_seed(1000 + p)

        def permuted(evid, _gen=gen):
            e = orig_forward(evid)                      # (crops, T, hid); test = full sequence, no padding
            T = e.shape[1]
            perm = torch.randperm(T, generator=_gen).to(e.device)
            return e[:, perm, :]
        model.enc.forward = permuted
        score_and_eval("perm%d" % p)
    model.enc.forward = orig_forward
    perms = [results["perm%d" % p] for p in range(args.n_perm)]
    results["perm_mean"] = {k: float(np.mean([r[k] for r in perms])) for k in perms[0]}
    results["drop_baseline_minus_perm"] = {k: results["baseline"][k] - results["perm_mean"][k]
                                           for k in perms[0]}
    results["trial_dir"] = args.trial_dir
    results["ablation"] = ablation
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("drop (baseline - permuted mean):", json.dumps(results["drop_baseline_minus_perm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
