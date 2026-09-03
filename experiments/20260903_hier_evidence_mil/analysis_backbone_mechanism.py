"""What does the trained revision-1 backbone actually compute?  (analysis, no training)

Loads a search trial's checkpoint and answers, on TEST (developmental evidence):
  1. test-time occlusion: zero one input group at inference (trained weights fixed)
     and re-score -> which inputs the trained network relies on;
  2. verdict-cell analysis: mean content logit z_t and GT rate per
     (b_fine, b_coarse) cell, vs the HMM posterior -> did the network learn a
     reliability correction that the HMM lacks;
  3. verdict-context probe: ridge regression of z_t on hand-made local verdict
     context features (own verdict, neighbours, run length, block agreement,
     video-level fire rate, HMM posterior) -> how much of z_t is a function of
     the verdict sequence alone, and which context features carry it;
  4. within-video content contribution: z_t residual after the verdict probe
     vs the content-only (no_verdict) model's score.

    python experiments/20260903_hier_evidence_mil/analysis_backbone_mechanism.py \
        --corpus hatemm --trial runs/20260903_hier_evidence_mil/hatemm/seed234/trial3
Outputs runs/20260903_hier_evidence_mil/analysis/backbone_mechanism/<corpus>_seed<seed>.{txt,json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from hate_common import data as hdata          # noqa: E402
from macilsd import align                       # noqa: E402
from torch.utils.data import DataLoader         # noqa: E402
import vlm_verdict                              # noqa: E402
import verdict_hmm                              # noqa: E402
import dataset as ds                            # noqa: E402
import train as tr                              # noqa: E402

K, J = tr.K_FINE, tr.J_COARSE


def load_model(trial_dir, device):
    cfg = json.load(open(os.path.join(trial_dir, "config.json")))
    a = tr.Args(cfg["hparams"] if "hparams" in cfg else cfg)
    a["a_feature_size"] = ds.A_EXT_DIM
    a["v_feature_size"] = align.V_DIM
    model = tr.Candidate(a, a.prior_scale).to(device)
    model.load_state_dict(torch.load(os.path.join(trial_dir, "model.pth"),
                                     map_location=device))
    model.eval()
    return model, a


def forward_rows(model, f_a, f_v, occlude=None, prior=True):
    """(z rows, av rows) five-crop mean of logits; occlude: list of (lo, hi) f_a
    column ranges or 'visual'."""
    f_a = f_a.clone()
    f_v = f_v.clone()
    if occlude:
        for o in occlude:
            if o == "visual":
                f_v.zero_()
            else:
                f_a[..., o[0]:o[1]] = 0.0
    ps = model.prior_scale
    if not prior:
        model.prior_scale = 0.0
    with torch.no_grad():
        _, _, _, av_log, _, _ = model(f_a, f_v, seq_len=None)
    model.prior_scale = ps
    z = model.last_content_logit.squeeze(-1).mean(0).cpu().numpy()
    av = av_log.squeeze(-1).mean(0).cpu().numpy()
    return z, av


def ridge_r2(X, y, lam=1.0):
    Xm, ym = X.mean(0), y.mean()
    Xc, yc = X - Xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    pred = Xc @ w
    return 1.0 - ((yc - pred) ** 2).sum() / (yc ** 2).sum(), w


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--content-only-scores", default=None,
                    help="scores_test.jsonl of the no_verdict ablation (same seed)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    corpus = args.corpus
    seed = int(json.load(open(os.path.join(args.trial, "config.json"))).get("seed", 234))
    out_dir = os.path.join(ROOT, "runs", "20260903_hier_evidence_mil", "analysis",
                           "backbone_mechanism")
    os.makedirs(out_dir, exist_ok=True)
    tag = "%s_seed%d" % (corpus, seed)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    model, a = load_model(args.trial, args.device)
    hmm = verdict_hmm.HierEvidenceHMM.load(os.path.join(args.trial, "hmm_params.json"))
    labels = hdata.load_labels(corpus)
    test_gt = hdata.gt_arrays(corpus, "test")
    test_ids = [v for v in tr.usable(corpus, hdata.load_split(corpus, "test")) if v in test_gt]
    hate_ids = {v for v, l in labels.items() if l == 1}
    V = {k: vlm_verdict.load_verdicts(corpus, k=k, tag="qwen") for k in (K, J)}
    binary = {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
              for v in V[K] if v in V[J]}
    cache = ds.ScaffoldCache(corpus, test_ids,
                             tr.make_scaffold_fn(hmm, binary, "full", a.w_fine))
    loader = DataLoader(ds.EvalDataset(corpus, test_ids, cache), batch_size=1,
                        shuffle=False, num_workers=2)
    say("%s trial %s | prior_scale %.3f w_fine %.3f lambda_block %.3f | %d test videos"
        % (corpus, args.trial, a.prior_scale, a.w_fine, a.lambda_block, len(test_ids)))

    # ---------------------------------------------------------------- 1. occlusion
    A0, T0, S0 = 0, align.A_DIM, ds.SCAF_OFFSET
    groups = {
        "full": [],
        "occl_visual": ["visual"],
        "occl_audio": [(A0, T0)],
        "occl_text": [(T0, S0)],
        "occl_all_content": ["visual", (A0, S0)],
        "occl_ell": [(S0 + ds.COL_ELL, S0 + ds.COL_ELL + 1)],
        "occl_ps": [(S0 + ds.COL_PS, S0 + ds.COL_PS + 1)],
        "occl_hmm_cols": [(S0 + ds.COL_ELL, S0 + ds.COL_PS + 1)],
        "occl_bfine": [(S0 + ds.COL_BF, S0 + ds.COL_BF + 1)],
        "occl_bcoarse": [(S0 + ds.COL_BC, S0 + ds.COL_BC + 1)],
        "occl_raw_cols": [(S0 + ds.COL_BF, S0 + ds.COL_BC + 1)],
        "occl_all_verdict_cols": [(S0, S0 + ds.N_INPUT_SCAF)],
    }
    batches = []
    for f_v, f_a, index_map, n_seconds, vid in loader:
        batches.append((vid[0], f_v[0].to(args.device), f_a[0].to(args.device),
                        index_map[0].numpy(), int(n_seconds)))
    results = {}
    z_full, av_full = {}, {}
    for name, occ in groups.items():
        for prior in (True, False):
            key = name + ("" if prior else "+no_prior")
            if not prior and name not in ("full", "occl_all_content", "occl_all_verdict_cols"):
                continue
            sc, scz = {}, {}
            for vid, f_v, f_a, imap, n in batches:
                z, av = forward_rows(model, f_a, f_v, occ, prior)
                sc[vid] = 1.0 / (1.0 + np.exp(-av))[imap][:n]
                scz[vid] = z[imap][:n]
                if key == "full":
                    z_full[vid], av_full[vid] = z[imap][:n], av[imap][:n]
            m = tr.frame_metrics(sc, test_gt, hate_ids)
            mz = tr.frame_metrics({v: 1 / (1 + np.exp(-scz[v])) for v in scz}, test_gt, hate_ids)
            results[key] = {"final": m, "content_logit_only": mz}
    say("\n1. test-time occlusion (trained weights fixed). final = sigmoid(z + prior); z = content logit alone")
    say("   %-28s %-24s %-24s" % ("variant", "final AP/ROC/within", "z-only AP/ROC/within"))
    for key, r in results.items():
        m, mz = r["final"], r["content_logit_only"]
        say("   %-28s %.3f/%.3f/%.3f        %.3f/%.3f/%.3f" % (
            key, m["pooled_ap"], m["pooled_roc"], m["within_roc"],
            mz["pooled_ap"], mz["pooled_roc"], mz["within_roc"]))

    # ------------------------------------------------- 2. verdict cells on test
    rows = []   # per second: bf, bc, ell, z, gt, vid, window index, block
    for vid in test_ids:
        f_a, n, snip = cache[vid]
        bf, bc = binary[vid]
        lo = hmm.posterior_log_odds(bf, bc, a.w_fine)
        widx = np.clip(((np.arange(n) + 0.5) * K / max(n, 1)).astype(int), 0, K - 1)
        gt = np.asarray(test_gt[vid])[:n]
        for t in range(n):
            w = widx[t]
            rows.append((bf[w], bc[hmm.block[w]], lo[w], z_full[vid][t], gt[t], vid, w))
    R = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], float)
    say("\n2. verdict cells on test seconds: GT rate, mean HMM log-odds, mean z (content logit)")
    cells = {}
    for bf_ in (0, 1):
        for bc_ in (0, 1):
            sel = (R[:, 0] == bf_) & (R[:, 1] == bc_)
            if sel.sum() == 0:
                continue
            cells["bf%d_bc%d" % (bf_, bc_)] = {
                "n": int(sel.sum()), "gt_rate": float(R[sel, 4].mean()),
                "hmm_logodds": float(R[sel, 2].mean()), "z_mean": float(R[sel, 3].mean())}
            say("   b_fine=%d b_coarse=%d n=%6d  GT %.3f  HMM ell %+.2f  z %+.2f"
                % (bf_, bc_, sel.sum(), R[sel, 4].mean(), R[sel, 2].mean(), R[sel, 3].mean()))
    # rank agreement of cell means with GT: does z order the cells like GT does?
    order_gt = sorted(cells, key=lambda c: cells[c]["gt_rate"])
    order_z = sorted(cells, key=lambda c: cells[c]["z_mean"])
    order_h = sorted(cells, key=lambda c: cells[c]["hmm_logodds"])
    say("   cell order by GT: %s | by z: %s | by HMM: %s" % (order_gt, order_z, order_h))

    # ------------------------------------------- 3. verdict-context probe of z
    feats, names = [], ["b_f", "b_f-1", "b_f+1", "b_f-2", "b_f+2", "b_c", "b_c_prev", "b_c_next",
                        "run_len", "n_fired_in_block", "frac_fired_video", "frac_blocks_fired",
                        "hmm_ell", "pos"]
    for vid in test_ids:
        f_a, n, snip = cache[vid]
        bf, bc = binary[vid]
        lo = hmm.posterior_log_odds(bf, bc, a.w_fine)
        widx = np.clip(((np.arange(n) + 0.5) * K / max(n, 1)).astype(int), 0, K - 1)
        # run length of the fine verdict run containing window w
        run = np.zeros(K)
        i = 0
        while i < K:
            j = i
            while j + 1 < K and bf[j + 1] == bf[i]:
                j += 1
            run[i:j + 1] = j - i + 1
            i = j + 1
        for t in range(n):
            w = widx[t]
            b = hmm.block[w]
            g = lambda k: bf[k] if 0 <= k < K else 0  # noqa: E731
            feats.append([bf[w], g(w - 1), g(w + 1), g(w - 2), g(w + 2), bc[b],
                          bc[b - 1] if b > 0 else 0, bc[b + 1] if b < J - 1 else 0,
                          run[w], bf[hmm.block == b].sum(), bf.mean(), bc.mean(),
                          lo[w], w / K])
    X = np.array(feats, float)
    y = R[:, 3]
    r2_all, w_all = ridge_r2(X, y)
    r2_own, _ = ridge_r2(X[:, [0, 5]], y)
    r2_hmm, _ = ridge_r2(X[:, [12]], y)
    r2_ctx, _ = ridge_r2(X[:, :12], y)
    say("\n3. verdict-context probe: ridge R^2 of z_t on verdict features (test seconds, n=%d)" % len(y))
    say("   own verdicts (b_f, b_c) only R^2 = %.3f | HMM log-odds only R^2 = %.3f | "
        "local context (12 feats, no HMM) R^2 = %.3f | all 14 R^2 = %.3f"
        % (r2_own, r2_hmm, r2_ctx, r2_all))
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, ws = ridge_r2(Xs, y)
    top = np.argsort(-np.abs(ws))[:8]
    say("   standardised ridge weights (largest |w|): " +
        ", ".join("%s %+.2f" % (names[i], ws[i]) for i in top))
    # variance decomposition: within-video vs between-video share of z, and how much of within is verdict-explained
    vids = np.array([r[5] for r in rows])
    zm = {v: y[vids == v].mean() for v in set(vids)}
    y_within = y - np.array([zm[v] for v in vids])
    Xw = X - np.array([X[vids == v].mean(0) for v in vids])
    r2_within, _ = ridge_r2(Xw, y_within)
    say("   z variance: between-video share %.2f, within-video share %.2f; within-video z explained by "
        "verdict context R^2 = %.3f" % (1 - y_within.var() / y.var(), y_within.var() / y.var(), r2_within))

    # -------------------------------- 4. content contribution within video
    if args.content_only_scores and os.path.exists(args.content_only_scores):
        co = {}
        for line in open(args.content_only_scores):
            r = json.loads(line)
            co[r["video_id"]] = np.asarray(r["score_av"], float)
        resid = y - (Xs @ ws + y.mean())
        c = []
        for vid in test_ids:
            n = cache[vid][1]
            c.extend(list(co[vid][:n]))
        c = np.array(c)
        cw = c - np.array([c[vids == v].mean() for v in vids])
        rw = resid - np.array([resid[vids == v].mean() for v in vids])
        corr = np.corrcoef(cw, rw)[0, 1]
        gt = R[:, 4]
        say("\n4. within-video content contribution: corr(z residual after verdict probe, content-only "
            "model score) = %.3f (within-video, mean-removed)" % corr)
        # does the residual carry GT information within video?
        from sklearn.metrics import roc_auc_score
        aucs, aucs_c = [], []
        for v in set(vids):
            sel = vids == v
            if gt[sel].min() == gt[sel].max():
                continue
            aucs.append(roc_auc_score(gt[sel], resid[sel]))
            aucs_c.append(roc_auc_score(gt[sel], c[sel]))
        say("   within-video AUC of z residual (verdict part removed) = %.3f; of content-only model = %.3f; "
            "n videos %d" % (np.mean(aucs), np.mean(aucs_c), len(aucs)))

    with open(os.path.join(out_dir, tag + ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, tag + ".json"), "w") as fh:
        json.dump({"occlusion": results, "cells": cells,
                   "probe": {"r2_own": r2_own, "r2_hmm": r2_hmm, "r2_ctx": r2_ctx,
                             "r2_all": r2_all, "r2_within": r2_within,
                             "weights_std": dict(zip(names, map(float, ws)))}},
                  fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
