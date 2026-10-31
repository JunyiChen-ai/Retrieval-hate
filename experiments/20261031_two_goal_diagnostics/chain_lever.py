"""Goal 2 pre-test: is a per-video chain transition worth a backbone that predicts it? (CPU, r5 checkpoints, no
training, 0 calls.)

r5's chain prior has one learned 2-state transition A shared by all videos; the per-second scores come from the
content logits s_t, the video logit g, the node potentials phi and this A (ctree.marginals). For the r5
test-selected trial of each seed, the test scores at 0 calls are recomputed with:
  - learned: the trial's own A (must reproduce scores_test_fixed0.jsonl);
  - oracle: for each positive test video, A fitted to its own GT sequence (+1 smoothing); the ceiling of a
    per-video transition, an upper bound because it also carries the video's GT hate fraction;
  - ridge_<features>: A of every test video predicted from content by a ridge regression fitted on the validation
    positive videos (targets: logit of the two stay probabilities of their GT-fitted A; alpha by leave-one-out on
    validation). Features: "rep" = attention-pooled and mean representation h = a_out + v_out (5-crop mean);
    "out" = the model's own outputs (g, mean / std / max of s, log T). Validation GT is used only to fit this
    diagnostic predictor (no gradient into any model).
Reported: pooled AP / ROC / within at 0 calls, and the Spearman correlation between predicted and GT-fitted stay
probabilities on the test positive videos. Rule-10 developmental evidence.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import ctree, data as qdata
from qtl.backbones import make_model
from qtl.checkpoint import load_trial
import hier_evidence_common as hc

R5 = "runs/20260929_query_paradigm_r5"
SEEDS = (234, 2025, 3407)


class VideoChain:
    """A fixed (2, 2) or per-video (B, 2, 2) log transition with the learned chain's boundary / form flags."""

    def __init__(self, logA, like):
        self._logA = torch.as_tensor(logA, dtype=torch.float64)
        self.closed, self.zero_inflated, self.normalized = like.closed, like.zero_inflated, like.normalized
        self._logpi = like.logpi().detach().double()

    def logA_for(self, forest):
        return self._logA

    def logpi(self):
        return self._logpi


@torch.no_grad()
def forward(model, store, v):
    """5-crop means of s (T,), g, phi (nodes,), attention-pooled h and mean h (as policy.video_prior_nodes)."""
    T = store.T[v]
    f_v = torch.from_numpy(np.stack([store.visual(v, c) for c in range(5)]))
    f_a = torch.from_numpy(np.repeat(store.at[v][None], 5, axis=0))
    mask = torch.ones(5, T, dtype=torch.bool)
    s, g, _, _, v_out, a_out = model(f_a, f_v, mask)
    phi = model.node_logits(v_out, a_out, ctree.Forest([T] * 5, T)).view(5, -1).mean(0).double().numpy()
    h = a_out + v_out
    w = torch.softmax(model.att(h).squeeze(-1), dim=1)
    pooled = torch.einsum("bt,bth->bh", w, h).mean(0).numpy()
    return s.mean(0).double().numpy(), float(g.mean()), phi, pooled, h.mean((0, 1)).numpy()


def scores0(s, g, phi, chain):
    T = len(s)
    fo = ctree.Forest([T], T)
    _, _, p = ctree.marginals(fo, torch.as_tensor(s)[None], torch.as_tensor([g], dtype=torch.float64),
                              torch.zeros(fo.N, 3, dtype=torch.float64), chain, torch.as_tensor(phi))
    return p[0].numpy()


def gt_logA(y):
    c = np.ones((2, 2))
    np.add.at(c, (y[:-1], y[1:]), 1)
    return np.log(c / c.sum(1, keepdims=True))


def stay_logits(logA):
    p = np.exp(logA)
    return np.array([np.log(p[0, 0] / p[0, 1]), np.log(p[1, 1] / p[1, 0])])


def logA_from_stay(z):
    st = 1 / (1 + np.exp(-z))
    return np.log(np.array([[st[0], 1 - st[0]], [1 - st[1], st[1]]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/chain_lever")
    a = ap.parse_args()
    torch.set_num_threads(4)
    labels, ids, gt_all, _ = hc.load_fixed_cohort(a.corpus)
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    vids = {sp: [v for v in ids[sp] if v in gt_all[sp]] for sp in ("val", "test")}
    store = qdata.Store(a.corpus, vids["val"] + vids["test"], ["bert"])
    gt = {v: np.asarray(gt_all[sp][v]).astype(np.int64) for sp in ("val", "test") for v in vids[sp]}
    hate = {v for v in vids["test"] if labels[v] == 1}
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "seeds": {}}
    for seed in SEEDS:
        study = ROOT / R5 / a.corpus / f"seed{seed}"
        trial = json.loads((study / "study_summary.json").read_text())["best"]["number"]
        _, cfg, model, _, chain = load_trial(study / f"trial{trial}", "cpu", answers, ids, model_factory=make_model)
        out = {v: forward(model, store, v) for v in vids["val"] + vids["test"]}
        saved = {}
        for line in (study / f"trial{trial}" / "scores_test_fixed0.jsonl").read_text().splitlines():
            r = json.loads(line)
            saved[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
        learned = chain.logA_for(None).detach().double().numpy()
        res = {"trial": trial, "learned_A": np.exp(learned).round(3).tolist()}
        sc = {"learned": {}, "oracle": {}}
        for v in vids["test"]:
            s, g, phi = out[v][:3]
            sc["learned"][v] = scores0(s, g, phi, VideoChain(learned, chain))
            sc["oracle"][v] = (scores0(s, g, phi, VideoChain(gt_logA(gt[v]), chain)) if labels[v] == 1
                               else sc["learned"][v])
        res["max_abs_diff_to_saved_fixed0"] = float(max(np.abs(sc["learned"][v] - saved[v]).max() for v in vids["test"]))
        feats = {"rep": lambda v: np.concatenate([out[v][3], out[v][4]]),
                 "out": lambda v: np.array([out[v][1], out[v][0].mean(), out[v][0].std(), out[v][0].max(),
                                            np.log(len(out[v][0]))])}
        vpos = [v for v in vids["val"] if labels[v] == 1]
        tpos = [v for v in vids["test"] if labels[v] == 1]
        for name, f in feats.items():
            X = np.stack([f(v) for v in vpos])
            mu, sd = X.mean(0), X.std(0) + 1e-8
            Y = np.stack([stay_logits(gt_logA(gt[v])) for v in vpos])
            reg = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit((X - mu) / sd, Y)
            pred = {v: reg.predict(((f(v) - mu) / sd)[None])[0] for v in vids["test"]}
            key = f"ridge_{name}"
            sc[key] = {v: scores0(out[v][0], out[v][1], out[v][2], VideoChain(logA_from_stay(pred[v]), chain))
                       for v in vids["test"]}
            ytrue = np.stack([stay_logits(gt_logA(gt[v])) for v in tpos])
            yhat = np.stack([pred[v] for v in tpos])
            res[f"{key}_alpha"] = float(reg.alpha_)
            res[f"{key}_spearman_stay00_stay11"] = [float(spearmanr(yhat[:, k], ytrue[:, k]).correlation) for k in (0, 1)]
        for k, v in sc.items():
            m = hc.frame_metrics(v, {u: gt[u] for u in vids["test"]}, hate)
            res[k] = [float(m[x]) for x in ("pooled_ap", "pooled_roc", "within_roc")]
        report["seeds"][seed] = res
        print(a.corpus, seed, json.dumps(res), flush=True)
    keys = ("learned", "oracle", "ridge_rep", "ridge_out")
    report["mean"] = {k: np.mean([report["seeds"][s][k] for s in SEEDS], 0).tolist() for k in keys}
    out_dir = ROOT / a.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["mean"]))
    print(out_dir / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
