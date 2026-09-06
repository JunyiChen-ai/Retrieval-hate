"""Adaptive VLM querying, replayed on the cached verdicts (0 new VLM calls).

The interval evidence HMM infers with missing verdicts, so a query policy can
be simulated by revealing cached verdicts one at a time: start from the 4
coarse blocks, then pick fine windows until a call budget is spent, and score
every test video by the HMM posterior alone (no localizer; training-free).

Policies (next fine window to reveal):
  uniform       fixed evenly-spread order (bit-reversal of 0..29)
  random        uniform random order (seed)
  coarse_pos    windows inside positive coarse blocks first (evenly spread), then the rest
  entropy       window whose segments have the largest current P(s)(1-P(s))
  localization  largest expected reduction of the within-video ranking
                uncertainty U = sum_g P(s_g)(1-P(s_g)), expectation over the
                two possible verdicts with the model's predictive probability
Budgets: total calls 4, 8, 12, 16, 22, 34 (34 = everything observed).

    python experiments/20260907_c3_rev3_interval_evidence/adaptive_query_replay.py --corpus hatemm

Writes runs/20260907_c3_rev3_interval_evidence/adaptive_replay/<corpus>/
    <policy>_b<budget>/{scores.jsonl,metrics.json}, summary.json, curve.png
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
BUDGETS = (4, 8, 12, 16, 22, 34)
POLICIES = ("uniform", "random", "coarse_pos", "entropy", "localization")


def bit_reversal_order(k):
    """Evenly spread refinement order of 0..k-1 (van der Corput on k)."""
    idx = []
    n = 1
    while n < k:
        n *= 2
    for i in range(n):
        r, x = 0, i
        for _ in range(n.bit_length() - 1):
            r = (r << 1) | (x & 1)
            x >>= 1
        w = int(r * k / n)
        if w not in idx and w < k:
            idx.append(w)
    for w in range(k):
        if w not in idx:
            idx.append(w)
    return idx


class Replay:
    def __init__(self, hmm, duration):
        self.hmm = hmm
        self.dur = duration
        self.Tm = hmm._transitions(duration)
        self.grid = hmm.grid
        self.fine_end = np.where(self.grid["fine_end"])[0]
        self.fine_end_of = {int(self.grid["fine_of"][g]): g for g in self.fine_end}

    def gamma(self, bf, bc):
        e = self.hmm._emissions(bf, bc)
        g, _, _ = self.hmm._fb(self.Tm, e)
        return g

    @staticmethod
    def p_s(gamma):
        return gamma[:, ieh.S_OF == 1].sum(1)

    def uncertainty(self, gamma):
        p = self.p_s(gamma)
        return float(np.sum(p * (1.0 - p)))

    def window_segments(self, w):
        return np.where(self.grid["fine_of"] == w)[0]

    def next_window(self, policy, bf, bc, unobserved, rng, order):
        if policy in ("uniform", "random"):
            for w in order:
                if w in unobserved:
                    return w
        if policy == "coarse_pos":
            pos_blocks = {j for j in range(J) if bc[j] == 1}
            for w in order:
                if w in unobserved and int(self.grid["coarse_of"][self.window_segments(w)[0]]) in pos_blocks:
                    return w
            for w in order:
                if w in unobserved:
                    return w
        gamma = self.gamma(bf, bc)
        p = self.p_s(gamma)
        if policy == "entropy":
            best, bw = -1.0, None
            for w in unobserved:
                u = float(np.sum(p[self.window_segments(w)] * (1.0 - p[self.window_segments(w)])))
                if u > best:
                    best, bw = u, w
            return bw
        if policy == "localization":
            u_now = float(np.sum(p * (1.0 - p)))
            best, bw = -np.inf, None
            for w in unobserved:
                g_end = self.fine_end_of[w]
                p_h1 = float(gamma[g_end, ieh.HF_OF == 1].sum())
                p_b1 = self.hmm.q_f * p_h1 + self.hmm.r_f * (1.0 - p_h1)
                exp_u = 0.0
                for b, pb in ((1, p_b1), (0, 1.0 - p_b1)):
                    if pb < 1e-9:
                        continue
                    bf2 = bf.copy()
                    bf2[w] = b
                    exp_u += pb * self.uncertainty(self.gamma(bf2, bc))
                gain = u_now - exp_u
                if gain > best:
                    best, bw = gain, w
            return bw
        raise ValueError(policy)

    def run(self, policy, bf_true, bc, seed):
        """Greedy reveal sequence; returns dict budget -> segment log-odds."""
        rng = np.random.RandomState(seed)
        order = bit_reversal_order(K) if policy != "random" else list(rng.permutation(K))
        bf = np.full(K, ieh.MISSING, dtype=int)
        unobserved = set(range(K))
        out = {}
        calls = J
        for budget in BUDGETS:
            while calls < budget and unobserved:
                w = self.next_window(policy, bf, bc, unobserved, rng, order)
                bf[w] = int(bf_true[w])
                unobserved.discard(w)
                calls += 1
            p = self.p_s(self.gamma(bf, bc))
            out[budget] = np.log(p + 1e-6) - np.log(1.0 - p + 1e-6)
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "runs", "20260907_c3_rev3_interval_evidence", "adaptive_replay"))
    args = ap.parse_args(argv)
    corpus = args.corpus
    V = {K: vlm_verdict.load_verdicts(corpus, k=K, tag="qwen"),
         J: vlm_verdict.load_verdicts(corpus, k=J, tag="qwen")}
    B = {v: (verdict_hmm.binarize(V[K][v]), verdict_hmm.binarize(V[J][v]))
         for v in V[K] if v in V[J]}
    labels = hdata.load_labels(corpus)
    train_ids = [v for v in hdata.load_split(corpus, "train") if v in B]
    hmm, n_pos, n_neg = hc.fit_hmm(corpus, train_ids, labels, B, model="interval",
                                   normalized_time=True, positive_constraint=True)
    out_dir = os.path.join(args.out_root, corpus)
    os.makedirs(out_dir, exist_ok=True)
    hmm.save(os.path.join(out_dir, "hmm_params.json"))
    gt = hdata.gt_arrays(corpus, args.split)
    vids = sorted(gt)
    summary = {"corpus": corpus, "split": args.split, "budgets": list(BUDGETS),
               "hmm": hmm.params(), "n_train_pos": n_pos, "n_train_neg": n_neg, "results": {}}
    for policy in args.policies.split(","):
        t0 = time.time()
        scores = {b: {} for b in BUDGETS}
        for vid in vids:
            n = len(gt[vid])
            if vid not in B:
                for b in BUDGETS:
                    scores[b][vid] = [0.0] * n
                continue
            bf, bc = B[vid]
            rp = Replay(hmm, 1.0)          # normalized time: duration irrelevant
            seq = rp.run(policy, bf, bc, args.seed)
            for b in BUDGETS:
                sc = ieh.rows_from_segments(seq[b], hmm.grid, align.second_bounds(n), float(n))
                scores[b][vid] = [round(float(x), 6) for x in sc]
        for b in BUDGETS:
            d = os.path.join(out_dir, "%s_b%d" % (policy, b))
            os.makedirs(d, exist_ok=True)
            sp = os.path.join(d, "scores.jsonl")
            with open(sp, "w") as fh:
                for vid in vids:
                    fh.write(json.dumps({"video_id": vid, "n_frames": len(gt[vid]),
                                         "score_av": scores[b][vid]}) + "\n")
            jo = os.path.join(d, "metrics.json")
            subprocess.run([sys.executable, EVALUATOR, "--corpus", corpus, "--split", args.split,
                            "--scores", sp, "--json-out", jo], check=True, cwd=ROOT,
                           stdout=subprocess.DEVNULL)
            r = json.load(open(jo))["results"]["score_av"]
            summary["results"].setdefault(policy, {})[str(b)] = dict(
                pooled_ap=r["pr_auc"], pooled_roc=r["roc_auc"], within_roc=r["per_video"]["macro_auc"])
            print("%s %-13s budget %2d  AP %.4f ROC %.4f within %.4f" % (
                corpus, policy, b, r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"]), flush=True)
        print("  %s done in %.0fs" % (policy, time.time() - t0), flush=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for policy, res in summary["results"].items():
            xs = [int(b) for b in res]
            for ax, key in zip(axes, ("pooled_ap", "pooled_roc", "within_roc")):
                ax.plot(xs, [res[str(b)][key] for b in xs], marker="o", label=policy)
                ax.set_xlabel("VLM calls per video")
                ax.set_title("%s %s (%s, HMM posterior only)" % (corpus, key, args.split))
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curve.png"), dpi=130)
    except Exception as exc:              # plotting is optional
        print("no plot: %s" % exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
