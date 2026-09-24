"""Where pooled AP / ROC comes from (README section 8; development evidence on test under rule 10, no training or
selection uses it). For each given test score file:
  - pooled AP / ROC, and the video-level AUC of the per-video mean and max score;
  - "constant per video": every second of a video gets the video's mean score (keeps the cross-video ranking,
    removes all within-video ranking); pooled minus this = what the within-video part adds;
  - "negatives zeroed": negative videos get 0 (what is left is the ranking inside and across positive videos);
  - mean score on GT-hateful seconds, on non-hateful seconds of positive videos, and on negative videos;
  - within-video macro ROC over positive videos that have both classes.
Also, per corpus, the within-video ROC of the raw VLM answers when every queryable node is asked (the answer of the
shortest asked node covering the second; the mean over all nodes covering it), for the hate level and for the max
level over the five categories.

    python experiments/20260925_query_paradigm/score_decomposition.py --corpus hatemm --scores a.jsonl b.jsonl ...
Output: printed; with --out, a JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402


def read_scores(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scores", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    vids = ids["test"]
    G = {v: np.asarray(gt["test"][v]) for v in vids}
    both = [v for v in vids if labels[v] == 1 and 0 < G[v].mean() < 1]

    def pooled(S):
        s = np.concatenate([S[v] for v in vids])
        y = np.concatenate([G[v] for v in vids])
        return float(average_precision_score(y, s)), float(roc_auc_score(y, s))

    def within(S):
        return float(np.mean([roc_auc_score(G[v], S[v]) for v in both]))

    res = {"corpus": a.corpus, "n_within_videos": len(both), "files": {}}
    for p in a.scores:
        S = read_scores(p)
        yv = [labels[v] for v in vids]
        r = {"pooled": pooled(S),
             "video_auc_mean": float(roc_auc_score(yv, [S[v].mean() for v in vids])),
             "video_auc_max": float(roc_auc_score(yv, [S[v].max() for v in vids])),
             "constant_per_video": pooled({v: np.full(len(S[v]), S[v].mean()) for v in vids}),
             "negatives_zeroed": pooled({v: S[v] if labels[v] == 1 else 0 * S[v] for v in vids}),
             "mean_score": {"hate": float(np.mean([S[v][G[v] > 0].mean() for v in vids if G[v].max() > 0])),
                            "positive_nonhate": float(np.mean([S[v][G[v] == 0].mean() for v in vids
                                                               if labels[v] == 1 and (G[v] == 0).any()])),
                            "negative": float(np.mean([S[v].mean() for v in vids if labels[v] == 0]))},
             "within": within(S)}
        res["files"][p] = r
        print("%s\n  pooled %.4f/%.4f | constant per video %.4f/%.4f | negatives zeroed %.4f/%.4f | video AUC "
              "mean %.3f max %.3f | within %.3f | mean score hate %.3f pos-nonhate %.3f neg %.3f" % (
                  p, *r["pooled"], *r["constant_per_video"], *r["negatives_zeroed"], r["video_auc_mean"],
                  r["video_auc_max"], r["within"], *r["mean_score"].values()))
    answers, _ = qdata.load_answers(a.corpus)
    raw = {}
    for v in vids:
        T = len(G[v])
        best = np.full(T, np.inf)
        leaf = np.zeros((2, T))
        tot = np.zeros((2, T))
        cnt = np.zeros(T)
        for (s, e), o in answers[v].items():
            if o is None:
                continue
            val = np.array([o[0], max(o)], dtype=np.float64)
            m = (e - s) < best[s:e]
            idx = np.arange(s, e)[m]
            best[idx] = e - s
            leaf[:, idx] = val[:, None]
            tot[:, s:e] += val[:, None]
            cnt[s:e] += 1
        jitter = np.random.RandomState(0).rand(T) * 1e-6       # ties inside a video broken the same way everywhere
        raw[v] = (leaf + jitter, tot / np.maximum(cnt, 1) + jitter)
    res["raw_answers_all_nodes_within"] = {
        "shortest_node_hate": within({v: raw[v][0][0] for v in vids}),
        "shortest_node_max": within({v: raw[v][0][1] for v in vids}),
        "mean_over_nodes_hate": within({v: raw[v][1][0] for v in vids}),
        "mean_over_nodes_max": within({v: raw[v][1][1] for v in vids})}
    print("raw answers, all nodes asked, within:", res["raw_answers_all_nodes_within"])
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
