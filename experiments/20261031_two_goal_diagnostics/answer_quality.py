"""Goal 2 diagnostic: how much localisation the VLM tree answers carry when every node is answered (CPU).

The content model of r5 is trained by the video label and the VLM soft answers (p_yes, `soft_both`) on every
queryable node of the training videos. On validation and test (per-second GT), for the answered nodes:
  - node AUC: p_yes against "the node contains a hate second", over all nodes and over the nodes of positive videos
    only (within-positive), by node length;
  - share: Spearman of p_yes with the node's hate share, over nodes that contain hate;
  - all-answers score: each second gets p_yes of the shortest answered node covering it (ties: mean); unified frame
    metrics on test, and the level / hate-fraction Spearman over positive videos.
If the all-answers score localises much better than r5 at 0 calls, the answers carry more than r5's content model
learns from them; if not, the training signal itself is the limit. Rule-10 developmental evidence.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata
import hier_evidence_common as hc

LEN_BINS = ((4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--source", default="soft_both")
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/answer_quality")
    a = ap.parse_args()
    labels, ids, gt_all, _ = hc.load_fixed_cohort(a.corpus)
    P, _, _ = qdata.load_soft_p(a.corpus, a.source)
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "source": a.source, "splits": {}}
    for split in ("val", "test"):
        vids = [v for v in ids[split] if v in gt_all[split] and v in P]
        rows = []                                   # (positive video, length, p, contains, share)
        scores = {}
        for v in vids:
            g = np.asarray(gt_all[split][v]).astype(np.int64)
            T = len(g)
            best = np.full(T, np.inf)
            acc, cnt = np.zeros(T), np.zeros(T)
            for (s, e), p in P[v].items():
                if p is None:
                    continue
                e = min(e, T)
                if e <= s:
                    continue
                L = e - s
                rows.append((labels[v], L, p, int(g[s:e].any()), float(g[s:e].mean())))
                shorter = L < best[s:e]
                same = L == best[s:e]
                seg_best, seg_acc, seg_cnt = best[s:e], acc[s:e], cnt[s:e]
                seg_acc[shorter], seg_cnt[shorter], seg_best[shorter] = p, 1, L
                seg_acc[same] += p
                seg_cnt[same] += 1
            scores[v] = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        r = np.array(rows)
        res = {"n_videos": len(vids), "n_nodes": int(len(r))}
        for name, sel in (("all", np.ones(len(r), bool)), ("positive_videos", r[:, 0] == 1)):
            sub = r[sel]
            res[f"node_auc_{name}"] = float(roc_auc_score(sub[:, 3], sub[:, 2]))
            res[f"node_auc_{name}_by_length"] = {
                f"{lo}-{hi}": (float(roc_auc_score(sub[m, 3], sub[m, 2])) if 0 < sub[m, 3].mean() < 1 else None)
                for lo, hi in LEN_BINS for m in [(sub[:, 1] >= lo) & (sub[:, 1] < hi)] if m.sum() > 20}
        h = r[r[:, 3] == 1]
        res["share_spearman_in_hate_nodes"] = float(spearmanr(h[:, 2], h[:, 4]).correlation)
        gt = {v: np.asarray(gt_all[split][v]) for v in vids}
        hate = {v for v in vids if labels[v] == 1}
        m = hc.frame_metrics(scores, gt, hate)
        pos = [v for v in vids if labels[v] == 1]
        res["all_answers_score"] = {"pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
                                    "within_roc": float(m["within_roc"]),
                                    "level_fraction_spearman": float(spearmanr([scores[v].mean() for v in pos],
                                                                               [gt[v].mean() for v in pos]).correlation)}
        report["splits"][split] = res
        print(a.corpus, split, json.dumps(res), flush=True)
    out = ROOT / a.out_root
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(out / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
