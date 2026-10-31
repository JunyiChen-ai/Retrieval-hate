"""Goal 2 diagnostic: where the inside-outside prior loses to r5 (CPU, saved test scores, no retraining).

For the test-selected trial of each seed of both methods, at fixed 0 and 8 calls, splits pooled AP/ROC into a
video-level part and a within-video part:
  - video ROC of the mean and of the max second score;
  - pooled AP/ROC over positive videos only (seconds of hateful videos; cross-video calibration + localisation);
  - hybrids: one method's within-video shape (logit minus its video mean logit) with the other method's video mean
    logit. If "IO shape + r5 level" recovers r5's pooled numbers, the gap is in the video-level component.
Rule-10 developmental evidence (test predictions and test GT read; nothing trained or selected).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc

METHODS = {"r5": "runs/20260929_query_paradigm_r5", "io": "runs/20261001_inside_outside_backbone"}
SEEDS = (234, 2025, 3407)


def load_scores(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def stats(scores, gt, labels, hate):
    vids = sorted(scores)
    y = [labels[v] for v in vids]
    out = {"official": [hc.frame_metrics(scores, gt, hate)[k] for k in ("pooled_ap", "pooled_roc", "within_roc")],
           "video_roc_mean": float(roc_auc_score(y, [scores[v].mean() for v in vids])),
           "video_roc_max": float(roc_auc_score(y, [scores[v].max() for v in vids]))}
    pos = [v for v in vids if labels[v] == 1]
    s = np.concatenate([scores[v] for v in pos])
    g = np.concatenate([np.asarray(gt[v]) for v in pos])
    out["positive_only_pooled"] = [float(average_precision_score(g, s)), float(roc_auc_score(g, s))]
    neg = [v for v in vids if labels[v] == 0]
    out["mean_score"] = {"negative_videos": float(np.mean([scores[v].mean() for v in neg])) if neg else None,
                         "hate_seconds": float(s[g == 1].mean()), "nonhate_seconds_in_positive": float(s[g == 0].mean())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/backbone_gap")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    gt = gt["test"]
    hate = {v for v in ids["test"] if labels[v] == 1}
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "seeds": {}}
    for seed in SEEDS:
        row = {}
        sc = {}
        for name, root in METHODS.items():
            study = ROOT / root / a.corpus / f"seed{seed}"
            trial = json.loads((study / "study_summary.json").read_text())["best"]["number"]
            row[name] = {"trial": trial}
            for b in (0, 8):
                sc[(name, b)] = load_scores(study / f"trial{trial}" / f"scores_test_fixed{b}.jsonl")
                assert set(sc[(name, b)]) == set(gt)
                row[name][f"fixed{b}"] = stats(sc[(name, b)], gt, labels, hate)
        for b in (0, 8):
            for shape, level in (("io", "r5"), ("r5", "io")):
                mixed = {}
                for v in gt:
                    ls, ll = logit(sc[(shape, b)][v]), logit(sc[(level, b)][v])
                    mixed[v] = 1 / (1 + np.exp(-(ls - ls.mean() + ll.mean())))
                row[f"shape_{shape}_level_{level}_fixed{b}"] = stats(mixed, gt, labels, hate)
        report["seeds"][seed] = row
        print(a.corpus, seed, "done", flush=True)
    out = ROOT / a.out_root
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(out / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
