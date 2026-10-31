"""Goal 2 diagnostic: where r5's remaining pooled error is (CPU, saved test scores, no retraining).

For the r5 test-selected trial of each seed at fixed 8 calls, pooled AP/ROC of the saved scores and of two oracle
edits, each fixing one component and keeping the rest:
  - negatives_zero: every second of a negative video gets score 0 (perfect rejection of negative videos; positive
    videos unchanged), the gain available from video-level discrimination of negatives;
  - shape_oracle: inside each positive video its own scores are reassigned so that the hate seconds get the highest
    ones (perfect within-video ranking with the same set of scores per video), the gain available from localisation.
Also counts where the non-hate seconds are (negative videos vs non-hate parts of positive videos).
Rule-10 developmental evidence (test predictions and test GT read; nothing trained or selected).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc

R5 = "runs/20260929_query_paradigm_r5"
SEEDS = (234, 2025, 3407)


def load_scores(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def metrics(scores, gt, hate):
    m = hc.frame_metrics(scores, gt, hate)
    return {k: float(m[k]) for k in ("pooled_ap", "pooled_roc", "within_roc")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/headroom")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    gt = gt["test"]
    hate = {v for v in ids["test"] if labels[v] == 1}
    nonhate_neg = sum(len(gt[v]) for v in gt if labels[v] == 0)
    nonhate_pos = sum(int((np.asarray(gt[v]) == 0).sum()) for v in gt if labels[v] == 1)
    hate_sec = sum(int((np.asarray(gt[v]) == 1).sum()) for v in gt if labels[v] == 1)
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "seconds": {"hate": hate_sec, "nonhate_in_positive_videos": nonhate_pos,
                          "nonhate_in_negative_videos": nonhate_neg}, "seeds": {}}
    for seed in SEEDS:
        study = ROOT / R5 / a.corpus / f"seed{seed}"
        trial = json.loads((study / "study_summary.json").read_text())["best"]["number"]
        sc = load_scores(study / f"trial{trial}" / "scores_test_fixed8.jsonl")
        assert set(sc) == set(gt)
        neg0 = {v: (np.zeros_like(s) if labels[v] == 0 else s) for v, s in sc.items()}
        shape = {}
        for v, s in sc.items():
            g = np.asarray(gt[v]).astype(np.int64)          # GT arrays may be unsigned
            if labels[v] == 1 and 0 < g.sum() < len(g):
                order = np.argsort(-g, kind="stable")          # hate seconds first
                t = np.empty_like(s)
                t[order] = np.sort(s)[::-1]
                shape[v] = t
            else:
                shape[v] = s
        report["seeds"][seed] = {"trial": trial, "official": metrics(sc, gt, hate),
                                 "negatives_zero": metrics(neg0, gt, hate), "shape_oracle": metrics(shape, gt, hate)}
        print(a.corpus, seed, json.dumps(report["seeds"][seed]), flush=True)
    keys = ("official", "negatives_zero", "shape_oracle")
    report["mean"] = {k: {m: float(np.mean([report["seeds"][s][k][m] for s in SEEDS]))
                          for m in ("pooled_ap", "pooled_roc", "within_roc")} for k in keys}
    out = ROOT / a.out_root
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.corpus}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["mean"], indent=1))
    print(out / f"{a.corpus}.json")


if __name__ == "__main__":
    main()
