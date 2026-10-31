"""Goal 1 diagnostic: how calls could be allocated across videos (CPU, saved 32-call trajectories, no retraining).

Within-video ROC only depends on the calls made in positive videos, so a rule that keeps positive videos at the
fixed-8 trajectory keeps within unchanged; the question is whether the video posterior P(G = 1 | answers) separates
positives early enough and what pooled AP/ROC cost fewer calls on negatives have. Per trial this reports:
  - video-level ROC of p_G after k calls (val and test);
  - an oracle allocation (true labels: positives fixed 8, negatives fixed k) as an upper bound only;
  - label-free rules calibrated on validation mean calls (no labels) to several targets, with their test metrics.
Screening metrics use the shared core evaluator in-process (hc.frame_metrics); rows quoted as results are re-run
through the unified evaluator by the caller. Rule-10 developmental evidence (test labels read, no training).
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import socket
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import policy
from qtl.stop_rules import rule_values, add_quantile_rules, add_relaxed_rules
import hier_evidence_common as hc

TARGETS = (2, 3, 4, 5, 6, 8)
FIXED = (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 32)


def logit_change_prob(run, n):
    """Per call k (value read before call k+1): sum over seconds of |p_t(k) - p_t(k-1)| on the probability scale;
    +inf before the first call (the first call is always made under this statistic)."""
    s = [np.asarray(x, dtype=np.float64) for x in run["scores"]]
    return [np.inf] + [float(np.abs(s[k] - s[k - 1]).sum()) for k in range(1, n)]


def rules(run, vals):
    n = len(run["eig"])
    pg = [float(x) for x in run["p_G"][:n]]
    q = vals["qmixSG_rt10"]
    dp = logit_change_prob(run, n)
    return {
        "qmixSG_rt10": q,
        # continue while the video is likely hateful (no change term): the allocation follows P(G) only
        "pG": pg,
        # the existing rule weighted by P(G): likely-negative videos stop sooner
        "qmixSG_rt10_x_pG": [x * g if math.isfinite(x) else x for x, g in zip(q, pg)],
        # change on the probability scale (seconds already near 0 move little), relaxed like rt10
        "dpsum_rt10": [x * 0.9 ** k if math.isfinite(x) else x for k, x in enumerate(dp)],
        "dpsum": dp,
        # the label-split analysis (mix_budgets.py) found that negatives, not positives, need calls 5-8 for
        # pooled AP/ROC on HateMM: ask while the video does NOT yet look hateful, relaxed like rt10
        "negG_rt10": [(1.0 - g) * 0.9 ** k for k, g in enumerate(pg)],
        "qmixSG_rt10_x_negG": [x * (1.0 - g) if math.isfinite(x) else x for x, g in zip(q, pg)],
    }


def metrics(scores, gt, hate):
    m = hc.frame_metrics(scores, gt, hate)
    return [m["pooled_ap"], m["pooled_roc"], m["within_roc"]]


def analyse_trial(trial_dir, labels, ids, gt):
    runs = pickle.loads((trial_dir / "runs.pkl").read_bytes())["runs"]
    vals = {split: {v: rule_values(r, len(r["scores"][0])) for v, r in runs[split].items()} for split in runs}
    add_quantile_rules(vals)
    add_relaxed_rules(vals)
    R = {split: {v: rules(runs[split][v], vals[split][v]) for v in runs[split]} for split in runs}
    hate = {split: {v for v in ids[split] if labels[v] == 1} for split in ("val", "test")}
    out = {"trial": trial_dir.name, "source": str(trial_dir)}

    def score_at(split, calls):
        return {v: runs[split][v]["scores"][calls[v]] for v in ids[split]}

    def summary(split, calls):
        c = np.array([calls[v] for v in ids[split]])
        pos = np.array([labels[v] == 1 for v in ids[split]])
        return {"metrics": metrics(score_at(split, calls), gt[split], hate[split]), "mean_calls": float(c.mean()),
                "mean_calls_positive": float(c[pos].mean()), "mean_calls_negative": float(c[~pos].mean())}

    # 1. video-level separation by p_G after k calls
    out["pG_video_roc"] = {}
    for split in ("val", "test"):
        y = [labels[v] for v in ids[split]]
        out["pG_video_roc"][split] = {
            k: float(roc_auc_score(y, [runs[split][v]["p_G"][min(k, len(runs[split][v]["eig"]))] for v in ids[split]]))
            for k in (0, 1, 2, 3, 4, 6, 8)}
    # 2. fixed curve
    out["fixed"] = {k: summary("test", {v: min(k, len(runs["test"][v]["eig"])) for v in ids["test"]}) for k in FIXED}
    # 3. oracle: positives fixed 8, negatives fixed k (upper bound, uses test labels)
    out["oracle_pos8_negk"] = {}
    for k in (0, 1, 2, 3, 4):
        calls = {v: min(8 if labels[v] == 1 else k, len(runs["test"][v]["eig"])) for v in ids["test"]}
        out["oracle_pos8_negk"][k] = summary("test", calls)
    # 4. label-free rules calibrated on validation mean calls
    out["rules"] = {}
    for name in R["val"][ids["val"][0]]:
        rows = {}
        for t in TARGETS:
            thr, val_mean = policy.calibrate([R["val"][v][name] for v in ids["val"]], float(t))
            calls = {v: policy.stop_calls(R["test"][v][name], thr) for v in ids["test"]}
            row = summary("test", calls)
            row.update(threshold=thr if math.isfinite(thr) else str(thr), validation_mean_calls=val_mean)
            rows[t] = row
        out["rules"][name] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--results-dir", nargs="+", required=True, help="Dirs holding seed<s>_trial<k>/runs.pkl")
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/call_allocation")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    out_dir = (ROOT / a.out_root / a.corpus).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for results in a.results_dir:
        for trial_dir in sorted(Path(results).resolve().glob("seed*_trial*")):
            trials.append(analyse_trial(trial_dir, labels, ids, gt))
            print(trial_dir.name, "done", flush=True)
    path = out_dir / f"partial_{socket.gethostname()}.json"
    path.write_text(json.dumps({"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
                                "trials": trials}, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
