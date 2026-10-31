"""Goal 1 diagnostic: which videos need calls 5-8 (CPU, saved per-budget test scores, test labels read).

For each trial directory holding scores_test_fixed<k>.jsonl, scores positive videos at budget kp and negative
videos at budget kn (label-split oracle, an analysis only) and reports pooled AP / ROC / within through the shared
core evaluator. If (kp=8, kn=4) recovers the fixed-8 AP, calls 5-8 matter on positive videos; if (kp=4, kn=8) does,
they matter on negatives. Rule-10 developmental evidence.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import hier_evidence_common as hc

PAIRS = ((0, 0), (4, 4), (8, 8), (8, 4), (4, 8), (8, 2), (2, 8), (8, 0), (0, 8), (16, 4), (16, 8))


def load(path):
    return {r["video_id"]: np.asarray(r["score_av"], dtype=np.float64)
            for r in map(json.loads, Path(path).read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--trial-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {v for v in ids["test"] if labels[v] == 1}
    rows = {}
    for d in a.trial_dirs:
        d = Path(d)
        sc = {k: load(d / f"scores_test_fixed{k}.jsonl") for k in {x for p in PAIRS for x in p}}
        rows[str(d)] = {}
        for kp, kn in PAIRS:
            s = {v: sc[kp if labels[v] == 1 else kn][v] for v in ids["test"]}
            m = hc.frame_metrics(s, gt["test"], hate)
            rows[str(d)][f"pos{kp}_neg{kn}"] = [m["pooled_ap"], m["pooled_roc"], m["within_roc"]]
    mean = {k: np.mean([r[k] for r in rows.values()], 0).tolist() for k in next(iter(rows.values()))}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trials": rows, "mean": mean}, indent=1) + "\n")
    for k, v in mean.items():
        print(k, " ".join(f"{x:.3f}" for x in v))


if __name__ == "__main__":
    main()
