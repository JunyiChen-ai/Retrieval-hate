#!/usr/bin/env python
"""In-job smoke gate: verify arbitration JSON health before the full run.

Reads the first N records of an arb JSONL; requires >= min_json strict-JSON
verdicts. Prints the verdict distribution + a compact per-sample line so the
human check of verdict plausibility can happen post-hoc from the job log.
Exit code 1 aborts the sbatch (set -e) if the gate fails.
"""
import argparse
import json
import sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--min_json", type=int, default=8)
args = ap.parse_args()

recs = []
with open(args.jsonl) as f:
    for line in f:
        line = line.strip()
        if line:
            recs.append(json.loads(line))
        if len(recs) >= args.n:
            break

ok = [r for r in recs if r.get("verdict_bin") is not None]
print("[smoke_check] {}: {} records, {} strict-JSON verdicts (need >= {})".format(
    args.jsonl, len(recs), len(ok), args.min_json))
print("[smoke_check] verdict distribution:", Counter(r["verdict"] for r in ok))
agree_knn = sum(1 for r in ok if r["verdict_bin"] == r["pred_knn"])
agree_gt = sum(1 for r in ok if r["verdict_bin"] == r["label"])
print("[smoke_check] agree with kNN: {}/{}; agree with gt: {}/{}".format(
    agree_knn, len(ok), agree_gt, len(ok)))
for r in recs:
    print("  id={} verdict={} bin={} knn={} gt={} margin={:.3f} err={}".format(
        r["id"], r.get("verdict"), r.get("verdict_bin"), r["pred_knn"],
        r["label"], r["margin"], (r.get("parse_error") or "")[:60]))
if len(recs) < args.n:
    print("[smoke_check] FAIL: only {} records present".format(len(recs)))
    sys.exit(1)
if len(ok) < args.min_json:
    print("[smoke_check] FAIL: JSON rate too low")
    sys.exit(1)
print("[smoke_check] PASS")
