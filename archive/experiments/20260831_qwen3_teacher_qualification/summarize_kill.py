#!/usr/bin/env python3
"""Record the frozen early kill after complete HateMM qualification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


def dense(row):
    total = np.zeros(row["length"], dtype=np.float64)
    count = np.zeros(row["length"], dtype=np.float64)
    for (start, end), score in zip(row["spans"], row["scores"]):
        total[start:end] += score
        count[start:end] += 1
    return total / np.maximum(count, 1)


def main():
    run = REPO / "runs/20260831_qwen3_teacher_qualification"
    rows = [
        json.loads(line)
        for line in (run / "teacher_hatemm.jsonl").read_text().splitlines()
    ]
    scores = {row["video_id"]: dense(row) for row in rows}
    complete_gt = hdata.gt_arrays("hatemm", "val")
    gt = {video_id: complete_gt[video_id] for video_id in scores}
    report = evaluate_scores(scores, gt, set(scores))
    within = report["per_video"]["macro_auc"]
    raw_rows = [
        json.loads(line)
        for line in (run / "teacher_hatemm_raw.jsonl").read_text().splitlines()
    ]
    unparsed = sum(
        not any(
            token.isdigit() and 0 <= int(token) <= 10
            for token in row["generation"].replace(",", " ").split()
        )
        for row in raw_rows
    )
    hcs_rows = (
        (run / "teacher_hateclipseg.jsonl").read_text().splitlines()
        if (run / "teacher_hateclipseg.jsonl").is_file() else []
    )
    payload = {
        "date": "2026-08-31",
        "stage": "dense_teacher_validation_qualification",
        "verdict": "KILL_AFTER_COMPLETE_HATEMM",
        "pass": False,
        "test_labels_read": False,
        "hatemm": {
            "coverage_complete": len(rows) == 43,
            "videos": len(rows),
            "windows": len(raw_rows),
            "unparsed": unparsed,
            "unparsed_rate": unparsed / max(1, len(raw_rows)),
            "teacher_within_roc": within,
            "within_n": report["per_video"]["n_videos_both_classes"],
            "powa_within_reference": 0.5719312723877894,
            "absolute_gate": 0.60,
            "failed": ["absolute_within_gate", "gain_over_powa_at_least_0.020"],
        },
        "hateclipseg": {
            "status": "partial_non_authoritative_stopped_after_hatemm_kill",
            "completed_video_rows": len(hcs_rows),
        },
    }
    out = run / "summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
