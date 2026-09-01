#!/usr/bin/env python3
"""Rule-10 test diagnostic for the fixed train-fit audio teacher alone."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import test_teacher_diagnostic as core
import train
from eval_baseline_scores import evaluate_scores
from hate_common import data as hdata
from macilsd import align
from powa_macil.dataset import usable_text_ids


REPO = Path(__file__).resolve().parents[2]


def analyze(corpus):
    labels = hdata.load_labels(corpus)
    train_ids = usable_text_ids(corpus, hdata.load_split(corpus, "train"))
    if corpus == "hateclipseg":
        raw_ids = {path.stem for path in core.HCS_TRAIN_RAW.glob("*.json")}
        train_ids = [video_id for video_id in train_ids if video_id in raw_ids]
    expected = 744 if corpus == "hatemm" else 238
    if len(train_ids) != expected:
        raise RuntimeError(f"unexpected train coverage: {corpus}/{len(train_ids)}")

    args = SimpleNamespace(seed=234, max_audio_rows=200, audio_epochs=5)
    audio_model = train.fit_audio_model(corpus, train_ids, labels, args)
    anchors = core.load_jsonl(core.ANCHOR[corpus])
    gt = hdata.gt_arrays(corpus, "test")
    if set(anchors) != set(gt):
        raise RuntimeError(f"test coverage mismatch: {corpus}")

    transported = {}
    anchor_scores = {}
    for video_id in sorted(gt):
        anchor = np.asarray(anchors[video_id]["score_powa"], dtype=np.float64)
        audio, n_seconds, snippets = align.aligned_audio(
            corpus, video_id, "snippet"
        )
        if len(anchor) != n_seconds:
            raise RuntimeError(f"test timeline mismatch: {corpus}/{video_id}")
        order = train.percentile(
            audio_model.decision_function(train.normalize(audio))
        )
        order = core.fixed_grid_order(order)
        second_order = align.scores_to_gold_grid(
            order, snippets, n_seconds, "snippet"
        )
        transported[video_id] = core.transport(anchor, second_order)
        anchor_scores[video_id] = anchor

    positives = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {
        "score_anchor": evaluate_scores(anchor_scores, gt, positives),
        "transport_audio_fixed200": evaluate_scores(
            transported, gt, positives
        ),
    }
    return {
        "train_videos": len(train_ids),
        "test_videos": len(gt),
        "metrics": {
            name: train.metric_summary(report) for name, report in reports.items()
        },
        "sota_thresholds": core.SOTA[corpus],
    }


def main():
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "status": "rule10_iterative_developmental_diagnostic",
        "purpose": "measure the fixed-grid audio-only teacher ceiling while K16 generation runs",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "corpora": {corpus: analyze(corpus) for corpus in core.CORPORA},
    }
    output = REPO / (
        "runs/20260831_powa_consensus_distillation_pilot/"
        "test_audio_only_diagnostic/analysis.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
