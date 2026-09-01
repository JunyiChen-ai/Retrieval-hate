#!/usr/bin/env python3
"""Rule-10 test diagnostic for the fixed train-only audio+VERA teacher."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO))

import train  # noqa: E402
import prepare_vera_k16 as producer  # noqa: E402
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402
from powa_macil.dataset import usable_text_ids  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
ANCHOR = {
    "hatemm": Path(
        "/home/jehc223/Hate-follow-up/results/reproduction/powa_macil/"
        "final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl"
    ),
    "hateclipseg": REPO / (
        "runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl"
    ),
}
VERA_K16 = {
    corpus: REPO / (
        "runs/20260831_powa_consensus_distillation_pilot/teacher_cache/"
        f"{corpus}_test/raw"
    ) for corpus in CORPORA
}
HCS_TRAIN_RAW = REPO / (
    "results/reproduction/official_val/final/vera/hateclipseg/seed_234/"
    "train_sparse_k16/raw"
)
SOTA = {
    "hatemm": {"pooled_ap": .5938316, "pooled_roc": .8161838,
               "within_roc": .6315317},
    "hateclipseg": {"pooled_ap": .6193711, "pooled_roc": .6050225,
                    "within_roc": .5619079},
}


def load_jsonl(path):
    rows = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["video_id"]] = row
    return rows


def transport(anchor, order):
    anchor = np.asarray(anchor, dtype=np.float64)
    order = np.asarray(order, dtype=np.float64)
    output = np.empty_like(anchor)
    # Preserve the anchor's own ordering inside teacher ties. Using temporal
    # index as an implicit tie-break would manufacture a monotonic time prior.
    output[np.lexsort((anchor, order))] = np.sort(anchor, kind="stable")
    return output


def fixed_grid_order(order, max_seqlen=200):
    """Apply the train-time uniform grid, then lift to the native grid."""
    order = np.asarray(order, dtype=np.float64)
    valid_length = min(len(order), max_seqlen)
    fixed = train.process_order(order, max_seqlen, valid_length)
    if len(order) <= max_seqlen:
        return fixed
    index = np.linspace(
        0, len(order) - 1, max_seqlen, dtype=np.uint16
    ).astype(np.int64)
    return np.interp(np.arange(len(order)), index, fixed)


def analyze(corpus):
    labels = hdata.load_labels(corpus)
    train_ids = usable_text_ids(corpus, hdata.load_split(corpus, "train"))
    if corpus == "hateclipseg":
        raw_ids = {path.stem for path in HCS_TRAIN_RAW.glob("*.json")}
        train_ids = [video_id for video_id in train_ids if video_id in raw_ids]
    expected = 744 if corpus == "hatemm" else 238
    if len(train_ids) != expected:
        raise RuntimeError(f"unexpected train coverage: {corpus}/{len(train_ids)}")
    args = SimpleNamespace(seed=234, max_audio_rows=200, audio_epochs=5)
    audio_model = train.fit_audio_model(corpus, train_ids, labels, args)

    anchors = load_jsonl(ANCHOR[corpus])
    gt = hdata.gt_arrays(corpus, "test")
    vera_ids = {path.stem for path in VERA_K16[corpus].glob("*.json")}
    if set(anchors) != set(gt) or vera_ids != set(gt):
        raise RuntimeError(f"test coverage mismatch: {corpus}")
    branches = {name: {} for name in (
        "score_anchor", "transport_audio", "transport_vera_k16",
        "transport_audio_vera_k16",
    )}
    for video_id in sorted(gt):
        anchor = np.asarray(anchors[video_id]["score_powa"], dtype=np.float64)
        audio, n_seconds, snippets = align.aligned_audio(
            corpus, video_id, "snippet"
        )
        audio_order = train.percentile(
            audio_model.decision_function(train.normalize(audio))
        )
        vera_source = json.loads((VERA_K16[corpus] / f"{video_id}.json").read_text())
        if vera_source.get("video_id") != video_id:
            raise RuntimeError(f"VERA source identity mismatch: {corpus}/{video_id}")
        if len(anchor) != n_seconds:
            raise RuntimeError(f"test timeline mismatch: {corpus}/{video_id}")
        support_length = max(
            1, int(np.ceil(min(float(n_seconds), float(vera_source["duration"]))))
        )
        starts = np.unique(np.rint(np.linspace(
            0, support_length - 1, min(16, support_length)
        )).astype(np.int64))
        media = Path(producer.vera.video_path(corpus, video_id)).resolve()
        if not producer.valid_result(
                VERA_K16[corpus] / f"{video_id}.json", video_id, starts,
                float(vera_source["duration"]), media, corpus, "test"):
            raise RuntimeError(f"K16 recipe mismatch: {corpus}/{video_id}")
        segments = vera_source.get("segments", [])
        stored_starts = np.asarray([float(row["start"]) for row in segments])
        stored_scores = np.asarray([float(row["score"]) for row in segments])
        if (not np.array_equal(stored_starts, starts.astype(float))
                or not np.isin(stored_scores, [0.0, 1.0]).all()):
            raise RuntimeError(f"invalid fixed K16 source: {corpus}/{video_id}")
        vera_snippet = np.interp(snippets.mean(1), stored_starts, stored_scores)
        audio_order = fixed_grid_order(audio_order)
        vera_order = fixed_grid_order(train.percentile(vera_snippet))
        consensus = .5 * audio_order + .5 * vera_order
        second_orders = {
            "transport_audio": align.scores_to_gold_grid(
                audio_order, snippets, n_seconds, "snippet"
            ),
            "transport_vera_k16": align.scores_to_gold_grid(
                vera_order, snippets, n_seconds, "snippet"
            ),
            "transport_audio_vera_k16": align.scores_to_gold_grid(
                consensus, snippets, n_seconds, "snippet"
            ),
        }
        branches["score_anchor"][video_id] = anchor
        for name, order in second_orders.items():
            branches[name][video_id] = transport(anchor, order)

    positives = {video_id for video_id in gt if labels[video_id] == 1}
    reports = {name: evaluate_scores(scores, gt, positives)
               for name, scores in branches.items()}
    metrics = {name: train.metric_summary(report)
               for name, report in reports.items()}
    core = metrics["transport_audio_vera_k16"]
    gates = {name: core[name] >= threshold
             for name, threshold in SOTA[corpus].items()}
    return {
        "train_videos": len(train_ids), "test_videos": len(gt),
        "metrics": metrics, "sota_thresholds": SOTA[corpus],
        "core_sota_gates": gates, "all_core_metrics_sota": all(gates.values()),
    }


def main():
    corpora = {corpus: analyze(corpus) for corpus in CORPORA}
    payload = {
        "date": "2026-08-31", "split": "test",
        "status": "rule10_iterative_developmental_diagnostic",
        "purpose": "authorize or reject the fixed audio+VERA train-only teacher",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "test_artifacts": {
            corpus: {"anchor_predictions": str(ANCHOR[corpus]),
                     "vera_k16_predictions": str(VERA_K16[corpus]),
                     "ground_truth": str(REPO / f"results/reproduction/gt/{corpus}_test.npz")}
            for corpus in CORPORA
        },
        "corpora": corpora,
        "continue_to_student": all(
            row["all_core_metrics_sota"] for row in corpora.values()
        ),
    }
    output = REPO / (
        "runs/20260831_powa_consensus_distillation_pilot/"
        "test_teacher_diagnostic/analysis.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
