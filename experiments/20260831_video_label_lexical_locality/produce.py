#!/usr/bin/env python
"""Train-only lexical probe and label-blind test score producer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
sys.path.insert(0, str(ROOT / "src"))
from hate_common import data as hdata  # noqa: E402
from lexical_locality import (load_asr, local_texts, new_classifier,
                              new_vectorizer)  # noqa: E402

CORPORA = ("hatemm", "hateclipseg")
ASR_PATHS = {
    corpus: ROOT / "results" / "reproduction" / "asr" / f"{corpus}_all" /
    "timestamped_chunks.jsonl"
    for corpus in CORPORA
}
LABEL_PATHS = {
    corpus: ROOT / "results" / "reproduction" / "splits" /
    "scoped_labels" / f"{corpus}_train.json"
    for corpus in CORPORA
}
GRID_ROOT = ROOT / "results" / "reproduction" / "features" / "clip_b16_1fps"
LEFT_CONTEXT = 2.0
RIGHT_CONTEXT = 3.0


def load_scoped_labels(corpus: str) -> dict[str, int]:
    payload = json.loads(LABEL_PATHS[corpus].read_text())
    if payload.get("corpus") != corpus or payload.get("split") != "train":
        raise ValueError(f"{corpus}: scoped label identity mismatch")
    labels = {str(k): int(v) for k, v in payload["labels"].items()}
    expected = set(hdata.load_split(corpus, "train"))
    if set(labels) != expected:
        raise ValueError(f"{corpus}: scoped labels do not exactly match train split")
    if set(labels.values()) != {0, 1}:
        raise ValueError(f"{corpus}: scoped labels are not binary with both classes")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "corpora": list(CORPORA),
        "training": "independent target-corpus train video labels only",
        "representation": "char_wb TF-IDF 3-5 gram",
        "min_df": 2,
        "max_features": 50000,
        "classifier": "LogisticRegression C=1 class_weight=balanced",
        "local_window_seconds": [-LEFT_CONTEXT, RIGHT_CONTEXT],
        "score_grid": "label-blind clip_b16_1fps row count",
        "test_gt_read_by_producer": False,
        "metadata_summary_fields_read": False,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {}
    for corpus in CORPORA:
        asr, asr_stats = load_asr(ASR_PATHS[corpus])
        labels = load_scoped_labels(corpus)
        train_ids = hdata.load_split(corpus, "train")
        missing_train_asr = sorted(set(train_ids) - set(asr))
        if missing_train_asr:
            raise FileNotFoundError(
                f"{corpus}: {len(missing_train_asr)} train ASR records missing")
        train_texts = [asr[v]["text"] for v in train_ids]
        train_y = np.asarray([labels[v] for v in train_ids], dtype=np.int64)
        vectorizer = new_vectorizer()
        train_x = vectorizer.fit_transform(train_texts)
        model = new_classifier()
        model.fit(train_x, train_y)

        manifest_test_ids = hdata.load_split(corpus, "test")
        gt_archive = ROOT / "results" / "reproduction" / "gt" / f"{corpus}_test.npz"
        with np.load(gt_archive) as gt_index:
            # Only member names define the evaluator cohort.  No GT array is
            # accessed by the producer.
            gt_keys = set(gt_index.files)
        test_ids = [v for v in manifest_test_ids if v in gt_keys]
        if set(test_ids) != gt_keys:
            raise ValueError(f"{corpus}: GT cohort contains IDs outside test manifest")
        missing_test_asr = sorted(set(test_ids) - set(asr))
        if missing_test_asr:
            raise FileNotFoundError(
                f"{corpus}: {len(missing_test_asr)} test ASR records missing")
        out_dir = run_dir / corpus
        out_dir.mkdir(parents=True, exist_ok=True)
        n_empty_windows = 0
        n_seconds = 0
        with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as fh:
            for vid in test_ids:
                grid_path = GRID_ROOT / corpus / f"{vid}.npy"
                grid = np.load(grid_path, mmap_mode="r")
                if grid.ndim != 2 or grid.shape[0] <= 0:
                    raise ValueError(f"invalid score grid: {grid_path}")
                length = int(grid.shape[0])
                texts, speech = local_texts(
                    asr[vid]["chunks"], length, LEFT_CONTEXT, RIGHT_CONTEXT)
                lexical = model.decision_function(vectorizer.transform(texts))
                if lexical.shape != (length,) or not np.isfinite(lexical).all():
                    raise RuntimeError(f"invalid lexical scores for {corpus}/{vid}")
                record = {
                    "video_id": vid,
                    "score_lexical": lexical.astype(float).tolist(),
                    "score_speech": speech.tolist(),
                }
                fh.write(json.dumps(record) + "\n")
                n_empty_windows += int((speech == 0).sum())
                n_seconds += length
        report[corpus] = {
            "n_train_videos": len(train_ids),
            "train_class_counts": {str(c): int((train_y == c).sum())
                                   for c in (0, 1)},
            "n_train_empty_transcripts": int(sum(not x for x in train_texts)),
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "classifier_n_iter": int(model.n_iter_[0]),
            "asr_chunk_filtering": asr_stats,
            "n_test_manifest": len(manifest_test_ids),
            "n_test_videos": len(test_ids),
            "n_test_manifest_without_gt": len(set(manifest_test_ids) - gt_keys),
            "n_test_seconds": n_seconds,
            "n_empty_test_windows": n_empty_windows,
        }
        print(corpus, report[corpus], flush=True)
    (run_dir / "producer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
