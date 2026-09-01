#!/usr/bin/env python
"""Create OOF train and train-only val/test lexical evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
sys.path.insert(0, str(ROOT / "src"))
from hate_common import data as hdata  # noqa: E402
from lexical_locality import (load_asr, local_texts, new_classifier,
                              new_vectorizer)  # noqa: E402
from scoped_video_protocol import (evaluator_test_ids,
                                   scoped_video_labels)  # noqa: E402

CORPORA = ("hatemm", "hateclipseg")
ASR_PATHS = {
    corpus: ROOT / "results/reproduction/asr" / f"{corpus}_all" /
    "timestamped_chunks.jsonl"
    for corpus in CORPORA
}
GRID_ROOT = ROOT / "results/reproduction/features/clip_b16_1fps"
N_FOLDS = 5


def grid_length(corpus: str, vid: str) -> int:
    arr = np.load(GRID_ROOT / corpus / f"{vid}.npy", mmap_mode="r")
    if arr.ndim != 2 or arr.shape[0] <= 0:
        raise ValueError(f"invalid score grid for {corpus}/{vid}: {arr.shape}")
    return int(arr.shape[0])


def score_video(vectorizer, model, asr_row: dict, length: int
                ) -> tuple[np.ndarray, np.ndarray]:
    texts, speech = local_texts(asr_row["chunks"], length)
    evidence = model.decision_function(vectorizer.transform(texts)).astype(np.float64)
    if evidence.shape != (length,) or not np.isfinite(evidence).all():
        raise RuntimeError("invalid lexical evidence")
    return evidence, speech


def fit(texts: list[str], labels: np.ndarray):
    vectorizer = new_vectorizer()
    x = vectorizer.fit_transform(texts)
    model = new_classifier()
    model.fit(x, labels)
    return vectorizer, model


def save_npz(path: Path, rows: dict[str, np.ndarray]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty evidence archive: {path}")
    np.savez_compressed(path, **rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    config = {
        "corpora": list(CORPORA), "oof_folds": N_FOLDS,
        "lexical_representation": "char_wb TF-IDF 3-5 gram",
        "local_window_seconds": [-2.0, 3.0],
        "test_gt_values_read": False,
        "code_version": "working-tree source reviewed before formal run",
    }
    (out_root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_root / "code_version.txt").write_text(
        "Reviewed working-tree implementation: "
        "experiments/20260831_lexical_posterior_regularization plus "
        "src/lexical_locality.py; 2026-08-31.\n")
    report = {}
    for corpus in CORPORA:
        corpus_dir = out_root / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)
        asr, asr_stats = load_asr(ASR_PATHS[corpus])
        train_ids = hdata.load_split(corpus, "train")
        val_ids = hdata.load_split(corpus, "val")
        test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
        labels = scoped_video_labels(corpus, "train", train_ids)
        needed = set(train_ids) | set(val_ids) | set(test_ids)
        if not needed.issubset(asr):
            raise FileNotFoundError(f"{corpus}: ASR does not cover required cohort")
        y = np.asarray([labels[v] for v in train_ids], dtype=np.int64)
        whole_text = [asr[v]["text"] for v in train_ids]
        oof_evidence: dict[str, np.ndarray] = {}
        oof_speech: dict[str, np.ndarray] = {}
        folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=234)
        fold_rows = []
        for fold, (fit_idx, held_idx) in enumerate(folds.split(train_ids, y)):
            vectorizer, model = fit([whole_text[i] for i in fit_idx], y[fit_idx])
            for i in held_idx:
                vid = train_ids[int(i)]
                evidence, speech = score_video(
                    vectorizer, model, asr[vid], grid_length(corpus, vid))
                oof_evidence[vid] = evidence
                oof_speech[vid] = speech
            fold_rows.append({"fold": fold, "n_fit": int(len(fit_idx)),
                              "n_held": int(len(held_idx)),
                              "vocabulary_size": int(len(vectorizer.vocabulary_)),
                              "classifier_n_iter": int(model.n_iter_[0])})
        if set(oof_evidence) != set(train_ids):
            raise RuntimeError(f"{corpus}: OOF evidence coverage failure")

        vectorizer, model = fit(whole_text, y)
        split_rows = {
            "train": (oof_evidence, oof_speech),
            "val": ({}, {}),
            "test": ({}, {}),
        }
        for split, ids in (("val", val_ids), ("test", test_ids)):
            for vid in ids:
                evidence, speech = score_video(
                    vectorizer, model, asr[vid], grid_length(corpus, vid))
                split_rows[split][0][vid] = evidence
                split_rows[split][1][vid] = speech
        for split, (evidence, speech) in split_rows.items():
            save_npz(corpus_dir / f"{split}_evidence.npz", evidence)
            save_npz(corpus_dir / f"{split}_speech.npz", speech)
        report[corpus] = {
            "n_train": len(train_ids), "n_val": len(val_ids),
            "n_test": len(test_ids), "n_folds": N_FOLDS,
            "train_class_counts": {str(c): int((y == c).sum()) for c in (0, 1)},
            "folds": fold_rows,
            "full_vocabulary_size": int(len(vectorizer.vocabulary_)),
            "full_classifier_n_iter": int(model.n_iter_[0]),
            "asr_chunk_filtering": asr_stats,
            "test_gt_values_read": False,
        }
        print(corpus, report[corpus], flush=True)
    (out_root / "evidence_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
