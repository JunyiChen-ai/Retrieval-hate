#!/usr/bin/env python
"""Reproduce the fixed lexical probe with strict word timing, then test it."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
sys.path.insert(0, str(ROOT / "src"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from lexical_locality import (load_asr, local_texts, new_classifier,
                              new_vectorizer)  # noqa: E402

EXPECTED_RECIPE_BASE = {
    "model": "openai/whisper-large-v3",
    "language": "en",
    "decoding": "greedy",
    "timestamp_mode": "word_strict_no_fallback",
    "long_form": "whisper_native",
    "implementation_version": "content-locked-word-timing-v1",
}


def read_score_branch(path, branch):
    scores = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scores[str(row["video_id"])] = np.asarray(row[branch], dtype=float)
    return scores


def normalized_token(token):
    return re.sub(r"[^\w]+", "", token, flags=re.UNICODE).lower()


def content_locked_chunks(old_chunks, new_chunks, duration):
    """Assign new ASR time evidence to the exact old transcript tokens."""
    old_text = " ".join(text for _, _, text in old_chunks)
    old_tokens = old_text.split()
    if not old_tokens:
        return [], {"old_tokens": len(old_tokens), "matched_tokens": 0,
                    "match_fraction": 1.0, "identity_fallback": False}
    if not new_chunks:
        return list(old_chunks), {"old_tokens": len(old_tokens),
                                  "matched_tokens": 0,
                                  "match_fraction": 0.0,
                                  "identity_fallback": True}
    new_tokens = [str(chunk[2]).strip() for chunk in new_chunks]
    old_norm = [normalized_token(token) for token in old_tokens]
    new_norm = [normalized_token(token) for token in new_tokens]
    matcher = difflib.SequenceMatcher(None, old_norm, new_norm, autojunk=False)
    links = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            if old_norm[block.a + offset]:
                links[block.a + offset] = block.b + offset
    anchors = {i: 0.5 * (new_chunks[j][0] + new_chunks[j][1])
               for i, j in links.items()}
    centers = np.zeros(len(old_tokens), dtype=float)
    anchor_ids = sorted(anchors)
    if not anchor_ids:
        centers = np.linspace(0.0, max(duration, 0.0), len(old_tokens) + 2)[1:-1]
    else:
        knots_i = [-1] + anchor_ids + [len(old_tokens)]
        knots_t = [0.0] + [anchors[i] for i in anchor_ids] + [duration]
        centers = np.interp(np.arange(len(old_tokens)), knots_i, knots_t)
    chunks = []
    epsilon = min(0.01, max(duration, 1e-3) / 1000.0)
    starts = np.clip(centers, 0.0, max(duration - epsilon, 0.0))
    for index, token in enumerate(old_tokens):
        start = float(starts[index])
        end = start + epsilon
        chunks.append((float(start), float(end), token))
    reconstructed = " ".join(token for _, _, token in chunks)
    if reconstructed != " ".join(old_tokens):
        raise RuntimeError("content lock failed to preserve old transcript tokens")
    return chunks, {"old_tokens": len(old_tokens),
                    "matched_tokens": len(links),
                    "match_fraction": len(links) / len(old_tokens),
                    "identity_fallback": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    parser.add_argument("--word-asr", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    old_asr_path = (ROOT / "results/reproduction/asr" /
                    f"{args.corpus}_all/timestamped_chunks.jsonl")
    old_asr, _ = load_asr(old_asr_path)
    word_asr, word_filter_stats = load_asr(Path(args.word_asr))
    labels_path = (ROOT / "results/reproduction/splits/scoped_labels" /
                   f"{args.corpus}_train.json")
    label_payload = json.loads(labels_path.read_text())
    if label_payload.get("corpus") != args.corpus or label_payload.get("split") != "train":
        raise ValueError("scoped label corpus/split identity mismatch")
    labels = {str(k): int(v) for k, v in label_payload["labels"].items()}
    train_ids = hdata.load_split(args.corpus, "train")
    if set(labels) != set(train_ids) or set(labels.values()) != {0, 1}:
        raise ValueError("scoped labels do not exactly match binary train split")
    vectorizer = new_vectorizer()
    train_x = vectorizer.fit_transform([old_asr[v]["text"] for v in train_ids])
    model = new_classifier().fit(
        train_x, np.asarray([labels[v] for v in train_ids], dtype=int))

    test_ids = hdata.load_split(args.corpus, "test")
    gt = hdata.gt_arrays(args.corpus, "test")
    test_ids = [v for v in test_ids if v in gt]
    if set(word_asr) != set(hdata.load_split(args.corpus, "test")):
        raise ValueError("word ASR does not exactly cover test manifest")
    new_scores = {}
    speech_scores = {}
    statuses = {}
    raw_rows = {}
    with Path(args.word_asr).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id in raw_rows:
                raise ValueError("duplicate raw word ASR row")
            raw_rows[video_id] = row
            status = row.get("status", "UNKNOWN")
            statuses[status] = statuses.get(status, 0) + 1
            if row.get("corpus") != args.corpus or row.get("split") != "test":
                raise ValueError("word ASR row corpus/split mismatch")
            expected_recipe = {"corpus": args.corpus, "split": "test",
                               **EXPECTED_RECIPE_BASE}
            if row.get("recipe") != expected_recipe:
                raise ValueError("word ASR row generation recipe mismatch")
            if row.get("timestamp_mode") != "word_strict_no_fallback":
                raise ValueError("word ASR row is not strict word timestamp")
            chunks = row.get("chunks", [])
            if status == "OK" and not chunks:
                raise ValueError("OK word ASR row has no chunks")
            if status in {"NO_AUDIO", "EMPTY_SPEECH"} and chunks:
                raise ValueError("non-OK word ASR row unexpectedly has chunks")
    allowed_statuses = {"OK", "NO_AUDIO", "EMPTY_SPEECH"}
    if not set(statuses).issubset(allowed_statuses):
        raise ValueError(f"word ASR has rejected statuses: {sorted(set(statuses)-allowed_statuses)}")
    if any(word_filter_stats.values()):
        raise ValueError(f"word loader dropped invalid chunks: {word_filter_stats}")
    alignment_rows = []
    score_path = run_dir / "scores.jsonl"
    with score_path.open("w", encoding="utf-8") as handle:
        for video_id in test_ids:
            grid = np.load(ROOT / "results/reproduction/features/clip_b16_1fps" /
                           args.corpus / f"{video_id}.npy", mmap_mode="r")
            duration = float(raw_rows[video_id].get("analyzed_duration") or len(grid))
            locked, alignment = content_locked_chunks(
                old_asr[video_id]["chunks"], word_asr[video_id]["chunks"], duration)
            if not alignment["identity_fallback"]:
                if any(not (np.isfinite(start) and np.isfinite(end) and
                            0 <= start < end <= duration + 1e-8)
                       for start, end, _ in locked):
                    raise ValueError("content-locked token interval is invalid")
                if any(locked[i][0] > locked[i + 1][0]
                       for i in range(len(locked) - 1)):
                    raise ValueError("content-locked token starts are not monotonic")
            alignment_rows.append(alignment)
            texts, speech = local_texts(locked, len(grid), 2.0, 3.0)
            score = model.decision_function(vectorizer.transform(texts))
            new_scores[video_id] = np.asarray(score, dtype=float)
            speech_scores[video_id] = speech
            handle.write(json.dumps({"video_id": video_id,
                                     "score_lexical": score.tolist(),
                                     "score_speech": speech.tolist()}) + "\n")

    labels_all = hdata.load_labels(args.corpus)
    hate_ids = {v for v in gt if labels_all.get(v) == 1}
    old_score_path = (ROOT / "runs/20260831_video_label_lexical_locality/premise" /
                      args.corpus / "scores.jsonl")
    old_scores = read_score_branch(old_score_path, "score_lexical")
    old_metrics = evaluate_scores(old_scores, gt, hate_ids)
    new_metrics = evaluate_scores(new_scores, gt, hate_ids)
    speech_metrics = evaluate_scores(speech_scores, gt, hate_ids)
    deltas = {
        "pr_auc": float(new_metrics["pr_auc"] - old_metrics["pr_auc"]),
        "roc_auc": float(new_metrics["roc_auc"] - old_metrics["roc_auc"]),
        "within_video_macro_roc_auc": float(
            new_metrics["per_video"]["macro_auc"] -
            old_metrics["per_video"]["macro_auc"]),
    }
    gate = {
        "exact_test_gt_coverage": set(new_scores) == set(gt),
        "within_improves_at_least_020":
            deltas["within_video_macro_roc_auc"] >= 0.020,
        "pooled_ap_drop_at_most_020": deltas["pr_auc"] >= -0.020,
        "pooled_roc_drop_at_most_020": deltas["roc_auc"] >= -0.020,
        "only_allowed_asr_statuses": set(statuses).issubset(allowed_statuses),
        "no_invalid_chunks_dropped": not any(word_filter_stats.values()),
        "mean_exact_token_match_fraction_at_least_070":
            float(np.mean([x["match_fraction"] for x in alignment_rows])) >= 0.70,
    }
    gate["pass"] = all(gate.values())
    artifact = {
        "developmental_test_evidence": True,
        "corpus": args.corpus,
        "training": "same-corpus train video labels; frozen old whole transcripts",
        "changed_factor": "test local timing only; exact old transcript tokens are content-locked",
        "word_asr_status_counts": statuses,
        "word_chunk_filtering": word_filter_stats,
        "content_lock_alignment": {
            "mean_exact_token_match_fraction": float(np.mean(
                [x["match_fraction"] for x in alignment_rows])),
            "total_old_tokens": int(sum(x["old_tokens"] for x in alignment_rows)),
            "total_matched_tokens": int(sum(x["matched_tokens"] for x in alignment_rows)),
            "identity_fallback_videos": int(sum(
                x["identity_fallback"] for x in alignment_rows)),
        },
        "old_chunk_lexical": old_metrics,
        "word_aligned_lexical": new_metrics,
        "word_speech_presence_control": speech_metrics,
        "word_minus_old": deltas,
        "gate": gate,
        "decision": "CONTINUE" if gate["pass"] else "STOP",
    }
    (run_dir / "metrics.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"corpus": args.corpus, "deltas": deltas,
                      "gate": gate, "decision": artifact["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
