"""Shared model-free OmniVTG prediction and temporal-control utilities.

This module does not read splits or labels.  It validates the frozen prediction
schema, converts intervals to the canonical one-second grid through the shared
frame utility, and defines the deterministic block-rotation corruption.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


MODEL_ID = "zhengmh/OmniVTG-7B"
QUERY = (
    "content that attacks, dehumanizes, threatens, or expresses hatred toward "
    "a person or group based on identity"
)
RAW_CONTRACT = "omnivtg_positive_test_teacher_v1"
CORRUPTION_CONTRACT = "omnivtg_block_rotation_v1"
BLOCK_SECONDS = 8.0
RAW_ROW_FIELDS = {
    "contract_version", "video_id", "corpus", "split", "model", "query",
    "source_video", "parse_ok", "interval_seconds", "completion",
    "error_type", "error_message", "traceback",
}
CORRUPTED_ROW_FIELDS = {
    "contract_version", "video_id", "corpus", "split", "model", "query",
    "source_video", "corruption", "raw_parse_ok", "parse_ok",
    "interval_seconds", "completion", "error_type", "error_message", "traceback",
}


def parse_interval(text):
    if not isinstance(text, str):
        return None
    blocks = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not blocks:
        return None
    matches = re.findall(
        r"From\s+([0-9]+(?:\.[0-9]+)?)\s+seconds\s+to\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s+seconds",
        blocks[-1], flags=re.IGNORECASE,
    )
    if not matches:
        return None
    start, end = map(float, matches[-1])
    return [start, end] if end >= start else None


def validate_raw_row(row: dict, corpus: str | None = None) -> None:
    if set(row) != RAW_ROW_FIELDS:
        raise RuntimeError("invalid frozen OmniVTG row schema")
    if (
        row["contract_version"] != RAW_CONTRACT
        or row["split"] != "test"
        or row["model"] != MODEL_ID
        or row["query"] != QUERY
        or (corpus is not None and row["corpus"] != corpus)
        or not isinstance(row["video_id"], str)
        or not row["video_id"]
        or not isinstance(row["source_video"], str)
        or not row["source_video"]
        or not isinstance(row["parse_ok"], bool)
    ):
        raise RuntimeError("frozen OmniVTG row provenance mismatch")
    parsed = parse_interval(row["completion"])
    if row["parse_ok"]:
        if parsed is None or row["interval_seconds"] != parsed:
            raise RuntimeError("successful frozen row has invalid interval")
        if any(row[key] is not None for key in (
            "error_type", "error_message", "traceback"
        )):
            raise RuntimeError("successful frozen row carries an error")
    elif row["interval_seconds"] is not None:
        raise RuntimeError("failed frozen row carries an interval")


def load_raw_rows(path: Path, corpus: str) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_raw_row(row, corpus)
            except Exception as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
            video_id = row["video_id"]
            if video_id in rows:
                raise RuntimeError(f"duplicate prediction row: {video_id}")
            rows[video_id] = row
    if not rows:
        raise RuntimeError(f"empty prediction file: {path}")
    return rows


def validate_corrupted_row(row: dict, corpus: str,
                           raw_rows: dict[str, dict]) -> None:
    if set(row) != CORRUPTED_ROW_FIELDS:
        raise RuntimeError("corruption row schema mismatch")
    video_id = row.get("video_id")
    if (
        row.get("contract_version") != CORRUPTION_CONTRACT
        or row.get("corpus") != corpus
        or row.get("split") != "test"
        or row.get("model") != MODEL_ID
        or row.get("query") != QUERY
        or video_id not in raw_rows
        or Path(row.get("source_video", "")).resolve()
           != Path(raw_rows[video_id]["source_video"]).resolve()
        or row.get("raw_parse_ok") is not raw_rows[video_id]["parse_ok"]
        or not isinstance(row.get("parse_ok"), bool)
    ):
        raise RuntimeError("corruption row provenance mismatch")
    corruption = row.get("corruption")
    if not isinstance(corruption, dict):
        raise RuntimeError("missing corruption plan")
    expected = block_rotation_plan(float(corruption["duration_seconds"]), BLOCK_SECONDS)
    if corruption != expected:
        raise RuntimeError("saved corruption plan is not canonical")
    if row["parse_ok"]:
        interval = row.get("interval_seconds")
        if (not (isinstance(interval, list) and len(interval) == 2)
                or parse_interval(row.get("completion")) != interval):
            raise RuntimeError("successful corruption row has invalid interval")
        if any(row[key] is not None for key in (
            "error_type", "error_message", "traceback"
        )):
            raise RuntimeError("successful corruption row carries error")
    elif row.get("interval_seconds") is not None:
        raise RuntimeError("failed corruption row carries interval")


def load_corrupted_rows(path: Path, corpus: str,
                        raw_rows: dict[str, dict]) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_corrupted_row(row, corpus, raw_rows)
            except Exception as error:
                raise RuntimeError(f"{path}:{line_number}: {error}") from error
            video_id = row["video_id"]
            if video_id in rows:
                raise RuntimeError(f"duplicate corrupted prediction row: {video_id}")
            rows[video_id] = row
    if set(rows) != set(raw_rows):
        raise RuntimeError("corrupted predictions do not exactly cover raw cohort")
    return rows


def interval_score(interval, length: int, frame_eval_module) -> np.ndarray:
    if length <= 0:
        raise ValueError("score length must be positive")
    if interval is None:
        return np.zeros(length, dtype=np.float64)
    start, end = map(float, interval)
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start <= end):
        raise ValueError("invalid interval")
    score, _ = frame_eval_module.spans_to_frame_scores(
        [(start, end)], [1.0], duration=length, fps=1.0, uncovered=0.0
    )
    return np.asarray(score, dtype=np.float64)


def block_rotation_plan(duration: float, width: float = BLOCK_SECONDS) -> dict:
    if not (math.isfinite(duration) and duration > 0 and width > 0):
        raise ValueError("duration and block width must be positive finite values")
    n_blocks = int(math.ceil(duration / width))
    blocks = [
        [index * width, min((index + 1) * width, duration)]
        for index in range(n_blocks)
    ]
    pivot = n_blocks // 2
    order = list(range(pivot, n_blocks)) + list(range(0, pivot))
    permuted_starts = {}
    cursor = 0.0
    for block_index in order:
        permuted_starts[block_index] = cursor
        cursor += blocks[block_index][1] - blocks[block_index][0]
    return {
        "block_seconds": float(width),
        "duration_seconds": float(duration),
        "n_blocks": n_blocks,
        "order": order,
        "blocks": blocks,
        "permuted_starts": [float(permuted_starts[i]) for i in range(n_blocks)],
    }


def inverse_mapped_interval_score(interval, length: int, plan: dict) -> np.ndarray:
    if interval is None:
        return np.zeros(length, dtype=np.float64)
    if int(plan["n_blocks"]) < 2:
        raise ValueError("block corruption requires at least two blocks")
    start, end = map(float, interval)
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start <= end):
        raise ValueError("invalid corrupted interval")
    blocks = plan["blocks"]
    permuted_starts = plan["permuted_starts"]
    duration = float(plan["duration_seconds"])
    score = np.zeros(length, dtype=np.float64)
    for second in range(length):
        if second >= duration:
            raise ValueError("one-second grid extends beyond source media duration")
        block_index = min(int(second // float(plan["block_seconds"])), len(blocks) - 1)
        original_start, original_end = map(float, blocks[block_index])
        if not (original_start <= second < original_end):
            raise ValueError("invalid original-to-block assignment")
        permuted_time = float(permuted_starts[block_index]) + second - original_start
        score[second] = float(start <= permuted_time < end)
    return score
