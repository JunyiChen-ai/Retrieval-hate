"""Shared read-only 1 fps frame/ASR window access for VLM diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FRAME_DIRS = {
    "hatemm": "HateMM",
    "mhclip_en": "MHC",
    "mhclip_zh": "MHC_zh",
    "hateclipseg": "HateClipSeg",
}
ASR_RUNS = {
    "hatemm": "hatemm_all",
    "mhclip_en": "mhclip_en_all",
    "mhclip_zh": "mhclip_zh_all",
    "hateclipseg": "hateclipseg_all",
}


def load_timestamped_asr(repo: Path, corpus: str):
    table = {}
    path = (
        repo / "results/reproduction/asr" / ASR_RUNS[corpus]
        / "timestamped_chunks.jsonl"
    )
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            table[row["video_id"]] = row.get("chunks") or []
    return table, path


def window_asr(chunks, start: int, end: int, cap: int = 600):
    parts = [
        chunk["text"]
        for chunk in chunks
        if chunk.get("start") is not None
        and chunk.get("end") is not None
        and chunk["end"] > start
        and chunk["start"] < end
    ]
    return " ".join(part.strip() for part in parts).strip()[:cap]


def window_frame_paths(
    repo: Path,
    corpus: str,
    video_id: str,
    start: int,
    end: int,
    count: int,
):
    directory = repo / "data/frames_1fps" / FRAME_DIRS[corpus] / video_id
    if not directory.is_dir():
        return []
    timestamps = np.linspace(start, max(start, end - 1), count)
    paths = []
    for timestamp in timestamps:
        path = directory / f"{int(round(timestamp)):06d}.jpg"
        if path.is_file():
            paths.append(path)
    return sorted(set(paths))


def temporal_windows(length: int, width: int = 16, stride: int = 8):
    windows = []
    for start in range(0, max(1, int(length)), stride):
        end = min(start + width, int(length))
        windows.append((start, end))
        if end >= length:
            break
    return windows
