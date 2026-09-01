#!/usr/bin/env python3
"""Run frozen OmniVTG on deterministic within-video block rotations.

The cohort is inherited from the already frozen prediction rows.  This script
does not read any split label, frame label, span label, or evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from omnivtg_protocol import (  # noqa: E402
    BLOCK_SECONDS, CORRUPTION_CONTRACT, MODEL_ID, QUERY,
    block_rotation_plan, load_raw_rows, validate_corrupted_row,
)
from omnivtg_runtime import build_model, infer_one  # noqa: E402


RAW_ROOT = ROOT / "runs/20260831_omnivtg_grounder_diagnostic/formal"
RUN_ROOT = ROOT / "runs/20260831_omnivtg_reopening_controls/main"


def media_duration(path: Path) -> float:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], check=True, capture_output=True, text=True)
    value = float(completed.stdout.strip())
    if not (value > 0):
        raise RuntimeError(f"nonpositive media duration: {path}")
    return value


def make_rotated_video(source: Path, output: Path, plan: dict) -> None:
    if int(plan["n_blocks"]) < 2:
        raise RuntimeError("source has fewer than two fixed-width blocks")
    filters = []
    for index, (start, end) in enumerate(plan["blocks"]):
        filters.append(
            f"[0:v]trim=start={float(start):.6f}:end={float(end):.6f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
    inputs = "".join(f"[v{index}]" for index in plan["order"])
    filters.append(f"{inputs}concat=n={int(plan['n_blocks'])}:v=1:a=0[outv]")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
        "-filter_complex", ";".join(filters), "-map", "[outv]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-movflags", "+faststart", str(output),
    ], check=True)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError("ffmpeg did not create a usable rotated video")


def load_completed(path: Path, corpus: str, raw_rows: dict[str, dict]) -> set[str]:
    if not path.exists():
        return set()
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row.get("video_id")
            if video_id not in raw_rows or video_id in rows:
                raise RuntimeError(f"invalid resumed row at {path}:{line_number}")
            rows[video_id] = row
    # The shared loader requires full coverage; validate partial rows one at a
    # time by restricting the raw mapping to exactly the observed IDs.
    for video_id, row in rows.items():
        validate_corrupted_row(row, corpus, {video_id: raw_rows[video_id]})
    return set(rows)


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("hatemm", "hateclipseg"), required=True)
    args = parser.parse_args()
    raw_path = RAW_ROOT / args.corpus / "predictions.jsonl"
    raw_rows = load_raw_rows(raw_path, args.corpus)
    run_dir = RUN_ROOT / args.corpus
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "corrupted_predictions.jsonl"
    config = {
        "contract_version": CORRUPTION_CONTRACT,
        "corpus": args.corpus,
        "source_predictions": str(raw_path.resolve()),
        "model": MODEL_ID,
        "query": QUERY,
        "block_seconds": BLOCK_SECONDS,
        "permutation": "half rotation of fixed-width blocks",
        "cohort_source": "IDs and media paths from frozen positive-test predictions",
        "reads_test_video_labels": False,
        "reads_frame_or_span_ground_truth": False,
    }
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("existing corruption config mismatch")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    version_path = run_dir / "code_version.txt"
    version_text = "2026-08-31 reviewed OmniVTG fixed block-rotation premise\n"
    if version_path.exists() and version_path.read_text() != version_text:
        raise RuntimeError("existing code version description mismatch")
    if not version_path.exists():
        version_path.write_text(version_text)

    completed = load_completed(output, args.corpus, raw_rows)
    pending = [video_id for video_id in sorted(raw_rows) if video_id not in completed]
    print(json.dumps({"event": "cohort", "corpus": args.corpus,
                      "total": len(raw_rows), "pending": len(pending)}), flush=True)
    if not pending:
        return
    processor, llm, sampling = build_model()
    temporary_dir = run_dir / "temporary_media"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    for index, video_id in enumerate(pending, 1):
        source = Path(raw_rows[video_id]["source_video"]).resolve()
        temporary = temporary_dir / f"{video_id}.block-rotated.mp4"
        plan = None
        try:
            duration = media_duration(source)
            plan = block_rotation_plan(duration, BLOCK_SECONDS)
            make_rotated_video(source, temporary, plan)
            completion, interval = infer_one(temporary, processor, llm, sampling)
            row = {
                "contract_version": CORRUPTION_CONTRACT,
                "video_id": video_id, "corpus": args.corpus, "split": "test",
                "model": MODEL_ID, "query": QUERY, "source_video": str(source),
                "corruption": plan, "raw_parse_ok": raw_rows[video_id]["parse_ok"],
                "parse_ok": interval is not None, "interval_seconds": interval,
                "completion": completion,
                "error_type": None if interval is not None else "ParseFailure",
                "error_message": (None if interval is not None else
                                  "completion did not contain a valid final interval"),
                "traceback": None,
            }
        except Exception as error:
            if plan is None:
                raise
            row = {
                "contract_version": CORRUPTION_CONTRACT,
                "video_id": video_id, "corpus": args.corpus, "split": "test",
                "model": MODEL_ID, "query": QUERY, "source_video": str(source),
                "corruption": plan, "raw_parse_ok": raw_rows[video_id]["parse_ok"],
                "parse_ok": False, "interval_seconds": None, "completion": None,
                "error_type": type(error).__name__,
                "error_message": str(error) or repr(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            if temporary.is_file():
                temporary.unlink()
        append_row(output, row)
        print(json.dumps({"event": "video_complete", "corpus": args.corpus,
                          "index": len(completed) + index, "total": len(raw_rows),
                          "video_id": video_id, "parse_ok": row["parse_ok"],
                          "error_type": row["error_type"]}), flush=True)
    if load_completed(output, args.corpus, raw_rows) != set(raw_rows):
        raise RuntimeError("corruption producer ended without exact cohort")


if __name__ == "__main__":
    main()
