#!/usr/bin/env python3
"""Generate fixed Qwen3 pointwise scores without opening temporal test GT."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from src.vlm_windows import (  # noqa: E402
    load_timestamped_asr,
    temporal_windows,
    window_asr,
    window_frame_paths,
)
from protocol import (  # noqa: E402
    CODE_VERSION_DESCRIPTION,
    CONTRACT_VERSION,
    CORPORA,
    FRAMES_PER_WINDOW,
    MODEL_ID,
    WINDOW_STRIDE,
    WINDOW_WIDTH,
    VIDEO_LABEL_SOURCES,
    expected_config,
    parse_score,
    positive_test_cohort,
    prompt,
    validate_prediction_row,
)


RUN_ROOT = REPO / "runs/20260831_qwen3_test_teacher_diagnostic/formal"


def generate(model, processor, content):
    message = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    images = [item["image"] for item in content if item["type"] == "image"]
    inputs = processor(
        text=[rendered], images=images or None, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def cohort(corpus: str):
    split_ids = hdata.load_split(corpus, "test")
    payload = json.loads(VIDEO_LABEL_SOURCES[corpus].read_text())
    if payload.get("corpus") != corpus or payload.get("split") != "test":
        raise RuntimeError(f"scoped video-label source mismatch for {corpus}")
    raw_labels = payload.get("labels")
    if not isinstance(raw_labels, dict):
        raise RuntimeError(f"invalid scoped video-label source for {corpus}")
    if set(raw_labels) != set(split_ids):
        raise RuntimeError(f"scoped video-label coverage mismatch for {corpus}")
    labels = {}
    for video_id, value in raw_labels.items():
        if type(value) is not int or value not in (0, 1):
            raise RuntimeError(f"invalid scoped video label for {corpus}/{video_id}")
        labels[video_id] = value
    return positive_test_cohort(corpus, split_ids, labels)


def load_existing(path: Path, corpus: str, expected_ids, expected_lengths):
    rows = []
    if path.is_file():
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                raise RuntimeError("blank resume row")
            if len(rows) >= len(expected_ids):
                raise RuntimeError("resume output contains extra rows")
            row = json.loads(line)
            expected_video_id = expected_ids[len(rows)]
            validate_prediction_row(
                row,
                corpus,
                expected_video_id=expected_video_id,
                expected_length=expected_lengths[expected_video_id],
            )
            rows.append(row)
    ids = [row["video_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate resume video")
    if ids != expected_ids[:len(ids)]:
        raise RuntimeError("resume rows are not an exact cohort prefix")
    return rows


def ensure_run_metadata(output_dir: Path, corpus: str, output_path: Path) -> None:
    config = expected_config(corpus, output_path)
    config_path = output_dir / "config.json"
    version_path = output_dir / "code_version.txt"
    if output_path.exists() and (not config_path.is_file() or not version_path.is_file()):
        raise RuntimeError("predictions exist without complete readable run metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError("formal run config mismatch")
    else:
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        os.replace(temporary, config_path)
    expected_version = CODE_VERSION_DESCRIPTION + "\n"
    if version_path.exists():
        if version_path.read_text() != expected_version:
            raise RuntimeError("formal code version description mismatch")
    else:
        temporary = version_path.with_suffix(".txt.tmp")
        temporary.write_text(expected_version)
        os.replace(temporary, version_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=CORPORA)
    args = parser.parse_args(argv)

    video_ids = cohort(args.corpus)
    lengths = {}
    for video_id in video_ids:
        path = Path(hdata.feature_path(args.corpus, video_id))
        if not path.is_file():
            raise FileNotFoundError(f"missing length feature for {video_id}")
        shape = np.load(path, mmap_mode="r").shape
        if len(shape) != 2 or shape[0] <= 0:
            raise RuntimeError(f"invalid length feature for {video_id}")
        lengths[video_id] = int(shape[0])
    output_dir = RUN_ROOT / args.corpus
    output_path = output_dir / "predictions.jsonl"
    ensure_run_metadata(output_dir, args.corpus, output_path)
    existing = load_existing(output_path, args.corpus, video_ids, lengths)
    print(f"cohort={len(video_ids)} already_complete={len(existing)}", flush=True)

    asr, _ = load_timestamped_asr(REPO, args.corpus)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    with output_path.open("a", encoding="utf-8") as handle:
        for index, video_id in enumerate(video_ids[len(existing):], len(existing) + 1):
            length = lengths[video_id]
            windows = []
            video_failed = False
            for start, end in temporal_windows(
                length, width=WINDOW_WIDTH, stride=WINDOW_STRIDE
            ):
                try:
                    paths = window_frame_paths(
                        REPO, args.corpus, video_id, start, end, FRAMES_PER_WINDOW
                    )
                    images = []
                    for path in paths:
                        with Image.open(path) as image:
                            images.append(image.convert("RGB"))
                    transcript = window_asr(asr.get(video_id, []), start, end)
                    content = [
                        *({"type": "image", "image": image} for image in images),
                        {"type": "text", "text": prompt(transcript, end - start)},
                    ]
                    generation = generate(model, processor, content)
                    parsed = parse_score(generation)
                    status = "ok" if parsed is not None else "parse_failure"
                    score = int(parsed) if parsed is not None else 0
                except Exception as error:
                    generation = f"{type(error).__name__}: {error}"[:300]
                    score = 0
                    status = "inference_failure"
                    video_failed = True
                windows.append({"span": [start, end], "generation": generation,
                                "parsed_score": score, "status": status})
            row = {"contract_version": CONTRACT_VERSION,
                   "corpus": args.corpus, "split": "test_positive",
                   "video_id": video_id, "length": length, "model": MODEL_ID,
                   "windows": windows,
                   "status": "inference_failure" if video_failed else "ok"}
            validate_prediction_row(
                row, args.corpus, expected_video_id=video_id,
                expected_length=length,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"PROGRESS {args.corpus} {index}/{len(video_ids)} {video_id} "
                  f"windows={len(windows)} status={row['status']}", flush=True)
    final = load_existing(output_path, args.corpus, video_ids, lengths)
    if len(final) != len(video_ids):
        raise RuntimeError("cohort incomplete after producer exit")
    print(f"cohort_complete={len(final)}", flush=True)


if __name__ == "__main__":
    main()
