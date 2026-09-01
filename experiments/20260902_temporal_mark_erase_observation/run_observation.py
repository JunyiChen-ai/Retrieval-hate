#!/usr/bin/env python3
"""Generate complete HMM/HCS positive-test mark/erase scores with one Qwen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from hate_common import data as hdata  # noqa: E402
from src.scoped_video_protocol import (  # noqa: E402
    evaluator_test_ids,
    scoped_video_labels,
)
from src.vlm_windows import (  # noqa: E402
    load_timestamped_asr,
    window_asr,
    window_frame_paths,
)
from protocol import (  # noqa: E402
    CODE_VERSION_DESCRIPTION,
    CONTRACT_VERSION,
    CORPORA,
    EXPECTED_POSITIVE_COUNTS,
    FRAMES_PER_SECTION,
    MODEL_ID,
    SCORE_MAX,
    candidate_windows,
    context_span,
    parse_score,
    prompt,
)


RUN_ROOT = REPO / "runs/20260902_temporal_mark_erase_observation/formal"


def cohort(corpus: str):
    split_ids = hdata.load_split(corpus, "test")
    labels = scoped_video_labels(corpus, "test", split_ids)
    eligible = evaluator_test_ids(corpus, split_ids)
    result = sorted(video_id for video_id in eligible if labels[video_id] == 1)
    if len(result) != EXPECTED_POSITIVE_COUNTS[corpus]:
        raise RuntimeError(f"positive cohort count changed for {corpus}: {len(result)}")
    return result


def marked_image(image: Image.Image):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width = max(4, min(result.size) // 40)
    for offset in range(width):
        draw.rectangle(
            [offset, offset, result.width - 1 - offset, result.height - 1 - offset],
            outline=(255, 0, 0),
        )
    return result


def erased_image(size):
    result = Image.new("RGB", size, color=(128, 128, 128))
    draw = ImageDraw.Draw(result)
    draw.text((10, 10), "ERASED CANDIDATE", fill=(255, 255, 255))
    return result


def load_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def generate(model, processor, images, text):
    content = [
        *({"type": "image", "image": image} for image in images),
        {"type": "text", "text": text},
    ]
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[rendered], images=images, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def arm_result(model, processor, images, text):
    try:
        generation = generate(model, processor, images, text)
        parsed = parse_score(generation)
        return {
            "generation": generation,
            "score": int(parsed) if parsed is not None else 0,
            "status": "ok" if parsed is not None else "parse_failure",
        }
    except Exception as error:  # preserve coverage; fail closed
        return {
            "generation": f"{type(error).__name__}: {error}"[:300],
            "score": 0,
            "status": "inference_failure",
        }


def validate_row(row, corpus, expected_id, expected_length):
    required = {"contract_version", "corpus", "split", "video_id", "length", "model", "windows"}
    if set(row) != required or row["contract_version"] != CONTRACT_VERSION:
        raise RuntimeError("row schema/contract mismatch")
    if row["corpus"] != corpus or row["split"] != "test_positive" or row["model"] != MODEL_ID:
        raise RuntimeError("row provenance mismatch")
    if row["video_id"] != expected_id or row["length"] != expected_length:
        raise RuntimeError("row identity/length mismatch")
    spans = [window["span"] for window in row["windows"]]
    expected_spans = [list(span) for span in candidate_windows(expected_length)]
    if spans != expected_spans:
        raise RuntimeError("candidate window coverage mismatch")
    for window in row["windows"]:
        if set(window) != {"span", "context", "marked", "erased"}:
            raise RuntimeError("window schema mismatch")
        start, end = window["span"]
        if window["context"] != list(context_span(expected_length, start, end)):
            raise RuntimeError("context span mismatch")
        for arm in ("marked", "erased"):
            value = window[arm]
            if set(value) != {"generation", "score", "status"}:
                raise RuntimeError("arm schema mismatch")
            if value["status"] not in {"ok", "parse_failure", "inference_failure"}:
                raise RuntimeError("arm status mismatch")
            if type(value["score"]) is not int or not 0 <= value["score"] <= SCORE_MAX:
                raise RuntimeError("arm score mismatch")
            parsed = parse_score(value["generation"])
            if value["status"] == "ok" and parsed != value["score"]:
                raise RuntimeError("arm generation/score mismatch")
            if value["status"] != "ok" and value["score"] != 0:
                raise RuntimeError("failed arm must score zero")


def load_existing(path, corpus, ids, lengths):
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip() or len(rows) >= len(ids):
                raise RuntimeError("invalid resume rows")
            row = json.loads(line)
            video_id = ids[len(rows)]
            validate_row(row, corpus, video_id, lengths[video_id])
            rows.append(row)
    if [row["video_id"] for row in rows] != ids[:len(rows)]:
        raise RuntimeError("resume is not exact cohort prefix")
    return rows


def metadata(output_dir, corpus, output_path):
    config = {
        "contract_version": CONTRACT_VERSION,
        "corpus": corpus,
        "split": "test_positive",
        "model": MODEL_ID,
        "window_width": 16,
        "window_stride": 8,
        "context_before_after": 16,
        "frames_per_section": FRAMES_PER_SECTION,
        "decoding": {"do_sample": False, "max_new_tokens": 8},
        "predictions": str(output_path.resolve()),
        "test_labels_used_for_training_or_selection": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    version_path = output_dir / "code_version.txt"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("existing config differs")
    if version_path.exists() and version_path.read_text() != CODE_VERSION_DESCRIPTION + "\n":
        raise RuntimeError("existing version description differs")
    if not config_path.exists():
        temporary = config_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        os.replace(temporary, config_path)
    if not version_path.exists():
        temporary = version_path.with_suffix(".tmp")
        temporary.write_text(CODE_VERSION_DESCRIPTION + "\n")
        os.replace(temporary, version_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=CORPORA)
    args = parser.parse_args(argv)

    ids = cohort(args.corpus)
    lengths = {}
    for video_id in ids:
        shape = np.load(hdata.feature_path(args.corpus, video_id), mmap_mode="r").shape
        if len(shape) != 2 or shape[0] <= 0:
            raise RuntimeError(f"invalid length feature for {video_id}")
        lengths[video_id] = int(shape[0])
    output_dir = RUN_ROOT / args.corpus
    output_path = output_dir / "predictions.jsonl"
    metadata(output_dir, args.corpus, output_path)
    existing = load_existing(output_path, args.corpus, ids, lengths)
    print(f"cohort={len(ids)} already_complete={len(existing)}", flush=True)

    asr, _ = load_timestamped_asr(REPO, args.corpus)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    with output_path.open("a", encoding="utf-8") as handle:
        for index, video_id in enumerate(ids[len(existing):], len(existing) + 1):
            length = lengths[video_id]
            windows = []
            chunks = asr.get(video_id, [])
            for start, end in candidate_windows(length):
                context_start, context_end = context_span(length, start, end)
                before_paths = window_frame_paths(
                    REPO, args.corpus, video_id, context_start, start, FRAMES_PER_SECTION
                ) if context_start < start else []
                candidate_paths = window_frame_paths(
                    REPO, args.corpus, video_id, start, end, FRAMES_PER_SECTION
                )
                after_paths = window_frame_paths(
                    REPO, args.corpus, video_id, end, context_end, FRAMES_PER_SECTION
                ) if end < context_end else []
                before_images = load_images(before_paths)
                candidate_images = load_images(candidate_paths)
                after_images = load_images(after_paths)
                marked_images = before_images + [marked_image(image) for image in candidate_images] + after_images
                fallback_size = marked_images[0].size if marked_images else (448, 448)
                erased_images = before_images + [erased_image(image.size) for image in candidate_images] + after_images
                if not candidate_images:
                    marked_images = before_images + [marked_image(Image.new("RGB", fallback_size))] + after_images
                    erased_images = before_images + [erased_image(fallback_size)] + after_images
                before_text = window_asr(chunks, context_start, start)
                candidate_text = window_asr(chunks, start, end)
                after_text = window_asr(chunks, end, context_end)
                marked = arm_result(
                    model, processor, marked_images,
                    prompt(before_text, candidate_text, after_text, erased=False),
                )
                erased = arm_result(
                    model, processor, erased_images,
                    prompt(before_text, candidate_text, after_text, erased=True),
                )
                windows.append({
                    "span": [start, end],
                    "context": [context_start, context_end],
                    "marked": marked,
                    "erased": erased,
                })
            row = {
                "contract_version": CONTRACT_VERSION,
                "corpus": args.corpus,
                "split": "test_positive",
                "video_id": video_id,
                "length": length,
                "model": MODEL_ID,
                "windows": windows,
            }
            validate_row(row, args.corpus, video_id, length)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            failures = sum(
                arm["status"] != "ok" for window in windows
                for arm in (window["marked"], window["erased"])
            )
            print(
                f"PROGRESS {args.corpus} {index}/{len(ids)} {video_id} "
                f"windows={len(windows)} failures={failures}", flush=True,
            )
    final = load_existing(output_path, args.corpus, ids, lengths)
    if len(final) != len(ids):
        raise RuntimeError("cohort incomplete")
    print(f"cohort_complete={len(final)}", flush=True)


if __name__ == "__main__":
    main()
