#!/usr/bin/env python3
"""Generate full-test Qwen3-VL pointwise window scores without reading GT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
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


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CHOICES = tuple(str(value) for value in range(11))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "hateclipseg"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def prompt(transcript: str) -> str:
    return (
        "Judge hateful content in THIS 16-second window only. Hateful means "
        "an attack, slur, dehumanization, exclusion, or incitement against a "
        "protected group. Generic profanity, aggression, criticism, sexual "
        "content, or violence without a protected target is not enough. Use "
        "the frames, visible text, and transcript. Rate explicit hateful "
        "evidence from 0 (none) to 10 (unambiguous). Your first output token "
        "must be exactly one integer from 0 through 10.\n"
        f'Transcript: "{transcript}"'
    )


def load_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def choice_token_ids(tokenizer):
    ids = []
    for choice in CHOICES:
        encoded = tokenizer.encode(choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"choice {choice!r} is not one token: {encoded}"
            )
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError("choice token ids are not unique")
    return ids


@torch.inference_mode()
def score_window(model, processor, token_ids, images, transcript, device):
    content = [
        *({"type": "image", "image": image} for image in images),
        {"type": "text", "text": prompt(transcript)},
    ]
    messages = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[rendered], images=images, return_tensors="pt"
    ).to(device)
    output = model(**inputs, use_cache=False)
    logits = output.logits[0, -1, token_ids].float()
    probabilities = torch.softmax(logits, dim=0)
    values = torch.arange(11, dtype=torch.float32, device=logits.device)
    expected = float((probabilities * values).sum().cpu()) / 10.0
    return expected, [float(value) for value in probabilities.cpu()]


def densify(windows, window_scores, length):
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.int64)
    for (start, end), score in zip(windows, window_scores):
        total[start:end] += score
        count[start:end] += 1
    if not np.all(count > 0):
        missing = np.flatnonzero(count == 0)[:10].tolist()
        raise RuntimeError(f"window grid left seconds uncovered: {missing}")
    return total / count


def read_completed(path: Path):
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = (row["video_id"], int(row["window_index"]))
            if key in completed:
                raise RuntimeError(
                    f"duplicate raw window at {path}:{line_number}: {key}"
                )
            completed[key] = row
    return completed


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "windows.jsonl"
    scores_path = args.out_dir / "scores.jsonl"
    config_path = args.out_dir / "config.json"
    completed = read_completed(raw_path)

    test_ids = hdata.load_split(args.corpus, "test")
    lengths = {
        video_id: int(np.load(
            hdata.feature_path(args.corpus, video_id), mmap_mode="r"
        ).shape[0])
        for video_id in test_ids
    }
    asr, asr_path = load_timestamped_asr(REPO, args.corpus)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    token_ids = choice_token_ids(processor.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        local_files_only=True,
    )
    model.eval()

    config = {
        "date": "2026-08-31",
        "status": "iterative/developmental test diagnostic",
        "corpus": args.corpus,
        "split": "test",
        "model": MODEL_ID,
        "model_local_only": True,
        "window_seconds": 16,
        "stride_seconds": 8,
        "frames_per_window": 2,
        "choice_strings": list(CHOICES),
        "choice_token_ids": token_ids,
        "choice_normalization": "softmax restricted to the 11 choice logits",
        "score": "expected integer choice divided by 10",
        "asr_path": str(asr_path.resolve()),
        "test_gt_read_by_generator": False,
        "validation_read": False,
        "code_version": "workspace source snapshot dated 2026-08-31",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    started = time.time()
    total_windows = sum(len(temporal_windows(length))
                        for length in lengths.values())
    done = len(completed)
    with raw_path.open("a") as raw:
        for video_number, video_id in enumerate(test_ids, 1):
            windows = temporal_windows(lengths[video_id])
            chunks = asr.get(video_id, [])
            for window_index, (start, end) in enumerate(windows):
                key = (video_id, window_index)
                if key in completed:
                    continue
                paths = window_frame_paths(
                    REPO, args.corpus, video_id, start, end, 2
                )
                if not paths:
                    raise RuntimeError(
                        f"no frame for {video_id} window {start}:{end}"
                    )
                images = load_images(paths)
                transcript = window_asr(chunks, start, end)
                score, probabilities = score_window(
                    model, processor, token_ids, images, transcript, args.device
                )
                row = {
                    "video_id": video_id,
                    "window_index": window_index,
                    "span": [start, end],
                    "frame_paths": [str(path.resolve()) for path in paths],
                    "transcript": transcript,
                    "choice_probabilities": probabilities,
                    "score_qwen3_pointwise": score,
                }
                raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw.flush()
                completed[key] = row
                done += 1
                if done == 1 or done % 25 == 0 or done == total_windows:
                    elapsed = time.time() - started
                    print(json.dumps({
                        "progress_windows": done,
                        "total_windows": total_windows,
                        "video": f"{video_number}/{len(test_ids)}",
                        "video_id": video_id,
                        "elapsed_seconds_this_process": elapsed,
                    }), flush=True)

    scores = {}
    for video_id in test_ids:
        windows = temporal_windows(lengths[video_id])
        window_scores = []
        for window_index in range(len(windows)):
            key = (video_id, window_index)
            if key not in completed:
                raise RuntimeError(f"missing completed window {key}")
            window_scores.append(
                float(completed[key]["score_qwen3_pointwise"])
            )
        scores[video_id] = densify(windows, window_scores, lengths[video_id])

    temporary = scores_path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as handle:
        for video_id in test_ids:
            handle.write(json.dumps({
                "video_id": video_id,
                "score_qwen3_pointwise": scores[video_id].tolist(),
            }) + "\n")
    temporary.replace(scores_path)
    print(json.dumps({
        "status": "complete",
        "corpus": args.corpus,
        "videos": len(test_ids),
        "windows": total_windows,
        "scores": str(scores_path.resolve()),
    }), flush=True)


if __name__ == "__main__":
    main()

