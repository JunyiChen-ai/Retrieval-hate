#!/usr/bin/env python3
"""Qwen3-VL pointwise vs within-video listwise teacher validation smoke."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from src.vlm_windows import (  # noqa: E402
    load_timestamped_asr,
    temporal_windows,
    window_asr,
    window_frame_paths,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
LABELS = "ABCD"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True,
                   choices=("hatemm", "hateclipseg"))
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def point_prompt(transcript):
    return (
        "Judge hateful content in THIS 16-second window only. Hateful means an "
        "attack, slur, dehumanization, or incitement against a protected group; "
        "do not confuse generic profanity or criticism with hate. Use frames, "
        "on-screen text, and transcript. Return one integer 0 to 10 only.\n"
        f'Transcript: "{transcript}"'
    )


def rank_instruction(labels):
    return (
        "Rank these windows from MOST to LEAST hateful evidence. Compare them "
        "within this same video, so ignore the video's general topic/style. "
        "Hateful means attack, slur, dehumanization, or incitement against a "
        "protected group; generic profanity is not enough. Return every one of "
        f"these labels exactly once: {', '.join(labels)}. Separate the labels "
        "with > and give no explanation. Do not preserve presentation order "
        "unless the evidence actually warrants it."
    )


def generate(model, processor, content, max_new_tokens):
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = [item["image"] for item in content if item["type"] == "image"]
    inputs = processor(
        text=[prompt], images=images or None, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def parse_integer(text):
    for token in re.findall(r"\b\d+\b", text):
        value = int(token)
        if 0 <= value <= 10:
            return value
    return None


def parse_ranking(text, labels):
    seen = []
    for token in re.findall(r"\b[A-D]\b", text.upper()):
        if token in labels and token not in seen:
            seen.append(token)
    return seen if len(seen) == len(labels) else None


def load_images(paths):
    return [Image.open(path).convert("RGB") for path in paths]


def densify(windows, values, length):
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for (start, end), value in zip(windows, values):
        total[start:end] += value
        count[start:end] += 1
    return total / np.maximum(count, 1)


def main(argv=None):
    args = parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = hdata.load_labels(args.corpus)
    _, val_ids = hdata.load_train_val(args.corpus, labels)
    gt = hdata.gt_arrays(args.corpus, "val")
    videos = sorted(
        video_id for video_id in val_ids
        if labels[video_id] == 1 and video_id in gt
    )[:args.limit]
    asr, asr_path = load_timestamped_asr(REPO, args.corpus)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    point_scores = {}
    list_scores = {}
    rows = []
    raw_path = args.out_dir / f"{args.corpus}_raw.jsonl"
    point_unparsed = rank_unparsed = rank_calls = 0
    with raw_path.open("w") as raw:
        for video_index, video_id in enumerate(videos):
            length = len(gt[video_id])
            windows = temporal_windows(length)
            chunks = asr.get(video_id, [])
            materials = []
            points = []
            for window_index, (start, end) in enumerate(windows):
                paths = window_frame_paths(
                    REPO, args.corpus, video_id, start, end, 2
                )
                transcript = window_asr(chunks, start, end)
                images = load_images(paths)
                content = [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": point_prompt(transcript)},
                ]
                generated = generate(model, processor, content, 8)
                score = parse_integer(generated)
                if score is None:
                    point_unparsed += 1
                    score = 0
                points.append(score)
                materials.append((images, transcript))
                raw.write(json.dumps({
                    "kind": "point", "corpus": args.corpus,
                    "video_id": video_id, "window_index": window_index,
                    "span": [start, end], "generation": generated,
                    "parsed": score,
                }) + "\n")

            borda = np.zeros(len(windows), dtype=np.float64)
            appearances = np.zeros(len(windows), dtype=np.float64)
            starts = list(range(0, len(windows), 3))
            for group_start in starts:
                indices = list(range(group_start, min(group_start + 4, len(windows))))
                if len(indices) < 2:
                    continue
                for presented in (indices, list(reversed(indices))):
                    local_labels = LABELS[:len(presented)]
                    content = []
                    for local_label, window_index in zip(local_labels, presented):
                        images, transcript = materials[window_index]
                        content.append({
                            "type": "text",
                            "text": f"WINDOW {local_label}:"
                        })
                        content.extend(
                            {"type": "image", "image": image}
                            for image in images
                        )
                        content.append({
                            "type": "text",
                            "text": f'Transcript: "{transcript}"'
                        })
                    content.append({
                        "type": "text",
                        "text": rank_instruction(list(local_labels)),
                    })
                    generated = generate(model, processor, content, 16)
                    ranking = parse_ranking(generated, set(local_labels))
                    rank_calls += 1
                    if ranking is None:
                        rank_unparsed += 1
                    else:
                        label_to_index = dict(zip(local_labels, presented))
                        for rank, label in enumerate(ranking):
                            index = label_to_index[label]
                            borda[index] += len(ranking) - 1 - rank
                            appearances[index] += 1
                    raw.write(json.dumps({
                        "kind": "list", "corpus": args.corpus,
                        "video_id": video_id, "presented": presented,
                        "generation": generated, "parsed": ranking,
                    }) + "\n")
                    raw.flush()
            window_list_score = borda / np.maximum(appearances, 1)
            point_scores[video_id] = densify(windows, points, length)
            list_scores[video_id] = densify(windows, window_list_score, length)
            rows.append({
                "video_id": video_id,
                "length": length,
                "windows": windows,
                "point_scores": points,
                "listwise_scores": window_list_score.tolist(),
                "appearances": appearances.tolist(),
            })
            print(
                f"PROGRESS {args.corpus} {video_index + 1}/{len(videos)} "
                f"{video_id} windows={len(windows)}", flush=True
            )

    subset_gt = {video_id: gt[video_id] for video_id in videos}
    positive_ids = set(videos)
    reports = {
        "pointwise": evaluate_scores(point_scores, subset_gt, positive_ids),
        "listwise": evaluate_scores(list_scores, subset_gt, positive_ids),
    }
    payload = {
        "stage": "validation_smoke",
        "corpus": args.corpus,
        "videos": videos,
        "model": MODEL_ID,
        "window_seconds": 16,
        "stride_seconds": 8,
        "frames_per_window": 2,
        "asr_artifact": str(asr_path.resolve()),
        "point_unparsed": point_unparsed,
        "rank_unparsed": rank_unparsed,
        "rank_calls": rank_calls,
        "metrics": {
            name: {
                "within_roc": report["per_video"]["macro_auc"],
                "within_n": report["per_video"]["n_videos_both_classes"],
            }
            for name, report in reports.items()
        },
        "rows": rows,
        "test_labels_read": False,
    }
    out = args.out_dir / f"{args.corpus}_summary.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
