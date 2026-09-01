#!/usr/bin/env python3
"""Qualify Qwen3-VL dense pointwise teacher on complete validation positives."""

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
CORPORA = ("hatemm", "hateclipseg")
BASELINE_WITHIN = {"hatemm": 0.5719312723877894,
                   "hateclipseg": 0.5270723962968439}
ABSOLUTE_GATE = {"hatemm": 0.60, "hateclipseg": 0.56}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def prompt(transcript, seconds):
    return (
        f"Judge hateful content in THIS {seconds}-second window only. Hateful "
        "means an attack, slur, dehumanization, or incitement against a "
        "protected group; do not confuse generic profanity or criticism with "
        "hate. Use frames, on-screen text, and transcript. Return one integer "
        "0 to 10 only.\n"
        f'Transcript: "{transcript}"'
    )


def parse_score(text):
    for token in re.findall(r"\b\d+\b", text):
        value = int(token)
        if 0 <= value <= 10:
            return value
    return None


def generate(model, processor, content):
    message = [{"role": "user", "content": content}]
    rendered = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    images = [item["image"] for item in content if item["type"] == "image"]
    inputs = processor(
        text=[rendered], images=images or None, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def densify(row):
    total = np.zeros(row["length"], dtype=np.float64)
    count = np.zeros(row["length"], dtype=np.float64)
    for (start, end), score in zip(row["spans"], row["scores"]):
        total[start:end] += score
        count[start:end] += 1
    return total / np.maximum(count, 1)


def qualify_corpus(corpus, model, processor, out_dir):
    labels = hdata.load_labels(corpus)
    _, val_ids = hdata.load_train_val(corpus, labels)
    gt = hdata.gt_arrays(corpus, "val")
    video_ids = sorted(
        video_id for video_id in val_ids
        if labels[video_id] == 1 and video_id in gt
    )
    asr, asr_path = load_timestamped_asr(REPO, corpus)
    teacher_path = out_dir / f"teacher_{corpus}.jsonl"
    raw_path = out_dir / f"teacher_{corpus}_raw.jsonl"
    existing = []
    done = set()
    if teacher_path.is_file():
        existing = [json.loads(line) for line in teacher_path.read_text().splitlines()]
        done = {row["video_id"] for row in existing}
    unparsed = 0
    calls = 0
    with teacher_path.open("a") as teacher, raw_path.open("a") as raw:
        for video_index, video_id in enumerate(video_ids):
            if video_id in done:
                continue
            length = len(gt[video_id])
            spans = temporal_windows(length)
            scores = []
            for window_index, (start, end) in enumerate(spans):
                paths = window_frame_paths(
                    REPO, corpus, video_id, start, end, 4
                )
                images = [Image.open(path).convert("RGB") for path in paths]
                transcript = window_asr(asr.get(video_id, []), start, end)
                content = [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt(transcript, end - start)},
                ]
                generated = generate(model, processor, content)
                score = parse_score(generated)
                calls += 1
                if score is None:
                    unparsed += 1
                    score = 0
                scores.append(score)
                raw.write(json.dumps({
                    "corpus": corpus, "video_id": video_id,
                    "window_index": window_index, "span": [start, end],
                    "frame_paths": [str(path) for path in paths],
                    "generation": generated, "parsed_score": score,
                }) + "\n")
            teacher.write(json.dumps({
                "video_id": video_id, "length": length,
                "spans": spans, "scores": scores,
            }) + "\n")
            teacher.flush()
            raw.flush()
            print(
                f"PROGRESS {corpus} {video_index + 1}/{len(video_ids)} "
                f"{video_id} windows={len(spans)}", flush=True
            )
    rows = [json.loads(line) for line in teacher_path.read_text().splitlines()]
    row_by_id = {row["video_id"]: row for row in rows}
    if set(row_by_id) != set(video_ids):
        raise RuntimeError(f"incomplete teacher coverage for {corpus}")
    scores = {video_id: densify(row_by_id[video_id]) for video_id in video_ids}
    subset_gt = {video_id: gt[video_id] for video_id in video_ids}
    report = evaluate_scores(scores, subset_gt, set(video_ids))
    total_calls = sum(len(row_by_id[video_id]["scores"]) for video_id in video_ids)
    total_unparsed = sum(
        1 for line in raw_path.read_text().splitlines()
        if json.loads(line).get("generation") is not None
        and parse_score(json.loads(line)["generation"]) is None
    )
    within = report["per_video"]["macro_auc"]
    checks = {
        "full_coverage": len(row_by_id) == len(video_ids),
        "parse_failure_below_0.01": total_unparsed / max(1, total_calls) < 0.01,
        "absolute_within_gate": within >= ABSOLUTE_GATE[corpus],
        "gain_over_powa_at_least_0.020": (
            within >= BASELINE_WITHIN[corpus] + 0.020
        ),
    }
    return {
        "corpus": corpus,
        "split": "val_positive_videos",
        "videos": len(video_ids),
        "calls": total_calls,
        "unparsed": total_unparsed,
        "unparsed_rate": total_unparsed / max(1, total_calls),
        "teacher_artifact": str(teacher_path.resolve()),
        "raw_artifact": str(raw_path.resolve()),
        "asr_artifact": str(asr_path.resolve()),
        "powa_within_reference": BASELINE_WITHIN[corpus],
        "teacher_within_roc": within,
        "within_n": report["per_video"]["n_videos_both_classes"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def main(argv=None):
    args = parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    corpora = {
        corpus: qualify_corpus(corpus, model, processor, args.out_dir)
        for corpus in CORPORA
    }
    payload = {
        "date": "2026-08-31",
        "stage": "dense_teacher_validation_qualification",
        "model": MODEL_ID,
        "window_seconds": 16,
        "stride_seconds": 8,
        "frames_per_window": 4,
        "test_labels_read": False,
        "corpora": corpora,
        "pass": all(record["pass"] for record in corpora.values()),
    }
    out = args.out_dir / "summary.json"
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
