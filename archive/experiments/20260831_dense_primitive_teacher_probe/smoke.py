#!/usr/bin/env python3
"""Dense Qwen3-VL typed-primitive teacher smoke on validation positives."""

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
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from src.vlm_windows import (  # noqa: E402
    load_timestamped_asr, temporal_windows, window_asr, window_frame_paths,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
AXES = ("hostile", "target", "violence", "sexual", "self_harm", "context")
CORPORA = ("hatemm", "hateclipseg")
FULL_GATES = {"hatemm": 0.59193, "hateclipseg": 0.54707}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--full-qualification", action="store_true")
    return p


def prompt(transcript, seconds):
    return (
        f"Annotate observable evidence in THIS {seconds}-second video window. "
        "Return exactly one compact JSON object with integer values 0 to 4 and "
        "these keys: hostile,target,violence,sexual,self_harm,context. "
        "hostile=attack, insult, abuse, contempt, or dehumanization; "
        "target=protected identity group; violence=violent act, threat, or "
        "incitement; sexual=sexual content; self_harm=self-harm content; "
        "context=quotation, reporting, education, condemnation, or counterspeech. "
        "Use frames, on-screen text, and transcript. Do not infer absent evidence. "
        "No explanation.\n"
        f'Transcript: "{transcript}"'
    )


def parse_primitives(text):
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        values = np.asarray([float(obj[name]) / 4.0 for name in AXES])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        return None
    return values


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
        output = model.generate(
            **inputs, max_new_tokens=64, do_sample=False
        )
    return processor.batch_decode(
        output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def noisy_or(values):
    return 1.0 - np.prod([1.0 - np.clip(value, 0, 1) for value in values])


def compile_policy(corpus, primitive):
    hostile, target, violence, sexual, self_harm, context = primitive
    targeted = hostile * target * (1.0 - context)
    abuse = hostile * (1.0 - target) * (1.0 - context)
    if corpus == "hatemm":
        return targeted
    if corpus == "hateclipseg":
        return noisy_or((targeted, abuse, violence, sexual, self_harm))
    raise ValueError(corpus)


def densify(length, windows, values):
    total = np.zeros(length, dtype=float)
    count = np.zeros(length, dtype=float)
    for (start, end), value in zip(windows, values):
        total[start:end] += value
        count[start:end] += 1
    return total / np.maximum(count, 1)


def run_corpus(corpus, model, processor, out_dir, limit, minimum_within):
    labels = hdata.load_labels(corpus)
    _, val_ids = hdata.load_train_val(corpus, labels)
    gt = hdata.gt_arrays(corpus, "val")
    ids = sorted(
        video_id for video_id in val_ids
        if labels[video_id] == 1 and video_id in gt
        and len(np.unique(gt[video_id])) == 2
    )[:limit]
    asr, asr_path = load_timestamped_asr(REPO, corpus)
    raw_path = out_dir / f"{corpus}_raw.jsonl"
    scores = {}
    rows = []
    failures = calls = 0
    with raw_path.open("w") as raw:
        for video_index, video_id in enumerate(ids):
            length = len(gt[video_id])
            windows = temporal_windows(length, 16, 8)
            values = []
            primitives = []
            for window_index, (start, end) in enumerate(windows):
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
                primitive = parse_primitives(generated)
                calls += 1
                if primitive is None:
                    failures += 1
                    primitive = np.zeros(len(AXES), dtype=float)
                value = compile_policy(corpus, primitive)
                values.append(float(value))
                primitives.append(primitive.tolist())
                raw.write(json.dumps({
                    "corpus": corpus, "video_id": video_id,
                    "window_index": window_index, "span": [start, end],
                    "frame_paths": [str(path) for path in paths],
                    "transcript": transcript, "generation": generated,
                    "primitive": primitive.tolist(), "compiled": float(value),
                }) + "\n")
                raw.flush()
                for image in images:
                    image.close()
            scores[video_id] = densify(length, windows, values)
            rows.append({
                "video_id": video_id, "length": length,
                "windows": windows, "primitives": primitives,
                "compiled": values,
            })
            print(
                f"PROGRESS {corpus} {video_index + 1}/{len(ids)} "
                f"{video_id} windows={len(windows)}", flush=True
            )
    report = evaluate_scores(
        scores, {video_id: gt[video_id] for video_id in ids}, set(ids)
    )
    within = report["per_video"]["macro_auc"]
    checks = {
        "coverage_complete": len(scores) == len(ids),
        "parse_failure_below_0.01": failures / max(1, calls) < .01,
        f"compiled_within_at_least_{minimum_within:.5f}": within >= minimum_within,
    }
    return {
        "corpus": corpus, "split": "val_positive_smoke",
        "videos": ids, "calls": calls, "parse_failures": failures,
        "parse_failure_rate": failures / max(1, calls),
        "compiled_within_roc": within,
        "within_n": report["per_video"]["n_videos_both_classes"],
        "asr_artifact": str(asr_path.resolve()),
        "raw_artifact": str(raw_path.resolve()),
        "checks": checks, "pass": all(checks.values()), "rows": rows,
    }


def main():
    args = parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    limits = None if args.full_qualification else args.limit
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    corpora = {
        corpus: run_corpus(
            corpus, model, processor, args.out_dir, limits,
            FULL_GATES[corpus] if args.full_qualification else .55,
        )
        for corpus in CORPORA
    }
    payload = {
        "date": "2026-08-31",
        "stage": ("dense_primitive_teacher_full_validation"
                  if args.full_qualification else "dense_primitive_teacher_smoke"),
        "model": MODEL_ID, "window_seconds": 16, "stride_seconds": 8,
        "frames_per_window": 4, "test_used": False,
        "full_gates": FULL_GATES if args.full_qualification else None,
        "corpora": corpora, "pass": all(row["pass"] for row in corpora.values()),
    }
    out = args.out_dir / "summary.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps({c: {"within": r["compiled_within_roc"], "pass": r["pass"]}
                      for c, r in corpora.items()}, indent=2))


if __name__ == "__main__":
    main()
