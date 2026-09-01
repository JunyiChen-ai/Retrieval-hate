#!/usr/bin/env python3
"""Generate fixed-recipe train/test VERA K16 caches for the pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
FROZEN = Path("/home/jehc223/Hate-follow-up/scripts/reproduction_baselines")
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(BASE))

import vera_adapter as vera  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402
from powa_macil.dataset import usable_text_ids  # noqa: E402
from vera_fast_infer import ReusableVideoReader, predict_batch  # noqa: E402


MODEL_ID = "OpenGVLab/InternVL2-8B"
K = 16
WINDOW = 10.0
BATCH_SIZE = 2
PROMPTS = [
    "Does anyone attack, threaten, demean, or promote hostility toward a person or protected group?",
    "Are hateful symbols, slurs, insults, or dehumanizing claims visibly communicated?",
    "Does the scene endorse violence, exclusion, discrimination, or harm against a person or group?",
    "Do gestures, displayed text, or interactions convey targeted hateful or abusive meaning?",
    "Taken in context, is hateful or severely offensive content present in this video segment?",
]


def sparse_starts(length):
    if length <= 0:
        raise ValueError("empty timeline")
    return np.unique(
        np.rint(np.linspace(0, length - 1, min(K, length))).astype(np.int64)
    )


def valid_result(path, video_id, starts, duration, media, corpus, split):
    try:
        row = json.loads(path.read_text())
        segments = row["segments"]
        provenance = row["provenance"]
        return (
            row.get("video_id") == video_id
            and float(row["duration"]) == float(duration)
            and len(segments) == len(starts)
            and all(
                float(segment["start"]) == float(start)
                and float(segment["end"]) == min(
                    float(duration), float(start) + WINDOW
                )
                and segment.get("score") in (0, 1)
                and isinstance(segment.get("response"), str)
                for segment, start in zip(segments, starts)
            )
            and provenance.get("corpus") == corpus
            and provenance.get("split") == split
            and provenance.get("labels_read") is False
            and provenance.get("model") == MODEL_ID
            and provenance.get("model_version") == (
                "local cached model available 2026-08-31"
            )
            and provenance.get("prompt") == PROMPTS
            and provenance.get("batch_size") == BATCH_SIZE
            and float(provenance.get("window_seconds")) == WINDOW
            and provenance.get("k") == K
            and provenance.get("media_path") == str(media)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.generation import GenerationConfig, GenerationMixin

    common = {
        "local_files_only": True, "trust_remote_code": True,
    }
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=False, **common,
    ).eval().cuda()
    language_model = model.language_model
    if not isinstance(language_model, GenerationMixin):
        compatible_class = type(
            "InternLM2GenerationCompatible",
            (language_model.__class__, GenerationMixin),
            {
                "_supports_default_dynamic_cache": classmethod(
                    lambda cls: False
                )
            },
        )
        language_model.__class__ = compatible_class
    if language_model.generation_config is None:
        language_model.generation_config = GenerationConfig.from_model_config(
            language_model.config
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **common)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True,
                        choices=("hatemm", "hateclipseg"))
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--raw-root", required=True, type=Path)
    args = parser.parse_args()
    if args.corpus == "hateclipseg" and args.split == "train":
        raise ValueError("reuse the independently audited 238-video HCS train cache")
    args.raw_root = args.raw_root.resolve()
    args.raw_root.mkdir(parents=True, exist_ok=True)

    video_ids = usable_text_ids(
        args.corpus, hdata.load_split(args.corpus, args.split)
    )
    if args.split == "test":
        gt_ids = set(hdata.gt_arrays(args.corpus, "test"))
        video_ids = [video_id for video_id in video_ids if video_id in gt_ids]
    expected = {("hatemm", "train"): 744, ("hatemm", "test"): 214,
                ("hateclipseg", "test"): 79}[(args.corpus, args.split)]
    if len(video_ids) != expected or len(set(video_ids)) != expected:
        raise RuntimeError(f"unexpected cohort: {args.corpus}/{args.split}")
    if args.split == "train" and set(video_ids) & set(
            hdata.load_split(args.corpus, "test")):
        raise RuntimeError("train/test overlap")

    pending = []
    for video_id in video_ids:
        media = Path(vera.video_path(args.corpus, video_id)).resolve()
        reader = ReusableVideoReader(media)
        n_seconds = np.load(
            align.audio_path(args.corpus, video_id), mmap_mode="r"
        ).shape[0]
        length = max(1, int(np.ceil(min(float(n_seconds), reader.duration))))
        starts = sparse_starts(length)
        target = args.raw_root / f"{video_id}.json"
        if valid_result(target, video_id, starts, reader.duration, media,
                        args.corpus, args.split):
            print(f"{video_id}: already complete", flush=True)
        else:
            pending.append((video_id, media, reader.duration, starts, target))

    print(json.dumps({"eligible": len(video_ids), "pending": len(pending)}),
          flush=True)
    if pending:
        model, tokenizer = load_model()
        for position, (video_id, media, duration, starts, target) in enumerate(
                pending, 1):
            reader = ReusableVideoReader(media)
            records = []
            for offset in range(0, len(starts), BATCH_SIZE):
                batch_starts = starts[offset:offset + BATCH_SIZE]
                images = [reader.frames(float(start), WINDOW, 8)
                          for start in batch_starts]
                predictions = predict_batch(
                    model, tokenizer, images, PROMPTS, BATCH_SIZE
                )
                for start, (score, response) in zip(batch_starts, predictions):
                    records.append({
                        "start": float(start),
                        "end": min(float(duration), float(start) + WINDOW),
                        "score": int(score), "response": response,
                    })
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "video_id": video_id, "duration": float(duration),
                "segments": records,
                "provenance": {
                    "corpus": args.corpus, "split": args.split,
                    "labels_read": False,
                    "model": MODEL_ID,
                    "model_version": "local cached model available 2026-08-31",
                    "prompt": PROMPTS, "batch_size": BATCH_SIZE,
                    "window_seconds": WINDOW, "k": K,
                    "media_path": str(media),
                },
            }, ensure_ascii=False))
            os.replace(temporary, target)
            if not valid_result(target, video_id, starts, duration, media,
                                args.corpus, args.split):
                raise RuntimeError(f"generated invalid cache: {video_id}")
            print(f"[{position}/{len(pending)}] {video_id}: complete", flush=True)

    actual = {path.stem for path in args.raw_root.glob("*.json")}
    if actual != set(video_ids):
        raise RuntimeError("final cache coverage mismatch")
    manifest = {
        "corpus": args.corpus, "split": args.split, "labels_read": False,
        "model": MODEL_ID,
        "model_version": "local cached model available 2026-08-31",
        "prompt": PROMPTS, "batch_size": BATCH_SIZE,
        "window_seconds": WINDOW, "k": K,
        "raw_root": str(args.raw_root), "video_ids": video_ids,
        "complete_videos": len(video_ids),
    }
    (args.raw_root.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"complete_videos": len(video_ids)}), flush=True)


if __name__ == "__main__":
    main()
