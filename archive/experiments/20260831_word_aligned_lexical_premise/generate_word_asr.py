#!/usr/bin/env python
"""Label-blind, resumable strict word-level ASR for one fixed split."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402
from src.utils.generate_segment_asr_HF import decode_audio_pyav  # noqa: E402

VIDEO_ROOTS = {
    "hatemm": Path("/home/jehc223/data/HateMM/video"),
    "hateclipseg": Path("/home/jehc223/data/HateClipSeg/videos"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=sorted(VIDEO_ROOTS), required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def validate_words(chunks, duration):
    previous = -1.0
    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        if not (math.isfinite(start) and math.isfinite(end)):
            return False
        if not (0 <= start < end <= duration + 0.1 and start >= previous):
            return False
        previous = start
    return True


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = {
        "corpus": args.corpus,
        "split": args.split,
        "model": args.model,
        "language": "en",
        "decoding": "greedy",
        "timestamp_mode": "word_strict_no_fallback",
        "long_form": "whisper_native",
        "implementation_version": "content-locked-word-timing-v1",
    }
    recipe_path = output.parent / "asr_config.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != recipe:
            raise RuntimeError("existing ASR output directory has a different recipe")
    else:
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    ids = hdata.load_split(args.corpus, args.split)
    if len(ids) != len(set(ids)):
        raise RuntimeError("split manifest contains duplicate video IDs")
    manifest_ids = set(ids)
    done = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                video_id = str(row.get("video_id"))
                if video_id not in manifest_ids:
                    raise RuntimeError("existing ASR row is outside split manifest")
                if video_id in done:
                    raise RuntimeError("existing ASR output has duplicate video ID")
                if row.get("corpus") != args.corpus or row.get("split") != args.split:
                    raise RuntimeError("existing ASR row has a different corpus/split")
                if row.get("recipe") != recipe:
                    raise RuntimeError("existing ASR row has a different generation recipe")
                if row.get("timestamp_mode") != recipe["timestamp_mode"]:
                    raise RuntimeError("existing ASR row has a different timestamp mode")
                if row.get("status") not in {"OK", "NO_AUDIO", "EMPTY_SPEECH"}:
                    raise RuntimeError("existing ASR row is incomplete or failed")
                chunks = row.get("chunks", [])
                if row["status"] == "OK":
                    if not chunks or not validate_words(
                            chunks, float(row["analyzed_duration"])):
                        raise RuntimeError("existing OK row fails strict validation")
                elif chunks:
                    raise RuntimeError("existing non-OK row unexpectedly has chunks")
                done.add(video_id)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
        local_files_only=True).to(device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    asr = pipeline(
        "automatic-speech-recognition", model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype, device=device, chunk_length_s=0, batch_size=1,
        return_timestamps="word")

    root = VIDEO_ROOTS[args.corpus]
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for index, video_id in enumerate(ids, start=1):
            if video_id in done:
                continue
            video_path = root / f"{video_id}.mp4"
            row = {"video_id": video_id, "corpus": args.corpus,
                   "split": args.split, "text": "", "chunks": [],
                   "recipe": recipe,
                   "timestamp_mode": "word_strict_no_fallback"}
            try:
                audio, duration = decode_audio_pyav(str(video_path))
                row["duration"] = duration
                if audio is None or len(audio) <= 160:
                    row["status"] = "NO_AUDIO"
                else:
                    analyzed_duration = len(audio) / 16000.0
                    result = asr(
                        {"raw": audio, "sampling_rate": 16000},
                        generate_kwargs={"task": "transcribe", "language": "en",
                                         "num_beams": 1})
                    chunks = []
                    for item in result.get("chunks") or []:
                        stamp = item.get("timestamp")
                        if not stamp or stamp[0] is None or stamp[1] is None:
                            raise ValueError("word timestamp missing endpoint")
                        text = str(item.get("text", ""))
                        if not text.strip():
                            continue
                        chunks.append({
                            "start": float(stamp[0]), "end": float(stamp[1]),
                            "text": text,
                        })
                    if chunks and not validate_words(chunks, analyzed_duration):
                        raise ValueError("word timestamps fail strict monotonic validation")
                    status = "OK" if chunks else "EMPTY_SPEECH"
                    row.update({"status": status, "text": result.get("text", ""),
                                "chunks": chunks,
                                "analyzed_duration": analyzed_duration})
            except Exception as error:  # fail closed per video, keep resumability
                row["status"] = "ERROR"
                row["error_type"] = type(error).__name__
                row["error"] = str(error)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{args.corpus}/{args.split} {index}/{len(ids)} "
                  f"{video_id} {row['status']} words={len(row['chunks'])} "
                  f"elapsed={time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
