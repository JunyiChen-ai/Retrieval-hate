import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

from src.utils.generate_segment_asr_HF import decode_audio_pyav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, source_duration = decode_audio_pyav(args.video)
    if audio is None:
        raise RuntimeError("The fixed smoke video has no decodable audio")
    sample_rate = 16000
    audio = audio[: int(args.max_seconds * sample_rate)]
    analyzed_duration = len(audio) / sample_rate

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=0,
        batch_size=1,
        return_timestamps="word",
    )
    result = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "num_beams": 1,
        },
    )

    words = []
    for chunk in result.get("chunks") or []:
        timestamp = chunk.get("timestamp")
        if not timestamp or timestamp[0] is None or timestamp[1] is None:
            raise RuntimeError("Strict word timestamp missing an endpoint")
        start, end = float(timestamp[0]), float(timestamp[1])
        words.append({"start": start, "end": end, "text": chunk.get("text", "")})

    nonempty = [w for w in words if w["text"].strip()]
    durations = [w["end"] - w["start"] for w in nonempty]
    finite_valid = all(
        math.isfinite(w["start"])
        and math.isfinite(w["end"])
        and 0.0 <= w["start"] <= w["end"] <= analyzed_duration + 0.1
        for w in nonempty
    )
    monotone = all(
        nonempty[i]["start"] <= nonempty[i + 1]["start"]
        for i in range(len(nonempty) - 1)
    )
    positive_durations = [d for d in durations if d > 0]
    median_duration = (
        float(np.median(positive_durations)) if positive_durations else None
    )
    passed = (
        len(nonempty) >= 10
        and finite_valid
        and monotone
        and median_duration is not None
        and median_duration < 2.0
    )
    artifact = {
        "status": "PASS" if passed else "FAIL",
        "video_id": Path(args.video).stem,
        "source_duration_seconds": source_duration,
        "analyzed_duration_seconds": analyzed_duration,
        "timestamp_mode": "word_strict_no_fallback",
        "word_count": len(nonempty),
        "finite_valid_intervals": finite_valid,
        "nondecreasing_start_times": monotone,
        "positive_duration_count": len(positive_durations),
        "word_duration_seconds": {
            "median": median_duration,
            "p90": float(np.quantile(positive_durations, 0.9))
            if positive_durations
            else None,
            "max": max(positive_durations) if positive_durations else None,
        },
        "transcript": result.get("text", ""),
        "words": words,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in artifact.items() if k != "words"},
                     ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
