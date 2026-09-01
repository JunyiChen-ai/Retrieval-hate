#!/usr/bin/env python
"""Build/read full-cohort K=16 VERA train observations and 1fps scores."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/home/jehc223/Hate-follow-up/scripts/reproduction_baselines")
BASE = ROOT / "scripts/reproduction_baselines"
sys.path[:0] = [str(EXTERNAL), str(BASE), str(ROOT)]
from hate_common import data as hdata  # noqa: E402
from src.hate_local_features import aligned_local_features  # noqa: E402
import vera_adapter as vera  # noqa: E402
from vera_fast_infer import ReusableVideoReader, predict_batch  # noqa: E402


def sparse_starts(length):
    return np.unique(np.rint(np.linspace(0, length - 1, min(16, length))).astype(int))


def bounded_starts(length, duration):
    bounded_length = max(1, min(int(length), int(np.ceil(duration))))
    return sparse_starts(bounded_length).astype(float)


def valid_record(path, video_id, length):
    try:
        row = json.loads(path.read_text())
        segments = row["segments"]
        duration = float(row["duration"])
        expected = bounded_starts(length, duration)
        starts = np.asarray([float(x["start"]) for x in segments])
        return (row["video_id"] == video_id and np.isfinite(duration)
                and duration > 0 and starts.shape == expected.shape
                and np.allclose(starts, expected, rtol=0, atol=1e-6)
                and all(np.isfinite(float(x["end"]))
                        and 0 <= float(x["start"]) < float(x["end"]) <= duration + 1e-6
                        and x["score"] in (0, 1) for x in segments))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def postprocess(corpus, video_id, length, record):
    segments = record["segments"]
    starts = np.asarray([x["start"] for x in segments], dtype=float)
    values = np.asarray([x["score"] for x in segments], dtype=float)
    order = np.argsort(starts); starts = starts[order]; values = values[order]
    raw = np.interp(np.arange(length), starts, values,
                    left=values[0], right=values[-1])
    clip_path = ROOT / "results/reproduction/features/clip_b16_1fps" / corpus / f"{video_id}.npy"
    visual = np.load(clip_path).astype(np.float64)
    if len(visual) != length:
        raise RuntimeError(f"CLIP/1fps mismatch {corpus}/{video_id}")
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    similarity = visual @ visual.T
    top_n = max(1, int(.15 * length)); propagated = np.empty(length)
    for t in range(length):
        ix = np.argpartition(similarity[t], -top_n)[-top_n:]
        weight = np.exp((similarity[t, ix] - similarity[t, ix].max()) * 10.)
        weight /= weight.sum()
        propagated[t] = weight @ raw[ix]
    neighbor = gaussian_filter1d(propagated, sigma=10, radius=7, mode="nearest")
    x = np.arange(length)
    center = np.exp(-.5 * ((x - length // 2) / max(length / 2, 1)) ** 2)
    return raw, neighbor * center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--existing-raw-root")
    ap.add_argument("--prompt-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate-missing", action="store_true")
    args = ap.parse_args()
    labels = hdata.load_labels(args.corpus)
    train_ids, _ = hdata.load_train_val(args.corpus, labels, val_frac=.1, seed=234)
    lengths = {v: len(aligned_local_features(args.corpus, v)["audio"])
               for v in train_ids}
    raw_root = Path(args.raw_root); raw_root.mkdir(parents=True, exist_ok=True)
    existing = Path(args.existing_raw_root) if args.existing_raw_root else None
    selection = json.loads(Path(args.prompt_json).read_text())
    missing = []
    for video_id in train_ids:
        local = raw_root / f"{video_id}.json"
        external = existing / f"{video_id}.json" if existing else local
        if (not valid_record(local, video_id, lengths[video_id])
                and valid_record(external, video_id, lengths[video_id])):
            # Data-preparation copy remains readable JSON; no digest is used.
            local.write_text(external.read_text())
        if not valid_record(local, video_id, lengths[video_id]):
            missing.append(video_id)
    if missing and args.generate_missing:
        model, tokenizer, _ = vera.load_model(selection["attention_backend"])
        for position, video_id in enumerate(missing, 1):
            target = raw_root / f"{video_id}.json"
            try:
                reader = ReusableVideoReader(vera.video_path(args.corpus, video_id))
            except Exception as error:
                print(f"[{position}/{len(missing)}] {video_id}: unavailable {error}", flush=True)
                continue
            starts = bounded_starts(lengths[video_id], reader.duration)
            rows = []
            for offset in range(0, len(starts), 2):
                batch = starts[offset:offset + 2]
                images = [reader.frames(float(start), 10., 8) for start in batch]
                predictions = predict_batch(model, tokenizer, images,
                                            selection["prompts"], 2)
                for start, (score, response) in zip(batch, predictions):
                    rows.append({"start": float(start),
                                 "end": min(reader.duration, float(start) + 10.),
                                 "score": score, "response": response})
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps({"video_id": video_id,
                                             "duration": reader.duration,
                                             "segments": rows}, ensure_ascii=False))
            os.replace(temporary, target)
            print(f"[{position}/{len(missing)}] {video_id}: {len(rows)} windows", flush=True)
    target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    written = 0; unavailable = []
    with target.open("w") as handle:
        for video_id in train_ids:
            path = raw_root / f"{video_id}.json"
            if not valid_record(path, video_id, lengths[video_id]):
                unavailable.append(video_id); continue
            record = json.loads(path.read_text())
            raw, score = postprocess(args.corpus, video_id, lengths[video_id], record)
            handle.write(json.dumps({"video_id": video_id,
                                     "score_raw": raw.tolist(),
                                     "score_official_postprocessed": score.tolist()}) + "\n")
            written += 1
    provenance = {"corpus": args.corpus, "split": "train-fit",
                  "producer": "VERA/LLaVA-Next-Video-7B-DPO K=16 uniform windows",
                  "selected_prompt": str(Path(args.prompt_json).resolve()),
                  "raw_root": str(raw_root.resolve()), "output": str(target.resolve()),
                  "n_expected": len(train_ids), "n_written": written,
                  "unavailable_video_ids": unavailable,
                  "frame_labels_read": False, "validation_or_test_read": False,
                  "verification": "parsed JSON, finite scores, 1fps feature shape, split isolation"}
    (target.parent / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"written": written, "unavailable": len(unavailable)}, indent=2))


if __name__ == "__main__":
    main()
