#!/usr/bin/env python
"""Build frozen TalkNet speech-to-face assignments and face CLIP features.

This is data preparation, not method training.  It invokes the unmodified
official TalkNet demo for each raw video, converts its 25-fps track decisions
to the project's one-second grid, and writes only compact arrays.  Project
labels and localization annotations are never read.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "reproduction_baselines"
sys.path.insert(0, str(BASE))

from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402


CORPUS_LAYOUT = {
    "hatemm": (Path("/home/jehc223/data/HateMM/video"), "HateMM"),
    "hateclipseg": (Path("/home/jehc223/data/HateClipSeg/videos"),
                    "HateClipSeg"),
}
TEXT_ROOT = ROOT / "results" / "reproduction" / "features" / "bert_sentence_1fps"
TALKNET_ROOT = ROOT / "third_party" / "TalkNet-ASD"
TALKNET_MODEL = TALKNET_ROOT / "pretrain_TalkSet.model"
CLIP_NAME = "openai/clip-vit-base-patch16"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=tuple(CORPUS_LAYOUT))
    ap.add_argument("--out-root", default=str(ROOT / "data" / "active_speaker_bound"))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--clip-batch-size", type=int, default=128)
    ap.add_argument("--face-batch-size", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    return ap.parse_args()


def raw_video(corpus: str, video_id: str) -> Path:
    folder, _ = CORPUS_LAYOUT[corpus]
    matches = sorted(folder.glob(video_id + ".*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one raw video for {corpus}/{video_id}, got {matches}")
    return matches[0]


def clip_encode(images, model, processor, device, batch_size):
    if not images:
        return np.zeros((0, 512), dtype=np.float32)
    parts = []
    for start in range(0, len(images), batch_size):
        batch = processor(images=images[start:start + batch_size],
                          return_tensors="pt")
        pixels = batch["pixel_values"].to(device)
        with torch.no_grad():
            features = model.get_image_features(pixel_values=pixels)
            features = torch.nn.functional.normalize(features, dim=-1)
        parts.append(features.float().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def crop_track_frame(frame_path: Path, track, local_index: int):
    image = cv2.imread(str(frame_path))
    if image is None:
        raise RuntimeError(f"could not read {frame_path}")
    box = np.asarray(track["track"]["bbox"][local_index], dtype=np.float64)
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    pad = .40
    left = max(0, int(np.floor(cx - side * (0.5 + pad))))
    right = min(image.shape[1], int(np.ceil(cx + side * (0.5 + pad))))
    top = max(0, int(np.floor(cy - side * .5)))
    bottom = min(image.shape[0], int(np.ceil(cy + side * (0.5 + 2 * pad))))
    if right <= left or bottom <= top:
        raise RuntimeError(f"empty face crop in {frame_path}")
    rgb = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def run_talknet(video: Path, work: Path, face_batch_size: int):
    link = work / ("input" + video.suffix.lower())
    link.symlink_to(video.resolve())
    command = [
        sys.executable, str(TALKNET_ROOT / "demoTalkNet.py"),
        "--videoName", "input", "--videoFolder", str(work),
        "--pretrainModel", str(TALKNET_MODEL), "--nDataLoaderThread", "8",
        "--facedetStride", "5", "--minTrack", "2",
        "--facedetBatchSize", str(face_batch_size),
        "--skipVisualization",
    ]
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    completed = subprocess.run(command, cwd=TALKNET_ROOT, env=env,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"TalkNet failed for {video}:\n{tail}")
    output = work / "input" / "pywork"
    with (output / "tracks.pckl").open("rb") as handle:
        tracks = pickle.load(handle)
    with (output / "scores.pckl").open("rb") as handle:
        scores = pickle.load(handle)
    return tracks, scores, work / "input" / "pyframes"


def run_talknet_with_oom_retry(video: Path, work: Path,
                               face_batch_size: int):
    """Run TalkNet, reducing only detector batch size after a CUDA OOM."""
    batch_sizes = []
    current = face_batch_size
    while current >= 16:
        if current not in batch_sizes:
            batch_sizes.append(current)
        current //= 2
    if 16 not in batch_sizes:
        batch_sizes.append(16)

    for attempt, batch_size in enumerate(batch_sizes):
        attempt_dir = work / f"attempt_{attempt}"
        attempt_dir.mkdir()
        try:
            return run_talknet(video, attempt_dir, batch_size)
        except RuntimeError as error:
            message = str(error).lower()
            cuda_allocation_failure = (
                "out of memory" in message
                or "cublas_status_alloc_failed" in message
            )
            if not cuda_allocation_failure:
                raise
            print(json.dumps({
                "video": video.stem,
                "status": "retry_cuda_oom",
                "face_batch_size": batch_size,
            }), flush=True)
    raise RuntimeError(
        f"TalkNet CUDA OOM for {video} at minimum face batch size 16")


def convert(video_id, n_seconds, tracks, scores, frames_dir, clip_model,
            clip_processor, device, clip_batch_size):
    candidates = [[] for _ in range(n_seconds)]
    representative_images = []
    representative_keys = []
    for track_id, (track, score) in enumerate(zip(tracks, scores)):
        frame_ids = np.asarray(track["track"]["frame"], dtype=np.int64)
        score = np.asarray(score, dtype=np.float32)
        usable = min(len(frame_ids), len(score))
        frame_ids, score = frame_ids[:usable], score[:usable]
        second_ids = frame_ids // 25
        for second in np.unique(second_ids):
            second = int(second)
            if not 0 <= second < n_seconds:
                continue
            locations = np.flatnonzero(second_ids == second)
            middle = int(locations[len(locations) // 2])
            candidates[second].append((track_id, float(score[locations].mean())))
            representative_keys.append((second, track_id))
            representative_images.append(crop_track_frame(
                frames_dir / f"{int(frame_ids[middle]) + 1:06d}.jpg",
                track, middle))

    encoded = clip_encode(representative_images, clip_model, clip_processor,
                          device, clip_batch_size)
    face_by_key = {key: encoded[i] for i, key in enumerate(representative_keys)}
    core_face = np.zeros((n_seconds, 512), dtype=np.float32)
    permuted_face = np.zeros_like(core_face)
    state = np.zeros(n_seconds, dtype=np.int64)  # 0=no speech, 1=null, 2=visible
    assigned_track = np.full(n_seconds, -1, dtype=np.int32)
    permuted_track = np.full(n_seconds, -1, dtype=np.int32)
    active_score = np.full(n_seconds, -np.inf, dtype=np.float32)
    return (candidates, face_by_key, core_face, permuted_face, state,
            assigned_track, permuted_track, active_score)


def finalize(corpus, video_id, n_seconds, converted):
    (candidates, face_by_key, core_face, permuted_face, state,
     assigned_track, permuted_track, active_score) = converted
    text = np.load(TEXT_ROOT / corpus / f"{video_id}.npy").astype(np.float32)
    speech = np.zeros(n_seconds, dtype=bool)
    speech[:min(n_seconds, len(text))] = (
        np.linalg.norm(text[:min(n_seconds, len(text))], axis=-1) > 1e-6)
    eligible_multiface = np.zeros(n_seconds, dtype=bool)
    for second in np.flatnonzero(speech):
        rows = sorted(candidates[second], key=lambda row: row[0])
        if not rows:
            state[second] = 1
            continue
        ranked = sorted(rows, key=lambda row: (-row[1], row[0]))
        best_track, best_score = ranked[0]
        unique = len(ranked) == 1 or best_score > ranked[1][1]
        if best_score <= 0.0 or not unique:
            state[second] = 1
            continue
        state[second] = 2
        active_score[second] = best_score
        assigned_track[second] = best_track
        core_face[second] = face_by_key[(second, best_track)]
        visible_ids = [row[0] for row in rows]
        if len(visible_ids) >= 2:
            eligible_multiface[second] = True
            index = visible_ids.index(best_track)
            other = visible_ids[(index + 1) % len(visible_ids)]
        else:
            other = best_track
        permuted_track[second] = other
        permuted_face[second] = face_by_key[(second, other)]
    return {
        "core_face": core_face, "permuted_face": permuted_face,
        "source_state": state, "speech_mask": speech.astype(np.uint8),
        "assigned_track": assigned_track, "permuted_track": permuted_track,
        "active_score": active_score,
        "eligible_multiface": eligible_multiface.astype(np.uint8),
    }


def no_speech_arrays(n_seconds):
    return {
        "core_face": np.zeros((n_seconds, 512), dtype=np.float32),
        "permuted_face": np.zeros((n_seconds, 512), dtype=np.float32),
        "source_state": np.zeros(n_seconds, dtype=np.int64),
        "speech_mask": np.zeros(n_seconds, dtype=np.uint8),
        "assigned_track": np.full(n_seconds, -1, dtype=np.int32),
        "permuted_track": np.full(n_seconds, -1, dtype=np.int32),
        "active_score": np.full(n_seconds, -np.inf, dtype=np.float32),
        "eligible_multiface": np.zeros(n_seconds, dtype=np.uint8),
    }


def write_arrays(destination, arrays):
    temporary_out = destination.with_suffix(".npz.tmp")
    with temporary_out.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary_out, destination)


def main():
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not TALKNET_MODEL.is_file():
        raise FileNotFoundError(TALKNET_MODEL)
    out_dir = Path(args.out_root) / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ids = []
    for split in ("train", "val", "test"):
        all_ids.extend(hdata.load_split(args.corpus, split))
    all_ids = list(dict.fromkeys(all_ids))
    all_ids = [video_id for index, video_id in enumerate(all_ids)
               if index % args.num_shards == args.shard_index]
    model = CLIPModel.from_pretrained(CLIP_NAME).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    for number, video_id in enumerate(all_ids, 1):
        destination = out_dir / f"{video_id}.npz"
        if destination.exists() and not args.overwrite:
            print(json.dumps({"video": video_id, "status": "exists",
                              "index": number, "total": len(all_ids)}), flush=True)
            continue
        text_path = TEXT_ROOT / args.corpus / f"{video_id}.npy"
        if not text_path.is_file():
            raise FileNotFoundError(text_path)
        n_seconds = int(align.load_audio(args.corpus, video_id).shape[0])
        text = np.load(text_path, mmap_mode="r")
        usable = min(n_seconds, len(text))
        if not np.any(np.linalg.norm(text[:usable], axis=-1) > 1e-6):
            arrays = no_speech_arrays(n_seconds)
            write_arrays(destination, arrays)
            print(json.dumps({
                "video": video_id, "status": "written_no_speech",
                "index": number, "total": len(all_ids),
                "seconds": n_seconds, "speech_seconds": 0,
                "visible_assigned_seconds": 0,
                "multiface_control_seconds": 0,
            }), flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix="talknet_") as temporary:
            tracks, scores, frames = run_talknet_with_oom_retry(
                raw_video(args.corpus, video_id), Path(temporary),
                args.face_batch_size)
            converted = convert(
                video_id, n_seconds, tracks, scores, frames, model, processor,
                args.device, args.clip_batch_size)
            arrays = finalize(args.corpus, video_id, n_seconds, converted)
        write_arrays(destination, arrays)
        print(json.dumps({
            "video": video_id, "status": "written", "index": number,
            "total": len(all_ids), "seconds": n_seconds,
            "speech_seconds": int(arrays["speech_mask"].sum()),
            "visible_assigned_seconds": int((arrays["source_state"] == 2).sum()),
            "multiface_control_seconds": int(arrays["eligible_multiface"].sum()),
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
