import argparse
import json
import os

import numpy as np
import torch

# Segment-aligned ASR for the multimodal sub-clip keys (EXP_mm_segment_keys).
#
# W2 attribution: MHC-EN hate is predominantly carried by SPEECH / on-screen
# text, so the purely-visual K=4 sub-clip keys vote noise in the consensus
# E-step. This script gives every sub-clip WINDOW its own transcript:
#
#   1. Decode the audio track of each video with PyAV (no ffmpeg binary
#      needed), resampled to 16 kHz mono float32.
#   2. Run Whisper (default openai/whisper-large-v3) via the transformers
#      ASR pipeline with chunked long-form decoding and timestamps.
#   3. Map every timestamped ASR chunk to one of the K=4 sub-clip windows.
#      The windows are derived from the SAME frame sampling contract as
#      generate_subclip_embedding_HF.py: M=16 frames uniformly sampled over
#      the video, split into K=4 contiguous groups (window k = frames
#      [k*M/K, (k+1)*M/K)). Frame i sits at time ~ duration * i/(M-1), so the
#      time boundary between window k-1 and k is
#          duration * (first_frame_of_k - 0.5) / (M - 1)
#      (midpoint between the two adjacent sampled frames). A chunk belongs to
#      the window containing its midpoint.
#
# Output (one JSONL per split, one line per video):
#   data/ASR/<DS>/{train,dev_seen,test_seen}_asrK<K>_<model_tag>.jsonl
#   {"id", "label", "duration", "audio_ok", "language",
#    "chunks": [[start, end, text], ...],          # raw whisper chunks
#    "window_bounds": [[t0, t1], ...],             # K time spans (seconds)
#    "window_text": [str, ...]}                    # K per-window transcripts
#
# Read-only w.r.t. every existing cache; writes ONLY the new ASR JSONL.

SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}

# Forced decode language per dataset (--language auto). None = whisper detect.
DATASET_LANGUAGE = {
    "MHC": "en",
    "MHC_zh": "zh",
    "HateMM": "en",
    "ImpliHateVid": "en",
    "HateClipSeg": "en",
}


def parse_args_sys(args_list=None):
    ap = argparse.ArgumentParser(
        description="Whisper ASR aligned to the K sub-clip windows.")
    ap.add_argument("--dataset", type=str, default="MHC")
    ap.add_argument("--gt_dir", type=str, default="./data/gt")
    ap.add_argument("--video_dir", type=str, default="./data/video")
    ap.add_argument("--out_dir", type=str, default="./data/ASR")
    ap.add_argument("--model", type=str, default="openai/whisper-large-v3")
    ap.add_argument("--language", type=str, default="auto",
                    help="'auto' = per-dataset table (MHC->en, MHC_zh->zh); "
                         "'detect' = let whisper detect; else a language code.")
    ap.add_argument("--num_frames", type=int, default=16,
                    help="M of the sub-clip cache frame sampler (window map).")
    ap.add_argument("--num_subclips", type=int, default=4,
                    help="K of the sub-clip cache (windows per video).")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N videos per split (smoke). "
                         "0 = all.")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Whisper pipeline batch size over 30 s chunks.")
    ap.add_argument("--chunk_length_s", type=float, default=30.0)
    ap.add_argument("--timestamps", type=str, default="word",
                    choices=["word", "chunk"],
                    help="Timestamp granularity. 'word' (default) assigns "
                         "each word to its window by midpoint (precise); "
                         "'chunk' uses whisper segment chunks (a long chunk "
                         "is dumped whole into its midpoint window).")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=lambda x: str(x).lower() == "true",
                    default=True,
                    help="Skip ids already present in the output JSONL.")
    return ap.parse_args(args_list)


def read_gt(gt_path):
    items = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append({"id": str(obj["id"]), "label": obj["label"]})
    return items


def decode_audio_pyav(video_path, target_rate=16000):
    """Decode the first audio stream -> (float32 mono [T], duration_s).

    duration_s comes from the container (fallback: decoded audio length).
    Returns (None, duration) when the file has no decodable audio.
    """
    import av

    container = av.open(video_path)
    duration = None
    if container.duration:
        duration = float(container.duration) / av.time_base
    if not container.streams.audio:
        container.close()
        return None, duration
    resampler = av.AudioResampler(format="s16", layout="mono",
                                  rate=target_rate)
    pcm = []
    try:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                arr = out.to_ndarray()  # [1, samples] int16
                pcm.append(arr.reshape(-1))
        # flush the resampler
        for out in resampler.resample(None):
            arr = out.to_ndarray()
            pcm.append(arr.reshape(-1))
    finally:
        container.close()
    if not pcm:
        return None, duration
    audio = np.concatenate(pcm).astype(np.float32) / 32768.0
    if duration is None:
        duration = len(audio) / float(target_rate)
    return audio, duration


def window_time_bounds(duration, num_frames, num_subclips):
    """Time spans of the K sub-clip windows (matches _window_bounds frames).

    Frame i (of M uniformly sampled) sits at ~ duration * i/(M-1). Window k
    covers frames [start_k, end_k) from generate_subclip_embedding_HF's
    _window_bounds; the time boundary between consecutive windows is the
    midpoint between the last frame of one and the first frame of the next.
    """
    M, K = num_frames, num_subclips
    base, rem = M // K, M % K
    starts, s = [], 0
    for k in range(K):
        starts.append(s)
        s += base + (1 if k < rem else 0)
    denom = max(M - 1, 1)
    cuts = [0.0]
    for k in range(1, K):
        cuts.append(duration * (starts[k] - 0.5) / denom)
    cuts.append(duration)
    return [[cuts[k], cuts[k + 1]] for k in range(K)]


def assign_chunks_to_windows(chunks, bounds, word_level):
    """chunks: [[start, end, text], ...] -> K joined window texts (midpoint).

    word_level=True: chunks are whisper words; join with '' so whisper's own
    leading-space convention survives for EN and ZH stays unspaced.
    """
    texts = [[] for _ in bounds]
    for s, e, text in chunks:
        raw = text or ""
        if not raw.strip():
            continue
        mid = 0.5 * (s + e)
        target = len(bounds) - 1
        for k, (t0, t1) in enumerate(bounds):
            if mid < t1 or k == len(bounds) - 1:
                target = k
                break
        texts[target].append(raw if word_level else raw.strip())
    joiner = "" if word_level else " "
    return [joiner.join(t).strip() for t in texts]


def normalise_chunks(raw_chunks, duration):
    """Whisper pipeline 'chunks' -> [[start, end, text], ...] with finite,
    duration-clipped timestamps."""
    out = []
    for ch in raw_chunks or []:
        ts = ch.get("timestamp") or (None, None)
        s = ts[0] if ts[0] is not None else 0.0
        e = ts[1] if ts[1] is not None else duration
        s = float(min(max(s, 0.0), duration))
        e = float(min(max(e, s), duration))
        # keep raw text (leading spaces matter for word-level re-joining)
        out.append([s, e, ch.get("text") or ""])
    return out


def main(args):
    from transformers import pipeline as hf_pipeline

    if args.language == "auto":
        language = DATASET_LANGUAGE.get(args.dataset)
        if language is None:
            print("[WARN] no language table entry for dataset '{}'; letting "
                  "whisper detect.".format(args.dataset))
    elif args.language == "detect":
        language = None
    else:
        language = args.language

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("Loading ASR pipeline: {} (device={}, dtype={}, language={})".format(
        args.model, device, dtype, language))
    return_ts = "word" if args.timestamps == "word" else True
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=args.model,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        return_timestamps=return_ts,
    )
    generate_kwargs = {"task": "transcribe"}
    if language is not None:
        generate_kwargs["language"] = language

    model_tag = str(args.model).split("/")[-1]
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    video_root = os.path.join(args.video_dir, args.dataset, "All")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' unmapped; skipping.".format(split))
            continue
        gt_path = os.path.join(args.gt_dir, args.dataset,
                               "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] missing gt {}; skipping.".format(gt_path))
            continue
        items = read_gt(gt_path)
        if args.limit > 0:
            items = items[: args.limit]
        outname = SPLIT_TO_OUTNAME[split]
        out_path = os.path.join(out_dir, "{}_asrK{}_{}.jsonl".format(
            outname, args.num_subclips, model_tag))

        done = set()
        if args.resume and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:  # noqa: BLE001
                        pass
            print("[resume] {} already-done ids in {}".format(
                len(done), out_path))

        n_no_audio, n_empty, n_done = 0, 0, 0
        with open(out_path, "a") as fout:
            for n, item in enumerate(items):
                vid = item["id"]
                if vid in done:
                    continue
                video_path = os.path.join(video_root, "{}.mp4".format(vid))
                audio, duration = None, None
                if os.path.exists(video_path):
                    try:
                        audio, duration = decode_audio_pyav(video_path)
                    except Exception as e:  # noqa: BLE001
                        print("[WARN] audio decode failed for {} ({})".format(
                            vid, repr(e)))
                else:
                    print("[WARN] missing video file: {}".format(video_path))
                if duration is None:
                    duration = (len(audio) / 16000.0) if audio is not None else 0.0

                chunks = []
                ts_mode = args.timestamps
                audio_ok = audio is not None and len(audio) > 160  # >10 ms
                if audio_ok:
                    result = None
                    try:
                        result = asr({"raw": audio, "sampling_rate": 16000},
                                     generate_kwargs=generate_kwargs)
                    except Exception as e:  # noqa: BLE001
                        if args.timestamps == "word":
                            # word-timestamp DTW occasionally crashes inside
                            # transformers; retry at sentence granularity.
                            print("[WARN] word-ts ASR failed for {} ({}); "
                                  "retrying chunk-level.".format(vid, repr(e)))
                            ts_mode = "chunk"
                            try:
                                result = asr(
                                    {"raw": audio, "sampling_rate": 16000},
                                    return_timestamps=True,
                                    generate_kwargs=generate_kwargs)
                            except Exception as e2:  # noqa: BLE001
                                print("[WARN] ASR failed for {} ({})".format(
                                    vid, repr(e2)))
                                audio_ok = False
                        else:
                            print("[WARN] ASR failed for {} ({})".format(
                                vid, repr(e)))
                            audio_ok = False
                    if result is not None:
                        chunks = normalise_chunks(
                            result.get("chunks"), duration)
                        if not chunks and (result.get("text") or "").strip():
                            # timestamp-less fallback: whole text, full span
                            ts_mode = "chunk"
                            chunks = [[0.0, duration,
                                       result["text"].strip()]]
                else:
                    n_no_audio += 1

                bounds = window_time_bounds(
                    duration if duration > 0 else 1.0,
                    args.num_frames, args.num_subclips)
                window_text = assign_chunks_to_windows(
                    chunks, bounds, word_level=(ts_mode == "word"))
                if not any(t for t in window_text):
                    n_empty += 1

                fout.write(json.dumps({
                    "id": vid,
                    "label": item["label"],
                    "duration": duration,
                    "audio_ok": bool(audio_ok),
                    "language": language or "detect",
                    "timestamps": ts_mode,
                    "chunks": chunks,
                    "window_bounds": bounds,
                    "window_text": window_text,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
                if (n + 1) % 25 == 0:
                    print("  [{}] {}/{} videos (no-audio={}, all-empty={})".format(
                        split, n + 1, len(items), n_no_audio, n_empty))

        print("[{}] wrote {} new records -> {} (no-audio={}, all-empty-text={})".format(
            split, n_done, out_path, n_no_audio, n_empty))


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
