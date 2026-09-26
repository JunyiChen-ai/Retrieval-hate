"""Word-level Whisper transcripts with reliable timestamps for HateMM, HateClipSeg and DeHate.

Why (experiments/20260925_query_paradigm/README.md section 14): the node transcripts of the query tree were cut from
data/ASR/<Corpus>/*_asrK30_whisper-large-v3.jsonl by the midpoint of each Whisper chunk. The word-timestamp run
behind that cache crashed for 51% of HateMM and 62% of HateClipSeg videos and fell back to chunk timestamps, and
those chunks are often long or broken (hate_video_132: one chunk [0, 225.7] s; non_hate_video_58: 74 s of speech
stamped [73.8, 74.0] s). The whole text of such a chunk lands in one node and 57-71% of the short test nodes of
positive videos get no transcript at all.

Here every video's 16 kHz audio (data/AV2A_wav/<Corpus>/<id>.wav) is cut into n = ceil(duration / 30 s) equal,
non-overlapping windows (each <= 30 s, Whisper's context), and each window is transcribed on its own with
word timestamps (cross-attention alignment of whisper-large-v3, English, greedy). Windows never need long-form
stitching, which is where the old run crashed. A window whose word-timestamp decoding fails is retried alone; if it
still fails, it is decoded with segment timestamps and each segment's words are spread evenly over the segment
(counted in `fallback_windows`).

    python scripts/asr_words.py --corpus hatemm [--shard 0/2] [--batch 16]
Output: data/ASR_words/<Corpus>/words[.shard<i>of<n>].jsonl, one line per video:
    {"id", "duration", "n_windows", "fallback_windows", "words": [[start_s, end_s, text], ...]}
(text keeps Whisper's leading space; join with "" to rebuild the transcript). Resumes: videos already written are
skipped. Merge shards with --merge.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
SR = 16000
WIN = 30.0


def windows(duration):
    n = max(1, int(math.ceil(duration / WIN - 1e-9)))
    edges = np.linspace(0.0, duration, n + 1)
    return list(zip(edges[:-1], edges[1:]))


def words_from(result, t0, t1):
    out = []
    for c in result.get("chunks") or []:
        text = c.get("text") or ""
        if not text.strip():
            continue
        s, e = c.get("timestamp") or (None, None)
        if s is None:
            continue
        e = s if e is None else e
        s, e = min(max(t0 + s, t0), t1), min(max(t0 + e, t0), t1)
        out.append([round(float(s), 3), round(float(max(e, s)), 3), text])
    return out


def spread(result, t0, t1):
    """Segment-timestamp fallback: each segment's words evenly over the segment."""
    out = []
    for c in result.get("chunks") or []:
        w = (c.get("text") or "").split()
        s, e = c.get("timestamp") or (0.0, None)
        s = 0.0 if s is None else s
        e = (t1 - t0) if e is None else e
        s, e = min(max(t0 + s, t0), t1), min(max(t0 + e, t0), t1)
        e = max(e, s)
        for i, tok in enumerate(w):
            a = s + (e - s) * i / len(w)
            b = s + (e - s) * (i + 1) / len(w)
            out.append([round(float(a), 3), round(float(b), 3), " " + tok])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=tuple(CORPUS_DIR))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    d = CORPUS_DIR[a.corpus]
    out_dir = os.path.join(ROOT, "data", "ASR_words", d)
    os.makedirs(out_dir, exist_ok=True)
    if a.merge:
        seen = {}
        for p in sorted(glob.glob(os.path.join(out_dir, "words.shard*.jsonl"))):
            for line in open(p):
                r = json.loads(line)
                seen.setdefault(r["id"], line)
        with open(os.path.join(out_dir, "words.jsonl"), "w") as fh:
            for v in sorted(seen):
                fh.write(seen[v])
        print("merged %d videos" % len(seen))
        return

    import soundfile as sf
    import torch
    from transformers import pipeline

    si, sn = (int(x) for x in a.shard.split("/"))
    out_path = os.path.join(out_dir, "words.jsonl" if sn == 1 else "words.shard%dof%d.jsonl" % (si, sn))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    wavs = sorted(glob.glob(os.path.join(ROOT, "data", "AV2A_wav", d, "*.wav")))
    todo = [p for i, p in enumerate(wavs) if i % sn == si and os.path.basename(p)[:-4] not in done]
    print("%s shard %s: %d videos to do, %d done" % (a.corpus, a.shard, len(todo), len(done)), flush=True)
    asr = pipeline("automatic-speech-recognition", model=a.model, torch_dtype=torch.float16, device="cuda:0")
    gk = {"task": "transcribe", "language": "en"}
    t0, n_win = time.time(), 0
    with open(out_path, "a") as fh:
        for k, p in enumerate(todo):
            v = os.path.basename(p)[:-4]
            audio, sr = sf.read(p, dtype="float32")
            assert sr == SR, (p, sr)
            if audio.ndim > 1:
                audio = audio.mean(1)
            dur = len(audio) / SR
            wins = windows(dur) if dur > 0.1 else []
            clips = [audio[int(round(s * SR)):int(round(e * SR))] for s, e in wins]
            words, fb = [], 0
            if clips:
                try:
                    res = asr([{"raw": c, "sampling_rate": SR} for c in clips], batch_size=a.batch,
                              return_timestamps="word", generate_kwargs=gk)
                except Exception:
                    res = [None] * len(clips)
                for (s, e), c, r in zip(wins, clips, res):
                    if r is None:
                        try:
                            r = asr({"raw": c, "sampling_rate": SR}, return_timestamps="word", generate_kwargs=gk)
                        except Exception:
                            r = None
                    if r is not None:
                        words += words_from(r, s, e)
                        continue
                    fb += 1
                    try:
                        words += spread(asr({"raw": c, "sampling_rate": SR}, return_timestamps=True,
                                            generate_kwargs=gk), s, e)
                    except Exception as ex:  # noqa: BLE001
                        print("[WARN] %s window %.1f-%.1f failed: %r" % (v, s, e, ex), flush=True)
            fh.write(json.dumps({"id": v, "duration": round(dur, 3), "n_windows": len(wins), "fallback_windows": fb,
                                 "words": words}) + "\n")
            fh.flush()
            n_win += len(wins)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                print("%d/%d videos, %d windows, %.1f windows/s" % (k + 1, len(todo), n_win, n_win / (time.time() - t0)),
                      flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
