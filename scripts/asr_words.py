"""Word-level Whisper transcripts with reliable timestamps for HateMM, HateClipSeg and DeHate.

Why (experiments/20260925_query_paradigm/README.md section 14): the node transcripts of the query tree were cut from
data/ASR/<Corpus>/*_asrK30_whisper-large-v3.jsonl by the midpoint of each Whisper chunk. The word-timestamp run
behind that cache crashed for 51% of HateMM and 62% of HateClipSeg videos and fell back to chunk timestamps, and
those chunks are often long or broken (hate_video_132: one chunk [0, 225.7] s; non_hate_video_58: 74 s of speech
stamped [73.8, 74.0] s). The whole text of such a chunk lands in one node and 57-71% of the short test nodes of
positive videos get no transcript at all.

Method (the WhisperX recipe: Whisper for the words, a CTC model for their times):
1. Each video's 16 kHz audio (data/AV2A_wav/<Corpus>/<id>.wav) is cut into n = ceil(duration / 30 s) equal,
   non-overlapping windows (each <= 30 s, Whisper's context). Each window is transcribed on its own by
   whisper-large-v3 (English, greedy, text only); no long-form stitching. (Segment timestamps and 5-beam search were
   15x slower on the 5090s: 6.5 s against 0.4 s per window.) A batch that runs out of GPU memory is retried with
   smaller batches.
2. The window's words are aligned to the window audio by CTC forced alignment with torchaudio's
   WAV2VEC2_ASR_BASE_960H (letters and apostrophe; torchaudio.functional.forced_align). Words without a letter
   (numbers, symbols) take the time between their aligned neighbours. A window whose text cannot be aligned (no
   letters, or more letters than audio frames, e.g. a Whisper repetition loop) has its words spread evenly over the
   window (counted in `spread_windows`).

    python scripts/asr_words.py --corpus hatemm --splits test [--shard 0/2] [--batch 24]
Only videos of the listed splits (scripts/reproduction_baselines/hate_common split lists) are transcribed, in the
listed order. Output: data/ASR_words/<Corpus>/words[.shard<i>of<n>].jsonl, one line per video:
    {"id", "duration", "n_windows", "spread_windows", "align_score", "words": [[start_s, end_s, text]]}
(text has a leading space; join with "" to rebuild the transcript; align_score = mean CTC probability of the
aligned letters). Resumes: videos already written to any words*.jsonl of the corpus are skipped. Merge shards with
--merge.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
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


class Aligner:
    def __init__(self, device):
        import torch
        import torchaudio
        self.torch, self.F = torch, torchaudio.functional
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(device).eval()
        self.dict = {c: i for i, c in enumerate(bundle.get_labels()) if c not in "-|"}
        self.device = device

    def norm(self, w):
        return [self.dict[c] for c in w.upper().replace("’", "'") if c in self.dict]

    def emission(self, clip):
        with self.torch.inference_mode():
            em, _ = self.model(self.torch.from_numpy(np.ascontiguousarray(clip)).to(self.device)[None])
        return self.torch.log_softmax(em.float(), -1)[0]

    def words(self, lp, dur, text):
        """Word times (seconds from the window start) of one window's text; None when it cannot be aligned."""
        toks = text.split()
        s, e = 0.0, dur
        fps = lp.shape[0] / dur
        f0, f1 = 0, lp.shape[0]
        ids = [self.norm(t) for t in toks]
        keep = [i for i, x in enumerate(ids) if x]
        target = [c for i in keep for c in ids[i]]
        rep = sum(1 for a, b in zip(target, target[1:]) if a == b)
        if not target or f1 - f0 < len(target) + rep:
            return None, None
        ali, sc = self.F.forced_align(lp[f0:f1][None], self.torch.tensor([target], dtype=self.torch.int32,
                                                                         device=lp.device), blank=0)
        spans = self.F.merge_tokens(ali[0], sc[0].exp())
        if len(spans) != len(target):
            return None, None
        t = [None] * len(toks)
        k = 0
        for i in keep:
            sp = spans[k:k + len(ids[i])]
            k += len(ids[i])
            t[i] = [(f0 + sp[0].start) / fps, (f0 + sp[-1].end) / fps]
        # words without letters: between the neighbours' times
        for i in range(len(toks)):
            if t[i] is None:
                prev = next((t[j][1] for j in range(i - 1, -1, -1) if t[j] is not None), s)
                nxt = next((t[j][0] for j in range(i + 1, len(toks)) if t[j] is not None), e)
                t[i] = [prev, max(prev, nxt)]
        return [(a, b, " " + w) for (a, b), w in zip(t, toks)], float(np.mean([x.score for x in spans]))


def spread(s, e, text):
    w = text.split()
    return [(s + (e - s) * i / len(w), s + (e - s) * (i + 1) / len(w), " " + x) for i, x in enumerate(w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=tuple(CORPUS_DIR))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--splits", default="test,val,train")
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
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    from hate_common import data as hdata

    si, sn = (int(x) for x in a.shard.split("/"))
    out_path = os.path.join(out_dir, "words.jsonl" if sn == 1 else "words.shard%dof%d.jsonl" % (si, sn))
    done = set()
    for p in glob.glob(os.path.join(out_dir, "words*.jsonl")):
        for line in open(p):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    wav_dir = os.path.join(ROOT, "data", "AV2A_wav", d)
    todo = []
    for split in a.splits.split(","):
        ids = [v for v in hdata.load_split(a.corpus, split) if os.path.exists(os.path.join(wav_dir, v + ".wav"))]
        todo += [os.path.join(wav_dir, v + ".wav") for i, v in enumerate(ids) if i % sn == si and v not in done]
    print("%s shard %s: %d videos to do, %d done" % (a.corpus, a.shard, len(todo), len(done)), flush=True)
    asr = pipeline("automatic-speech-recognition", model=a.model, dtype=torch.float16, device="cuda:0")
    al = Aligner("cuda:0")
    gk = {"task": "transcribe", "language": "en", "num_beams": 1}
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
            words, n_spread, scores = [], 0, []
            if clips:
                for bs in (a.batch, 4, 1):
                    try:
                        res = asr([{"raw": c, "sampling_rate": SR} for c in clips], batch_size=bs,
                                  generate_kwargs=gk)
                        break
                    except torch.OutOfMemoryError:
                        if bs == 1:
                            raise
                        print("[OOM] %s at batch %d, retrying smaller" % (v, bs), flush=True)
                        torch.cuda.empty_cache()
                for (ws, we), c, r in zip(wins, clips, res):
                    text = (r.get("text") or "").strip()
                    if not text:
                        continue
                    w, sc = al.words(al.emission(c), we - ws, text)
                    if w is None:
                        n_spread += 1
                        w = spread(0.0, we - ws, text)
                    else:
                        scores.append(sc)
                    words += [[round(ws + x, 3), round(ws + y, 3), t] for x, y, t in w]
            fh.write(json.dumps({"id": v, "duration": round(dur, 3), "n_windows": len(wins),
                                 "spread_windows": n_spread,
                                 "align_score": round(float(np.mean(scores)), 4) if scores else None,
                                 "words": words}) + "\n")
            fh.flush()
            n_win += len(wins)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                print("%d/%d videos, %d windows, %.2f windows/s" % (k + 1, len(todo), n_win,
                                                                   n_win / (time.time() - t0)), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
