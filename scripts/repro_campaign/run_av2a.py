#!/usr/bin/env python
"""REPRO campaign Wave 1 — AV²A (CVPR 2025) on the four hate-video corpora.

Runs `third_party/AV2A/video_parser_optmizer.py::VideoParserOptimizer` **unmodified**
(the published pipeline: filter_classes -> per-second similarity -> optimize with
dynamic thresholds -> refine_segments) under its own venv
`third_party/_venv/av2a/bin/python`, with the README's LanguageBind
hyper-parameters verbatim.

Everything that is *ours* is in this file and is frozen before the run:

  * the open-vocabulary label list and the hate subset  (LABELS / HATE_LABELS below)
  * the windowing that turns a 40-240 s hate video into the 10-second, 10-bin unit
    the published code hard-codes (`VisionTransform`: `fps = len(vr) // 10`,
    `images_num = 10`; `late_fusion`: `max_time = 10`).  AV²A was published on LLP
    and AVE, whose clips are all exactly 10 s.
  * the read-out: a continuous per-bin similarity curve (max over the hate subset)
    and a binary rasterisation of the events the pipeline actually kept.

Outputs, per dataset:
  idea-stage/repro_av2a/curves/<DS>/<vid>.npz
      sim_video sim_audio sim_combined   float32 (10*n_win,)  continuous
      evt_video evt_audio evt_combined   float32 (10*n_win,)  binary 0/1
      rate                               float64 scalar, bins per second
  idea-stage/repro_av2a/raw/av2a_events_<DS>.jsonl   one line per video, all events
  idea-stage/repro_av2a/raw/failures_<DS>.jsonl      one line per dropped video
  idea-stage/repro_av2a/curves/<DS>_intervals_<v>.json   (--build-intervals)

CLI
  python scripts/repro_campaign/run_av2a.py --demux-only --datasets HateMM,...
  python scripts/repro_campaign/run_av2a.py --datasets HateMM --limit 2      # smoke
  python scripts/repro_campaign/run_av2a.py --datasets HateMM,MHC,MHC_zh,HateClipSeg
  python scripts/repro_campaign/run_av2a.py --build-intervals
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
AV2A = ROOT / "third_party/AV2A"
sys.path.insert(0, str(AV2A))

RUN_DIR = ROOT / "idea-stage/repro_av2a"
CURVE_DIR = RUN_DIR / "curves"
RAW_DIR = RUN_DIR / "raw"
WAV_ROOT = ROOT / "data/AV2A_wav"

DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]
VIDEO_DIRS = {
    "HateMM": Path("/home/jehc223/data/HateMM/video"),
    "MHC": Path("/home/jehc223/data/Multihateclip/English/video_mp4"),
    "MHC_zh": Path("/home/jehc223/data/Multihateclip/Chinese/video"),
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
# freeze §12 D2: two released HateMM containers hold a single AAC stream and no
# video stream at all.  There is no frame to encode; permanent failure, never faked.
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}

SEED = 20250819

# --------------------------------------------------------------------------
# FROZEN LABEL VOCABULARY.  Written into this file and committed before the run
# that produces any number; never tuned against any metric.
#
# AV²A is open-vocabulary and localises *audio-visual events*, and its
# `norm_similarities` z-scores the text-similarity vector across the label axis
# before the sigmoid, so the scores are relative to the rest of the vocabulary and
# its absolute thresholds (filter 0.55, stage-1/2 0.75) are meaningless for a
# hate-only list.  The vocabulary therefore pairs hate-related event words with
# ordinary background event words, in the style of the LLP 25-class list the paper
# ships with.  Strings enter the LanguageBind text encoder through the repo's own
# template, `f"A {label.replace('_',' ').lower()}"`, unmodified.
HATE_LABELS = [
    "racial slur",
    "hate speech",
    "angry shouting insults",
    "swastika symbol",
    "nazi salute",
    "white supremacist rally",
    "racist joke",
    "homophobic remark",
    "misogynistic rant",
    "threat of violence",
    "physical assault",
    "hateful caption on screen",
]
NEUTRAL_LABELS = [
    "speech",
    "conversation",
    "music",
    "singing",
    "laughter",
    "applause",
    "cheering crowd",
    "car engine",
    "dog barking",
    "cooking food",
    "typing on a keyboard",
    "news broadcast",
    "video game play",
    "cartoon animation",
    "sports match",
    "dancing",
    "telephone ringing",
    "silence",
    "walking outdoors",
    "person talking to camera",
]
LABELS = HATE_LABELS + NEUTRAL_LABELS
HATE_IDX = list(range(len(HATE_LABELS)))
HATE_SET = {s.lower() for s in HATE_LABELS}

# README hyper-parameters, LanguageBind row, verbatim.
HP = dict(method="bbse-cosine", backbone="language_bind", alpha=0.5,
          filter_threshold=0.55, threshold_stage1=0.75, threshold_stage2=0.75,
          gamma=2.5, fusion="early")
# `dataset` only selects refine_segments' AVE argmax branch; LLP is the
# multi-label branch, which is the one a hate corpus needs.
HP_DATASET = "LLP"

BINS_PER_WINDOW = 10          # VisionTransform.images_num, hard-coded upstream
WINDOW_SEC = 10.0             # LLP/AVE clip length the published code assumes

VARIANTS = ["sim_video", "sim_audio", "sim_combined",
            "evt_video", "evt_audio", "evt_combined"]


# ----------------------------------------------------------------- helpers ---
def find_video(ds: str, vid: str) -> Path | None:
    d = VIDEO_DIRS[ds]
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".flv", ".mov"):
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    hits = sorted(d.glob(f"{vid}.*"))
    return hits[0] if hits else None


def demux_wav(src: Path, dst: Path) -> str:
    """16 kHz mono PCM wav next to the video, idempotent.

    16 kHz is `AudioTransform.sample_rate`, so the transform's resample step is a
    no-op and nothing is resampled twice.
    """
    if dst.exists() and dst.stat().st_size > 44:
        return "cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-vn", "-map", "0:a:0", "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le", "-f", "wav", str(tmp)],
        capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        return "no_audio"
    os.replace(tmp, dst)
    return "demuxed"


class WindowVR:
    """A view of a decord VideoReader restricted to [i0, i1).

    `VisionTransform` only uses `len()` and `get_batch()`, so a view is enough to
    hand the unmodified pipeline the 10-second unit it expects without cutting new
    files.  Frame indices inside the window are 0-based, as they would be for a
    standalone 10 s clip.
    """

    def __init__(self, vr, i0: int, i1: int):
        self.vr, self.i0, self.i1 = vr, i0, i1

    def __len__(self):
        return self.i1 - self.i0

    def get_batch(self, idx):
        return self.vr.get_batch([int(i) + self.i0 for i in idx])


class PyAVReader:
    """decord-compatible whole-video reader for containers decord refuses.

    Same fallback the Qwen2.5-VL run needed (`REPRO_CAMPAIGN_RESULTS §I`): a
    substantial share of the MHC containers do not expose a video stream index to
    decord.  Frames are decoded once and held as uint8; at 224-px-bound inputs the
    videos in these corpora fit comfortably.
    """

    def __init__(self, path: Path, max_frames: int = 4000):
        import av
        import torch
        frames = []
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            n = st.frames or 0
            step = max(1, math.ceil(n / max_frames)) if n else 1
            for k, f in enumerate(c.decode(video=0)):
                if k % step == 0:
                    frames.append(f.to_ndarray(format="rgb24"))
                if len(frames) >= max_frames:
                    break
        if not frames:
            raise RuntimeError("pyav decoded 0 frames")
        self.arr = torch.from_numpy(np.stack(frames))

    def __len__(self):
        return self.arr.shape[0]

    def get_batch(self, idx):
        import torch
        return self.arr[torch.as_tensor([int(i) for i in idx])]


def open_video(path: Path):
    from decord import VideoReader, cpu
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
        if len(vr) > 0:
            _ = vr.get_batch([0])
            return vr, "decord"
    except Exception:
        pass
    return PyAVReader(path), "pyav"


def jsonl_ids(p: Path) -> set:
    out = set()
    if p.exists():
        with open(p) as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)["video_id"])
                except Exception:
                    continue
    return out


def append_jsonl(p: Path, rec: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez appends .npz otherwise
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- the run ---
def process_video(vparser, model, vision_tf, audio_tf, ds, vid, path, duration, device):
    """One video -> (curves dict, all-events dict, meta).  Raises on failure."""
    import torch
    import torchaudio

    vr, backend = open_video(path)
    n_frames_total = len(vr)
    if n_frames_total < BINS_PER_WINDOW:
        raise RuntimeError(f"too_few_frames:{n_frames_total}")

    wav, sr = torchaudio.load(str(WAV_ROOT / ds / f"{vid}.wav"))
    # the wav is written by us at 16 kHz mono; keep the check honest anyway
    n_samp = wav.shape[1]

    # windowing: as many equal windows of at most 10 s as the video needs, and
    # never so many that a window holds fewer than 10 frames (the pipeline's
    # `fps = len(vr) // 10` would degenerate to 0).
    n_win = max(1, math.ceil(duration / WINDOW_SEC))
    n_win = max(1, min(n_win, n_frames_total // BINS_PER_WINDOW))
    W = duration / n_win
    n_bins = n_win * BINS_PER_WINDOW
    rate = n_bins / duration

    sim = {k: np.zeros(n_bins, dtype=np.float32) for k in ("video", "audio", "combined")}
    evt = {k: np.zeros(n_bins, dtype=np.float32) for k in ("video", "audio", "combined")}
    all_events = {"video": [], "audio": [], "combined": []}
    hate_iv = {"video": [], "audio": [], "combined": []}

    fps_native = n_frames_total / duration
    for k in range(n_win):
        t0, t1 = k * W, (k + 1) * W
        i0 = int(round(t0 * fps_native))
        i1 = int(round(t1 * fps_native)) if k < n_win - 1 else n_frames_total
        i1 = min(max(i1, i0 + BINS_PER_WINDOW), n_frames_total)
        i0 = min(i0, i1 - BINS_PER_WINDOW)
        win_vr = WindowVR(vr, i0, i1)

        a0 = int(round(t0 * sr))
        a1 = min(int(round(t1 * sr)), n_samp)
        # clone: AudioTransform.get_mel subtracts the mean *in place*, and a slice
        # of the full waveform is a view, so an un-cloned slice would corrupt the
        # neighbouring windows.
        wav_win = wav[:, a0:a1].clone()
        if wav_win.shape[1] < sr // 10:  # < 0.1 s of audio in this window
            wav_win = torch.zeros((wav.shape[0], max(1, sr // 10)), dtype=wav.dtype)

        # ---- the published pipeline, unmodified -------------------------------
        combined_res, video_res, audio_res = vparser(
            LABELS, win_vr, (wav_win, sr), vid)

        # ---- our continuous read-out, one extra forward with the FULL label set.
        # The pipeline's own per-second matrices are computed on the *filtered*
        # label subset, which differs per window and per channel; a curve built on
        # a moving vocabulary is not comparable across windows, so the continuous
        # variant uses the frozen list throughout.  Same transforms, same model,
        # same `norm_similarities`.
        vt_img = vision_tf(win_vr, transform_type="image").to(device)
        at = audio_tf((wav_win.clone(), sr)).to(device)
        s = model(LABELS, vt_img, at, similarity_type="combined", vision_mode="image")
        b0 = k * BINS_PER_WINDOW
        for ch, key in (("video", "image"), ("audio", "audio"), ("combined", "combined")):
            m = s[key].detach().float().cpu().numpy()          # (10, L)
            sim[ch][b0:b0 + BINS_PER_WINDOW] = m[:, HATE_IDX].max(axis=1)

        # ---- rasterise the events the pipeline kept ---------------------------
        bin_sec = W / BINS_PER_WINDOW
        for ch, res in (("video", video_res), ("audio", audio_res),
                        ("combined", combined_res)):
            for e in res.get(vid, []):
                lab = str(e["event_label"]).lower()
                gs = t0 + float(e["start"]) * bin_sec
                ge = t0 + float(e["end"]) * bin_sec
                all_events[ch].append([LABELS.index(lab) if lab in LABELS else -1,
                                       round(gs, 3), round(ge, 3)])
                if lab in HATE_SET:
                    j0 = b0 + int(e["start"])
                    j1 = b0 + int(e["end"])
                    evt[ch][max(b0, j0):min(b0 + BINS_PER_WINDOW, j1)] = 1.0
                    hate_iv[ch].append([round(gs, 3), round(min(ge, duration), 3)])

    # merge touching hate intervals so the proposal metric sees one per stretch
    for ch in hate_iv:
        ivs = sorted(hate_iv[ch])
        merged = []
        for a, b in ivs:
            if merged and a <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        hate_iv[ch] = merged

    curves = {f"sim_{c}": sim[c] for c in sim}
    curves.update({f"evt_{c}": evt[c] for c in evt})
    meta = dict(n_win=n_win, window_sec=round(W, 4), rate=rate, n_bins=n_bins,
                backend=backend, n_frames=n_frames_total, sr=sr,
                audio_sec=round(n_samp / sr, 3))
    return curves, rate, all_events, hate_iv, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--demux-only", action="store_true")
    ap.add_argument("--build-intervals", action="store_true")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--out-root", default="",
                    help="redirect all outputs (a dry run writes nowhere real)")
    args = ap.parse_args()

    if args.out_root:
        global RUN_DIR, CURVE_DIR, RAW_DIR
        RUN_DIR = Path(args.out_root)
        CURVE_DIR = RUN_DIR / "curves"
        RAW_DIR = RUN_DIR / "raw"

    dss = [d for d in args.datasets.split(",") if d]

    if args.build_intervals:
        for ds in dss:
            ev = RAW_DIR / f"av2a_events_{ds}.jsonl"
            if not ev.exists():
                continue
            per = {v: {} for v in ("video", "audio", "combined")}
            with open(ev) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    for ch in per:
                        per[ch][r["video_id"]] = [
                            [a, b, 1.0] for a, b in r["hate_intervals"][ch]]
            for ch in per:
                p = CURVE_DIR / f"{ds}_intervals_evt_{ch}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(per[ch]))
                print(f"[intervals] {p} videos={len(per[ch])}")
        return 0

    # ---------------------------------------------------------------- plan ---
    plan = []
    want = set(args.ids.split(",")) if args.ids else None
    for ds in dss:
        z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
        ids = [str(v) for v in z["video_ids"]]
        durs = {i: float(d) for i, d in zip(ids, z["duration"])}
        done = {p.stem for p in (CURVE_DIR / ds).glob("*.npz")}
        failed = jsonl_ids(RAW_DIR / f"failures_{ds}.jsonl")
        for vid in ids:
            if want is not None and vid not in want:
                continue
            if vid in done or vid in failed:
                continue
            plan.append((ds, vid, durs[vid]))
    if args.limit:
        keep, seen = [], {}
        for ds, vid, d in plan:
            seen[ds] = seen.get(ds, 0) + 1
            if seen[ds] <= args.limit:
                keep.append((ds, vid, d))
        plan = keep
    print(f"[plan] {len(plan)} videos over {dss}", flush=True)

    # -------------------------------------------------------------- demux ---
    counts = {}
    todo = []
    for ds, vid, dur in plan:
        path = find_video(ds, vid)
        if path is None or vid in NO_VIDEO_STREAM:
            reason = "no_video_stream" if vid in NO_VIDEO_STREAM else "missing_file"
            append_jsonl(RAW_DIR / f"failures_{ds}.jsonl",
                         dict(video_id=vid, dataset=ds, reason=reason))
            counts[reason] = counts.get(reason, 0) + 1
            continue
        st = demux_wav(path, WAV_ROOT / ds / f"{vid}.wav")
        counts[st] = counts.get(st, 0) + 1
        if st == "no_audio":
            # AV²A is an audio-visual method: `filter_classes` and every combined
            # score need the audio branch.  No silence is fabricated; the video is
            # recorded as a failure and dropped from the pool (freeze §14).
            append_jsonl(RAW_DIR / f"failures_{ds}.jsonl",
                         dict(video_id=vid, dataset=ds, reason="no_audio_stream"))
            continue
        todo.append((ds, vid, dur, path))
        if len(todo) % 200 == 0:
            print(f"PROGRESS demux {len(todo)} ok {counts}", flush=True)
    print(f"[demux] {counts}", flush=True)
    if args.demux_only:
        return 0

    # ------------------------------------------------------ crash recovery ---
    inflight = RUN_DIR / ".inflight"
    if inflight.exists():
        crashed = inflight.read_text().strip()
        if crashed:
            ds_c, vid_c = crashed.split("\t", 1)
            append_jsonl(RAW_DIR / f"failures_{ds_c}.jsonl",
                         dict(video_id=vid_c, dataset=ds_c,
                              reason="decoder_crashed_the_process"))
            print(f"[CRASH-RETIRED] {crashed}", flush=True)
            todo = [t for t in todo if not (t[0] == ds_c and t[1] == vid_c)]
        inflight.unlink()

    if not todo:
        print("[done] nothing to do", flush=True)
        return 0

    # --------------------------------------------------------------- model ---
    import torch
    from data_transforms import AudioTransform, VisionTransform
    from utils import set_random_seed
    from video_parser_optmizer import VideoParserOptimizer

    os.chdir(AV2A)  # backbones.py hard-codes cache_dir='./cache_dir'
    set_random_seed(SEED)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    vparser = VideoParserOptimizer(
        HP["method"], HP["backbone"], LABELS, device, HP["alpha"],
        HP["filter_threshold"], HP["threshold_stage1"], HP["threshold_stage2"],
        HP["gamma"], False, False, HP_DATASET, HP["fusion"])
    model = vparser.model
    vision_tf = VisionTransform(model=HP["backbone"])
    audio_tf = AudioTransform(model=HP["backbone"])
    print(f"[model] loaded on {device}", flush=True)

    meta_path = RUN_DIR / "run_meta.json"
    if not meta_path.exists():
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(dict(
            method="AV2A", repo="third_party/AV2A", hyperparams=HP,
            hp_dataset=HP_DATASET, seed=SEED, labels=LABELS,
            hate_labels=HATE_LABELS, bins_per_window=BINS_PER_WINDOW,
            window_sec=WINDOW_SEC, torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            git=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip(),
        ), indent=1))

    t0 = time.time()
    n_ok = n_fail = 0
    for i, (ds, vid, dur, path) in enumerate(todo, 1):
        inflight.write_text(f"{ds}\t{vid}")
        # per-video reseed: the published video transform keeps a
        # RandomHorizontalFlipVideo, so filter_classes and refine_segments are
        # stochastic.  Seeding from the id makes a resumed run bit-identical to an
        # uninterrupted one.
        set_random_seed(SEED ^ (zlib.crc32(vid.encode()) & 0xFFFFFFFF))
        tv = time.time()
        try:
            curves, rate, all_events, hate_iv, meta = process_video(
                vparser, model, vision_tf, audio_tf, ds, vid, path, dur, device)
        except Exception as e:
            append_jsonl(RAW_DIR / f"failures_{ds}.jsonl",
                         dict(video_id=vid, dataset=ds,
                              reason=f"{type(e).__name__}: {str(e)[:200]}"))
            n_fail += 1
            inflight.write_text("")
            continue
        save_npz(CURVE_DIR / ds / f"{vid}.npz",
                 rate=np.float64(rate), **curves)
        append_jsonl(RAW_DIR / f"av2a_events_{ds}.jsonl", dict(
            video_id=vid, dataset=ds, duration=round(dur, 3), **meta,
            sec=round(time.time() - tv, 2),
            events=all_events, hate_intervals=hate_iv,
            sim_mean={c: round(float(curves[f"sim_{c}"].mean()), 4)
                      for c in ("video", "audio", "combined")},
            sim_std={c: round(float(curves[f"sim_{c}"].std()), 4)
                     for c in ("video", "audio", "combined")},
            evt_frac={c: round(float(curves[f"evt_{c}"].mean()), 4)
                      for c in ("video", "audio", "combined")}))
        n_ok += 1
        inflight.write_text("")
        if i % args.progress_every == 0 or i == len(todo):
            el = time.time() - t0
            print(f"PROGRESS {i}/{len(todo)} ok={n_ok} fail={n_fail} ds={ds} "
                  f"vid={vid} sec/vid={el/i:.2f} eta_h={(len(todo)-i)*el/i/3600:.2f}",
                  flush=True)
    inflight.unlink(missing_ok=True)
    print(f"[done] ok={n_ok} fail={n_fail} wall={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
