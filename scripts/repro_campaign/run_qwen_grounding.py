#!/usr/bin/env python
"""REPRO campaign Wave 0 — Qwen2.5-VL-7B native temporal grounding, full corpora.

Driver around the verified harness `scripts/repro_campaign/qwen25vl_grounding.py`.
The harness's prompt (lmms-eval `temporal_grounding_charades` `pre_prompt` /
`post_prompt`), its greedy decoding settings and its `extract_time` parser are
imported unchanged; this file only adds the corpus loop, resume, per-video video
decode reuse across queries, and progress logging.

Query text (freeze §9 does not fix one, so it is recorded here):
  main row  -> "the moment containing hateful content targeting a person or group"
  HateClipSeg extra rows -> one query per released class
                            (0 normal, 1 hateful, 2 insulting, 3 sexual,
                             4 violence, 5 harm)

Output: one JSONL per (dataset, query key) under
  idea-stage/repro_qwen_ground/raw/qwen_<DS>_<qkey>.jsonl
with fields video_id / query_key / query / raw / span / n_frames_seen / sec.
A model refusal or an unparseable generation stores span=null; those videos are
counted and reported as missing, never interpolated (freeze §14 `notes`).

Idempotent: an id already present in the output file is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts/repro_campaign"))
from qwen25vl_grounding import build_text, extract_time  # noqa: E402

HOME = Path.home()
VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")

# freeze §12 D2: no video stream at all, nothing to ground on.
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}

MAIN_QUERY = "the moment containing hateful content targeting a person or group"
HCS_CLASS_QUERIES = {
    "c0_normal": "the moment containing normal, non-toxic content",
    "c1_hateful": "the moment containing hateful content",
    "c2_insulting": "the moment containing insulting content",
    "c3_sexual": "the moment containing sexual content",
    "c4_violence": "the moment containing violent content",
    "c5_harm": "the moment containing harmful content",
}

OUT_DIR = ROOT / "idea-stage/repro_qwen_ground/raw"


def find_video(ds: str, vid: str):
    d = VIDEO_DIR[ds]
    for ext in EXTS:
        p = d / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def dataset_ids(ds: str) -> list[str]:
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return sorted(str(v) for v in z["video_ids"])


PERMANENT_ERRORS = {"no_video_stream", "missing_file", "decode_all_backends_failed"}


def done_ids(path: Path) -> set[str]:
    """Ids that need no further work: a real generation, or a permanent failure.

    A transient failure (OOM, a decode error that a later backend fix repairs) is
    NOT counted, so a restart retries it.  The evaluator keeps the last record per
    id, so a retried id's earlier failure line is superseded.
    """
    if not path.exists():
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("raw") is not None or r.get("error") in PERMANENT_ERRORS:
                out.add(r["video_id"])
            else:
                out.discard(r["video_id"])
    return out


# ------------------------------------------------------------ fallback decode ---
# qwen_vl_utils tries decord and falls back to torchvision, which reports a
# nonsense frame count on these containers and then refuses nframes=32.  ~27% of
# MHC-EN, ~7% of MHC-ZH and ~7% of HateClipSeg videos hit this.  PyAV opens all of
# them.  Frames are sampled at the same uniform indices decord would have used:
# round(linspace(0, total-1, nframes)).
def decode_frames_pyav(path: Path, nframes: int):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames or 0
        if total <= 0:  # container has no frame count: estimate from duration*rate
            dur = float(stream.duration * stream.time_base) if stream.duration else None
            if dur is None and container.duration:
                dur = container.duration / av.time_base
            rate = float(stream.average_rate or 25.0)
            total = int(dur * rate) if dur else 0
        if total >= nframes:
            want = sorted(set(int(round(x)) for x in
                              np.linspace(0, total - 1, nframes).tolist()))
        else:
            want = None  # take everything, pad afterwards
        frames, idx = [], 0
        for frame in container.decode(video=0):
            if want is None or idx in want:
                frames.append(frame.to_ndarray(format="rgb24"))
            idx += 1
            if want is not None and len(frames) >= len(want) and idx > max(want):
                break
    if not frames:
        raise RuntimeError("pyav decoded 0 frames")
    if len(frames) < nframes:  # short/blank clip: repeat the last frame
        frames += [frames[-1]] * (nframes - len(frames))
    frames = frames[:nframes]
    if len(frames) % 2:  # Qwen FRAME_FACTOR = 2
        frames = frames[:-1]
    arr = np.stack(frames)  # (T, H, W, 3) uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--hcs-classes", action="store_true",
                    help="also run the six per-class HateClipSeg queries")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--max-pixels", type=int, default=151200)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N ids per dataset")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(torch_dtype=getattr(torch, args.dtype), device_map="cuda:0",
              attn_implementation="sdpa")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw.pop("torch_dtype")

    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kw).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    print(f"[load] {time.time()-t0:.1f}s mem={torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
          flush=True)

    # ---- build the work plan -------------------------------------------------
    plan = []  # (ds, vid, [(qkey, qtext), ...])
    for ds in args.datasets.split(","):
        ids = dataset_ids(ds)
        if args.limit:
            ids = ids[: args.limit]
        qs = [("main", MAIN_QUERY)]
        if ds == "HateClipSeg" and args.hcs_classes:
            qs += list(HCS_CLASS_QUERIES.items())
        done = {qk: done_ids(out_dir / f"qwen_{ds}_{qk}.jsonl") for qk, _ in qs}
        for vid in ids:
            todo = [(qk, qt) for qk, qt in qs if vid not in done[qk]]
            if todo:
                plan.append((ds, vid, todo))
    n_calls = sum(len(t[2]) for t in plan)
    print(f"[plan] videos={len(plan)} generations={n_calls} datasets={args.datasets} "
          f"hcs_classes={args.hcs_classes}", flush=True)

    handles: dict[str, object] = {}

    def fh(ds, qk):
        key = f"{ds}_{qk}"
        if key not in handles:
            handles[key] = open(out_dir / f"qwen_{ds}_{qk}.jsonl", "a")
        return handles[key]

    durations = {}
    for ds in args.datasets.split(","):
        p = ROOT / f"data/gt/frame_gt_4fps/durations_{ds}.json"
        durations[ds] = json.loads(p.read_text()) if p.exists() else {}

    n_done = n_err = n_unparsed = n_fallback = 0
    t0 = time.time()
    for i, (ds, vid, todo) in enumerate(plan, 1):
        path = find_video(ds, vid)
        if path is None or vid in NO_VIDEO_STREAM:
            for qk, qt in todo:
                rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                       "raw": None, "span": None,
                       "error": "no_video_stream" if vid in NO_VIDEO_STREAM else "missing_file"}
                fh(ds, qk).write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh(ds, qk).flush()
            n_err += 1
            continue

        t1 = time.time()
        backend = "qwen_vl_utils"
        try:
            probe = [{"role": "user", "content": [
                {"type": "video", "video": str(path), "nframes": args.nframes,
                 "max_pixels": args.max_pixels},
                {"type": "text", "text": "x"}]}]
            _, video_inputs, video_kwargs = process_vision_info(probe, return_video_kwargs=True)
        except Exception as e_primary:
            try:
                backend = "pyav_fallback"
                vt, total = decode_frames_pyav(path, args.nframes)
                dur = float(durations.get(ds, {}).get(vid, 0.0)) or 0.0
                fps_val = (vt.shape[0] / dur) if dur > 0 else 1.0
                video_inputs, video_kwargs = [vt], {"fps": [fps_val]}
                n_fallback += 1
            except Exception as e:
                for qk, qt in todo:
                    rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                           "raw": None, "span": None,
                           "error": "decode_all_backends_failed",
                           "detail": f"{type(e_primary).__name__}:{e_primary}|"
                                     f"{type(e).__name__}:{e}"[:300]}
                    fh(ds, qk).write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh(ds, qk).flush()
                n_err += 1
                print(f"[DECODE-FAIL] {ds}/{vid}: {e}"[:200], flush=True)
                continue

        nseen = int(video_inputs[0].shape[0]) if video_inputs else 0
        for qk, qt in todo:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": str(path), "nframes": args.nframes,
                 "max_pixels": args.max_pixels},
                {"type": "text", "text": build_text(qt)}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            try:
                # OOM here is a transient contention artefact on a shared card, not a
                # property of the video: back off and retry before recording a failure.
                for attempt in range(4):
                    try:
                        inputs = processor(text=[text], images=None, videos=video_inputs,
                                           padding=True, return_tensors="pt",
                                           **video_kwargs).to("cuda:0")
                        with torch.inference_mode():
                            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                                 do_sample=False, temperature=None,
                                                 top_p=None, top_k=None, num_beams=1)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if attempt == 3:
                            raise
                        print(f"[OOM-retry {attempt+1}] {ds}/{vid}/{qk}", flush=True)
                        time.sleep(20 * (attempt + 1))
                gen = processor.batch_decode(
                    out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]
                span = extract_time(gen)
                if span is None:
                    n_unparsed += 1
                rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                       "raw": gen.strip(), "span": span, "n_frames_seen": nseen,
                       "sec": round(time.time() - t1, 2)}
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                       "raw": None, "span": None, "error": f"oom:{e}"[:200]}
                n_err += 1
            except Exception as e:
                rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                       "raw": None, "span": None, "error": f"{type(e).__name__}:{e}"[:300]}
                n_err += 1
            fh(ds, qk).write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh(ds, qk).flush()
            n_done += 1

        del video_inputs
        if i % 10 == 0 or i == len(plan):
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            print(f"PROGRESS {i}/{len(plan)} gens={n_done}/{n_calls} ds={ds} vid={vid} "
                  f"elapsed={el:.0f}s rate={rate:.2f}gen/s "
                  f"eta={(n_calls-n_done)/max(rate,1e-9):.0f}s "
                  f"err={n_err} unparsed={n_unparsed} pyav={n_fallback} "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)

    for h in handles.values():
        h.close()
    print(f"[done] generations={n_done} errors={n_err} unparsed={n_unparsed} "
          f"pyav_fallback={n_fallback} wall={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
