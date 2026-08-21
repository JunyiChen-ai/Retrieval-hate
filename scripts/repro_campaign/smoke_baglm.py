#!/usr/bin/env python
"""Smoke test: BaGLM's video-step-grounding (VSG) stage on one of our videos.

BaGLM has no single-video entry point; `src/{coin,htstep,crosstask,ego4d_goalstep}_eval.py`
each bind a dataset class (COIN.json + taxonomy.xlsx, HT-Step, ...) to the scorer. This
driver keeps the scorer path exactly — `t2v_metrics.get_score_model(...)`,
`score.forward_vsg(video_data, ...)`, `prompts/vsg/question.txt`, `segment_duration=2`,
`sampling_fps=2` — and replaces only `dataset.COIN.__getitem__`, building the same
`{"video_uid", "videos", "task_id", "texts"}` dict from one of our mp4s.

Backbone substitution: the paper uses `internvl2.5-8b` (~16 GB bf16). The shared RTX 5090
is held by another job, so this smoke uses `internvl2.5-1b`, which is in the repo's own
`INTERNVL2_MODELS` table and fits the ~3 GiB headroom. The 8B row is the one that would be
reported; this only checks that the pipeline runs.

Also patched at runtime: `use_flash_attn=True` -> False. flash-attn 2.7.4.post1 (the repo's
pin) has no sm_120 wheel; InternVL's ViT degrades on its own but the LLM config would ask
transformers for flash_attention_2 and raise.
"""
import argparse
import os
import sys

import numpy as np
import torch

BAGLM = "/home/jehc223/Retrieval-hate/third_party/baglm"

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="/home/jehc223/data/HateClipSeg/videos/bit_0dcMcI6hYjhw.mp4")
ap.add_argument("--model", default="internvl2.5-1b")
ap.add_argument("--device", default="cuda")
ap.add_argument("--mem-frac", type=float, default=0.09)
ap.add_argument("--segment-duration", type=int, default=2)
ap.add_argument("--sampling-fps", type=int, default=2)
ap.add_argument("--max-segments", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()

os.chdir(BAGLM)
sys.path.insert(0, os.path.join(BAGLM, "src"))
sys.path.insert(0, os.path.join(BAGLM, "t2v_metrics"))

if args.device.startswith("cuda"):
    torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)

import t2v_metrics  # noqa: E402
from t2v_metrics.models.vqascore_models import internvl_model  # noqa: E402
from utils.video_utils import get_frame_indices, get_video_metadata, load_video  # noqa: E402

for _k, _v in internvl_model.INTERNVL2_MODELS.items():
    _v["model"]["use_flash_attn"] = False  # no sm_120 flash-attn wheel

meta = get_video_metadata(args.video, device="cpu")
print(f"[smoke] video metadata {meta}")
frame_indices = get_frame_indices(meta["num_frames"], meta["fps"], args.sampling_fps)
frames_per_segment = args.segment_duration * args.sampling_fps
frame_indices = frame_indices[: args.max_segments * frames_per_segment]
print(f"[smoke] {len(frame_indices)} frames at {args.sampling_fps} fps "
      f"-> {len(frame_indices) // frames_per_segment} segments of {args.segment_duration} s")

score_func = t2v_metrics.get_score_model(model=args.model, device=args.device)
preprocess_fn = score_func.model.get_preprocessor()
print("[smoke] scorer loaded")

all_frames = load_video(args.video, frame_indices, device="cpu")
all_frames = [preprocess_fn(f.data, nframes=frames_per_segment) for f in all_frames]
segments = []
for i in range(0, len(all_frames), frames_per_segment):
    seg = all_frames[i: i + frames_per_segment]
    if len(seg) == frames_per_segment:
        segments.append(torch.cat(seg))
print(f"[smoke] {len(segments)} segments, each {tuple(segments[0].shape)}")

# BaGLM's "goal" is the procedure and the "steps" are its ordered actions. The analogous
# hate-domain instantiation is the video-level target as the goal and the campaign's six
# hate categories as the step set; forward_vsg appends "None of the above." itself.
GOAL = "post a video expressing hostility toward a group of people"
STEPS = [
    "the speaker attacks people by race or ethnicity",
    "the speaker attacks people by religion",
    "the speaker attacks people by gender or sexuality",
    "the speaker attacks people by nationality or immigration status",
    "the speaker attacks people by disability",
    "the speaker says nothing hostile",
]
video_data = {
    "video_uid": os.path.splitext(os.path.basename(args.video))[0],
    "videos": segments,
    "task_id": 0,
    "texts": [{"goal": GOAL, "step": s} for s in STEPS],
}

with open(os.path.join(BAGLM, "prompts/vsg/question.txt")) as f:
    question_template = f.read()

scores = score_func.forward_vsg(video_data, 1, question_template=question_template).cpu()
scores = scores.repeat_interleave(args.segment_duration, dim=0)
print(f"[smoke] vsg scores {tuple(scores.shape)} "
      f"(segments x (len(STEPS)+1 'none of the above')) "
      f"range {scores.min().item():.4f}..{scores.max().item():.4f}")
print(f"[smoke] row sums {scores.sum(-1)[:4].tolist()}")
print(f"[smoke] per-second argmax {scores.argmax(-1).tolist()}")
if args.device.startswith("cuda"):
    print(f"[smoke] peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

if args.out:
    np.savez(args.out, vsg=scores.numpy(), rate=1.0, steps=np.array(STEPS + ["none"]))
    print(f"[smoke] wrote {args.out}")
