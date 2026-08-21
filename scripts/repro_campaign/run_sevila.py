#!/usr/bin/env python
"""REPRO campaign Wave 2 — SeViLA Localizer (aux-temporal-pretrain) on the four corpora.

The published mechanism: the Localizer asks a yes/no question about a single frame
and ranks frames by the "yes" token probability at the first decoding step
(`third_party/SeViLA/lavis/models/blip2_models/blip2_fmr.py::Blip2FMR.generate`,
which is the model/arch the repo's own QVHighlights moment-retrieval config
`lavis/projects/sevila/eval/qvh_eval.yaml` evaluates).  Weights:
`third_party/_ckpt/sevila/sevila_pretrained.pth` (Yui010206/SeViLA @ 419e7281),
whose Localizer Q-Former was pre-trained with temporal supervision on QVHighlights.

Frozen before the run (nothing below is chosen on any metric):

  * the localizer prompt template is the repo's own QVHighlights template from
    `lavis/datasets/datasets/mc_video_vqa_datasets.py` (the `'QVHighlight' in qid`
    branch); the ONLY substitution is the question text.
  * the question texts are the yes/no phrasings of `MAIN_QUERY` and
    `HCS_CLASS_QUERIES` in `scripts/repro_campaign/run_unitime.py`, so every
    grounding method in the campaign answers the same thing.
  * the frame grid is the campaign's frozen 1 fps JPEG grid
    (`data/frames_1fps/<DS>/<vid>/%06d.jpg`, content at t = k seconds), the same
    grid the LAVAD chain uses.  native_rate = 1.0.
  * the read-out is the repo's own `yes_score` (the raw "yes" logit at step 0).
    `*_margin` (yes - no) is written from the same forward pass and reported as a
    separate, clearly-marked variant of ours; it never replaces the base row.

Output, per dataset:
  idea-stage/repro_sevila/curves/<DS>/<vid>.npz
      main            float32 (N,)   raw "yes" logit, one per second   [base]
      main_margin     float32 (N,)   yes - no logit                    [ours]
      c0_normal ... c5_harm          HateClipSeg only, same read-out
      c*_margin                      HateClipSeg only
      rate            float64 scalar = 1.0 (samples per second)
  idea-stage/repro_sevila/raw/failures_<DS>.jsonl

CLI
  python scripts/repro_campaign/run_sevila.py --selftest            # tiny, CPU or GPU
  python scripts/repro_campaign/run_sevila.py --datasets HateMM --limit 2
  python scripts/repro_campaign/run_sevila.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jehc223/Retrieval-hate")
SEVILA = ROOT / "third_party/SeViLA"
sys.path.insert(0, str(SEVILA))
sys.path.insert(0, str(ROOT / "scripts/repro_campaign"))

CKPT = ROOT / "third_party/_ckpt/sevila/sevila_pretrained.pth"
RUN_DIR = ROOT / "idea-stage/repro_sevila"
CURVE_DIR = RUN_DIR / "curves"
RAW_DIR = RUN_DIR / "raw"

DATASETS = ["HateMM", "MHC", "MHC_zh", "HateClipSeg"]

# --------------------------------------------------------------- frozen text ---
# repo template, `mc_video_vqa_datasets.py` lines 78-81 (the QVHighlight branch):
#   loc_prompt = 'Question: ' + q + ' ' + pseudo_options + ' ' + LOC_SUFFIX
PSEUDO_OPTIONS = "Option A: yes. Option B: no."
LOC_SUFFIX = ("Does the information within the frame provide the necessary "
              "details to accurately answer the given question?")

# yes/no phrasing of run_unitime.py::MAIN_QUERY
MAIN_QUESTION = "Does this frame contain hateful content targeting a person or group?"
# yes/no phrasings of run_unitime.py::HCS_CLASS_QUERIES
HCS_QUESTIONS = {
    "c0_normal": "Does this frame contain normal, non-toxic content?",
    "c1_hateful": "Does this frame contain hateful content?",
    "c2_insulting": "Does this frame contain insulting content?",
    "c3_sexual": "Does this frame contain sexual content?",
    "c4_violence": "Does this frame contain violent content?",
    "c5_harm": "Does this frame contain harmful content?",
}


def loc_input(question: str) -> str:
    return f"Question: {question} {PSEUDO_OPTIONS} {LOC_SUFFIX}"


NATIVE_RATE = 1.0  # one sample per second, the campaign's frozen 1 fps grid

# SeViLA model config, verbatim from `app.py`
IMG_SIZE = 224
NUM_QUERY_TOKEN = 32
T5_MODEL = "google/flan-t5-xl"
VIT_PRECISION = "fp16"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


# ------------------------------------------------------------------- frames ---
def frame_paths(ds: str, vid: str, duration: float):
    """The campaign's frozen 1 fps JPEGs; extracted here if this video has none."""
    from blip2_caption import FRAME_ROOT, extract_1fps, find_video

    d = FRAME_ROOT / ds / vid
    if d.exists():
        f = sorted(d.glob("*.jpg"))
        if f:
            return f
    p = find_video(ds, vid)
    if p is None:
        return []
    return extract_1fps(p, d, min(int(duration) + 2, 6000))


def load_clip(files, transform):
    """JPEG -> the repo's own eval transform.  Returns (T, C, 224, 224) float."""
    import torch
    from PIL import Image

    arr = np.empty((len(files), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr[i] = np.asarray(im)
    clip = torch.from_numpy(arr).permute(3, 0, 1, 2)          # C, T, H, W  (uint8)
    clip = transform(clip)                                    # C, T, H, W  (norm)
    return clip.permute(1, 0, 2, 3).contiguous()              # T, C, H, W


# -------------------------------------------------------------------- model ---
def build_model(device: str):
    import torch
    from lavis.models.blip2_models.blip2_fmr import Blip2FMR

    m = Blip2FMR(img_size=IMG_SIZE, drop_path_rate=0, use_grad_checkpoint=False,
                 vit_precision="fp32" if device == "cpu" else VIT_PRECISION,
                 freeze_vit=True, num_query_token=NUM_QUERY_TOKEN, t5_model=T5_MODEL,
                 prompt="", max_txt_len=77, apply_lemmatizer=False,
                 frame_num=4, answer_num=5, task="freeze_loc")
    msg = m.load_checkpoint(url_or_filename=str(CKPT))
    m = m.eval()
    if device == "cpu":
        m = m.float()
    else:
        m = m.to(device)
    return m


def build_transform():
    """The repo's own eval transform, `blip_video_eval` with image_size 224 --
    the processor `lavis/projects/sevila/eval/qvh_eval.yaml` names for the
    Localizer's QVHighlights evaluation."""
    from lavis.processors.blip_processors import BlipVideoEvalProcessor

    return BlipVideoEvalProcessor(image_size=IMG_SIZE, mean=MEAN, std=STD).transform


# ------------------------------------------------------------------ scoring ---
# The published `Blip2FMR.generate` recomputes the whole visual tower for every
# prompt.  HateClipSeg needs seven prompts per frame, so the body is split here at
# the Q-Former boundary: `vision_feats` is the first half of that method copied
# line for line, `prompt_logits` the second half.  `--selftest` checks this split
# against the unmodified `generate` and prints the max absolute difference.
def vision_feats(model, frames):
    """frames: (t, c, h, w) on device.  Returns inputs_t5 (t, 32, d)."""
    import torch

    with torch.cuda.amp.autocast(enabled=(frames.device.type == "cuda")):
        image_embeds = model.ln_vision_loc(model.visual_encoder(frames))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long,
                            device=frames.device)
    query_tokens = model.query_tokens_loc.expand(image_embeds.shape[0], -1, -1)
    query_output = model.Qformer_loc.bert(
        query_embeds=query_tokens, encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts, return_dict=True)
    return model.t5_proj_loc(query_output.last_hidden_state)


def prompt_logits(model, inputs_t5, text_input, max_length=1):
    """Returns (t, 2) = [no_logit, yes_logit] at the first decoding step."""
    import torch

    device = inputs_t5.device
    t = inputs_t5.shape[0]
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
        frame_prefix = model.t5_tokenizer(
            model.frame_prefix, padding="longest", add_special_tokens=False,
            truncation=True, max_length=model.max_txt_len,
            return_tensors="pt").to(device)
        frame_prefix_id = torch.repeat_interleave(frame_prefix.input_ids, t, 0)
        frame_prefix_mask = torch.repeat_interleave(frame_prefix.attention_mask, t, 0)
        input_tokens = model.t5_tokenizer(
            [text_input], padding="longest", truncation=True,
            max_length=model.max_txt_len, return_tensors="pt").to(device)
        input_ids = torch.repeat_interleave(input_tokens.input_ids, t, 0)
        input_attention_mask = torch.repeat_interleave(input_tokens.attention_mask, t, 0)

        frame_predix_embed = model.t5_model.encoder.embed_tokens(frame_prefix_id)
        inputs_embeds = model.t5_model.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([frame_predix_embed, inputs_t5, inputs_embeds], dim=1)
        encoder_atts = torch.cat([frame_prefix_mask, atts_t5, input_attention_mask], dim=1)

        outputs = model.t5_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=encoder_atts,
            do_sample=False, top_p=0.9, temperature=1, num_beams=1,
            max_new_tokens=max_length, min_length=1, repetition_penalty=1.0,
            length_penalty=1.0, num_return_sequences=1,
            return_dict_in_generate=True, output_hidden_states=True,
            output_scores=True)
        pred = outputs.scores[0][:, [model.no_id, model.yes_id]]
    return pred.float()


def score_video(model, clip, questions, chunk: int, device):
    """clip: (T, C, H, W) on cpu.  Returns {key: (T,) yes logit, key+_margin}."""
    import torch

    out = {k: [] for k in questions}
    for i in range(0, clip.shape[0], chunk):
        # float32 frames, exactly as the repo's own eval feeds `samples["video"]`;
        # the fp16 vision tower is reached through its own autocast block.
        fr = clip[i:i + chunk].to(device, non_blocking=True)
        feats = vision_feats(model, fr)
        for k, q in questions.items():
            p = prompt_logits(model, feats, loc_input(q))
            out[k].append(p.cpu().numpy())
        del feats, fr
    res = {}
    for k in questions:
        a = np.concatenate(out[k], axis=0).astype(np.float32)
        res[k] = a[:, 1]
        res[k + "_margin"] = a[:, 1] - a[:, 0]
    return res


# ------------------------------------------------------------------- driver ---
def dataset_rows(ds: str, split: str):
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    rows = []
    for i, v in enumerate(z["video_ids"]):
        if split != "all" and str(z["split"][i]) != split:
            continue
        rows.append((str(v), float(z["duration"][i])))
    return sorted(rows)


def selftest(device, chunk):
    """Shapes, score range, and the split-vs-published-generate equality check.

    Freeze §10 red line 3: this prints tensor shapes, score ranges and a numerical
    agreement, and never a metric.
    """
    import torch

    ds, split = "HateClipSeg", "test"
    rows = dataset_rows(ds, split)[:1]
    vid, dur = rows[0]
    files = frame_paths(ds, vid, dur)[:8]
    print(f"[selftest] video={vid} duration={dur:.1f}s frames_used={len(files)}")
    tf = build_transform()
    clip = load_clip(files, tf)
    print(f"[selftest] clip {tuple(clip.shape)} dtype={clip.dtype} "
          f"min={clip.min():.3f} max={clip.max():.3f}")
    model = build_model(device)
    dev = torch.device(device)
    with torch.no_grad():
        qs = {"main": MAIN_QUESTION}
        r = score_video(model, clip, qs, chunk, dev)
        print(f"[selftest] loc_input = {loc_input(MAIN_QUESTION)!r}")
        for k, v in r.items():
            print(f"[selftest] {k}: shape={v.shape} min={v.min():.4f} "
                  f"max={v.max():.4f} mean={v.mean():.4f}")
        # published path, unmodified
        fr = clip.to(dev)
        samples = dict(video=fr.unsqueeze(0), question_id=[vid],
                       loc_input=[loc_input(MAIN_QUESTION)],
                       qa_output=["_".join(["no"] * len(files))])
        pub = model.generate(samples, max_length=30)
        pub_yes = np.asarray(pub["yes_score"], dtype=np.float32)
        d = np.abs(pub_yes - r["main"]).max()
        print(f"[selftest] published generate yes_score n={len(pub_yes)} "
              f"max|diff vs split path| = {d:.6f}")
    return 0


class Crash:
    """freeze §12 D3: an id is retired only after it takes the process down twice.

    A caught SIGTERM/SIGINT clears the in-flight marker and exits without retiring.
    """

    def __init__(self, path: Path):
        self.marker = path
        self.counts = path.with_suffix(".crashcount.json")
        self.killed = False

    def load_counts(self):
        if self.counts.exists():
            return json.loads(self.counts.read_text())
        return {}

    def pending(self):
        if not self.marker.exists():
            return None
        return self.marker.read_text().strip() or None

    def bump(self, key):
        c = self.load_counts()
        c[key] = c.get(key, 0) + 1
        self.counts.write_text(json.dumps(c))
        return c[key]

    def enter(self, key):
        self.marker.write_text(key)

    def clear(self):
        if self.marker.exists():
            self.marker.unlink()

    def install(self):
        def handler(signum, frame):
            self.killed = True
            self.clear()
            print(f"[signal] {signum} -> marker cleared, exiting without retiring",
                  flush=True)
            sys.exit(143)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-frac", type=float, default=0.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--timing-videos", type=int, default=0,
                    help="stop after N videos and print the measured throughput")
    ap.add_argument("--out-dir", default=None,
                    help="override the curve root (smoke runs write elsewhere so a "
                         "CPU-precision curve never lands in the corpus)")
    args = ap.parse_args()

    global CURVE_DIR
    if args.out_dir:
        CURVE_DIR = Path(args.out_dir)

    import torch

    if args.mem_frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        return selftest(device, args.chunk)

    crash = Crash(RUN_DIR / "inflight.marker")
    crash.install()
    retired = set()
    pend = crash.pending()
    if pend:
        n = crash.bump(pend)
        print(f"[crash] in-flight marker '{pend}' -> crash count {n}", flush=True)
        if n >= 2:
            retired.add(pend)
            print(f"[crash] retiring {pend} after two process deaths", flush=True)
        crash.clear()

    plan = []
    for ds in args.datasets.split(","):
        (CURVE_DIR / ds).mkdir(parents=True, exist_ok=True)
        rows = dataset_rows(ds, args.split)
        if args.limit:
            rows = rows[:args.limit]
        for vid, dur in rows:
            if (CURVE_DIR / ds / f"{vid}.npz").exists():
                continue
            if f"{ds}/{vid}" in retired:
                continue
            plan.append((ds, vid, dur))
    print(f"[plan] videos={len(plan)} split={args.split} rate={NATIVE_RATE} "
          f"chunk={args.chunk} device={device}", flush=True)
    if not plan:
        return 0

    model = build_model(device)
    tf = build_transform()
    dev = torch.device(device)
    t0 = time.time()
    ok = err = nframe = 0
    fails = {}
    for i, (ds, vid, dur) in enumerate(plan, 1):
        key = f"{ds}/{vid}"
        if key in retired:
            continue
        crash.enter(key)
        try:
            files = frame_paths(ds, vid, dur)
            if not files:
                raise RuntimeError("no_frames")
            clip = load_clip(files, tf)
            qs = {"main": MAIN_QUESTION}
            if ds == "HateClipSeg":
                qs.update(HCS_QUESTIONS)
            # An unattended run must not lose a video to a transient allocator
            # failure: halve the chunk and retry before recording a failure.
            ch = args.chunk
            while True:
                try:
                    with torch.no_grad():
                        res = score_video(model, clip, qs, ch, dev)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if ch <= 1:
                        raise
                    ch = max(1, ch // 2)
                    print(f"[oom] {key}: retrying at chunk={ch}", flush=True)
            res["rate"] = np.float64(NATIVE_RATE)
            p = CURVE_DIR / ds / f"{vid}.npz"
            tmp = p.with_name(f"{vid}.tmp.npz")  # np.savez appends .npz otherwise
            np.savez(tmp, **res)
            os.replace(tmp, p)
            ok += 1
            nframe += len(files)
        except Exception as e:  # noqa: BLE001
            err += 1
            fails.setdefault(ds, []).append(dict(video_id=vid, error=f"{type(e).__name__}: {e}"))
            with open(RAW_DIR / f"failures_{ds}.jsonl", "a") as fh:
                fh.write(json.dumps(dict(video_id=vid, dataset=ds,
                                         error=f"{type(e).__name__}: {e}")) + "\n")
            print(f"[fail] {key}: {type(e).__name__}: {e}", flush=True)
        crash.clear()
        if i % 5 == 0 or i == len(plan):
            el = time.time() - t0
            peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else 0.0
            print(f"PROGRESS {i}/{len(plan)} ds={ds} ok={ok} err={err} "
                  f"frames={nframe} elapsed={el:.0f}s "
                  f"rate={nframe / max(el, 1e-9):.2f}frame/s "
                  f"eta={el / i * (len(plan) - i):.0f}s peak={peak:.1f}GiB", flush=True)
        if args.timing_videos and i >= args.timing_videos:
            el = time.time() - t0
            print(f"TIMING videos={i} frames={nframe} elapsed={el:.1f}s "
                  f"rate={nframe / max(el, 1e-9):.2f}frame/s", flush=True)
            break
    el = time.time() - t0
    print(f"[done] ok={ok} err={err} frames={nframe} elapsed={el:.0f}s", flush=True)
    write_meta(args, device, ok, err, nframe, el)
    return 0


def write_meta(args, device, ok, err, nframe, elapsed):
    """freeze §6: every run records its provenance."""
    import subprocess

    import torch

    def sh(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True,
                                  cwd=str(ROOT)).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    meta = dict(
        method="SeViLA Localizer", wave=2, supervision="aux-temporal-pretrain",
        repo="Yui010206/SeViLA", repo_commit=sh("git", "-C", str(SEVILA),
                                                "rev-parse", "HEAD"),
        checkpoint=str(CKPT), model_class="lavis.models.blip2_models.blip2_fmr.Blip2FMR",
        venv=str(ROOT / "third_party/_venv/sevila"),
        campaign_commit=sh("git", "rev-parse", "HEAD"),
        torch=torch.__version__,
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        device=device, seed=None, deterministic=True,
        native_rate=NATIVE_RATE, frame_source="data/frames_1fps (ffmpeg fps=1)",
        chunk=args.chunk, split=args.split, datasets=args.datasets,
        loc_template=f"Question: <Q> {PSEUDO_OPTIONS} {LOC_SUFFIX}",
        question_main=MAIN_QUESTION, questions_hcs=HCS_QUESTIONS,
        readout="scores[0][:, yes_id] (repo yes_score); *_margin = yes - no (ours)",
        n_ok=ok, n_err=err, n_frames=nframe, wall_clock_s=round(elapsed, 1),
        finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    p = RUN_DIR / f"run_meta_{args.split}.json"
    old = json.loads(p.read_text()) if p.exists() else None
    if old:
        meta["n_ok"] += old.get("n_ok", 0)
        meta["n_frames"] += old.get("n_frames", 0)
        meta["wall_clock_s"] = round(meta["wall_clock_s"] + old.get("wall_clock_s", 0), 1)
        meta["resumed_from"] = old.get("finished")
    p.write_text(json.dumps(meta, indent=1))
    print(f"[meta] {p}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
