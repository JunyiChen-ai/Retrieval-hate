#!/usr/bin/env python
"""P8b — VISION-GROUNDED evidence summaries (EXP_p8 extra arm).

Qwen2.5-VL-7B reads sampled frames + the video's Title+Transcript and writes the
SAME ≤60-word evidence-dense summary as P8's text-only arm, PLUS an explicit
instruction to transcribe on-screen text verbatim (the channel CLIP's image
encoder cannot read on-screen text, especially Chinese captions/memes). Output
schema is byte-identical to P8 ({id, label, orig_text, summary}) so p1-prior's
caches / probe / training harness consume it unchanged — just a different
summaries dir.

Writes ONLY data/Summaries_vision/<DS>/<split>.jsonl (parallel to P8's
data/Summaries/<DS>/; never clobbers the text-only arm). Resume-safe. Greedy,
fixed prompt, label-free (labels never read for generation) → no leakage.
"""
import argparse
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from utils.generate_subclip_embedding_HF import load_video_frames  # noqa: E402

SPLIT_TO_OUTNAME = {"train": "train", "val": "val", "test": "test"}
MAX_TEXT_CHARS = 4000

# Frozen prompt: P8's condensation prompt + the on-screen-text transcription
# clause (the only P8b-specific addition). Kept as close to P8's wording as
# possible so B(vision) vs B(text-only) isolates the vision grounding.
SYSTEM_PROMPT = (
    "You are shown sampled frames of a short video together with its "
    "Title+Transcript. Your job is to write one evidence-dense summary of the "
    "video for a hateful-content detector."
)
USER_PROMPT = (
    "Condense this short video into at most 60 words, preserving WHO is "
    "targeted, WHAT is said or shown that could be hateful or offensive, and "
    "the overall topic. Transcribe any visible ON-SCREEN TEXT (captions, memes, "
    "overlaid words, signs) verbatim into the summary if it could be hateful or "
    "offensive. Output only the condensed text, no commentary.\n\n"
    "Title+Transcript: {text}"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P8b vision-grounded summaries.")
    ap.add_argument("--dataset", type=str, default="MHC_zh")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--gt_dir", type=str, default="./data/gt")
    ap.add_argument("--video_dir", type=str, default="./data/video")
    ap.add_argument("--out_dir", type=str, default="./data/Summaries_vision")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0, help="If >0, first N videos (smoke).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    return ap.parse_args(argv)


def read_gt(gt_path):
    items = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            items.append({
                "id": str(o["id"]),
                "label": o.get("label"),
                "text": "" if o.get("text") is None else str(o["text"]),
            })
    return items


def build_messages(frames, text):
    t = (text or "").strip()
    if len(t) > MAX_TEXT_CHARS:
        t = t[:MAX_TEXT_CHARS] + " ...[truncated]"
    if not t:
        t = "(none)"
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": USER_PROMPT.format(text=t)},
        ]},
    ]


@torch.no_grad()
def generate_summary(frames, text, processor, model, device, max_new_tokens):
    messages = build_messages(frames, text)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=None, videos=[frames], return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return raw.strip()


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(str(json.loads(line)["id"]))
                except Exception:
                    pass
    return done


def main(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = torch.device(args.device)
    print("Loading Qwen2.5-VL: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    video_root = os.path.join(args.video_dir, args.dataset, "All")
    out_ds = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_ds, exist_ok=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' unknown; skip".format(split))
            continue
        gt_path = os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] gt missing, skip {}: {}".format(split, gt_path))
            continue
        items = read_gt(gt_path)
        if args.limit > 0:
            items = items[: args.limit]
        out_path = os.path.join(out_ds, "{}.jsonl".format(SPLIT_TO_OUTNAME[split]))
        done = load_done_ids(out_path) if args.resume else set()
        print("[{}] {} videos ({} done) -> {}".format(split, len(items), len(done), out_path),
              flush=True)
        n_empty_src = 0
        with open(out_path, "a") as fout:
            for vi, item in enumerate(items):
                vid = item["id"]
                if vid in done:
                    continue
                if not (item["text"] or "").strip():
                    n_empty_src += 1
                vpath = os.path.join(video_root, "{}.mp4".format(vid))
                frames, ok = load_video_frames(vpath, args.num_frames)
                if ok:
                    summary = generate_summary(
                        frames, item["text"], processor, model, device, args.max_new_tokens)
                else:
                    summary = ""  # unreadable video -> empty (logged)
                fout.write(json.dumps({
                    "id": vid, "label": item["label"],
                    "orig_text": item["text"], "summary": summary,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                if (vi + 1) % 25 == 0:
                    print("  [{}] {}/{} (empty-src so far {})".format(
                        split, vi + 1, len(items), n_empty_src), flush=True)
        print("[{}] done -> {} (empty source texts: {})".format(split, out_path, n_empty_src),
              flush=True)


if __name__ == "__main__":
    main(parse_args())
