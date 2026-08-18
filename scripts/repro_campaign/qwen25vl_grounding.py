#!/usr/bin/env python
"""Qwen2.5-VL-7B native temporal grounding harness (zero-shot, no fine-tuning).

Prompt / parsing convention is copied verbatim from the lmms-eval
`temporal_grounding_charades` task (EvolvingLMMs-Lab/lmms-eval,
lmms_eval/tasks/charades_sta/{charades.yaml,utils.py}) -- the harness
TempSamp-R1 (NeurIPS 2025, arXiv 2509.18056) reports its Qwen2.5-VL-7B
zero-shot Charades-STA row under.  The only substitution is the query
sentence: instead of a Charades activity caption we pass the hate-content
query.  Everything else (pre_prompt, post_prompt, greedy decoding,
max_new_tokens=50, the extract_time regex cascade) is unchanged, so the
numbers stay comparable to the published row.

Output: one JSON record per (video, query) with the raw generation and the
parsed [start, end] interval in seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------- prompt ---
# verbatim from lmms_eval/tasks/charades_sta/charades.yaml
PRE_PROMPT = (
    "Please find the visual event described by a sentence in the video, "
    "determining its starting and ending times. The format should be: "
    "'The event happens in the start time - end time'. For example, "
    "The event 'person turn a light on' happens in the 24.3 - 30.4 seonds. "
    "Now I will give you the textual sentence: "
)
POST_PROMPT = "Please return its start time and end time."

DEFAULT_QUERY = "a person expresses hateful content targeting a group of people"


def build_text(query: str) -> str:
    """lmms-eval temporal_grounding_doc_to_text: f'{pre}{question}. {post}'."""
    return f"{PRE_PROMPT}{query}. {POST_PROMPT}"


# ----------------------------------------------------------------- parse ---
# verbatim port of lmms_eval/tasks/charades_sta/utils.py::extract_time
def _time_to_seconds(t: str) -> float:
    parts = [float(p) for p in t.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return parts[0]


def extract_time(paragraph: str):
    prompt = "A specific example is : 20.8 - 30.0 seconds".lower()
    paragraph = paragraph.lower().replace(prompt, "").replace("to", "-")
    sentences = re.split(r"[!?\n]", paragraph)

    keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
    candidates = [s for s in sentences if any(k in s for k in keywords)]
    if not candidates:
        candidates = sentences

    timestamps = []
    time_format_range_pattern = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\s*[–-]\s*(\d{1,2}:\d{2}(?::\d{2})?)\b")
    main_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)")
    time_number_pattern = re.compile(r"\b(\d+(?:\.\d+)?)\b")
    time_format_pattern = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")

    for sentence in candidates:
        m = time_format_range_pattern.findall(sentence)
        if m:
            timestamps = [[_time_to_seconds(a), _time_to_seconds(b)] for a, b in m]
            break
    if not timestamps:
        for sentence in candidates:
            m = main_pattern.findall(sentence)
            if m:
                timestamps = [[float(a), float(b)] for a, b in m]
                break
    if not timestamps:
        for sentence in candidates:
            m = time_format_pattern.findall(sentence)
            if len(m) >= 2:
                timestamps = [[_time_to_seconds(m[0]), _time_to_seconds(m[1])]]
                break
    if not timestamps:
        for sentence in candidates:
            m = time_number_pattern.findall(sentence)
            if len(m) >= 2:
                timestamps = [[float(m[0]), float(m[1])]]
                break
    return timestamps[0] if timestamps else None


# ------------------------------------------------------------------ main ---
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="video paths")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--nframes", type=int, default=32,
                    help="lmms-eval qwen2_5_vl default max_num_frames")
    ap.add_argument("--max-new-tokens", type=int, default=50)  # charades.yaml
    ap.add_argument("--max-pixels", type=int, default=151200,
                    help="per-frame visual token budget; lmms-eval qwen2_5_vl video default")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--load-4bit", action="store_true",
                    help="4-bit NF4 quantisation (use when the GPU is shared)")
    ap.add_argument("--out", default=None, help="output .jsonl (default: stdout only)")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

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
    print(f"[load] {time.time()-t0:.1f}s  "
          f"mem={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", file=sys.stderr)

    fh = open(args.out, "a") if args.out else None
    for vp in args.videos:
        t1 = time.time()
        messages = [{"role": "user", "content": [
            {"type": "video", "video": vp, "nframes": args.nframes,
             "max_pixels": args.max_pixels},
            {"type": "text", "text": build_text(args.query)},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt", **video_kwargs).to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 top_k=None, num_beams=1)
        gen = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        span = extract_time(gen)
        rec = {"video": vp, "query": args.query, "raw": gen.strip(),
               "span": span, "nframes": args.nframes,
               "sec": round(time.time() - t1, 2),
               "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(json.dumps(rec, ensure_ascii=False))
        if fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
