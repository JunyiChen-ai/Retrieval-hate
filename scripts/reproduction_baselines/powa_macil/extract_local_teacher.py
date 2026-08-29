#!/usr/bin/env python3
"""Budget-gated local primitive teacher pilot (train split only).

This is intentionally a teacher-quality audit, not full extraction.  It uses
one cached local language model call per ASR time chunk and records every raw
generation so parse failures cannot be hidden.  No API client is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          Qwen2VLForConditionalGeneration)

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from hate_common import data as hdata  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(PARENT), "duplex"))
from extract_clip_features import CORPORA as MEDIA_SPECS, find_video_path  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
AXES = ("hostile", "target", "violence", "sexual", "self_harm", "context")
SYSTEM = (
    "You are an evidence annotator, not a content moderator. Score only the "
    "given transcript excerpt. Return exactly one compact JSON object with "
    "keys hostile,target,violence,sexual,self_harm,context and integer values "
    "0 to 4. hostile=attack/insult/abuse; target=mentions an identity group "
    "such as race,religion,nationality,gender,sexual orientation or disability; "
    "context=reporting,quotation,education,condemnation or counterspeech. "
    "Do not infer absent evidence. No explanation."
)
VIDEO_SYSTEM = (
    "You annotate temporal evidence, not the final video label. The user gives "
    "numbered transcript excerpts. Return exactly one compact JSON object: "
    "{\"scores\":[[h,t,v,s,r,c],...]}. Return one six-integer row per excerpt "
    "in the same order, each integer 0..4. h=hostile attack/insult, t=mentions "
    "a protected identity group, v=violence, s=sexual content, r=self-harm, "
    "c=reporting/quotation/education/condemnation/counterspeech. Score only "
    "observable evidence; no explanation and no extra keys."
)
AXIS_QUESTIONS = {
    "hostile": "How strongly does it express an attack, insult, abuse, contempt or dehumanization?",
    "target": "How clearly does it mention or depict a protected identity group (race, ethnicity, nationality, religion, gender, sexual orientation, gender identity or disability)?",
    "violence": "How clearly does it contain violence, a violent threat, or encouragement of violence?",
    "sexual": "How clearly does it contain sexual content?",
    "self_harm": "How clearly does it contain self-harm or encouragement of self-harm?",
    "context": "How clearly is harmful language merely quoted, reported, taught about, condemned, or countered?",
}


def asr_records(corpus):
    path = os.path.join("results", "reproduction", "asr", corpus + "_all",
                        "timestamped_chunks.jsonl")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["video_id"]] = rec
    return out


def select_ids(corpus, records, per_class):
    labels = hdata.load_labels(corpus)
    train = set(hdata.load_split(corpus, "train"))
    chosen = []
    for cls in (0, 1):
        eligible = sorted(v for v in train if labels.get(v) == cls and
                          records.get(v, {}).get("chunks"))
        chosen.extend(eligible if per_class <= 0 else eligible[:per_class])
    return chosen


def midpoint_image(video_path, seconds):
    """Decode one frame through the system ffmpeg (including AV1)."""
    from PIL import Image
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", "%.3f" % seconds,
         "-i", video_path, "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "png", "pipe:1"], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    if proc.returncode or not proc.stdout:
        return None
    import io
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def parse_object(text):
    match = re.search(r"\{[^{}]+\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        values = [float(obj[k]) / 4.0 for k in AXES]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if not all(np.isfinite(values)) or not all(0 <= x <= 1 for x in values):
        return None
    return dict(zip(AXES, values))


def parse_video_object(text, expected):
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        return None
    try:
        rows = json.loads(match.group(0))["scores"]
        if len(rows) != expected:
            return None
        out = []
        for row in rows:
            if len(row) != len(AXES):
                return None
            values = [float(x) / 4.0 for x in row]
            if not all(np.isfinite(values)) or not all(0 <= x <= 1 for x in values):
                return None
            out.append(dict(zip(AXES, values)))
        return out
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def uniform_chunks(chunks, maximum):
    if len(chunks) <= maximum:
        return list(chunks)
    indices = np.linspace(0, len(chunks) - 1, maximum).round().astype(int)
    return [chunks[i] for i in indices]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=list(hdata.CORPORA),
                    choices=list(hdata.CORPORA))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--max-chunks-per-video", type=int, default=2)
    ap.add_argument("--max-input-chars", type=int, default=1600)
    ap.add_argument("--mode", default="chunk",
                    choices=["chunk", "video", "logits", "binary", "vl_binary"])
    ap.add_argument("--max-video-chunks", type=int, default=12)
    ap.add_argument("--out", default="results/reproduction/powa_macil/teacher_audit_qwen05b.jsonl")
    args = ap.parse_args(argv)

    is_vl = "VL" in args.model.upper()
    if is_vl:
        processor = AutoProcessor.from_pretrained(args.model,
                                                  local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map="cuda:0").eval()
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map="cuda:0").eval()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, done = [], set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    rows.append(rec)
                    done.add((rec["corpus"], rec["video_id"]))
    out_handle = open(args.out, "a", encoding="utf-8")
    started = time.time()
    for corpus in args.corpora:
        records = asr_records(corpus)
        labels = hdata.load_labels(corpus)
        for vid in select_ids(corpus, records, args.per_class):
            if (corpus, vid) in done:
                continue
            if args.mode in ("logits", "binary", "vl_binary"):
                chunks = uniform_chunks(records[vid]["chunks"],
                                        args.max_video_chunks)
                prompts, kept, prompt_images = [], [], []
                image = None
                video_path = None
                if args.mode == "vl_binary":
                    if processor is None:
                        raise RuntimeError("vl_binary requires a VL model")
                    video_path = find_video_path(MEDIA_SPECS[corpus]["video_dir"], vid)
                    if not video_path:
                        continue
                for chunk in chunks:
                    excerpt = (chunk.get("text") or "").strip()[:args.max_input_chars]
                    if not excerpt:
                        continue
                    kept.append(chunk)
                    if args.mode == "vl_binary":
                        from PIL import Image
                        midpoint = (float(chunk.get("start") or 0) +
                                    float(chunk.get("end") or 0)) / 2.0
                        image = midpoint_image(video_path, midpoint)
                        if image is None:
                            kept.pop()
                            continue
                        image.thumbnail((224, 224), Image.Resampling.LANCZOS)
                    for axis in AXES:
                        if args.mode in ("binary", "vl_binary"):
                            content = ("Excerpt: " + excerpt + "\nQuestion: " +
                                       AXIS_QUESTIONS[axis] +
                                       "\nIs this evidence present? Answer exactly Yes or No.")
                            instruction = "Score observable evidence only. Output exactly Yes or No."
                        else:
                            content = ("Excerpt: " + excerpt + "\nQuestion: " +
                                       AXIS_QUESTIONS[axis] +
                                       "\nAnswer with exactly one digit: 0=absent, "
                                       "1=weak, 2=possible, 3=clear, 4=explicit.")
                            instruction = "Score observable evidence only. Output one digit 0,1,2,3,or 4."
                        user_content = ([{"type": "image"},
                                         {"type": "text", "text": content}]
                                        if args.mode == "vl_binary" else content)
                        prompts.append(tokenizer.apply_chat_template(
                            [{"role": "system", "content": instruction},
                             {"role": "user", "content": user_content}],
                            tokenize=False, add_generation_prompt=True))
                        if args.mode == "vl_binary":
                            prompt_images.append(image.copy())
                    if image is not None:
                        image.close()
                if not kept:
                    continue
                n_actual_prompts = len(prompts)
                if args.mode == "vl_binary":
                    from PIL import Image
                    null_image = Image.new("RGB", (224, 224), color=(127, 127, 127))
                    for axis in AXES:
                        content = ("Excerpt: [NO CONTENT]\nQuestion: " +
                                   AXIS_QUESTIONS[axis] +
                                   "\nIs this evidence present? Answer exactly Yes or No.")
                        user_content = [{"type": "image"},
                                        {"type": "text", "text": content}]
                        prompts.append(tokenizer.apply_chat_template(
                            [{"role": "system", "content":
                              "Score observable evidence only. Output exactly Yes or No."},
                             {"role": "user", "content": user_content}],
                            tokenize=False, add_generation_prompt=True))
                        prompt_images.append(null_image.copy())
                    null_image.close()
                batch = ((processor(text=prompts,
                                    images=prompt_images if prompt_images else None,
                                    padding=True, return_tensors="pt")
                          if processor is not None else
                          tokenizer(prompts, padding=True, return_tensors="pt"))
                         .to(model.device))
                with torch.inference_mode():
                    logits = model(**batch).logits
                positions = torch.arange(batch.attention_mask.shape[1],
                                         device=model.device)[None]
                last = (positions * batch.attention_mask).max(-1).values
                next_logits = logits[torch.arange(len(prompts), device=model.device), last]
                if args.mode in ("binary", "vl_binary"):
                    answer_ids = []
                    for answer in ("No", "Yes"):
                        ids = tokenizer.encode(answer, add_special_tokens=False)
                        if len(ids) != 1:
                            raise RuntimeError("answer %s is not one token: %s" %
                                               (answer, ids))
                        answer_ids.append(ids[0])
                    pair = next_logits[:, answer_ids].float()
                    if args.mode == "vl_binary":
                        log_odds = pair[:, 1] - pair[:, 0]
                        null = log_odds[n_actual_prompts:].reshape(len(AXES))
                        expected = torch.sigmoid(
                            log_odds[:n_actual_prompts].reshape(-1, len(AXES)) -
                            null[None]).reshape(-1)
                        readout = "null_calibrated_multimodal_binary_probability"
                    else:
                        expected = torch.softmax(pair, -1)[:, 1]
                        readout = "binary_yes_probability"
                else:
                    digit_ids = []
                    for digit in range(5):
                        ids = tokenizer.encode(str(digit), add_special_tokens=False)
                        if len(ids) != 1:
                            raise RuntimeError("digit %d is not one token: %s" % (digit, ids))
                        digit_ids.append(ids[0])
                    prob = torch.softmax(next_logits[:, digit_ids].float(), -1)
                    expected = (prob * torch.arange(5, device=prob.device)).sum(-1) / 4.0
                    readout = "categorical_next_token_expectation"
                expected = expected.cpu().numpy().reshape(len(kept), len(AXES))
                parsed_rows = [dict(zip(AXES, map(float, row))) for row in expected]
                rows.append({"corpus": corpus, "video_id": vid,
                             "video_label": int(labels[vid]),
                             "chunks": [{"start": x.get("start"),
                                         "end": x.get("end"),
                                         "text": (x.get("text") or "")[:args.max_input_chars]}
                                        for x in kept],
                             "primitive_prob": parsed_rows,
                             "readout": readout})
                out_handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                out_handle.flush()
                continue
            if args.mode == "video":
                chunks = uniform_chunks(records[vid]["chunks"],
                                        args.max_video_chunks)
                excerpts = []
                kept = []
                for chunk in chunks:
                    excerpt = (chunk.get("text") or "").strip()[:args.max_input_chars]
                    if excerpt:
                        kept.append(chunk)
                        excerpts.append("[%d | %.2f-%.2fs] %s" %
                                        (len(kept) - 1, float(chunk.get("start") or 0),
                                         float(chunk.get("end") or 0), excerpt))
                if not excerpts:
                    continue
                messages = [{"role": "system", "content": VIDEO_SYSTEM},
                            {"role": "user", "content": "\n".join(excerpts)}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = ((processor(text=[prompt], return_tensors="pt")
                           if processor is not None else
                           tokenizer(prompt, return_tensors="pt")).to(model.device))
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=384,
                                               do_sample=False)
                raw = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True).strip()
                parsed_rows = parse_video_object(raw, len(kept))
                rows.append({"corpus": corpus, "video_id": vid,
                             "video_label": int(labels[vid]),
                             "chunks": [{"start": x.get("start"),
                                         "end": x.get("end"),
                                         "text": (x.get("text") or "")[:args.max_input_chars]}
                                        for x in kept],
                             "primitive_prob": parsed_rows, "raw": raw})
                out_handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                out_handle.flush()
                continue
            chunks = records[vid]["chunks"][:args.max_chunks_per_video]
            for ci, chunk in enumerate(chunks):
                excerpt = (chunk.get("text") or "").strip()[:args.max_input_chars]
                if not excerpt:
                    continue
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": "Excerpt: " + excerpt}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = ((processor(text=[prompt], return_tensors="pt")
                           if processor is not None else
                           tokenizer(prompt, return_tensors="pt")).to(model.device))
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=96,
                                               do_sample=False)
                raw = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:],
                                       skip_special_tokens=True).strip()
                rows.append({"corpus": corpus, "video_id": vid,
                             "video_label": int(labels[vid]),
                             "chunk_index": ci, "start": chunk.get("start"),
                             "end": chunk.get("end"), "text": excerpt,
                             "primitive_prob": parse_object(raw), "raw": raw})
                out_handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                out_handle.flush()
    out_handle.close()
    parsed = [r for r in rows if r["primitive_prob"] is not None]
    summary = {"records": len(rows), "parsed": len(parsed),
               "parse_rate": len(parsed) / max(1, len(rows)),
               "seconds": round(time.time() - started, 2), "api_calls": 0,
               "estimated_api_cost": 0.0}
    print(json.dumps(summary, indent=2))
    return 0 if summary["parse_rate"] >= 0.9 else 2


if __name__ == "__main__":
    sys.exit(main())
