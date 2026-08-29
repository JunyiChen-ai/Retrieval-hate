#!/usr/bin/env python3
"""Stage T: dense VLM window scoring of TEST hate videos (teacher signal check).

One local Qwen2.5-VL-7B call per 16-s window (stride 8): 4 cached 1-fps frames
plus the window's ASR text, deterministic decoding, integer 0-10 hate-evidence
score. Raw generations are logged; unparseable generations score 0 and are
counted. Frames come from data/frames_1fps; a video with no cached frame dir is
scored from ASR text alone and flagged (`no_frames`).

Usage: python stage_t_teacher.py [--corpus hatemm] [--limit N] [--smoke]
Output: runs/20260830_vlm_order_pilot/teacher_<corpus>.jsonl (one row per video:
window scores + spans), teacher_<corpus>_raw.log
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402

FRAME_DIRS = {"hatemm": "HateMM", "mhclip_en": "MHC", "mhclip_zh": "MHC_zh",
              "hateclipseg": "HateClipSeg"}
ASR = {"hatemm": "hatemm_all", "mhclip_en": "mhclip_en_all",
       "mhclip_zh": "mhclip_zh_all", "hateclipseg": "hateclipseg_all"}
OUT_DIR = os.path.join(REPO, "runs", "20260830_vlm_order_pilot")
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
WIN, STRIDE, FRAMES_PER_WIN = 16, 8, 4

PROMPT = (
    "These images are frames from one {win}-second window of a video, and the "
    "text below is the speech transcribed inside this window. Rate the amount "
    "of hateful content evidence in THIS WINDOW ONLY (attacks, slurs, "
    "dehumanization or incitement against a protected group - race, religion, "
    "nationality, gender, sexual orientation, disability - in speech, imagery "
    "or on-screen text). Ignore what other parts of the video might contain. "
    "Answer with one integer 0 to 10 and nothing else. 0 = no evidence, "
    "10 = explicit strong evidence.\nTranscript: \"{asr}\""
)


def load_asr(corpus):
    table = {}
    path = os.path.join(REPO, "results", "reproduction", "asr", ASR[corpus],
                        "timestamped_chunks.jsonl")
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            table[d["video_id"]] = d.get("chunks") or []
    return table


def window_asr(chunks, a, b, cap=600):
    parts = [c["text"] for c in chunks
             if c.get("start") is not None and c.get("end") is not None
             and c["end"] > a and c["start"] < b]
    return " ".join(p.strip() for p in parts).strip()[:cap]


def window_frames(frame_dir, a, b):
    if not os.path.isdir(frame_dir):
        return []
    ts = np.linspace(a, b - 1, FRAMES_PER_WIN)
    out = []
    for t in ts:
        p = os.path.join(frame_dir, "%06d.jpg" % int(round(t)))
        if os.path.isfile(p):
            out.append(p)
    return sorted(set(out))


def parse_score(text):
    for tok in text.replace(",", " ").split():
        tok = tok.strip(".:;()[]")
        if tok.isdigit():
            v = int(tok)
            if 0 <= v <= 10:
                return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--limit", type=int, default=0, help="max videos per corpus")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    corpora = [args.corpus] if args.corpus else list(FRAME_DIRS)
    os.makedirs(OUT_DIR, exist_ok=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    for corpus in corpora:
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        vids = sorted(v for v in gt if labels.get(v) == 1)
        if args.limit:
            vids = vids[:args.limit]
        if args.smoke:
            vids = vids[:3]
        asr_table = load_asr(corpus)
        out_path = os.path.join(OUT_DIR, "teacher_%s.jsonl" % corpus)
        raw_log = open(os.path.join(OUT_DIR, "teacher_%s_raw.log" % corpus), "a")
        done = set()
        if os.path.exists(out_path):
            with open(out_path) as fh:
                for line in fh:
                    done.add(json.loads(line)["video_id"])
        out = open(out_path, "a")
        n_unparse = 0
        for i, vid in enumerate(vids):
            if vid in done:
                continue
            T = len(gt[vid])
            frame_dir = os.path.join(REPO, "data", "frames_1fps",
                                     FRAME_DIRS[corpus], vid)
            chunks = asr_table.get(vid, [])
            spans, scores = [], []
            for a in range(0, max(1, T), STRIDE):
                b = min(a + WIN, T)
                imgs = window_frames(frame_dir, a, b)
                text = PROMPT.format(win=b - a, asr=window_asr(chunks, a, b))
                content = ([{"type": "image", "image": Image.open(p).convert("RGB")}
                            for p in imgs] + [{"type": "text", "text": text}])
                messages = [{"role": "user", "content": content}]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                images = [c["image"] for c in content if c["type"] == "image"]
                inputs = processor(text=[prompt], images=images or None,
                                   return_tensors="pt").to("cuda")
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=8,
                                         do_sample=False)
                gen = processor.batch_decode(
                    ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True)[0]
                score = parse_score(gen)
                raw_log.write(json.dumps({"video_id": vid, "a": a, "b": b,
                                          "n_imgs": len(imgs), "gen": gen}) + "\n")
                if score is None:
                    n_unparse += 1
                    score = 0
                spans.append([a, b])
                scores.append(score)
                if b >= T:
                    break
            out.write(json.dumps({"video_id": vid, "T": T, "spans": spans,
                                  "scores": scores,
                                  "no_frames": not os.path.isdir(frame_dir)}) + "\n")
            out.flush(); raw_log.flush()
            print("PROGRESS %s %d/%d %s windows=%d" %
                  (corpus, i + 1, len(vids), vid, len(scores)), flush=True)
        out.close(); raw_log.close()
        print("DONE %s unparseable=%d" % (corpus, n_unparse), flush=True)


if __name__ == "__main__":
    main()
