"""Frozen-VLM answers on the query tree (README section 2.1). One question per queryable node of
data/vlm_tree/<Corpus>/manifest.jsonl: 4 frames of the node + the node's transcript, answered with a 0-3 level for
each of five harm categories (the top-level categories of the OpenAI moderation taxonomy: hate, harassment,
sexual, violence, self-harm). Same model and frame resolution as data/MLLM_scores (Qwen2.5-VL-7B-Instruct, greedy,
max_pixels 360*420 per frame); vLLM backend.

    python experiments/20260925_query_paradigm/extract_tree_answers.py --corpus hateclipseg --splits test,val,train
Output: data/vlm_tree/<Corpus>/answers_qwen7b_mod5.jsonl, one line per video:
    {"id", "split", "T", "nodes": [[a, b, [hate, harassment, sexual, violence, self_harm] or null, raw], ...]}
Resumes: videos already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg"}
CATEGORIES = ("hate", "harassment", "sexual", "violence", "self-harm")
MAX_PIXELS = 360 * 420
MAX_WORDS = 2000

SYSTEM_PROMPT = (
    "You are a careful content-moderation rater. You are shown a few frames sampled from ONE part of a video "
    "and the speech transcript of that part. Rate the harmful content present IN THIS PART in five categories:\n"
    "1. hate: content that attacks, dehumanises, demeans, threatens or incites against people for a protected "
    "attribute (race, ethnicity, religion, nationality, gender, sexual orientation, disability, caste, "
    "immigration status, etc.).\n"
    "2. harassment: content that insults, mocks, bullies or threatens a person or group, not for a protected "
    "attribute.\n"
    "3. sexual: sexual or sexually explicit content.\n"
    "4. violence: content that depicts, glorifies or threatens violence, injury or gore.\n"
    "5. self-harm: content that depicts or encourages self-harm, suicide or dangerous acts.\n"
    "Harmful content can be carried by any channel: spoken words, on-screen text, symbols, gestures, or imagery. "
    "Rate ONLY what is visible/audible in this part."
)
USER_PROMPT = (
    "Transcript of this part: \"{asr}\"\n\n"
    "Scale for each category: 0 = none; 1 = faint or ambiguous cue; 2 = clear but implicit or coded; "
    "3 = explicit and unambiguous.\n"
    "Answer with exactly five integers separated by spaces, in the order: hate harassment sexual violence "
    "self-harm. No other text."
)
NO_SPEECH = "(no speech in this part)"
_INT = re.compile(r"[0-3]")


def parse(raw):
    d = _INT.findall(raw or "")
    return [int(x) for x in d[:5]] if len(d) >= 5 else None


def smart_size(h, w, factor=28, max_pixels=MAX_PIXELS, min_pixels=56 * 56):
    """Qwen2-VL smart_resize: sides multiples of 28, area within [min_pixels, max_pixels]."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def load_frames(fdir, idx):
    ims = [Image.open(os.path.join(fdir, "%06d.jpg" % i)).convert("RGB") for i in idx]
    h, w = smart_size(ims[0].height, ims[0].width)
    return np.stack([np.asarray(im.resize((w, h), Image.BICUBIC)) for im in ims])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=tuple(CORPUS_DIR))
    ap.add_argument("--splits", default="test,val,train")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--videos-per-batch", type=int, default=24)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes videos with index %% n == i")
    a = ap.parse_args()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    d = CORPUS_DIR[a.corpus]
    base = os.path.join(ROOT, "data", "vlm_tree", d)
    si, sn = (int(x) for x in a.shard.split("/"))
    out_path = os.path.join(base, "answers_qwen7b_mod5%s.jsonl" % ("" if sn == 1 else ".shard%dof%d" % (si, sn)))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    splits = a.splits.split(",")
    todo = [json.loads(l) for l in open(os.path.join(base, "manifest.jsonl"))]
    todo = [r for i, r in enumerate(todo) if i % sn == si]
    todo = sorted([r for r in todo if r["split"] in splits and r["id"] not in done], key=lambda r: splits.index(r["split"]))
    print("%s shard %s: %d videos to do, %d done" % (a.corpus, a.shard, len(todo), len(done)), flush=True)
    if not todo:
        return
    proc = AutoProcessor.from_pretrained(a.model)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "{USER}"}]}]
    template = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert template.count("{USER}") == 1 and "<|video_pad|>" in template
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 0, "video": 1}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    pool = ThreadPoolExecutor(16)
    t0, n_calls = time.time(), 0
    with open(out_path, "a") as fh:
        for s in range(0, len(todo), a.videos_per_batch):
            batch = todo[s:s + a.videos_per_batch]
            jobs = []
            for r in batch:
                fdir = os.path.join(ROOT, "data", "frames_1fps", d, r["id"])
                for (na, nb, idx, text) in r["nodes"]:
                    words = text.split()
                    if len(words) > MAX_WORDS:
                        text = " ".join(words[:MAX_WORDS])
                    jobs.append((r["id"], na, nb, fdir, idx, text.strip() or NO_SPEECH))
            frames = list(pool.map(lambda j: load_frames(j[3], j[4]), jobs))
            inputs = [{"prompt": template.replace("{USER}", USER_PROMPT.format(asr=j[5].replace('"', "'"))),
                       "multi_modal_data": {"video": f}} for j, f in zip(jobs, frames)]
            outs = llm.generate(inputs, sp, use_tqdm=False)
            by_vid = {}
            for j, o in zip(jobs, outs):
                raw = o.outputs[0].text.strip()
                by_vid.setdefault(j[0], []).append([j[1], j[2], parse(raw), raw])
            for r in batch:
                fh.write(json.dumps({"id": r["id"], "split": r["split"], "T": r["T"], "nodes": by_vid[r["id"]]}) + "\n")
            fh.flush()
            n_calls += len(jobs)
            el = time.time() - t0
            print("%d/%d videos, %d calls, %.1f calls/s, parse failures in batch %d" % (
                min(s + a.videos_per_batch, len(todo)), len(todo), n_calls, n_calls / el,
                sum(1 for v in by_vid.values() for n in v if n[2] is None)), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
