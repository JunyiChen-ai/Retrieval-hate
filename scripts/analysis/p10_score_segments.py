#!/usr/bin/env python
"""P10 — MLLM window scorer with prompt/model variants (GPU).

Built from the frozen P3 scorer (score_segments_mllm.py); the `p6` prompt is
byte-identical to it (reproduces the frozen P6 config). Adds --prompt variants
{p6, fewshot, speech}, --quant {none, bnb4} (for the 32B tier), and works for
Qwen2.5-VL-7B and Qwen2.5-VL-32B. Same output contract
(data/MLLM_scores/<DS>/<outname>_segscoreK<K>_<tag>.jsonl). Labels never read.
"""
import argparse
import json
import os
import re
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from utils.generate_subclip_embedding_HF import (  # noqa: E402
    load_video_frames, _window_bounds)

SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen", "test": "test_seen"}

SYSTEM_P6 = (
    "You are a careful forensic rater assisting a hateful-video detection "
    "system. You are shown ONE short segment of a video (a few frames) and its "
    "spoken transcript for that segment. Your ONLY job is to rate how much "
    "HATE evidence is present IN THIS SEGMENT. Hate evidence = content that "
    "attacks, dehumanises, demeans, threatens or incites against people for a "
    "protected attribute (race, ethnicity, religion, nationality, gender, "
    "sexual orientation, disability, caste, immigration status, etc.), carried "
    "by any channel: spoken words, on-screen text, symbols, gestures, or "
    "imagery. Rate ONLY what is visible/audible in THIS segment; do not guess "
    "about other parts of the video. Neutral, benign, or merely offensive-but-"
    "not-hateful content is NOT hate evidence.")

SYSTEM_SPEECH = (
    "You are a careful forensic rater for a hateful-video detection system. You "
    "are shown ONE short segment of a video: a few frames AND, most importantly, "
    "the exact words SPOKEN in this segment (transcript). Hate in these videos is "
    "usually carried by SPEECH. Rate how much HATE evidence THIS segment contains "
    "— attacks, dehumanisation, slurs, threats or incitement against a protected "
    "group (race, ethnicity, religion, nationality, gender, sexual orientation, "
    "disability, etc.), in the SPOKEN WORDS first, then on-screen text or imagery. "
    "Rate ONLY this segment; neutral or merely-rude content is not hate.")

RUBRIC = (
    "Rating scale (output a single integer):\n"
    "0 = no hate evidence in this segment (neutral / benign / unrelated).\n"
    "1 = faint or ambiguous cue that could relate to hate but is unclear.\n"
    "2 = clear hate-related content targeting a protected group, but implicit "
    "or coded.\n"
    "3 = explicit, unambiguous hate content (slur, dehumanisation, threat, or "
    "incitement targeting a protected group).\n")

FEWSHOT = (
    "Examples (segment transcript -> rating):\n"
    "- \"...and then we went to the beach, it was sunny\" -> 0\n"
    "- \"these people are everywhere now, i don't know\" -> 1\n"
    "- \"they're basically animals, they don't belong in a civilised country\" -> 2\n"
    "- \"[slur] should be rounded up and shot, every last one of them\" -> 3\n")

USER_TAIL = (
    "\nRespond with ONLY the single integer 0, 1, 2, or 3 for this segment. "
    "No words, no explanation.")

NO_SPEECH = "(no speech in this segment)"
PROMPTS = {"p6": (SYSTEM_P6, ""), "fewshot": (SYSTEM_P6, FEWSHOT),
           "speech": (SYSTEM_SPEECH, "")}
_INT_RE = re.compile(r"[0-3]")


def parse_score(raw):
    if raw is None:
        return 0, False
    m = _INT_RE.search(raw)
    return (int(m.group(0)), True) if m else (0, False)


def build_messages(frames, asr_text, prompt):
    sys_p, fewshot = PROMPTS[prompt]
    asr = asr_text.strip() if asr_text and asr_text.strip() else NO_SPEECH
    user = ("Segment transcript (speech in this segment): \"{}\"\n\n{}{}{}".format(
        asr, RUBRIC, ("\n" + fewshot) if fewshot else "", USER_TAIL))
    return [
        {"role": "system", "content": [{"type": "text", "text": sys_p}]},
        {"role": "user", "content": [{"type": "video", "video": frames},
                                     {"type": "text", "text": user}]}]


def read_gt(p):
    return [{"id": str(json.loads(l)["id"])} for l in open(p) if l.strip()]


def load_asr(p, K):
    out = {}
    if not os.path.exists(p):
        print("[WARN] no ASR", p); return out
    for line in open(p):
        line = line.strip()
        if line:
            o = json.loads(line)
            wt = [(t or "").strip() for t in (o.get("window_text") or [])]
            out[str(o["id"])] = (wt + [""] * K)[:K]
    return out


def load_done(p):
    d = set()
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    d.add(str(json.loads(line)["id"]))
                except Exception:  # noqa: BLE001
                    pass
    return d


@torch.no_grad()
def score_window(frames, asr, proc, model, device, prompt, mnt):
    messages = build_messages(frames, asr, prompt)
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=None, videos=[frames], return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=mnt, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(new, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HateMM")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--gt_dir", default="./data/gt_p10hate")
    ap.add_argument("--video_dir", default="./data/video")
    ap.add_argument("--asr_dir", default="./data/ASR")
    ap.add_argument("--asr_tag", default="asrK30_whisper-large-v3")
    ap.add_argument("--out_dir", default="./data/MLLM_scores")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--prompt", default="p6", choices=sorted(PROMPTS))
    ap.add_argument("--quant", default="none", choices=["none", "bnb4"])
    ap.add_argument("--num_frames", type=int, default=120)
    ap.add_argument("--num_subclips", type=int, default=30)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    K, M = args.num_subclips, args.num_frames
    device = torch.device(args.device)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    if args.quant == "bnb4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw.pop("torch_dtype", None)
    print("Loading {} (prompt={}, quant={}, K={}/M={})".format(
        args.model, args.prompt, args.quant, K, M), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kw).eval()
    proc = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    # output tag: p6 -> qwen (P6-compatible); variants get their own tag so caches never collide
    size = "-32b" if "32B" in args.model else ("-72b" if "72B" in args.model else "")
    tag = "qwen" if (args.prompt == "p6" and args.quant == "none"
                     and "7B" in args.model) else "p10-{}{}{}".format(
        args.prompt, size,
        "-" + args.quant if args.quant != "none" else "")
    video_root = os.path.join(args.video_dir, args.dataset, "All")
    out_ds = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_ds, exist_ok=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        outname = SPLIT_TO_OUTNAME[split]
        items = read_gt(os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split)))
        if args.limit:
            items = items[:args.limit]
        asr = load_asr(os.path.join(args.asr_dir, args.dataset,
                                    "{}_{}.jsonl".format(outname, args.asr_tag)), K)
        out_path = os.path.join(out_ds, "{}_segscoreK{}_{}.jsonl".format(outname, K, tag))
        done = load_done(out_path)
        todo = [it for it in items if it["id"] not in done]
        print("[{}] {} videos, {} done, {} todo -> {}".format(
            split, len(items), len(done), len(todo), out_path), flush=True)
        with open(out_path, "a") as fout:
            for vi, it in enumerate(todo):
                vid = it["id"]
                frames, ok = load_video_frames(
                    os.path.join(video_root, "{}.mp4".format(vid)), M)
                wt = asr.get(vid, [""] * K)
                scores, oks = [], []
                if ok:
                    for k, (s, e) in enumerate(_window_bounds(len(frames), K)):
                        if e <= s:
                            e = s + 1
                        raw = score_window(frames[s:e], wt[k] if k < len(wt) else "",
                                           proc, model, device, args.prompt,
                                           args.max_new_tokens)
                        sc, pok = parse_score(raw)
                        scores.append(sc); oks.append(pok)
                else:
                    scores, oks = [0] * K, [False] * K
                fout.write(json.dumps(dict(id=vid, scores=scores, parse_ok=oks,
                                           video_ok=bool(ok)), ensure_ascii=False) + "\n")
                fout.flush()
                if (vi + 1) % 25 == 0:
                    print("  [{}] {}/{}".format(split, vi + 1, len(todo)), flush=True)
        print("[{}] done -> {}".format(split, out_path), flush=True)


if __name__ == "__main__":
    main()
