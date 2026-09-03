#!/usr/bin/env python
"""P3 — MLLM evidence-density segment scoring (EXP_p3_evidence_pooling §0.1).

For every video in the requested splits, decode the SAME M=16 uniformly-sampled
frames used by generate_subclip_embedding_HF.py, split into K=4 contiguous
windows, and ask a frozen Qwen2.5-VL-7B-Instruct to rate each window's
hate-EVIDENCE DENSITY on an integer 0..3 scale. Each window is scored IN
ISOLATION (only its <=4 frames + its Whisper-ASR transcript), so the score is
the density of hate evidence WITHIN that segment.

This is unsupervised input processing: labels are never read, so scoring
train+val+test carries no leakage (same status as CLIP/ASR/archive extraction).

Output (one JSONL per split, resume-safe, never overwrites a cache):
  data/MLLM_scores/<DS>/<outname>_segscoreK4_qwen.jsonl
  {"id", "duration", "scores": [s0..s3], "raw": [str..],
   "asr_used": [bool..], "n_frames": [int..], "parse_ok": [bool..]}

NO GPU cache is mutated. The pooling builder (build_pool_cache.py) consumes the
scores and the existing *_subclipK4_* cache to emit the weighted-pool caches.
"""
import argparse
import json
import os
import re
import sys

import torch

# Reuse the EXACT frame sampler + window bounds of the sub-clip cache so the
# scored frames are the pooled frames.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from utils.generate_subclip_embedding_HF import (  # noqa: E402
    load_video_frames,
    _window_bounds,
)

SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen", "test": "test_seen"}

SYSTEM_PROMPT = (
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
    "not-hateful content is NOT hate evidence."
)

RUBRIC = (
    "Rating scale (output a single integer):\n"
    "0 = no hate evidence in this segment (neutral / benign / unrelated).\n"
    "1 = faint or ambiguous cue that could relate to hate but is unclear.\n"
    "2 = clear hate-related content targeting a protected group, but implicit "
    "or coded.\n"
    "3 = explicit, unambiguous hate content (slur, dehumanisation, threat, or "
    "incitement targeting a protected group).\n"
)

USER_PROMPT = (
    "Segment transcript (speech in this segment): \"{asr}\"\n\n"
    + RUBRIC
    + "\nRespond with ONLY the single integer 0, 1, 2, or 3 for this segment. "
    "No words, no explanation."
)

NO_SPEECH = "(no speech in this segment)"

# --- context-conditioned elicitation (module 1 proposal, 2026-09-03) ---------
# The segment is still rated on its own frames + transcript, but the prompt also
# carries the transcript of the enclosing coarse block (K4 window) so that the
# rater can resolve who is being talked about and whether a coded or ambiguous
# cue in this segment is actually hateful in its immediate context.
CONTEXT_PROMPT = (
    "Surrounding context (speech in the wider part of the video that contains "
    "this segment; for reference only, do NOT rate it): \"{ctx}\"\n\n"
    "Segment transcript (speech in this segment): \"{asr}\"\n\n"
    + RUBRIC
    + "\nRate ONLY the hate evidence present in THIS segment (its frames and its "
    "own transcript). Use the surrounding context only to interpret ambiguous or "
    "coded cues in this segment: a cue that the context shows to be benign is "
    "NOT hate evidence; a cue that the context shows to target a protected group "
    "IS. Respond with ONLY the single integer 0, 1, 2, or 3 for this segment. "
    "No words, no explanation."
)
NO_CONTEXT = "(no speech in the surrounding part)"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MLLM evidence-density segment scoring (P3).")
    ap.add_argument("--dataset", type=str, default="MHC")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--gt_dir", type=str, default="./data/gt")
    ap.add_argument("--video_dir", type=str, default="./data/video")
    ap.add_argument("--asr_dir", type=str, default="./data/ASR")
    ap.add_argument("--asr_tag", type=str, default="asrK4_whisper-large-v3")
    ap.add_argument("--out_dir", type=str, default="./data/MLLM_scores")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="If >0, first N videos per split (smoke).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--context", type=str, default="none", choices=("none", "block_asr"),
                    help="block_asr: add the enclosing coarse block's transcript as context")
    ap.add_argument("--context_subclips", type=int, default=4,
                    help="number of coarse blocks whose ASR is used as context (K_c)")
    ap.add_argument("--context_asr_tag", type=str, default="asrK4_whisper-large-v3")
    ap.add_argument("--out_tag", type=str, default="qwen",
                    help="output file suffix tag (use e.g. qwenctx for context runs)")
    return ap.parse_args(argv)


def read_gt(gt_path):
    items = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            items.append({"id": str(o["id"]), "label": o.get("label")})
    return items


def load_asr_windows(asr_path, K):
    """id -> list[str] of K per-window transcripts (empty string when absent)."""
    out = {}
    if not os.path.exists(asr_path):
        print("[WARN] no ASR file: {} (all windows -> no-speech)".format(asr_path))
        return out
    with open(asr_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            wt = o.get("window_text") or []
            wt = [(t or "").strip() for t in wt]
            if len(wt) < K:
                wt = wt + [""] * (K - len(wt))
            out[str(o["id"])] = wt[:K]
    return out


def build_messages(frames, asr_text, context_text=None):
    asr = asr_text.strip() if asr_text and asr_text.strip() else NO_SPEECH
    if context_text is None:
        text = USER_PROMPT.format(asr=asr)
    else:
        ctx = context_text.strip() if context_text and context_text.strip() else NO_CONTEXT
        text = CONTEXT_PROMPT.format(ctx=ctx, asr=asr)
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": text},
            ],
        },
    ]


_INT_RE = re.compile(r"[0-3]")


def parse_score(raw):
    """First digit in {0,1,2,3} -> (score, ok). No digit -> (0, False)."""
    if raw is None:
        return 0, False
    m = _INT_RE.search(raw)
    if m is None:
        return 0, False
    return int(m.group(0)), True


@torch.no_grad()
def score_window(frames, asr_text, processor, model, device, max_new_tokens,
                 context_text=None):
    messages = build_messages(frames, asr_text, context_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return raw


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

    K = args.num_subclips
    M = args.num_frames
    device = torch.device(args.device)
    print("Loading Qwen2.5-VL: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None
    )
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    video_root = os.path.join(args.video_dir, args.dataset, "All")
    out_ds = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_ds, exist_ok=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' unknown; skip".format(split))
            continue
        outname = SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] gt missing, skip {}: {}".format(split, gt_path))
            continue
        items = read_gt(gt_path)
        if args.limit > 0:
            items = items[: args.limit]
        asr_path = os.path.join(
            args.asr_dir, args.dataset, "{}_{}.jsonl".format(outname, args.asr_tag))
        asr = load_asr_windows(asr_path, K)
        ctx_asr, Kc = {}, args.context_subclips
        if args.context == "block_asr":
            ctx_path = os.path.join(
                args.asr_dir, args.dataset, "{}_{}.jsonl".format(outname, args.context_asr_tag))
            if not os.path.exists(ctx_path):
                raise SystemExit("context ASR file missing: {}".format(ctx_path))
            ctx_asr = load_asr_windows(ctx_path, Kc)
            if not ctx_asr:
                raise SystemExit("context ASR file empty: {}".format(ctx_path))
            with open(ctx_path) as fh:
                for line in fh:
                    if line.strip():
                        n_ctx = len(json.loads(line).get("window_text") or [])
                        assert n_ctx == Kc, "context ASR has {} windows, expected {}".format(n_ctx, Kc)
            print("[{}] context = block ASR ({} ids, K_c={})".format(split, len(ctx_asr), Kc), flush=True)

        out_path = os.path.join(out_ds, "{}_segscoreK{}_{}.jsonl".format(outname, K, args.out_tag))
        done = load_done_ids(out_path) if args.resume else set()
        print("[{}] {} videos ({} already done), ASR windows for {} ids -> {}".format(
            split, len(items), len(done), len(asr), out_path), flush=True)

        n_win_txt = 0
        with open(out_path, "a") as fout:
            for vi, item in enumerate(items):
                vid = item["id"]
                if vid in done:
                    continue
                vpath = os.path.join(video_root, "{}.mp4".format(vid))
                frames, ok = load_video_frames(vpath, M)
                wtext = asr.get(vid, [""] * K)
                ctext = ctx_asr.get(vid, [""] * Kc) if args.context == "block_asr" else None
                scores, raws, used, nfr, oks = [], [], [], [], []
                if ok:
                    bounds = _window_bounds(len(frames), K)
                    for k, (s, e) in enumerate(bounds):
                        if e <= s:
                            e = s + 1
                        win_frames = frames[s:e]
                        atext = wtext[k] if k < len(wtext) else ""
                        ctxt = None
                        if ctext is not None:
                            # block index of window k (start-of-window rule, identical to src/verdict_hmm._block_map)
                            ctxt = ctext[min((k * Kc) // K, Kc - 1)]
                        raw = score_window(
                            win_frames, atext, processor, model, device, args.max_new_tokens,
                            context_text=ctxt)
                        sc, pok = parse_score(raw)
                        scores.append(sc)
                        raws.append(raw)
                        used.append(bool(atext and atext.strip()))
                        nfr.append(len(win_frames))
                        oks.append(pok)
                        if atext and atext.strip():
                            n_win_txt += 1
                else:
                    # Unreadable video -> zero scores (matches the sub-clip
                    # zero-vector guard; uniform weights -> mean == zero vec).
                    scores = [0] * K
                    raws = ["<no-frames>"] * K
                    used = [False] * K
                    nfr = [0] * K
                    oks = [False] * K
                rec = {
                    "id": vid,
                    "scores": scores,
                    "raw": raws,
                    "asr_used": used,
                    "n_frames": nfr,
                    "parse_ok": oks,
                    "video_ok": bool(ok),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if (vi + 1) % 25 == 0:
                    print("  [{}] {}/{} (win-with-text so far {})".format(
                        split, vi + 1, len(items), n_win_txt), flush=True)
        print("[{}] done -> {}".format(split, out_path), flush=True)


if __name__ == "__main__":
    main(parse_args())
