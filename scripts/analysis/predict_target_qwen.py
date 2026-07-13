#!/usr/bin/env python
"""TARC G2 — MLLM target-community prediction (exp-tarc-t0.md §6-G2).

For every requested HateMM video, ask a frozen Qwen2.5-VL-7B-Instruct to name
the single protected-group community the video is about / directed at, on the
CLOSED label set that mirrors the HateMM `target` column:
    Blacks Jews Whites Others LGBTQ Muslims Sexists Asian None
Input = the SAME M=16 uniformly-sampled whole-video frames used by the
sub-clip cache + the video's Whisper transcript (joined K4 windows). One
generation per video (no per-window loop).

GOLD-ANNOTATION ISOLATION (§5): this script NEVER reads target_map.json or any
GT target. It reads only frames + transcript. Scoring against GT happens in a
separate offline step (score_target_pred.py). The output feeds a
`--tarc_target_source mllm_pred` retrain; there is no code path from GT target
into this prediction.

Output (resume-safe, append JSONL; consolidated JSON rebuilt when a full,
non-limit run finishes):
  data/gt/HateMM/target_pred_<tag>.jsonl
    {"id", "primary": code, "raw": str, "parse_ok": bool, "video_ok": bool,
     "asr_used": bool}
  data/gt/HateMM/target_pred_<tag>.json    (deliverable: {stem: {primary, raw}})
"""
import argparse
import json
import os
import re
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
from utils.generate_subclip_embedding_HF import load_video_frames  # noqa: E402

# HateMM target code_dict (MUST match data/gt/HateMM/target_map.json _meta so a
# predicted code is directly comparable to the GT primary; "Sexits" is the
# dataset's own spelling of the gender/sexist class). None -> -1.
CODE = {"Blacks": 0, "Jews": 1, "Whites": 2, "Others": 3,
        "LGBTQ": 4, "Muslims": 5, "Sexits": 6, "Asian": 7}
NONE_CODE = -1

# Prompt label -> dataset code. Prompt uses "Sexists" (clearer) -> code 6.
LABEL_TO_CODE = {
    "blacks": 0, "black": 0,
    "jews": 1, "jew": 1, "jewish": 1,
    "whites": 2, "white": 2,
    "others": 3, "other": 3,
    "lgbtq": 4, "lgbt": 4,
    "muslims": 5, "muslim": 5,
    "sexists": 6, "sexist": 6,
    "asian": 7, "asians": 7,
    "none": NONE_CODE,
}

SPLIT_TO_ASR = {"train": "train", "val": "dev_seen", "test": "test_seen"}

SYSTEM_PROMPT = (
    "You are a careful forensic annotator for a hateful-video research "
    "dataset. Given a short video (a few frames) and its spoken transcript, "
    "you identify the single protected-group community the video most centrally "
    "concerns. If the video attacks or demeans a community, that is the target "
    "community; if the video is benign but is centrally about a protected "
    "group, that same community is the answer. You choose exactly ONE label "
    "from a fixed list and answer with a single JSON object, nothing else."
)

LABEL_MENU = (
    "Choose exactly ONE of these community labels:\n"
    "- Blacks: Black / African / African-American people.\n"
    "- Jews: Jewish people; antisemitic content.\n"
    "- Whites: White / Caucasian people.\n"
    "- Muslims: Muslim / Islamic people.\n"
    "- Asian: Asian people (East/South-East/South Asian).\n"
    "- LGBTQ: gay, lesbian, bisexual, transgender or queer people.\n"
    "- Sexists: content directed at people by sex/gender (e.g. misogyny, "
    "attacks on women), i.e. sexist content.\n"
    "- Others: some other protected group not in the list above.\n"
    "- None: the video is not centrally about any protected-group community.\n"
)

USER_TMPL = (
    "Video transcript (automatic speech recognition; may be noisy or empty):\n"
    '"""{asr}"""\n\n'
    + LABEL_MENU
    + "\nWhich single community does THIS video most centrally concern (as an "
    "attack target if it is hateful, otherwise as its central subject)? "
    'Respond with ONLY this JSON object and nothing else: '
    '{{"community": "<one label from the list>"}}'
)

NO_SPEECH = "(no speech recognised in this video)"
MAX_ASR_CHARS = 3000

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def load_asr(asr_dir, split):
    """id -> whole-video transcript (joined K4 window_text)."""
    outname = SPLIT_TO_ASR[split]
    path = os.path.join(asr_dir, "{}_asrK4_whisper-large-v3.jsonl".format(outname))
    out = {}
    if not os.path.exists(path):
        print("[WARN] no ASR file: {}".format(path), flush=True)
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            wt = [(t or "").strip() for t in (o.get("window_text") or [])]
            out[str(o["id"])] = " ".join([w for w in wt if w]).strip()
    return out


def read_split_ids(gt_dir, split):
    path = os.path.join(gt_dir, "{}.jsonl".format(split))
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
    return ids


def parse_label(raw):
    """Model text -> (code, ok). JSON {"community": ...} first, then a
    whole-word keyword scan over the closed label set. No match -> (-1, False)."""
    if not raw:
        return NONE_CODE, False
    # 1) strict-ish JSON object with a "community" field
    for m in _JSON_RE.finditer(raw):
        try:
            obj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            for k in ("community", "target", "label", "answer"):
                if k in obj and obj[k] is not None:
                    v = str(obj[k]).strip().lower()
                    if v in LABEL_TO_CODE:
                        return LABEL_TO_CODE[v], True
    # 2) whole-word keyword scan (priority order: specific groups before None
    #    and before the catch-all "Others")
    low = raw.lower()
    priority = ["blacks", "black", "jews", "jewish", "jew", "muslims", "muslim",
                "asian", "asians", "lgbtq", "lgbt", "sexists", "sexist",
                "whites", "white", "others", "other", "none"]
    for kw in priority:
        if re.search(r"\b" + re.escape(kw) + r"\b", low):
            return LABEL_TO_CODE[kw], True
    return NONE_CODE, False


def build_messages(frames, asr_text):
    asr = asr_text.strip() if asr_text and asr_text.strip() else NO_SPEECH
    if len(asr) > MAX_ASR_CHARS:
        asr = asr[:MAX_ASR_CHARS] + " ...[truncated]"
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": USER_TMPL.format(asr=asr)},
        ]},
    ]


@torch.no_grad()
def predict_one(frames, asr_text, processor, model, device, max_new_tokens):
    messages = build_messages(frames, asr_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
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
                except Exception:  # noqa: BLE001
                    pass
    return done


def assemble_json(jsonl_path, json_path):
    """Rebuild {stem: {primary, raw}} from the resume JSONL (last record wins)."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[str(o["id"])] = {"primary": int(o["primary"]), "raw": o.get("raw", "")}
    with open(json_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    return len(out)


def main(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = torch.device(args.device)
    print("Loading Qwen2.5-VL: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    gt_dir = os.path.join(args.gt_dir, args.dataset)
    asr_dir = os.path.join(args.asr_dir, args.dataset)
    video_root = os.path.join(args.video_dir, args.dataset, "All")

    jsonl_path = os.path.join(gt_dir, "target_pred_{}.jsonl".format(args.tag))
    json_path = os.path.join(gt_dir, "target_pred_{}.json".format(args.tag))

    if args.assemble_only:
        n = assemble_json(jsonl_path, json_path)
        print("[assemble] {} entries -> {}".format(n, json_path), flush=True)
        return

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    done = load_done_ids(jsonl_path) if args.resume else set()

    n_ok = n_fail = n_novideo = n_done = 0
    with open(jsonl_path, "a") as fout:
        for split in splits:
            asr = load_asr(asr_dir, split)
            ids = read_split_ids(gt_dir, split)
            if args.limit > 0:
                ids = ids[: args.limit]
            print("[{}] {} videos ({} already done)".format(split, len(ids), len(done)), flush=True)
            for vi, vid in enumerate(ids):
                if vid in done:
                    n_done += 1
                    continue
                vpath = os.path.join(video_root, "{}.mp4".format(vid))
                frames, ok = load_video_frames(vpath, args.num_frames)
                atext = asr.get(vid, "")
                if ok:
                    raw = predict_one(frames, atext, processor, model, device, args.max_new_tokens)
                    code, pok = parse_label(raw)
                    if pok:
                        n_ok += 1
                    else:
                        n_fail += 1
                else:
                    raw = "<no-frames>"
                    code, pok = NONE_CODE, False
                    n_novideo += 1
                rec = {"id": vid, "primary": int(code), "raw": raw,
                       "parse_ok": bool(pok), "video_ok": bool(ok),
                       "asr_used": bool(atext and atext.strip())}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(vid)
                if (vi + 1) % 25 == 0:
                    print("  [{}] {}/{} parse_ok={} parse_fail={} novideo={}".format(
                        split, vi + 1, len(ids), n_ok, n_fail, n_novideo), flush=True)

    total_new = n_ok + n_fail + n_novideo
    fail_rate = (n_fail + n_novideo) / total_new if total_new else 0.0
    print("[predict] new={} parse_ok={} parse_fail={} novideo={} (already_done={}) "
          "parse_fail_rate(incl novideo)={:.3f}".format(
              total_new, n_ok, n_fail, n_novideo, n_done, fail_rate), flush=True)

    # Rebuild the consolidated JSON only for a full (non-limit) run.
    if args.limit == 0:
        n = assemble_json(jsonl_path, json_path)
        print("[assemble] {} entries -> {}".format(n, json_path), flush=True)
    else:
        print("[assemble] skipped (smoke/limit run); JSONL only.", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="TARC G2/G3 MLLM target-community prediction.")
    ap.add_argument("--dataset", type=str, default="HateMM",
                    help="HateMM | MHC | MHC_zh (paths data/{gt,ASR,video}/<dataset>/).")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--gt_dir", type=str, default="./data/gt")
    ap.add_argument("--video_dir", type=str, default="./data/video")
    ap.add_argument("--asr_dir", type=str, default="./data/ASR")
    ap.add_argument("--tag", type=str, default="qwen7b")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="If >0, first N videos per split (smoke).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--assemble_only", action="store_true",
                    help="Rebuild the consolidated JSON from the JSONL and exit.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
