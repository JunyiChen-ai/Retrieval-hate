#!/usr/bin/env python
"""Build LLaMA-Factory multi-image ShareGPT data for hateful-video LoRA-SFT.

For a given --dataset (MHC | MHC_zh):
  * read data/gt/<DS>/{train,val}.jsonl (fields: id, text, label; label 1=hateful)
  * sample 8 frames per video via load_video_frames() and save JPGs to
      data/lora_frames/<DS>/<id>/frame_{0..7}.jpg   (idempotent; skip re-extract)
  * emit data/lora_sft/<DS>/{train,val}.json in the multi-image ShareGPT schema
    that Ver202512 dataset_info uses:
        {
          "messages": [
            {"role": "user",      "content": "<image>...*8 + instruction"},
            {"role": "assistant", "content": "<label word>"}
          ],
          "images": ["<abs frame path>", ...8]
        }
  * register mhc_lora_train/mhc_lora_val (or mhc_zh_...) in dataset_info.json.

Videos missing/unreadable are SKIPPED and counted.
"""
import argparse
import json
import os
import sys

from PIL import Image

# Reuse the project's verified frame sampler (decord + PyAV fallback).
_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_THIS)  # .../RGCL/src
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from utils.generate_VideoMLLM_embedding_HF import load_video_frames  # noqa: E402

RGCL_ROOT = "/data/jehc223/RGCL"
LF_ROOT = os.path.join(RGCL_ROOT, "RA-HMD", "LLAMA-FACTORY-Ver202512")
DATASET_INFO = os.path.join(LF_ROOT, "data", "dataset_info.json")

NUM_FRAMES = 8
MAX_TRANSCRIPT_CHARS = 1500

# id prefix per dataset for dataset_info registration.
DS_REGISTER = {
    "MHC": ("mhc_lora_train", "mhc_lora_val"),
    "MHC_zh": ("mhc_zh_lora_train", "mhc_zh_lora_val"),
}

IMG_TOKENS = "<image>" * NUM_FRAMES

INSTR_EN = (
    "Analyze this video using the 8 frames and the transcript below. Decide whether "
    "it is hateful/harmful (attacks or demeans people based on identity, or promotes "
    "harm) or normal. Consider visuals, on-screen text, gestures, symbols, and speech. "
    "Transcript: {text}. Answer with exactly one word: hateful or normal."
)
LABEL_EN = {1: "hateful", 0: "normal"}

INSTR_ZH = (
    "请结合这8帧画面和下面的文字记录分析这段视频。判断它是仇恨/有害内容"
    "（基于身份攻击或贬低他人，或宣扬伤害）还是正常内容。请综合考虑画面、"
    "屏幕文字、手势、符号以及语音。文字记录：{text}。请只用一个词回答：仇恨 或 正常。"
)
LABEL_ZH = {1: "仇恨", 0: "正常"}


def read_gt(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "id": str(obj["id"]),
                    "text": "" if obj.get("text") is None else str(obj["text"]),
                    "label": int(obj["label"]),
                }
            )
    return items


def ensure_frames(video_path, frame_dir):
    """Extract & save 8 JPGs if not already present. Returns list of abs paths or None."""
    paths = [os.path.join(frame_dir, "frame_{}.jpg".format(i)) for i in range(NUM_FRAMES)]
    if all(os.path.exists(p) for p in paths):
        return paths
    frames, ok = load_video_frames(video_path, NUM_FRAMES)
    if not ok:
        return None
    if len(frames) < NUM_FRAMES:
        # pad by repeating last frame so we always have exactly 8
        frames = list(frames) + [frames[-1]] * (NUM_FRAMES - len(frames))
    os.makedirs(frame_dir, exist_ok=True)
    for i, p in enumerate(paths):
        img = frames[i]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img.convert("RGB").save(p, format="JPEG", quality=90)
    return paths


def build_split(items, ds, video_root, frames_root, instr_tmpl, label_map):
    records = []
    n_skip = 0
    n_hate = 0
    n_norm = 0
    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))
        frame_dir = os.path.join(frames_root, vid)
        paths = ensure_frames(video_path, frame_dir)
        if paths is None:
            n_skip += 1
            continue
        text = item["text"][:MAX_TRANSCRIPT_CHARS]
        instruction = instr_tmpl.format(text=text if text else "(none)")
        target = label_map[item["label"]]
        records.append(
            {
                "messages": [
                    {"role": "user", "content": IMG_TOKENS + instruction},
                    {"role": "assistant", "content": target},
                ],
                "images": [os.path.abspath(p) for p in paths],
            }
        )
        if item["label"] == 1:
            n_hate += 1
        else:
            n_norm += 1
        if (n + 1) % 50 == 0:
            print("  [{}] {}/{} (skipped {})".format(ds, n + 1, len(items), n_skip), flush=True)
    return records, n_skip, n_hate, n_norm


def register_dataset_info(train_key, val_key, train_json, val_json):
    with open(DATASET_INFO, "r") as f:
        info = json.load(f)
    entry = lambda fn: {
        "file_name": fn,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    info[train_key] = entry(os.path.abspath(train_json))
    info[val_key] = entry(os.path.abspath(val_json))
    with open(DATASET_INFO, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[register] added '{}' and '{}' to {}".format(train_key, val_key, DATASET_INFO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    args = ap.parse_args()
    ds = args.dataset

    if ds == "MHC":
        instr_tmpl, label_map = INSTR_EN, LABEL_EN
    else:
        instr_tmpl, label_map = INSTR_ZH, LABEL_ZH

    gt_dir = os.path.join(RGCL_ROOT, "data", "gt", ds)
    video_root = os.path.join(RGCL_ROOT, "data", "video", ds, "All")
    frames_root = os.path.join(RGCL_ROOT, "data", "lora_frames", ds)
    out_dir = os.path.join(RGCL_ROOT, "data", "lora_sft", ds)
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for split in ("train", "val"):
        items = read_gt(os.path.join(gt_dir, "{}.jsonl".format(split)))
        records, n_skip, n_hate, n_norm = build_split(
            items, ds, video_root, frames_root, instr_tmpl, label_map
        )
        out_path = os.path.join(out_dir, "{}.json".format(split))
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        summary[split] = dict(
            n_in=len(items), n_out=len(records), n_skip=n_skip, hate=n_hate, norm=n_norm
        )
        print(
            "[{}] {}: n={} skipped={} hateful={} normal={} -> {}".format(
                ds, split, len(records), n_skip, n_hate, n_norm, out_path
            ),
            flush=True,
        )

    train_key, val_key = DS_REGISTER[ds]
    register_dataset_info(
        train_key,
        val_key,
        os.path.join(out_dir, "train.json"),
        os.path.join(out_dir, "val.json"),
    )

    print("\n==== SUMMARY {} ====".format(ds))
    for split, s in summary.items():
        bal = "n/a"
        tot = s["hate"] + s["norm"]
        if tot:
            bal = "{:.1f}% hateful ({}/{})".format(100.0 * s["hate"] / tot, s["hate"], tot)
        print(
            "  {}: n_in={} n_out={} n_skipped={} balance={}".format(
                split, s["n_in"], s["n_out"], s["n_skip"], bal
            )
        )


if __name__ == "__main__":
    main()
