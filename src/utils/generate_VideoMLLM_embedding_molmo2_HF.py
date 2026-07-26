import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

# Molmo2-8B (allenai/Molmo2-8B: Qwen3-8B LLM + SigLIP2-so400m vision tower) frozen
# hidden-state pooled embeddings, in the SAME RGCL cache contract as
# generate_VideoMLLM_embedding_HF.py (the deployed Qwen2.5-VL recipe).
#
# Recon + deviation ledger: refine-logs/MOLMO2_FORENSIC_RECON.md.
# Requires the HateVideoVLM env (transformers 4.57.x); HateVideo's 4.49 cannot load Molmo2.
#
# What is IDENTICAL to the Qwen recipe:
#   * the frame sampler (8 uniform frames, decord -> PyAV fallback) -- copied verbatim below,
#     exactly as that script copied it from generate_VideoCLIP_embedding_HF.py;
#   * the instruction strings, the Title:/Transcript:/(none) scaffolding, and their assembly;
#   * the two spans -- img_feats = mean over [0 : last <|im_start|>), i.e. the visual +
#     instruction context; text_feats = mean over [last <|im_start|> : end), the assistant
#     generation-prompt tail -- then mean-pool -> L2-normalise;
#   * bf16 frozen forward, torch.no_grad(), output_hidden_states=True, no generation;
#   * the on-disk contract ("ids" is a list holding ONE sublist, in img/text/label order).
#
# What is FORCED to differ by Molmo2's architecture (all three documented in the recon):
#   * fixed 378x378 per frame with 3x3 pooling -> 81 tokens/frame (648 for 8 frames), instead
#     of Qwen's dynamic max_pixels grid;
#   * the <|video|> placeholder is emitted BEFORE the <|im_start|>user turn by Molmo2's chat
#     template, not inside it;
#   * per-frame wall-clock timestamps are injected into the video string (Molmo2 is
#     video-native). We feed TRUE timestamps by passing the real source fps and the real
#     sampled frame indices, from which transformers derives metadata.timestamps.
#
# Vision tokens are merged IN PLACE (modeling_molmo2.py: x[is_image_patch] += image_features),
# so hidden-state length == input_ids length and the span indexing below is aligned; this is
# asserted at runtime exactly as in the Qwen script.

from transformers import AutoModelForImageTextToText, AutoProcessor

SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}

# Fixed instructions -- byte-identical to the deployed Qwen recipe.
IMG_INSTRUCTION = (
    "Describe the people, symbols, gestures, and on-screen text in this video."
)
TEXT_INSTRUCTION = (
    "You are analysing a short video for potentially hateful or offensive content. "
    "Considering the frames together with the provided title and transcript, "
    "summarise the targets, symbols, tone, and any harmful intent conveyed."
)


def parse_args_sys(args_list=None):
    p = argparse.ArgumentParser(
        description="Generate frozen Molmo2-8B hidden-state embeddings in the RGCL cache contract."
    )
    p.add_argument("--dataset", type=str, default="HateMM")
    p.add_argument("--EXP_FOLDER", type=str, default="./data/CLIP_Embedding")
    p.add_argument("--gt_dir", type=str, default="./data/gt")
    p.add_argument("--video_dir", type=str, default="./data/video")
    p.add_argument("--model", type=str, default="/data/jehc223/models/Molmo2-8B-bf16")
    p.add_argument("--out_model_tag", type=str, default="Molmo2-8B_HF")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(args_list)


def read_gt(gt_path):
    items = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "id": str(obj["id"]),
                    "text": "" if obj.get("text") is None else str(obj["text"]),
                    "title": "" if obj.get("title") is None else str(obj.get("title", "")),
                    "label": obj["label"],
                }
            )
    return items


# ----------------------------------------------------------------------------
# Frame sampler -- verbatim from generate_VideoMLLM_embedding_HF.py, extended ONLY to
# also return (fps, total_frames, indices), which Molmo2 needs for true timestamps.
# ----------------------------------------------------------------------------
def _sample_frame_indices(num_total, num_frames):
    if num_total <= 0:
        return None
    idx = np.linspace(0, num_total - 1, num_frames)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, num_total - 1)
    return idx.tolist()


def _decode_with_decord(video_path, num_frames):
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("native")
    vr = VideoReader(video_path, ctx=cpu(0))
    num_total = len(vr)
    indices = _sample_frame_indices(num_total, num_frames)
    if indices is None:
        return None
    batch = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(batch[i]).convert("RGB") for i in range(batch.shape[0])]
    try:
        fps = float(vr.get_avg_fps())
    except Exception:  # noqa: BLE001
        fps = 0.0
    return frames, indices, num_total, fps


def _decode_with_pyav(video_path, num_frames):
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    num_total = stream.frames
    try:
        fps = float(stream.average_rate)
    except Exception:  # noqa: BLE001
        fps = 0.0
    decoded = []
    if num_total and num_total > 0:
        target = set(_sample_frame_indices(num_total, num_frames))
        for i, frame in enumerate(container.decode(video=0)):
            if i in target:
                decoded.append((i, frame.to_image().convert("RGB")))
            if len(decoded) >= len(target) and i >= max(target):
                break
        container.close()
        if decoded:
            indices = _sample_frame_indices(num_total, num_frames)
            lookup = {i: img for i, img in decoded}
            avail = sorted(lookup.keys())
            frames = []
            for idx in indices:
                if idx in lookup:
                    frames.append(lookup[idx])
                else:
                    nearest = min(avail, key=lambda a: abs(a - idx))
                    frames.append(lookup[nearest])
            return frames, indices, num_total, fps
        return None
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_image().convert("RGB"))
    container.close()
    if not all_frames:
        return None
    indices = _sample_frame_indices(len(all_frames), num_frames)
    return [all_frames[i] for i in indices], indices, len(all_frames), fps


def load_video_frames(video_path, num_frames):
    """Return (frames, indices, num_total, fps, ok). Zero-vector guard on any failure."""
    if not os.path.exists(video_path):
        print("[WARN] missing video file: {}".format(video_path))
        return None, None, 0, 0.0, False

    out = None
    try:
        out = _decode_with_decord(video_path, num_frames)
    except Exception as e:  # noqa: BLE001
        print("[WARN] decord failed for {} ({}); trying PyAV.".format(video_path, repr(e)))
        out = None

    if out is None:
        try:
            out = _decode_with_pyav(video_path, num_frames)
        except Exception as e:  # noqa: BLE001
            print("[WARN] PyAV failed for {} ({}).".format(video_path, repr(e)))
            out = None

    if not out or not out[0]:
        print("[WARN] no decodable frames for {}.".format(video_path))
        return None, None, 0, 0.0, False
    frames, indices, num_total, fps = out
    return frames, indices, num_total, fps, True


# ----------------------------------------------------------------------------
# Molmo2 hidden-state pooling
# ----------------------------------------------------------------------------
def _build_messages(instruction):
    """One user turn: the video placeholder plus the instruction (mirrors the Qwen recipe)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": None},
                {"type": "text", "text": instruction},
            ],
        }
    ]


@torch.no_grad()
def _encode(frames, indices, num_total, fps, instruction, processor, model, device, span):
    """One frozen forward -> L2-normed pooled last-layer hidden state [D] on CPU.

    span "prefix"   -> mean over the visual + instruction context (img_feats)
    span "response" -> mean over the trailing assistant-header tokens (text_feats)
    """
    messages = _build_messages(instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    video = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)  # (T,H,W,3)
    h, w = video.shape[1], video.shape[2]
    # fps + frames_indices are the REAL source values; transformers derives
    # metadata.timestamps = frames_indices / fps from them, so the timestamps Molmo2 injects
    # into the prompt are the true wall-clock times of the sampled frames.
    safe_fps = fps if fps and fps > 0 else float(len(frames))
    meta = {
        "total_num_frames": int(num_total) if num_total else len(frames),
        "fps": safe_fps,
        "duration": (int(num_total) / safe_fps) if num_total else float(len(frames)) / safe_fps,
        "height": int(h),
        "width": int(w),
        "frames_indices": [int(i) for i in indices],
        "video_backend": "decord",
    }

    inputs = processor(
        text=[text],
        videos=video,
        video_metadata=[meta],
        do_sample_frames=False,  # frames are already sampled; do NOT let Molmo2 resample
        return_tensors="pt",
    )
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]  # [seq_len, D]
    input_ids = inputs["input_ids"][0]
    # Same invariant as the Qwen recipe: vision features are added in place at the
    # image-patch positions, so lengths must match for the spans to align.
    assert last_hidden.shape[0] == input_ids.numel(), (
        "hidden/input_ids length mismatch: {} vs {}".format(
            last_hidden.shape[0], input_ids.numel()
        )
    )

    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

    if span == "prefix":
        if len(positions) > 0:
            end = int(positions[-1].item())
        else:
            end = last_hidden.shape[0]
        end = max(end, 1)
        pooled = last_hidden[:end].mean(dim=0)
    else:  # "response"
        if len(positions) > 0:
            start = int(positions[-1].item())
        else:
            start = max(last_hidden.shape[0] - 4, 0)
        start = min(start, last_hidden.shape[0] - 1)
        pooled = last_hidden[start:].mean(dim=0)

    pooled = pooled.float()
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
    return pooled.detach().cpu()


def process_split(items, split_name, args, processor, model, device, d):
    ids, img_feats, text_feats, labels = [], [], [], []
    zero_guard = 0
    video_root = os.path.join(args.video_dir, args.dataset, "All")

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))

        frames, indices, num_total, fps, ok = load_video_frames(video_path, args.num_frames)
        if ok:
            img_vec = _encode(
                frames, indices, num_total, fps, args.img_instruction,
                processor, model, device, span="prefix",
            )
            title = item.get("title", "")
            transcript = item.get("text", "")
            text_prompt = (
                args.text_instruction
                + "\n" + args.title_label + (title if title else args.none_placeholder)
                + "\n" + args.transcript_label + (transcript if transcript else args.none_placeholder)
            )
            text_vec = _encode(
                frames, indices, num_total, fps, text_prompt,
                processor, model, device, span="response",
            )
        else:
            zero_guard += 1
            img_vec = torch.zeros(d, dtype=torch.float32)
            text_vec = torch.zeros(d, dtype=torch.float32)

        ids.append(vid)
        img_feats.append(img_vec)
        text_feats.append(text_vec)
        labels.append(item["label"])

        if (n + 1) % 20 == 0:
            print(
                "  [{}] processed {}/{} (zero-vector guards so far: {})".format(
                    split_name, n + 1, len(items), zero_guard
                ),
                flush=True,
            )

    img_feats = torch.stack(img_feats, dim=0).float()
    text_feats = torch.stack(text_feats, dim=0).float()

    if all(isinstance(l, int) or (isinstance(l, float) and float(l).is_integer()) for l in labels):
        labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    else:
        labels_t = torch.tensor([float(l) for l in labels], dtype=torch.float32)

    return ids, img_feats, text_feats, labels_t, zero_guard


def main(args):
    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Molmo2 model: {}".format(args.model), flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    d = model.config.text_config.hidden_size  # 4096 for Molmo2-8B
    print("hidden_size = {}".format(d), flush=True)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' has no output-name mapping; skipping.".format(split))
            continue
        outname = SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] gt file not found, skipping split '{}': {}".format(split, gt_path))
            continue

        items = read_gt(gt_path)
        print(
            "Processing split '{}' ({} items) -> outname '{}'".format(split, len(items), outname),
            flush=True,
        )

        ids, img_feats, text_feats, labels_t, zero_guard = process_split(
            items, split, args, processor, model, device, d
        )

        save_obj = {
            "ids": [ids],
            "img_feats": img_feats,
            "text_feats": text_feats,
            "labels": labels_t,
        }
        out_path = os.path.join(out_dir, "{}_{}.pt".format(outname, args.out_model_tag))
        torch.save(save_obj, out_path)

        print(
            "Saved '{}': N={}, Dv={}, Dt={}, zero-vector videos={} -> {}".format(
                outname, len(ids), d, d, zero_guard, out_path
            ),
            flush=True,
        )


if __name__ == "__main__":
    args = parse_args_sys()
    # Scaffolding constants, byte-identical to the deployed Qwen recipe's defaults.
    args.img_instruction = IMG_INSTRUCTION
    args.text_instruction = TEXT_INSTRUCTION
    args.title_label = "Title: "
    args.transcript_label = "Transcript: "
    args.none_placeholder = "(none)"
    print(args)
    main(args)
