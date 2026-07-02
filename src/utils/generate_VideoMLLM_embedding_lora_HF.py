import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

# LoRA-AWARE variant of generate_VideoMLLM_embedding_HF.py.
#
# This script generates *video* MLLM (Qwen2.5-VL-7B-Instruct) hidden-state pooled
# embeddings for the RGCL training pipeline, but from a LoRA-ADAPTED model:
# the frozen base Qwen2.5-VL is loaded, a LoRA adapter (produced by the Stage-1
# LoRA-SFT, saved to logging/lora/<DS>/) is attached via peft.PeftModel and merged
# into the base weights (merge_and_unload) BEFORE the frozen forward passes.
#
# When --lora_dir is empty/unset, this script behaves EXACTLY like the original
# frozen extractor (no peft is imported, no merge happens): it is a strict superset.
#
# It mirrors the exact cache contract produced by generate_VideoMLLM_embedding_HF.py
# so the loader in src/data_loader/dataset.py:load_feats_MHC can consume it directly.
# The loader flattens the "ids" field via `[item for sublist in ids for item in sublist]`,
# therefore "ids" MUST be a list containing exactly ONE sublist of all string ids,
# in the SAME order as img_feats / text_feats / labels.
#
# Two streams, both frozen (post-merge) Qwen2.5-VL-7B, bf16, torch.no_grad(),
# output_hidden_states=True:
#
#   img_feats  (Dv=3584): 8 frames (as a video) + a FIXED neutral instruction
#       ("Describe the people, symbols, gestures, and on-screen text in this video.")
#       -> mean of the LAST-layer hidden states over the VISUAL + INSTRUCTION span
#       (i.e. everything up to and including the instruction; excludes generation
#        prompt tail tokens) -> L2-norm.
#
#   text_feats (Dt=3584): same 8 frames + title + transcript + a FIXED analytic
#       instruction -> mean of the LAST-layer hidden states over the RESPONSE / last
#       tokens span (the assistant generation-prompt tail: <|im_start|>assistant\n ...)
#       -> L2-norm. Since we run a single forward with NO generation, the "response
#       span" is taken as the trailing tokens after the last <|im_start|> (assistant
#       turn header), which are the model's contextualised summary tokens.
#
# The frame sampler (decord + PyAV fallback) is reused verbatim from the CLIP script.
#
# To keep the LoRA cache DISTINCT from the frozen cache (never clobbers it), the
# default --out_model_tag is "Qwen2.5-VL-7B-Instruct-LoRA_HF", so the output path is
#   data/CLIP_Embedding/<DS>/{split}_Qwen2.5-VL-7B-Instruct-LoRA_HF.pt
# which run_rac.py reads as --model Qwen2.5-VL-7B-Instruct-LoRA_HF.

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Map the ground-truth split filenames to the output cache filenames the loader expects.
SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}

# Fixed instructions (never sampled from the model; pure encoder use).
IMG_INSTRUCTION = (
    "Describe the people, symbols, gestures, and on-screen text in this video."
)
TEXT_INSTRUCTION = (
    "You are analysing a short video for potentially hateful or offensive content. "
    "Considering the frames together with the provided title and transcript, "
    "summarise the targets, symbols, tone, and any harmful intent conveyed."
)


def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(
        description="Generate LoRA-adapted video MLLM (Qwen2.5-VL) hidden-state embeddings in the RGCL cache contract."
    )
    arg_parser.add_argument("--dataset", type=str, default="HateMM")
    arg_parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/CLIP_Embedding",
        help="The path to save results (loader reads from CLIP_Embedding).",
    )
    arg_parser.add_argument(
        "--gt_dir",
        type=str,
        default="./data/gt",
        help="Directory with <dataset>/<split>.jsonl ground-truth files.",
    )
    arg_parser.add_argument(
        "--video_dir",
        type=str,
        default="./data/video",
        help="Directory with <dataset>/All/<id>.mp4 video files.",
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="The Qwen2.5-VL BASE model (HF id) to use.",
    )
    arg_parser.add_argument(
        "--lora_dir",
        type=str,
        default="",
        help="Path to a LoRA adapter directory (peft). When set, the adapter is "
        "loaded onto the base model and merged (merge_and_unload) before the frozen "
        "forward passes. When empty, behaves EXACTLY like the frozen base extractor.",
    )
    arg_parser.add_argument(
        "--out_model_tag",
        type=str,
        default="Qwen2.5-VL-7B-Instruct-LoRA_HF",
        help="Tag used in the output filename: {split}_{out_model_tag}.pt. "
        "Defaults to a LoRA-distinct tag so it never clobbers the frozen cache.",
    )
    arg_parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames uniformly sampled per video.",
    )
    arg_parser.add_argument(
        "--max_pixels",
        type=int,
        default=360 * 420,
        help="max_pixels per frame for the Qwen vision preprocessor (memory control).",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    arg_parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma separated gt splits to process.",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N items of each split (debug/validation).",
    )
    args = arg_parser.parse_args(args_list)
    return args


def read_gt(gt_path):
    """Read a jsonl gt file -> list of dicts with id / text / label / title."""
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
# Frame sampler (reused verbatim from generate_VideoCLIP_embedding_HF.py)
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
    batch = vr.get_batch(indices).asnumpy()  # [num_frames, H, W, 3] RGB
    frames = [Image.fromarray(batch[i]).convert("RGB") for i in range(batch.shape[0])]
    return frames


def _decode_with_pyav(video_path, num_frames):
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    num_total = stream.frames
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
            return frames
        return None
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_image().convert("RGB"))
    container.close()
    if not all_frames:
        return None
    indices = _sample_frame_indices(len(all_frames), num_frames)
    return [all_frames[i] for i in indices]


def load_video_frames(video_path, num_frames):
    """Load num_frames RGB PIL frames; decord first, PyAV fallback.

    Returns (frames, ok). On any failure / missing file / 0 decodable frames,
    returns (None, False) and the caller substitutes a zero vector.
    """
    if not os.path.exists(video_path):
        print("[WARN] missing video file: {}".format(video_path))
        return None, False

    frames = None
    try:
        frames = _decode_with_decord(video_path, num_frames)
    except Exception as e:  # noqa: BLE001
        print("[WARN] decord failed for {} ({}); trying PyAV.".format(video_path, repr(e)))
        frames = None

    if frames is None:
        try:
            frames = _decode_with_pyav(video_path, num_frames)
        except Exception as e:  # noqa: BLE001
            print("[WARN] PyAV failed for {} ({}).".format(video_path, repr(e)))
            frames = None

    if not frames:
        print("[WARN] no decodable frames for {}.".format(video_path))
        return None, False
    return frames, True


# ----------------------------------------------------------------------------
# Qwen2.5-VL hidden-state pooling
# ----------------------------------------------------------------------------
def _build_messages(frames, instruction):
    """Build a Qwen chat message list: one user turn with the video (as frames) + text."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": instruction},
            ],
        }
    ]


@torch.no_grad()
def _encode(frames, instruction, processor, model, device, max_pixels, span):
    """Run one frozen forward; return an L2-normed pooled last-layer hidden state [D] on CPU.

    span:
      "prefix"  -> mean over the whole prompt (visual + instruction context); used for img_feats.
      "response"-> mean over the trailing assistant-turn tokens (after the last
                   <|im_start|>); used for text_feats.
    """
    messages = _build_messages(frames, instruction)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # frames are already sampled PIL images; pass them as the video's frame list.
    # NOTE: max_pixels is set at processor CONSTRUCTION time (from_pretrained), not
    # here — the __call__ ignores it under transformers 4.49.
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    last_hidden = out.hidden_states[-1][0]  # [seq_len, D]
    input_ids = inputs["input_ids"][0]  # [seq_len]
    # Preflight invariant: vision tokens are masked_scatter'd in place (not shifted),
    # so hidden-state length must equal input_ids length for span indexing to align.
    assert last_hidden.shape[0] == input_ids.numel(), (
        "hidden/input_ids length mismatch: {} vs {}".format(
            last_hidden.shape[0], input_ids.numel()
        )
    )
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    if span == "prefix":
        # img_feats: mean over the VISION + INSTRUCTION span, i.e. every token that is
        # a video-pad token OR lies inside the user turn up to <|im_end|>. Concretely we
        # mean over all tokens EXCEPT the trailing assistant generation-prompt tail
        # (<|im_start|>assistant\n), so the pooled vector is the contextualised
        # visual+instruction summary and not the (empty) response header.
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            end = int(positions[-1].item())  # start of the assistant header
        else:
            end = last_hidden.shape[0]
        end = max(end, 1)
        pooled = last_hidden[:end].mean(dim=0)
    else:  # "response" == last-token decoder embedding
        # text_feats: we do NOT generate, so the "response" is represented by the LAST
        # token's hidden state — the canonical causal-LM sentence embedding, which has
        # attended over the full frames+title+transcript+instruction context. We mean
        # over the final generation-prompt tail (assistant header tokens) for a touch of
        # smoothing; these are the tokens whose representations summarise the input.
        im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            start = int(positions[-1].item())  # the assistant <|im_start|> header
        else:
            start = max(last_hidden.shape[0] - 4, 0)
        # ensure at least the last token is included; span = assistant header .. end
        start = min(start, last_hidden.shape[0] - 1)
        pooled = last_hidden[start:].mean(dim=0)

    _ = video_pad_id  # (kept for clarity/debug; span uses im_start boundaries)
    pooled = pooled.float()
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
    return pooled.detach().cpu()


def process_split(items, split_name, args, processor, model, device):
    ids = []
    img_feats = []
    text_feats = []
    labels = []
    zero_guard = 0
    d = model.config.hidden_size  # 3584 for Qwen2.5-VL-7B

    video_root = os.path.join(args.video_dir, args.dataset, "All")

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))

        frames, ok = load_video_frames(video_path, args.num_frames)
        if ok:
            img_vec = _encode(
                frames, IMG_INSTRUCTION, processor, model, device,
                args.max_pixels, span="prefix",
            )
            title = item.get("title", "")
            transcript = item.get("text", "")
            text_prompt = (
                TEXT_INSTRUCTION
                + "\nTitle: " + (title if title else "(none)")
                + "\nTranscript: " + (transcript if transcript else "(none)")
            )
            text_vec = _encode(
                frames, text_prompt, processor, model, device,
                args.max_pixels, span="response",
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

    img_feats = torch.stack(img_feats, dim=0).float()  # [N, D]
    text_feats = torch.stack(text_feats, dim=0).float()  # [N, D]

    if all(isinstance(l, int) or (isinstance(l, float) and float(l).is_integer()) for l in labels):
        labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    else:
        labels_t = torch.tensor([float(l) for l in labels], dtype=torch.float32)

    return ids, img_feats, text_feats, labels_t, d, d, zero_guard


def main(args):
    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Qwen2.5-VL base model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn optional; sdpa is the safe default
        device_map=None,
    )

    # LoRA adapter (optional). When --lora_dir is set, attach the peft adapter and
    # merge it into the base weights so the rest of the pipeline is identical to the
    # frozen path (plain nn.Module forward, no peft wrapper active at inference).
    lora_dir = args.lora_dir.strip() if args.lora_dir else ""
    if lora_dir:
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(
                "--lora_dir '{}' is not a directory (expected a peft adapter dir "
                "with adapter_config.json + adapter_model.safetensors).".format(lora_dir)
            )
        from peft import PeftModel

        print("Attaching LoRA adapter from: {}".format(lora_dir), flush=True)
        model = PeftModel.from_pretrained(model, lora_dir)
        print("Merging LoRA adapter into base weights (merge_and_unload) ...", flush=True)
        model = model.merge_and_unload()
    else:
        print("No --lora_dir given; using FROZEN base model (original behavior).", flush=True)

    model.to(device).eval()
    # Set max_pixels at construction (the __call__ ignores it under transformers 4.49).
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
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
            "Processing split '{}' ({} items) -> outname '{}'".format(
                split, len(items), outname
            ),
            flush=True,
        )

        ids, img_feats, text_feats, labels_t, dv, dt, zero_guard = process_split(
            items, split, args, processor, model, device
        )

        # CONTRACT: ids is a list containing ONE sublist of all string ids.
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
                outname, len(ids), dv, dt, zero_guard, out_path
            ),
            flush=True,
        )


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
