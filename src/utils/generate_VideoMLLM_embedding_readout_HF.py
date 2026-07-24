import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

# READOUT-GRID variant of generate_VideoMLLM_embedding_lora_HF.py.
#
# This script is a strict clone of the banked LoRA extractor
# (src/utils/generate_VideoMLLM_embedding_lora_HF.py). It changes ONLY the READOUT:
# which transformer LAYER, which TOKEN/pooling span, and which PROMPT the video-level
# embedding is read from. The model load path (frozen base + merged LoRA adapter), the
# frame sampler, the cache contract, and the pooling MATH are byte-identical to the
# banked extractor. NOTHING about the encoder, the training, or the loader contract
# changes -- this is a pure extraction-time read of the SAME merged frozen forward.
#
# It harvests the pre-registered READOUT_PREREG.md grid (R0-R3) in ONE pass over the
# videos. Because output_hidden_states=True is already set, one forward materialises
# ALL layer hidden states and the full token sequence, so reading layer 24 in addition
# to layer 28 and re-pooling a last-token span are GPU-FREE. Only a different PROMPT
# forces a new forward. Per item the grid therefore costs 4 forwards (2 prompt-passes
# x 2 streams) = 2x the deployed per-item cost, and writes 4 distinct caches:
#
#   R0 (ro_L28)     baseline prompt, current span (img=prefix mean / text=response mean), layer 28 (final)
#                   -> MUST reproduce the banked deployed cache BIT-EXACT (clobber-guard / determinism gate)
#   R1 (ro_L24)     baseline prompt, current span,                                        layer 24 (intermediate)
#   R2 (ro_ow_L28)  one-word prompt, last-token @ generation position,                    layer 28 (final)
#   R3 (ro_ow_L24)  one-word prompt, last-token @ generation position,                    layer 24 (intermediate)
#
# Cache tags: given --out_model_base_tag <BASE> (the DEPLOYED tag for the dataset:
#   ZH  = Qwen2.5-VL-7B-Instruct-LoRA_HF
#   HM  = Qwen2.5-VL-7B-Instruct-LoRA-curric_HF),
# the four output files are {split}_<BASE>-ro_L28.pt / -ro_L24.pt / -ro_ow_L28.pt /
# -ro_ow_L24.pt. The banked deployed cache {split}_<BASE>.pt is NEVER written and never
# clobbered (distinct -ro_* suffix).
#
# When --lora_dir is empty, the model load falls back to the FROZEN base (identical to
# the frozen extractor); the readout grid is unaffected.

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Map the ground-truth split filenames to the output cache filenames the loader expects.
SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}

# ----------------------------------------------------------------------------
# FROZEN readout grid (pinned by READOUT_PREREG.md; DO NOT sweep).
# Qwen2.5-VL-7B LLM = 28 decoder layers, hidden_size 3584. out.hidden_states is a
# tuple of length 29: index 0 = embedding output, indices 1..28 = decoder-layer
# outputs, [-1] = index 28 = final layer. So hidden_states[28] == hidden_states[-1]
# (the deployed read) and hidden_states[24] is the intermediate layer (VidVec depth
# 24/28 ~= 0.857). Both indices are literature-pinned; NO layer sweep.
# ----------------------------------------------------------------------------
LAYER_FINAL = 28  # deployed = out.hidden_states[-1]
LAYER_MID = 24    # VidVec intermediate (2602.08099), single pinned layer

# Baseline prompts -- BYTE-IDENTICAL to the deployed extractor (do not edit; the R0
# cell must reproduce the banked cache bit-exact).
IMG_INSTRUCTION = (
    "Describe the people, symbols, gestures, and on-screen text in this video."
)
TEXT_INSTRUCTION = (
    "You are analysing a short video for potentially hateful or offensive content. "
    "Considering the frames together with the provided title and transcript, "
    "summarise the targets, symbols, tone, and any harmful intent conveyed."
)

# One-word (PromptEOL / E5-V style) readout prompts -- pinned frozen text.
#   img : a standalone one-word summarisation instruction (replaces IMG_INSTRUCTION).
#   text: the baseline analytic instruction + title/transcript content, with the
#         one-word constraint appended at the VERY END so the last token (generation
#         position) is forced to compress the whole input. The last-token span reads
#         exactly that position.
IMG_INSTRUCTION_OW = "Describe this video in one word:"
TEXT_OW_TAIL = "\nSummarise the above in one word:"

# Frozen cell table: (tag_suffix, prompt_kind, img_span, text_span, layer).
# Each row is one grid cell; the extractor amortises the two prompt-passes across
# both layers, so this drives which pooled vector lands in which cache.
CELLS = [
    ("ro_L28", "baseline", "prefix", "response", LAYER_FINAL),      # R0 (clobber-guard)
    ("ro_L24", "baseline", "prefix", "response", LAYER_MID),        # R1
    ("ro_ow_L28", "oneword", "last_token", "last_token", LAYER_FINAL),  # R2
    ("ro_ow_L24", "oneword", "last_token", "last_token", LAYER_MID),    # R3
]


def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(
        description="Generate READOUT-GRID (R0-R3) video MLLM (Qwen2.5-VL) hidden-state embeddings."
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
        help="Path to a LoRA adapter directory (peft). Merged before the frozen "
        "forward passes. When empty, behaves like the frozen base extractor.",
    )
    arg_parser.add_argument(
        "--out_model_base_tag",
        type=str,
        default="Qwen2.5-VL-7B-Instruct-LoRA_HF",
        help="DEPLOYED cache tag for this dataset; the four grid caches append "
        "-ro_L28 / -ro_L24 / -ro_ow_L28 / -ro_ow_L24 to it. Never clobbers the "
        "deployed cache {split}_{out_model_base_tag}.pt.",
    )
    arg_parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames uniformly sampled per video (held at the deployed 8).",
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
        help="If >0, only process the first N items of each split (smoke/validation).",
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
# Frame sampler (reused verbatim from generate_VideoMLLM_embedding_lora_HF.py)
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
    """Load num_frames RGB PIL frames; decord first, PyAV fallback."""
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
# Qwen2.5-VL hidden-state pooling (readout-grid)
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


def _pool_span(last_hidden, input_ids, span, im_start_id):
    """Pool a [seq_len, D] hidden state over the requested span.

    span == "prefix"     -> mean over vision + instruction span (up to last <|im_start|>);
                            BYTE-IDENTICAL to the deployed img_feats pooling.
    span == "response"   -> mean over the trailing assistant-header tail (from last
                            <|im_start|>); BYTE-IDENTICAL to the deployed text_feats pooling.
    span == "last_token" -> the single last token at the generation position (PromptEOL/E5-V).
    Returns a [D] float32 L2-normalised vector on CPU.
    """
    if span == "last_token":
        pooled = last_hidden[-1]
    else:
        positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
        if span == "prefix":
            if len(positions) > 0:
                end = int(positions[-1].item())  # start of the assistant header
            else:
                end = last_hidden.shape[0]
            end = max(end, 1)
            pooled = last_hidden[:end].mean(dim=0)
        else:  # "response"
            if len(positions) > 0:
                start = int(positions[-1].item())  # the assistant <|im_start|> header
            else:
                start = max(last_hidden.shape[0] - 4, 0)
            start = min(start, last_hidden.shape[0] - 1)
            pooled = last_hidden[start:].mean(dim=0)
    pooled = pooled.float()
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=0)
    return pooled.detach().cpu()


@torch.no_grad()
def _encode_readout(frames, instruction, processor, model, device, max_pixels, span, layers):
    """Run ONE frozen forward; return {layer: L2-normed pooled [D]} for each layer.

    The single forward materialises all 29 hidden states (output_hidden_states=True is
    already on for the deployed extractor), so multiple layers are read GPU-free from
    the same forward. For (span in {prefix, response}, layer == 28) the returned vector
    is BYTE-IDENTICAL to the deployed _encode (hidden_states[28] == hidden_states[-1]).
    """
    messages = _build_messages(frames, instruction)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    input_ids = inputs["input_ids"][0]  # [seq_len]
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    result = {}
    for L in layers:
        last_hidden = out.hidden_states[L][0]  # [seq_len, D]
        # Preflight invariant (deployed extractor L306): vision tokens are masked_scatter'd
        # in place, so hidden-state length must equal input_ids length for span indexing.
        assert last_hidden.shape[0] == input_ids.numel(), (
            "hidden/input_ids length mismatch at layer {}: {} vs {}".format(
                L, last_hidden.shape[0], input_ids.numel()
            )
        )
        result[L] = _pool_span(last_hidden, input_ids, span, im_start_id)
    return result


def process_split(items, split_name, args, processor, model, device):
    d = model.config.hidden_size  # 3584 for Qwen2.5-VL-7B

    ids = []
    labels = []
    # One accumulator pair (img/text) per grid cell (keyed by tag suffix).
    acc = {suffix: {"img": [], "text": []} for (suffix, _, _, _, _) in CELLS}
    zero_guard = 0

    video_root = os.path.join(args.video_dir, args.dataset, "All")
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    layers = [LAYER_FINAL, LAYER_MID]

    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))
        frames, ok = load_video_frames(video_path, args.num_frames)

        title = item.get("title", "")
        transcript = item.get("text", "")

        if ok:
            # ---- Pass A: baseline prompts (current span) ----
            imgA = _encode_readout(
                frames, IMG_INSTRUCTION, processor, model, device,
                args.max_pixels, span="prefix", layers=layers,
            )
            text_prompt_base = (
                TEXT_INSTRUCTION
                + "\nTitle: " + (title if title else "(none)")
                + "\nTranscript: " + (transcript if transcript else "(none)")
            )
            txtA = _encode_readout(
                frames, text_prompt_base, processor, model, device,
                args.max_pixels, span="response", layers=layers,
            )
            # ---- Pass B: one-word prompts (last-token @ generation position) ----
            imgB = _encode_readout(
                frames, IMG_INSTRUCTION_OW, processor, model, device,
                args.max_pixels, span="last_token", layers=layers,
            )
            text_prompt_ow = text_prompt_base + TEXT_OW_TAIL
            txtB = _encode_readout(
                frames, text_prompt_ow, processor, model, device,
                args.max_pixels, span="last_token", layers=layers,
            )
            pass_img = {"baseline": imgA, "oneword": imgB}
            pass_txt = {"baseline": txtA, "oneword": txtB}
            for (suffix, prompt_kind, _img_span, _txt_span, layer) in CELLS:
                acc[suffix]["img"].append(pass_img[prompt_kind][layer])
                acc[suffix]["text"].append(pass_txt[prompt_kind][layer])
        else:
            zero_guard += 1
            for (suffix, _, _, _, _) in CELLS:
                acc[suffix]["img"].append(torch.zeros(d, dtype=torch.float32))
                acc[suffix]["text"].append(torch.zeros(d, dtype=torch.float32))

        ids.append(vid)
        labels.append(item["label"])

        if (n + 1) % 20 == 0:
            print(
                "  [{}] processed {}/{} (zero-vector guards so far: {})".format(
                    split_name, n + 1, len(items), zero_guard
                ),
                flush=True,
            )

    if all(isinstance(l, int) or (isinstance(l, float) and float(l).is_integer()) for l in labels):
        labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    else:
        labels_t = torch.tensor([float(l) for l in labels], dtype=torch.float32)

    cells = {}
    for suffix in acc:
        img_feats = torch.stack(acc[suffix]["img"], dim=0).float()  # [N, D]
        text_feats = torch.stack(acc[suffix]["text"], dim=0).float()  # [N, D]
        cells[suffix] = (img_feats, text_feats)
    return ids, cells, labels_t, d, zero_guard


def main(args):
    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading Qwen2.5-VL base model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    )

    lora_dir = args.lora_dir.strip() if args.lora_dir else ""
    if lora_dir:
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(
                "--lora_dir '{}' is not a directory (expected a peft adapter dir).".format(lora_dir)
            )
        from peft import PeftModel

        print("Attaching LoRA adapter from: {}".format(lora_dir), flush=True)
        model = PeftModel.from_pretrained(model, lora_dir)
        print("Merging LoRA adapter into base weights (merge_and_unload) ...", flush=True)
        model = model.merge_and_unload()
    else:
        print("No --lora_dir given; using FROZEN base model.", flush=True)

    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    # Sanity: the merged model must expose 28 decoder layers (29 hidden states) so the
    # pinned L24/L28 indices are valid. Guard before any extraction.
    num_layers = model.config.num_hidden_layers
    assert num_layers == LAYER_FINAL, (
        "expected {} decoder layers, got {}; pinned readout layers invalid".format(
            LAYER_FINAL, num_layers
        )
    )

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

        ids, cells, labels_t, d, zero_guard = process_split(
            items, split, args, processor, model, device
        )

        for suffix, (img_feats, text_feats) in cells.items():
            out_tag = "{}-{}".format(args.out_model_base_tag, suffix)
            save_obj = {
                "ids": [ids],  # CONTRACT: one sublist of all string ids
                "img_feats": img_feats,
                "text_feats": text_feats,
                "labels": labels_t,
            }
            out_path = os.path.join(out_dir, "{}_{}.pt".format(outname, out_tag))
            torch.save(save_obj, out_path)
            print(
                "Saved '{}' [{}]: N={}, Dv={}, Dt={}, zero-vector videos={} -> {}".format(
                    outname, suffix, len(ids), d, d, zero_guard, out_path
                ),
                flush=True,
            )


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
