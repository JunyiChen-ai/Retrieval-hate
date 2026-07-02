import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel

# This script generates *video* CLIP pooler embeddings for the RGCL training pipeline.
#
# It mirrors the cache contract produced by generate_CLIP_embedding_HF.py so the
# loader in src/data_loader/dataset.py:load_feats_split can consume it directly.
# The loader flattens the "ids" field via `[item for sublist in ids for item in sublist]`,
# therefore "ids" MUST be a list containing exactly ONE sublist of all string ids,
# in the SAME order as img_feats / text_feats / labels.
#
# Image stream  : uniformly sample num_frames frames from each .mp4 (decord, with a
#                 PyAV fallback), run CLIPVisionModel per frame, take pooler_output,
#                 then MEAN-POOL over frames -> [Dv].
# Text stream   : tokenize transcript with CLIPTokenizer; if longer than CLIP's 77-token
#                 limit, CHUNK into consecutive windows of <=75 content tokens, re-add
#                 special tokens, run CLIPTextModel per chunk, take pooler_output, then
#                 MEAN-POOL across chunks -> [Dt]. Short text is encoded once.

# Map the ground-truth split filenames to the output cache filenames the loader expects.
SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}


def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(
        description="Generate video CLIP pooler embeddings in the RGCL cache contract."
    )
    arg_parser.add_argument("--dataset", type=str, default="MHC")
    arg_parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/CLIP_Embedding",
        help="The path to save results.",
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
        default="openai/clip-vit-large-patch14-336",
        help="The CLIP model to use.",
    )
    arg_parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames uniformly sampled per video.",
    )
    arg_parser.add_argument(
        "--image_size", type=int, default=336, help="The image size to use."
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the vision model over frames.",
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
    args = arg_parser.parse_args(args_list)
    return args


def read_gt(gt_path):
    """Read a jsonl gt file -> list of dicts with id / text / label."""
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
                    "label": obj["label"],
                }
            )
    return items


def _sample_frame_indices(num_total, num_frames):
    """Uniformly sample num_frames indices across [0, num_total - 1].

    np.linspace handles the short-video case by repeating indices (sampling with
    replacement), so a video shorter than num_frames simply repeats frames.
    """
    if num_total <= 0:
        return None
    idx = np.linspace(0, num_total - 1, num_frames)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, num_total - 1)
    return idx.tolist()


def _decode_with_decord(video_path, num_frames):
    """Return a list of RGB PIL frames using decord, or None on failure."""
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
    """Return a list of RGB PIL frames using PyAV, or None on failure."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    # First pass: count decodable frames (robust against unreliable stream.frames).
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
            # Expand to num_frames respecting the requested order.
            indices = _sample_frame_indices(num_total, num_frames)
            lookup = {i: img for i, img in decoded}
            # Nearest available frame for any index we could not decode.
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
    # Fallback: decode everything (stream.frames unreliable), then subsample.
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
        print(
            "[WARN] decord failed for {} ({}); trying PyAV.".format(video_path, repr(e))
        )
        frames = None

    if frames is None:
        try:
            frames = _decode_with_pyav(video_path, num_frames)
        except Exception as e:  # noqa: BLE001
            print(
                "[WARN] PyAV failed for {} ({}).".format(video_path, repr(e))
            )
            frames = None

    if not frames:
        print("[WARN] no decodable frames for {}.".format(video_path))
        return None, False
    return frames, True


@torch.no_grad()
def encode_frames(frames, preprocess, vision_model, image_size, device, batch_size):
    """frames: list of PIL images -> mean-pooled pooler_output [Dv] on CPU float32."""
    resized = [img.resize((image_size, image_size), Image.BICUBIC) for img in frames]
    pooled = []
    for start in range(0, len(resized), batch_size):
        chunk = resized[start : start + batch_size]
        inputs = preprocess(images=chunk, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = vision_model(pixel_values=pixel_values)
        pooled.append(out.pooler_output.detach().cpu().float())
    pooled = torch.cat(pooled, dim=0)  # [num_frames, Dv]
    return pooled.mean(dim=0)  # [Dv]


@torch.no_grad()
def encode_text(text, tokenizer, text_model, device):
    """Encode transcript -> mean-pooled pooler_output [Dt] on CPU float32.

    Long transcripts are chunked into windows of <=75 content tokens (leaving room
    for BOS/EOS), each chunk re-encoded with special tokens, then mean-pooled.
    """
    text = text if text is not None else ""

    # Content tokens only (no special tokens), so we can re-window cleanly.
    content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    max_len = getattr(tokenizer, "model_max_length", 77)
    if not max_len or max_len > 77:
        max_len = 77
    content_window = max_len - 2  # leave room for BOS + EOS

    if len(content_ids) <= content_window:
        windows = [content_ids] if content_ids else [[]]
    else:
        windows = [
            content_ids[i : i + content_window]
            for i in range(0, len(content_ids), content_window)
        ]

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    pooled = []
    for window in windows:
        ids = []
        if bos is not None:
            ids.append(bos)
        ids.extend(window)
        if eos is not None:
            ids.append(eos)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled.append(out.pooler_output.detach().cpu().float())
    pooled = torch.cat(pooled, dim=0)  # [num_chunks, Dt]
    return pooled.mean(dim=0)  # [Dt]


def process_split(
    items,
    split_name,
    args,
    preprocess,
    tokenizer,
    vision_model,
    text_model,
    device,
):
    ids = []
    img_feats = []
    text_feats = []
    labels = []
    zero_guard = 0
    dv = None
    dt = None

    video_root = os.path.join(args.video_dir, args.dataset, "All")

    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))

        frames, ok = load_video_frames(video_path, args.num_frames)
        if ok:
            img_vec = encode_frames(
                frames,
                preprocess,
                vision_model,
                args.image_size,
                device,
                args.batch_size,
            )
            if dv is None:
                dv = img_vec.shape[0]
        else:
            zero_guard += 1
            if dv is None:
                # Determine Dv lazily from the vision model config.
                dv = vision_model.config.hidden_size
            img_vec = torch.zeros(dv, dtype=torch.float32)

        txt_vec = encode_text(item["text"], tokenizer, text_model, device)
        if dt is None:
            dt = txt_vec.shape[0]

        ids.append(vid)
        img_feats.append(img_vec)
        text_feats.append(txt_vec)
        labels.append(item["label"])

        if (n + 1) % 50 == 0:
            print(
                "  [{}] processed {}/{} (zero-vector guards so far: {})".format(
                    split_name, n + 1, len(items), zero_guard
                )
            )

    img_feats = torch.stack(img_feats, dim=0).float()  # [N, Dv]
    text_feats = torch.stack(text_feats, dim=0).float()  # [N, Dt]

    # Labels: LongTensor when all integral, else FloatTensor.
    if all(isinstance(l, int) or (isinstance(l, float) and float(l).is_integer()) for l in labels):
        labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    else:
        labels_t = torch.tensor([float(l) for l in labels], dtype=torch.float32)

    return ids, img_feats, text_feats, labels_t, dv, dt, zero_guard


def main(args):
    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading CLIP model: {}".format(args.model))
    vision_model = CLIPVisionModel.from_pretrained(args.model)
    text_model = CLIPTextModel.from_pretrained(args.model)
    preprocess = CLIPProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    vision_model.to(device).eval()
    text_model.to(device).eval()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        if split not in SPLIT_TO_OUTNAME:
            print(
                "[WARN] split '{}' has no output-name mapping; skipping.".format(split)
            )
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
            )
        )

        ids, img_feats, text_feats, labels_t, dv, dt, zero_guard = process_split(
            items,
            split,
            args,
            preprocess,
            tokenizer,
            vision_model,
            text_model,
            device,
        )

        # CONTRACT: ids is a list containing ONE sublist of all string ids.
        save_obj = {
            "ids": [ids],
            "img_feats": img_feats,
            "text_feats": text_feats,
            "labels": labels_t,
        }
        out_path = os.path.join(
            out_dir,
            "{}_{}_HF.pt".format(outname, str(args.model).replace("/", "_")),
        )
        torch.save(save_obj, out_path)

        print(
            "Saved '{}': N={}, Dv={}, Dt={}, zero-vector videos={} -> {}".format(
                outname, len(ids), dv, dt, zero_guard, out_path
            )
        )


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
