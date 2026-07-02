import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

# This script generates *sub-clip* (fine-granularity) CLIP visual pooler
# embeddings for the multi-granularity annotation-free temporal retrieval
# method (DESIGN_iter1.md Delta 1).
#
# For each video we:
#   1. Uniformly sample M frames (default 16) with the SAME decord/PyAV sampler
#      used by generate_VideoCLIP_embedding_HF.py.
#   2. Split the M frames into K=4 CONTIGUOUS temporal windows.
#   3. Run the frozen CLIP vision tower per frame, take pooler_output, and
#      MEAN-POOL within each window -> K sub-clip visual embeddings [K, Dv].
#   Sub-clips INHERIT the parent video label (MIL assumption). NO gold spans.
#
# The sub-clip TEXT stream is NOT re-extracted here: sub-clips share the
# VIDEO-LEVEL text embedding already stored in the whole-video cache
# (title+transcript apply to the whole video). The training code fuses each
# sub-clip visual embedding with its parent's video-level text_feats.
#
# Output cache contract (one file per split), e.g.
#   data/CLIP_Embedding/<DS>/{train,dev_seen,test_seen}_subclipK4_<model>_HF.pt
# with:
#   {
#     "video_ids"        : [V]      # string id per parent video (order == whole-video cache)
#     "subclip_img_feats": [TotalSub, Dv]   # float32, TotalSub == V * K
#     "subclip_parent"   : [TotalSub]       # long, row index into the video-level cache
#     "labels"           : [TotalSub]       # long, inherited parent video label
#     "num_subclips"     : K,
#     "num_frames"       : M,
#   }
#
# Zero-vector guard: an unreadable/missing video contributes K zero sub-clip
# vectors (all rows zero), so downstream code can mask them out.

# Map the ground-truth split filenames to the whole-video cache filenames the
# loader / whole-video extractor uses.
SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}


def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(
        description="Generate sub-clip CLIP visual pooler embeddings (multi-granularity retrieval)."
    )
    arg_parser.add_argument("--dataset", type=str, default="MHC")
    arg_parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/CLIP_Embedding",
        help="The path to save results (mirrors the whole-video cache dir).",
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
        help="The CLIP model to use (must match the whole-video cache model).",
    )
    arg_parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames uniformly sampled per video (M).",
    )
    arg_parser.add_argument(
        "--num_subclips",
        type=int,
        default=4,
        help="Number of contiguous temporal windows per video (K).",
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
    returns (None, False) and the caller substitutes zero vectors.
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
            print("[WARN] PyAV failed for {} ({}).".format(video_path, repr(e)))
            frames = None

    if not frames:
        print("[WARN] no decodable frames for {}.".format(video_path))
        return None, False
    return frames, True


def _window_bounds(num_frames, num_subclips):
    """Contiguous, (near-)equal windows over [0, num_frames).

    Returns a list of (start, end) index pairs, len == num_subclips. When
    num_frames is not divisible by num_subclips, the earlier windows get the
    extra frame(s). Guarantees each window has >= 1 frame as long as
    num_frames >= num_subclips (which _sample_frame_indices ensures by
    sampling exactly M frames).
    """
    base = num_frames // num_subclips
    rem = num_frames % num_subclips
    bounds = []
    start = 0
    for k in range(num_subclips):
        size = base + (1 if k < rem else 0)
        if size <= 0:
            size = 1  # safety; only if M < K, which we avoid
        end = min(start + size, num_frames)
        if start >= num_frames:
            # Degenerate: reuse the last frame for any empty tail window.
            start = num_frames - 1
            end = num_frames
        bounds.append((start, end))
        start = end
    return bounds


@torch.no_grad()
def encode_frames_pooled(frames, preprocess, vision_model, image_size, device, batch_size):
    """frames: list of PIL images -> per-frame pooler_output [M, Dv] on CPU float32."""
    resized = [img.resize((image_size, image_size), Image.BICUBIC) for img in frames]
    pooled = []
    for start in range(0, len(resized), batch_size):
        chunk = resized[start : start + batch_size]
        inputs = preprocess(images=chunk, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = vision_model(pixel_values=pixel_values)
        pooled.append(out.pooler_output.detach().cpu().float())
    return torch.cat(pooled, dim=0)  # [M, Dv]


def process_split(
    items,
    split_name,
    args,
    preprocess,
    vision_model,
    device,
):
    video_ids = []
    subclip_img_feats = []  # list of [Dv]
    subclip_parent = []     # int parent row index
    labels = []             # inherited parent label
    zero_guard = 0
    dv = None

    K = args.num_subclips
    M = args.num_frames
    video_root = os.path.join(args.video_dir, args.dataset, "All")

    for parent_idx, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))

        frames, ok = load_video_frames(video_path, M)
        if ok:
            per_frame = encode_frames_pooled(
                frames, preprocess, vision_model, args.image_size, device, args.batch_size
            )  # [M, Dv]
            if dv is None:
                dv = per_frame.shape[1]
            bounds = _window_bounds(per_frame.shape[0], K)
            sub_vecs = []
            for (s, e) in bounds:
                if e <= s:
                    e = s + 1
                sub_vecs.append(per_frame[s:e].mean(dim=0))  # [Dv]
        else:
            zero_guard += 1
            if dv is None:
                dv = vision_model.config.hidden_size
            sub_vecs = [torch.zeros(dv, dtype=torch.float32) for _ in range(K)]

        for k in range(K):
            subclip_img_feats.append(sub_vecs[k])
            subclip_parent.append(parent_idx)
            labels.append(item["label"])
        video_ids.append(vid)

        if (parent_idx + 1) % 50 == 0:
            print(
                "  [{}] processed {}/{} videos (zero-vector guards so far: {})".format(
                    split_name, parent_idx + 1, len(items), zero_guard
                )
            )

    subclip_img_feats = torch.stack(subclip_img_feats, dim=0).float()  # [TotalSub, Dv]
    subclip_parent = torch.tensor(subclip_parent, dtype=torch.long)     # [TotalSub]

    if all(isinstance(l, int) or (isinstance(l, float) and float(l).is_integer()) for l in labels):
        labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    else:
        labels_t = torch.tensor([float(l) for l in labels], dtype=torch.float32)

    return video_ids, subclip_img_feats, subclip_parent, labels_t, dv, zero_guard


def main(args):
    device = torch.device(args.device)

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading CLIP vision model: {}".format(args.model))
    vision_model = CLIPVisionModel.from_pretrained(args.model)
    preprocess = CLIPProcessor.from_pretrained(args.model)
    vision_model.to(device).eval()

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
            "Processing split '{}' ({} videos, K={}, M={}) -> outname '{}'".format(
                split, len(items), args.num_subclips, args.num_frames, outname
            )
        )

        video_ids, subclip_img_feats, subclip_parent, labels_t, dv, zero_guard = process_split(
            items, split, args, preprocess, vision_model, device
        )

        save_obj = {
            "video_ids": video_ids,
            "subclip_img_feats": subclip_img_feats,
            "subclip_parent": subclip_parent,
            "labels": labels_t,
            "num_subclips": args.num_subclips,
            "num_frames": args.num_frames,
        }
        model_tag = str(args.model).replace("/", "_")
        out_path = os.path.join(
            out_dir,
            "{}_subclipK{}_{}_HF.pt".format(outname, args.num_subclips, model_tag),
        )
        torch.save(save_obj, out_path)

        print(
            "Saved '{}': V={}, TotalSub={}, Dv={}, zero-vector videos={} -> {}".format(
                outname,
                len(video_ids),
                subclip_img_feats.shape[0],
                dv,
                zero_guard,
                out_path,
            )
        )


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
