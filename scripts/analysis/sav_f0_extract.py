"""SAV (C2) F-G0 extraction: frozen Qwen2.5-VL-7B per-head + pooled read-out cache.

Authority: research-wiki/experiments/exp-sav-f0.md (Rev-2a, APPROVED), §3 + §4 F-G0.

For every TRAIN+VAL video of HateMM / MHC / MHC_zh this runs the SAME frozen forward
as the banked encoder-swap extractor (src/utils/generate_VideoMLLM_embedding_HF.py),
mirroring its decord->PyAV 8-frame sampler, message build, processor and span pooling
VERBATIM (line cites below), and additionally taps a forward-pre-hook on each of the
28 text-decoder self_attn.o_proj modules to capture the value-weighted PER-HEAD output
at the final token (SAV h^{l,m}, arXiv 2412.00142v3) — NOT output_attentions.

Per video it emits (IMG stream = PROBE_STREAM, the mean-pooling dilution target):
  img_head_final    [28,28,128] fp32 : per-(layer,head) final-token attention vector
  img_head_spanmean [28,28,128] fp32 : per-(layer,head) span-mean over the "prefix" span
                                       (same span as the cached img_feats) — for C-sparse
  img_hidden_final  [3584]      fp32 : final-token FULL last-layer hidden state — for C-pos
  img_pooled        [3584]      fp32 : L2-normed span-mean last-layer hidden (== cached img)
  text_pooled       [3584]      fp32 : L2-normed response-tail last-layer hidden (== cached text)
The last two are the pooled read-outs the two-tier reproduction guard checks; the text
stream is forward-passed but NOT per-head cached (storage: one stream, two variants, ~2.4GB).

Deterministic, resumable per-video (skip-if-exists), atomic writes, fail-closed on any
missing input; a per-(dataset,split) manifest with the symlink `followed_target` audit is
written only after every video in the split is cached.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sav_f0_common as C  # noqa: E402  (in-repo sibling module; top-level import)


# --------------------------------------------------------------------------- #
# Frame sampler — mirrored VERBATIM from generate_VideoMLLM_embedding_HF.py    #
#   _sample_frame_indices :146-152 ; _decode_with_decord :155-167 ;           #
#   _decode_with_pyav :170-205 ; load_video_frames :208-235                    #
# --------------------------------------------------------------------------- #
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
    if not os.path.exists(video_path):
        # fail-closed: a missing INPUT video is a hard error (not a silent zero-guard),
        # because the banked cache was produced from these exact symlinks.
        raise FileNotFoundError("missing video file: {}".format(video_path))
    frames = None
    try:
        frames = _decode_with_decord(video_path, num_frames)
    except Exception as e:  # noqa: BLE001
        print("[WARN] decord failed for {} ({}); trying PyAV.".format(video_path, repr(e)), flush=True)
        frames = None
    if frames is None:
        try:
            frames = _decode_with_pyav(video_path, num_frames)
        except Exception as e:  # noqa: BLE001
            print("[WARN] PyAV failed for {} ({}).".format(video_path, repr(e)), flush=True)
            frames = None
    if not frames:
        return None, False
    return frames, True


def _build_messages(frames, instruction):
    # generate_VideoMLLM_embedding_HF.py:241-251
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": instruction},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# Per-head capture buffer (forward-pre-hook on the 28 decoder o_proj inputs)   #
# --------------------------------------------------------------------------- #
class HeadCapture:
    """Captures o_proj INPUT (concatenated per-head value-weighted output) per layer."""

    def __init__(self):
        self.enabled = False
        self.buf = {}  # layer_idx -> [seq, HIDDEN] tensor (GPU, bf16/fp32)

    def reset(self):
        self.buf = {}

    def make_hook(self, layer_idx):
        def _hook(module, args):
            if not self.enabled:
                return None
            inp = args[0]  # [batch, seq, HIDDEN]
            assert inp.dim() == 3 and inp.shape[0] == 1 and inp.shape[-1] == C.HIDDEN, (
                "o_proj input shape unexpected at layer {}: {}".format(layer_idx, tuple(inp.shape))
            )
            self.buf[layer_idx] = inp[0].detach()
            return None
        return _hook


def register_head_hooks(model, capture):
    """Attach a forward-pre-hook to EXACTLY the 28 text-decoder self_attn.o_proj modules."""
    found = []
    for name, module in model.named_modules():
        if name.endswith("self_attn.o_proj"):
            # order by the integer layer index in '...layers.<i>.self_attn.o_proj'
            try:
                li = int(name.split("layers.")[1].split(".")[0])
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("cannot parse layer index from o_proj name '{}': {}".format(name, e))
            found.append((li, name, module))
    found.sort(key=lambda t: t[0])
    assert len(found) == C.NUM_LAYERS, (
        "expected {} text-decoder o_proj modules, found {} ({})".format(
            C.NUM_LAYERS, len(found), [n for _, n, _ in found]
        )
    )
    handles = []
    for li, name, module in found:
        in_features = getattr(module, "in_features", None)
        assert in_features == C.HIDDEN, (
            "o_proj[{}] in_features={} != HIDDEN {} (name={})".format(li, in_features, C.HIDDEN, name)
        )
        handles.append(module.register_forward_pre_hook(capture.make_hook(li)))
    return handles


# --------------------------------------------------------------------------- #
# One frozen forward: pooled read-out (+ optional per-head capture for img)    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_once(frames, instruction, processor, model, device, span, capture=None):
    """Return (pooled[3584] fp32 L2-normed, extras) mirroring _encode() :254-323.

    span == "prefix"  -> img_feats span (visual+instruction up to the assistant header).
    span == "response"-> text_feats span (assistant-header tail .. end).
    If `capture` is provided it is enabled for this forward; extras then also carries the
    per-head final-token / span-mean tensors and the final-token full-hidden vector.
    """
    messages = _build_messages(frames, instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    inputs = inputs.to(device)

    if capture is not None:
        capture.reset()
        capture.enabled = True
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    if capture is not None:
        capture.enabled = False

    last_hidden = out.hidden_states[-1][0]  # [seq, 3584]
    input_ids = inputs["input_ids"][0]
    assert last_hidden.shape[0] == input_ids.numel(), (
        "hidden/input_ids length mismatch: {} vs {}".format(last_hidden.shape[0], input_ids.numel())
    )
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]

    if span == "prefix":
        if len(positions) > 0:
            boundary = int(positions[-1].item())  # start of the assistant header
        else:
            boundary = last_hidden.shape[0]
        boundary = max(boundary, 1)
        pooled = last_hidden[:boundary].mean(dim=0)
        span_slice = slice(0, boundary)
    else:  # "response"
        if len(positions) > 0:
            start = int(positions[-1].item())
        else:
            start = max(last_hidden.shape[0] - 4, 0)
        start = min(start, last_hidden.shape[0] - 1)
        pooled = last_hidden[start:].mean(dim=0)
        span_slice = slice(start, last_hidden.shape[0])

    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=0).detach().cpu()

    extras = {"seq_len": int(last_hidden.shape[0]), "span_start": int(span_slice.start),
              "span_stop": int(span_slice.stop)}
    if capture is not None:
        # assemble [28, 28, 128] final-token and span-mean per-head tensors
        assert len(capture.buf) == C.NUM_LAYERS, (
            "captured {} layers, expected {}".format(len(capture.buf), C.NUM_LAYERS)
        )
        head_final = torch.empty(C.NUM_LAYERS, C.NUM_HEADS, C.HEAD_DIM, dtype=torch.float32)
        head_span = torch.empty(C.NUM_LAYERS, C.NUM_HEADS, C.HEAD_DIM, dtype=torch.float32)
        for li in range(C.NUM_LAYERS):
            t = capture.buf[li].float().view(-1, C.NUM_HEADS, C.HEAD_DIM)  # [seq,28,128]
            assert t.shape[0] == last_hidden.shape[0], (
                "layer {} seq {} != {}".format(li, t.shape[0], last_hidden.shape[0])
            )
            head_final[li] = t[-1].cpu()
            head_span[li] = t[span_slice].mean(dim=0).cpu()
        extras["img_head_final"] = head_final
        extras["img_head_spanmean"] = head_span
        extras["img_hidden_final"] = last_hidden[-1].float().detach().cpu()
        capture.reset()
    return pooled, extras


# --------------------------------------------------------------------------- #
# Extraction driver                                                            #
# --------------------------------------------------------------------------- #
def process_split(dataset, split, processor, model, device, capture, limit=0):
    items = C.read_gt(dataset, split)
    if limit and limit > 0:
        items = items[:limit]
    exp = C.EXPECTED_COUNTS[dataset][split]
    if not (limit and limit > 0):
        assert len(items) == exp, (
            "gt count drift {}/{}: got {} expected {}".format(dataset, split, len(items), exp)
        )
    out_dir = C.extract_split_dir(dataset, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    symlink_audit = []
    zero_guard_ids = []
    done = 0
    skipped = 0
    t0 = time.time()
    for n, item in enumerate(items):
        vid = item["id"]
        vp = C.video_path(dataset, vid)
        vps = str(vp)
        followed = os.path.realpath(vps) if os.path.exists(vps) else ""
        symlink_audit.append({
            "id": vid,
            "path": os.path.relpath(vps, C.REPO_ROOT),
            "is_symlink": os.path.islink(vps),
            "followed_target": followed,
            "followed_target_in_repo": bool(followed) and (
                followed == str(C.REPO_ROOT) or followed.startswith(str(C.REPO_ROOT) + os.sep)
            ),
        })

        op = C.extract_video_path(dataset, split, vid)
        if op.exists():  # resumable: skip already-cached videos
            skipped += 1
            done += 1
            continue

        frames, ok = load_video_frames(vps, C.NUM_FRAMES)
        if not ok:
            # mirror the banked extractor's zero-vector guard for undecodable videos
            zero_guard_ids.append(vid)
            payload = {
                "id": vid, "label": int(item["label"]), "ok": False,
                "img_pooled": torch.zeros(C.HIDDEN, dtype=torch.float32),
                "text_pooled": torch.zeros(C.HIDDEN, dtype=torch.float32),
                "img_hidden_final": torch.zeros(C.HIDDEN, dtype=torch.float32),
                "img_head_final": torch.zeros(C.NUM_LAYERS, C.NUM_HEADS, C.HEAD_DIM, dtype=torch.float32),
                "img_head_spanmean": torch.zeros(C.NUM_LAYERS, C.NUM_HEADS, C.HEAD_DIM, dtype=torch.float32),
                "meta": {"model": C.MODEL_ID, "num_frames": C.NUM_FRAMES, "max_pixels": C.MAX_PIXELS,
                         "stream": C.PROBE_STREAM, "followed_target": followed},
            }
            C.atomic_torch_save(op, payload)
            done += 1
            continue

        img_pooled, img_extras = forward_once(
            frames, C.IMG_INSTRUCTION, processor, model, device, span="prefix", capture=capture)
        title = item.get("title", "")
        transcript = item.get("text", "")
        text_prompt = (
            C.TEXT_INSTRUCTION
            + "\nTitle: " + (title if title else "(none)")
            + "\nTranscript: " + (transcript if transcript else "(none)")
        )
        text_pooled, _ = forward_once(
            frames, text_prompt, processor, model, device, span="response", capture=None)

        payload = {
            "id": vid,
            "label": int(item["label"]),
            "ok": True,
            "img_pooled": img_pooled,
            "text_pooled": text_pooled,
            "img_hidden_final": img_extras["img_hidden_final"],
            "img_head_final": img_extras["img_head_final"],
            "img_head_spanmean": img_extras["img_head_spanmean"],
            "meta": {"model": C.MODEL_ID, "num_frames": C.NUM_FRAMES, "max_pixels": C.MAX_PIXELS,
                     "stream": C.PROBE_STREAM, "seq_len": img_extras["seq_len"],
                     "span_start": img_extras["span_start"], "span_stop": img_extras["span_stop"],
                     "followed_target": followed},
        }
        C.atomic_torch_save(op, payload)
        done += 1
        if (n + 1) % 20 == 0:
            rate = (n + 1) / max(time.time() - t0, 1e-6)
            print("  [{}/{}] {}/{} done (skipped {}), {:.2f} vid/s".format(
                dataset, split, n + 1, len(items), skipped, rate), flush=True)

    manifest = {
        "schema": "sav_f0_extract_manifest_v1",
        "dataset": dataset,
        "split": split,
        "outname": C.SPLIT_TO_OUTNAME[split],
        "n": len(items),
        "n_expected": exp if not (limit and limit > 0) else len(items),
        "n_cached": done,
        "n_skipped_resumed": skipped,
        "limit": int(limit),
        "zero_guard_ids": zero_guard_ids,
        "n_zero_guard": len(zero_guard_ids),
        "symlink_audit": symlink_audit,
        "symlink_audit_sha256": C.sha256_obj(symlink_audit),
        "complete": bool(done == len(items)),
        "model": C.MODEL_ID,
        "num_frames": C.NUM_FRAMES,
        "max_pixels": C.MAX_PIXELS,
        "stream": C.PROBE_STREAM,
    }
    C.atomic_write_json(C.extract_manifest_path(dataset, split), manifest)
    print("[manifest] {}/{}: complete={} n={} zero_guard={} -> {}".format(
        dataset, split, manifest["complete"], manifest["n"], manifest["n_zero_guard"],
        C.extract_manifest_path(dataset, split)), flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="SAV F-G0 per-head + pooled extraction (frozen Qwen2.5-VL).")
    ap.add_argument("--datasets", type=str, default=",".join(C.DATASETS))
    ap.add_argument("--splits", type=str, default=",".join(C.SPLITS))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="debug/smoke: first N items/split only")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)

    print("[extract] loading {} (bf16, {}) ...".format(C.MODEL_ID, C.ATTN_IMPL), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, torch_dtype=C.TORCH_DTYPE, attn_implementation=C.ATTN_IMPL, device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(C.MODEL_ID, max_pixels=C.MAX_PIXELS)

    # geometry cross-check vs config (fail-closed)
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    nl = getattr(cfg, "num_hidden_layers", None) or getattr(text_cfg, "num_hidden_layers", None)
    nh = getattr(cfg, "num_attention_heads", None) or getattr(text_cfg, "num_attention_heads", None)
    hs = getattr(cfg, "hidden_size", None) or getattr(text_cfg, "hidden_size", None)
    assert (nl, nh, hs) == (C.NUM_LAYERS, C.NUM_HEADS, C.HIDDEN), (
        "geometry drift: config (L,H,hidden)=({},{},{}) != pinned ({},{},{})".format(
            nl, nh, hs, C.NUM_LAYERS, C.NUM_HEADS, C.HIDDEN)
    )

    capture = HeadCapture()
    handles = register_head_hooks(model, capture)
    print("[extract] registered {} o_proj hooks; head geometry {}x{}x{} = {} head positions".format(
        len(handles), C.NUM_LAYERS, C.NUM_HEADS, C.HEAD_DIM, C.NUM_HEAD_POSITIONS), flush=True)

    try:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        all_ok = True
        for ds in datasets:
            for sp in splits:
                m = process_split(ds, sp, processor, model, device, capture, limit=args.limit)
                all_ok = all_ok and m["complete"]
        print("[extract] ALL DONE complete={}".format(all_ok), flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
