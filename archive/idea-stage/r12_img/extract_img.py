#!/usr/bin/env python
"""R12-IMG -- image-stream position readout extraction (IMAGE stream only).

Frozen design: idea-stage/R12_FREEZE.md section 2.3.

Thin fork of src/utils/generate_VideoMLLM_embedding_readout_HF.py.  Everything
that touches the model, the frame sampler, the prompt string (RO.IMG_INSTRUCTION),
the LoRA attachment, the cache contract and the pooling MATH is imported VERBATIM
from that module -- this file re-implements only (a) which position spans are pooled
and (b) the fact that only the IMAGE-stream forward is run.

Per item: ONE forward (RO.IMG_INSTRUCTION, 8 frames).  From that single forward, at
layer 28 only, it pools six spans:

  PRE  mean(h[0:hdr])              -- the DEPLOYED "prefix" span (~1000 positions)
  VIS  mean(h[0:v_end))            -- the vision block
  INS  mean(h[v_end:hdr))          -- the instruction-text positions
  STD  std(h[0:hdr], unbiased=False) -- elementwise second moment over the SAME span
  RA   mean(h[S])                  -- fixed random position subset, |S| = v_end
  RB   mean(h[[0,hdr) \\ S])        -- its complement

where v_end = 1 + index of the LAST <|video_pad|> and hdr = index of the LAST
<|im_start|>.  If v_end >= hdr the degenerate guard sets v_end = 0 (identical to the
R10 fork).

BELT B1 (freeze 2.3): on the first 12 videos of each split, PRE must equal the frozen
RO._pool_span(..., span="prefix", ...) computed on the same forward with max abs diff
EXACTLY 0.0.  Failure raises and HALTs.

Output tag suffix is 'ip', never used before, so no banked cache can be clobbered.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))

import generate_VideoMLLM_embedding_readout_HF as RO  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

LAYER = RO.LAYER_FINAL  # 28 -- pinned, no sweep
SPANS = ["PRE", "VIS", "INS", "STD", "RA", "RB"]
RNG_BASE = 20261218  # frozen in R12_FREEZE.md 2.3


def _spans_from_hidden(last_hidden, input_ids, im_start_id, video_pad_id, item_index):
    """Return ({span_name: [D] float32 L2-normed cpu tensor}, (T, v_end, hdr))."""
    T = last_hidden.shape[0]
    pos_im = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    hdr = int(pos_im[-1].item()) if len(pos_im) else T
    hdr = max(min(hdr, T), 1)

    pos_v = (input_ids == video_pad_id).nonzero(as_tuple=True)[0]
    v_end = int(pos_v[-1].item()) + 1 if len(pos_v) else 0
    if v_end >= hdr:  # degenerate -- no instruction text between vision block and header
        v_end = 0

    pre = last_hidden[:hdr]
    out = {}
    out["PRE"] = pre.mean(dim=0)
    out["STD"] = pre.float().std(dim=0, unbiased=False)
    out["VIS"] = last_hidden[:v_end].mean(dim=0) if v_end > 0 else pre.mean(dim=0)
    out["INS"] = last_hidden[v_end:hdr].mean(dim=0) if hdr > v_end else pre.mean(dim=0)

    # Fixed random positional split with the SAME block sizes as VIS / INS.
    k = v_end if 0 < v_end < hdr else max(1, hdr // 2)
    rng = np.random.default_rng(RNG_BASE + int(item_index))
    perm = rng.permutation(hdr)
    sel = np.sort(perm[:k])
    com = np.sort(perm[k:])
    idx_a = torch.as_tensor(sel, dtype=torch.long, device=last_hidden.device)
    idx_b = torch.as_tensor(com if len(com) else sel, dtype=torch.long,
                            device=last_hidden.device)
    out["RA"] = pre.index_select(0, idx_a).mean(dim=0)
    out["RB"] = pre.index_select(0, idx_b).mean(dim=0)

    for k2 in out:
        v = out[k2].float()
        out[k2] = torch.nn.functional.normalize(v, p=2, dim=0).detach().cpu()
    return out, (T, v_end, hdr)


@torch.no_grad()
def encode_img(frames, instruction, processor, model, device, item_index, belt):
    """ONE image-stream forward -> ({span: [D]}, (T, v_end, hdr))."""
    messages = RO._build_messages(frames, instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    input_ids = inputs["input_ids"][0]
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    lh = out.hidden_states[LAYER][0]
    assert lh.shape[0] == input_ids.numel(), (
        "hidden/input_ids length mismatch at layer %d: %d vs %d"
        % (LAYER, lh.shape[0], input_ids.numel()))

    res, stat = _spans_from_hidden(lh, input_ids, im_start_id, video_pad_id, item_index)

    if belt:  # BELT B1 -- frozen deployed pooling on the SAME forward
        ref = RO._pool_span(lh, input_ids, "prefix", im_start_id)
        diff = float((res["PRE"] - ref).abs().max().item())
        if diff != 0.0:
            raise RuntimeError("BELT B1 FAILED: PRE vs deployed prefix span "
                               "max abs diff = %r (must be exactly 0.0)" % diff)
    return res, stat


def process_split(items, split_name, args, processor, model, device):
    d = model.config.hidden_size
    ids, labels = [], []
    acc = {s: [] for s in SPANS}
    stats = []
    zero_guard = 0
    belts_ok = 0

    video_root = os.path.join(args.video_dir, args.dataset, "All")
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))
        frames, ok = RO.load_video_frames(video_path, args.num_frames)

        if ok:
            belt = n < 12
            res, st = encode_img(frames, RO.IMG_INSTRUCTION, processor, model, device,
                                 n, belt)
            if belt:
                belts_ok += 1
            stats.append(st)
            for s in SPANS:
                acc[s].append(res[s])
        else:
            zero_guard += 1
            for s in SPANS:
                acc[s].append(torch.zeros(d, dtype=torch.float32))

        ids.append(vid)
        labels.append(item["label"])
        if (n + 1) % 25 == 0:
            print("  [%s] %d/%d (zero guards %d)" % (split_name, n + 1, len(items),
                                                     zero_guard), flush=True)

    print("  [%s] BELT B1 passed on %d items (max abs diff 0.0)" % (split_name, belts_ok),
          flush=True)
    labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    packed = {s: torch.stack(acc[s], 0).float() for s in SPANS}
    return ids, packed, labels_t, d, zero_guard, stats, belts_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC_zh")
    ap.add_argument("--EXP_FOLDER", default=os.path.join(ROOT, "data", "CLIP_Embedding"))
    ap.add_argument("--gt_dir", default=os.path.join(ROOT, "data", "gt"))
    ap.add_argument("--video_dir", default=os.path.join(ROOT, "data", "video"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--lora_dir", default="")
    ap.add_argument("--out_model_base_tag", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_suffix", default="ip")
    ap.add_argument("--smoke", action="store_true",
                    help="2 items per split, prints wall clock / belt only, writes nothing.")
    args = ap.parse_args()
    print(args, flush=True)

    if "ip" not in args.out_suffix:
        raise RuntimeError("out_suffix must contain 'ip' (clobber guard)")

    device = torch.device(args.device)
    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading %s" % args.model, flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    if args.lora_dir.strip():
        from peft import PeftModel
        print("Attaching + merging LoRA: %s" % args.lora_dir, flush=True)
        model = PeftModel.from_pretrained(model, args.lora_dir.strip())
        model = model.merge_and_unload()
    else:
        print("FROZEN base (no LoRA).", flush=True)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    assert model.config.num_hidden_layers == RO.LAYER_FINAL

    import time
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        outname = RO.SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "%s.jsonl" % split)
        items = RO.read_gt(gt_path)
        if args.smoke:
            items = items[:2]
        print("split '%s' (%d items) -> %s" % (split, len(items), outname), flush=True)
        t0 = time.time()
        ids, packed, labels_t, d, zg, stats, belts = process_split(
            items, split, args, processor, model, device)
        el = time.time() - t0
        nan = int(sum(int(torch.isnan(packed[s]).any()) for s in SPANS))
        if args.smoke:
            print("SMOKE split=%s n=%d wall=%.1fs belts=%d nan_flag=%d (nothing written)"
                  % (split, len(ids), el, belts, nan), flush=True)
            continue

        obj = {"ids": [ids], "labels": labels_t,
               "spans": {str(LAYER): {s: packed[s] for s in SPANS}},
               "meta": {"layer": LAYER, "spans": SPANS, "zero_guard": zg, "d": d,
                        "rng_base": RNG_BASE, "belts_passed_per_split": belts,
                        "freeze": "idea-stage/R12_FREEZE.md 2.3"}}
        op = os.path.join(out_dir, "%s_%s-%s.pt" % (outname, args.out_model_base_tag,
                                                    args.out_suffix))
        torch.save(obj, op)
        arr = np.array(stats) if stats else np.zeros((1, 3))
        print("saved %s  N=%d zero=%d nan=%d  span median: total=%.1f v_end=%.1f hdr=%.1f "
              "(instruction span len median %.1f)"
              % (op, len(ids), zg, nan, np.median(arr[:, 0]), np.median(arr[:, 1]),
                 np.median(arr[:, 2]), np.median(arr[:, 2] - arr[:, 1])), flush=True)
        with open(os.path.join(out_dir, "IPSTATS_%s_%s.json" % (args.dataset, outname)),
                  "w") as f:
            json.dump({"n": len(stats),
                       "median_total": float(np.median(arr[:, 0])),
                       "median_v_end": float(np.median(arr[:, 1])),
                       "median_hdr": float(np.median(arr[:, 2])),
                       "median_insspan": float(np.median(arr[:, 2] - arr[:, 1])),
                       "belts_passed": belts}, f, indent=1)


if __name__ == "__main__":
    main()
