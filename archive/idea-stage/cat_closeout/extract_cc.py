#!/usr/bin/env python
"""CAT close-out -- two-forward end-to-end extraction (BOTH streams).

Frozen design: idea-stage/CAT_CLOSEOUT_FREEZE.md sections 2.2 / 3.2 / 3.3.

Thin fork.  Everything that touches the model, the LoRA attachment, the frame
sampler, the prompt strings, the chat template and the pooling MATH is imported
VERBATIM:

  * image stream : RO._encode_readout(..., span="prefix", layers=[28])
                   -- the frozen DEPLOYED img read-out, used as the function object
  * text stream  : one forward, spans pooled by TP._spans_from_hidden
                   -- the frozen R10 span decomposition, used as the function object

This file re-implements only (a) that exactly two forwards are run per item and
(b) the item ORDER in which the split is traversed (--order reverse is Leg A's
perturbation; rows are permuted back to ground-truth order before saving).

Output: {split}_<BASE>-cc.pt  with
  ids     [ [id, ...] ]                     ground-truth order
  labels  LongTensor [N]
  img28   FloatTensor [N, D]                deployed prefix span, layer 28
  spans   {"28": {span: FloatTensor [N,D]}} A0/TXT/S1..S4/ALL, layer 28
  stats   {id: {"T":..,"v_end":..,"hdr":..}}
  meta    provenance

Suffix 'cc' is never used by any banked cache; a hard guard enforces it.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))
sys.path.insert(0, os.path.join(ROOT, "idea-stage", "r10_tokpos"))

import generate_VideoMLLM_embedding_readout_HF as RO  # noqa: E402
import extract_tokpos as TP  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

LAYER = RO.LAYER_FINAL  # 28, pinned
SPANS = TP.SPANS        # ["A0","TXT","S1","S2","S3","S4","ALL"] -- from the frozen module
BELT_N = 12


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def encode_text_spans(frames, instruction, processor, model, device, belt):
    """ONE text-stream forward -> ({span: [D]}, (T, v_end, hdr))."""
    messages = RO._build_messages(frames, instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    input_ids = inputs["input_ids"][0]
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    lh = out.hidden_states[LAYER][0]
    assert lh.shape[0] == input_ids.numel(), "hidden/input_ids length mismatch"
    sp, stat = TP._spans_from_hidden(lh, input_ids, im_start_id, video_pad_id)

    dev = None
    if belt:
        # BELT A1 (text side): the A0 span must equal the frozen deployed
        # RO._pool_span(span="response") on the SAME forward, max abs diff exactly 0.0.
        ref = RO._pool_span(lh, input_ids, "response", im_start_id)
        dev = float((sp["A0"] - ref).abs().max())
    return sp, stat, dev


@torch.no_grad()
def encode_img(frames, processor, model, device, max_pixels, belt):
    """ONE image-stream forward, frozen deployed prefix span at layer 28."""
    # BELT A1 (img side) is satisfied BY CONSTRUCTION: this call is the frozen deployed
    # read-out itself (RO._encode_readout -> RO._pool_span(span="prefix")), not a
    # re-implementation of it, so the max abs diff against the frozen path is 0.0
    # identically.  The frozen source files are sha256-recorded in the meta json.
    res = RO._encode_readout(frames, RO.IMG_INSTRUCTION, processor, model, device,
                             max_pixels, span="prefix", layers=[LAYER])
    v = res[LAYER]
    dev = 0.0 if belt else None
    return v, dev


def process_split(items_ordered, split_name, args, processor, model, device):
    d = model.config.hidden_size
    order_ids, labels_by_id = [], {}
    img_by_id, spans_by_id, stats_by_id = {}, {}, {}
    zero_guard = 0
    belts = {"text_a0_maxdiff": [], "img_maxdiff": []}
    video_root = os.path.join(args.video_dir, args.dataset, "All")
    t0 = time.time()

    for n, item in enumerate(items_ordered):
        vid = item["id"]
        belt = n < BELT_N
        frames, ok = RO.load_video_frames(
            os.path.join(video_root, "{}.mp4".format(vid)), args.num_frames)
        if ok:
            iv, idev = encode_img(frames, processor, model, device, args.max_pixels, belt)
            text_prompt = (RO.TEXT_INSTRUCTION
                           + "\nTitle: " + (item.get("title") or "(none)")
                           + "\nTranscript: " + (item.get("text") or "(none)"))
            sp, stat, tdev = encode_text_spans(frames, text_prompt, processor, model,
                                               device, belt)
            if belt:
                belts["img_maxdiff"].append(idev)
                belts["text_a0_maxdiff"].append(tdev)
        else:
            zero_guard += 1
            iv = torch.zeros(d, dtype=torch.float32)
            sp = {s: torch.zeros(d, dtype=torch.float32) for s in SPANS}
            stat = (0, 0, 0)

        img_by_id[vid] = iv
        spans_by_id[vid] = sp
        stats_by_id[vid] = {"T": int(stat[0]), "v_end": int(stat[1]), "hdr": int(stat[2])}
        labels_by_id[vid] = int(item["label"])
        order_ids.append(vid)
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            print("  PROGRESS [%s] %d/%d zero=%d elapsed=%.0fs eta=%.0fs"
                  % (split_name, n + 1, len(items_ordered), zero_guard, el,
                     el / (n + 1) * (len(items_ordered) - n - 1)), flush=True)

    mx_t = max(belts["text_a0_maxdiff"]) if belts["text_a0_maxdiff"] else 0.0
    mx_i = max(belts["img_maxdiff"]) if belts["img_maxdiff"] else 0.0
    print("  BELT A1 [%s] text A0 vs frozen _pool_span('response'): max abs diff %.3e "
          "(n=%d) | img vs frozen _pool_span('prefix'): %.3e"
          % (split_name, mx_t, len(belts["text_a0_maxdiff"]), mx_i), flush=True)
    if mx_t != 0.0 or mx_i != 0.0:
        raise SystemExit("HALT: BELT A1 failed on %s (text %.3e img %.3e)"
                         % (split_name, mx_t, mx_i))
    return (order_ids, img_by_id, spans_by_id, stats_by_id, labels_by_id, zero_guard,
            {"text_a0_maxdiff": mx_t, "img_maxdiff": mx_i, "n_belt": len(belts["img_maxdiff"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--EXP_FOLDER", default=os.path.join(ROOT, "data", "CLIP_Embedding"))
    ap.add_argument("--gt_dir", default=os.path.join(ROOT, "data", "gt"))
    ap.add_argument("--video_dir", default=os.path.join(ROOT, "data", "video"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--out_model_base_tag", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--num_frames", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--order", default="forward", choices=["forward", "reverse"])
    ap.add_argument("--out_suffix", default="cc")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE ONLY: process the first N items of each split. Writing "
                         "into the real EXP_FOLDER with --limit>0 is refused.")
    args = ap.parse_args()
    print(args, flush=True)

    if args.out_suffix != "cc":
        raise RuntimeError("out_suffix must be exactly 'cc' (clobber guard)")
    if args.limit and os.path.abspath(args.EXP_FOLDER) == os.path.join(ROOT, "data",
                                                                      "CLIP_Embedding"):
        raise RuntimeError("--limit>0 is smoke-only; point --EXP_FOLDER at a scratch dir")

    lora_sha = sha256_file(os.path.join(args.lora_dir, "adapter_model.safetensors"))
    print("LoRA %s adapter_model.safetensors sha256 = %s" % (args.lora_dir, lora_sha), flush=True)

    device = torch.device(args.device)
    out_dir = os.path.join(args.EXP_FOLDER, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading %s" % args.model, flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    from peft import PeftModel
    print("Attaching + merging LoRA: %s" % args.lora_dir, flush=True)
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model = model.merge_and_unload()
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    assert model.config.num_hidden_layers == RO.LAYER_FINAL

    src_sha = {
        "src/utils/generate_VideoMLLM_embedding_readout_HF.py": sha256_file(
            os.path.join(ROOT, "src", "utils",
                         "generate_VideoMLLM_embedding_readout_HF.py")),
        "idea-stage/r10_tokpos/extract_tokpos.py": sha256_file(
            os.path.join(ROOT, "idea-stage", "r10_tokpos", "extract_tokpos.py")),
    }
    print("frozen source sha256: %s" % json.dumps(src_sha), flush=True)

    allmeta = {"freeze": "idea-stage/CAT_CLOSEOUT_FREEZE.md 2.2/3.2",
               "frozen_source_sha256": src_sha,
               "lora_dir": args.lora_dir, "lora_sha256": lora_sha,
               "order": args.order, "layer": LAYER, "spans": SPANS,
               "num_frames": args.num_frames, "max_pixels": args.max_pixels,
               "model": args.model, "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        outname = RO.SPLIT_TO_OUTNAME[split]
        items = RO.read_gt(os.path.join(args.gt_dir, args.dataset, "%s.jsonl" % split))
        if args.limit and args.limit > 0:
            items = items[: args.limit]
        gt_ids = [it["id"] for it in items]
        traversal = items[::-1] if args.order == "reverse" else items
        print("split '%s' (%d items, order=%s) -> %s"
              % (split, len(items), args.order, outname), flush=True)

        (order_ids, img_by_id, spans_by_id, stats_by_id, labels_by_id, zg,
         belt) = process_split(traversal, split, args, processor, model, device)

        # BELT A2: permute back to ground-truth order and verify identity elementwise.
        assert sorted(order_ids) == sorted(gt_ids), "id set mismatch on %s" % split
        assert len(set(gt_ids)) == len(gt_ids), "duplicate ids in gt for %s" % split
        img = torch.stack([img_by_id[i] for i in gt_ids], 0).float()
        packed = {s: torch.stack([spans_by_id[i][s] for i in gt_ids], 0).float() for s in SPANS}
        labels = torch.tensor([labels_by_id[i] for i in gt_ids], dtype=torch.long)
        gt_labels = torch.tensor([int(it["label"]) for it in items], dtype=torch.long)
        assert torch.equal(labels, gt_labels), "BELT A2: label order mismatch on %s" % split
        print("  BELT A2 [%s] OK: %d rows restored to ground-truth order" % (split, len(gt_ids)),
              flush=True)

        obj = {"ids": [gt_ids], "labels": labels, "img28": img,
               "spans": {str(LAYER): packed},
               "stats": {i: stats_by_id[i] for i in gt_ids},
               "meta": {"layer": LAYER, "spans": SPANS, "zero_guard": zg,
                        "d": int(img.shape[1]), "order": args.order,
                        "lora_sha256": lora_sha, "belt": belt,
                        "freeze": "idea-stage/CAT_CLOSEOUT_FREEZE.md 2.2"}}
        op = os.path.join(out_dir, "%s_%s-%s.pt" % (outname, args.out_model_base_tag,
                                                    args.out_suffix))
        torch.save(obj, op)
        arr = np.array([[stats_by_id[i]["T"], stats_by_id[i]["v_end"], stats_by_id[i]["hdr"]]
                        for i in gt_ids], dtype=float)
        allmeta["splits"][outname] = {
            "path": os.path.relpath(op, ROOT), "n": len(gt_ids), "zero_guard": zg,
            "belt": belt,
            "median_total": float(np.median(arr[:, 0])),
            "median_v_end": float(np.median(arr[:, 1])),
            "median_hdr": float(np.median(arr[:, 2])),
            "median_textspan": float(np.median(arr[:, 2] - arr[:, 1]))}
        print("saved %s N=%d zero=%d median total=%.1f v_end=%.1f hdr=%.1f textspan=%.1f"
              % (op, len(gt_ids), zg, np.median(arr[:, 0]), np.median(arr[:, 1]),
                 np.median(arr[:, 2]), np.median(arr[:, 2] - arr[:, 1])), flush=True)

    mp = os.path.join(ROOT, "idea-stage", "cat_closeout",
                      "extract_meta_%s%s.json" % (args.dataset,
                                                  "_SMOKE" if args.limit else ""))
    json.dump(allmeta, open(mp, "w"), indent=1)
    print("wrote", mp, flush=True)


if __name__ == "__main__":
    main()
