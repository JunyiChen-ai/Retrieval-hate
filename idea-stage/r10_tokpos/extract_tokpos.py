#!/usr/bin/env python
"""R10 Task B -- token-position readout extraction (TEXT stream only).

Frozen design: idea-stage/R10_TOKPOS_FREEZE.md section 2.2.

Thin fork of src/utils/generate_VideoMLLM_embedding_readout_HF.py.  Everything
that touches the model, the frame sampler, the prompt strings, the cache contract
and the pooling MATH is imported VERBATIM from that module -- this file
re-implements only (a) which token spans are pooled and (b) the fact that only the
text-stream forward is run.

Per item: ONE forward (baseline text prompt).  From that single forward, at layers
28 and 24, it pools six spans:

  A0   [hdr_start, end)              -- the DEPLOYED span (assistant header, 3-4 tokens)
  TXT  [v_end, hdr_start)            -- title/transcript/instruction content positions
  S1..S4                              -- TXT cut into 4 contiguous equal segments
  ALL  [0, seq_len)                  -- naive all-position pool (reference point)

where v_end = 1 + index of the LAST <|video_pad|> token and hdr_start = index of the
LAST <|im_start|>.

img_feats are NOT recomputed: they are carried over verbatim from the banked
{split}_<BASE>-ro_L{28,24}.pt caches by id (see build_arms.py).  This file writes only
the text-side spans, into {split}_<BASE>-tp.pt with a dict of span->{layer->tensor}.

Output tag suffix is 'tp', never the deployed tag, so no banked cache can be clobbered.
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

LAYERS = [RO.LAYER_FINAL, RO.LAYER_MID]  # 28, 24 -- pinned, no sweep
SPANS = ["A0", "TXT", "S1", "S2", "S3", "S4", "ALL"]
N_SEG = 4


def _spans_from_hidden(last_hidden, input_ids, im_start_id, video_pad_id):
    """Return {span_name: [D] float32 L2-normed cpu tensor} for one layer."""
    T = last_hidden.shape[0]
    pos_im = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    hdr = int(pos_im[-1].item()) if len(pos_im) else max(T - 4, 0)
    hdr = min(hdr, T - 1)

    pos_v = (input_ids == video_pad_id).nonzero(as_tuple=True)[0]
    v_end = int(pos_v[-1].item()) + 1 if len(pos_v) else 0
    if v_end >= hdr:  # degenerate -- no text content between vision block and header
        v_end = 0

    out = {}
    out["A0"] = last_hidden[hdr:].mean(dim=0)
    out["ALL"] = last_hidden.mean(dim=0)
    txt = last_hidden[v_end:hdr]
    out["TXT"] = txt.mean(dim=0)
    bounds = np.array_split(np.arange(txt.shape[0]), N_SEG)
    for k, b in enumerate(bounds):
        name = "S%d" % (k + 1)
        out[name] = txt[int(b[0]):int(b[-1]) + 1].mean(dim=0) if len(b) else txt.mean(dim=0)

    for k in out:
        v = out[k].float()
        out[k] = torch.nn.functional.normalize(v, p=2, dim=0).detach().cpu()
    return out, (T, v_end, hdr)


@torch.no_grad()
def encode_tokpos(frames, instruction, processor, model, device):
    """ONE text-stream forward -> {layer: {span: [D]}}, plus the span decomposition."""
    messages = RO._build_messages(frames, instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    inputs = inputs.to(device)

    out = model(**inputs, output_hidden_states=True, use_cache=False)
    input_ids = inputs["input_ids"][0]
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    res = {}
    stat = None
    for L in LAYERS:
        lh = out.hidden_states[L][0]
        assert lh.shape[0] == input_ids.numel(), (
            "hidden/input_ids length mismatch at layer %d: %d vs %d"
            % (L, lh.shape[0], input_ids.numel()))
        res[L], stat = _spans_from_hidden(lh, input_ids, im_start_id, video_pad_id)
    return res, stat


def process_split(items, split_name, args, processor, model, device):
    d = model.config.hidden_size
    ids, labels = [], []
    acc = {L: {s: [] for s in SPANS} for L in LAYERS}
    stats = []
    zero_guard = 0

    video_root = os.path.join(args.video_dir, args.dataset, "All")
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    for n, item in enumerate(items):
        vid = item["id"]
        video_path = os.path.join(video_root, "{}.mp4".format(vid))
        frames, ok = RO.load_video_frames(video_path, args.num_frames)
        title = item.get("title", "")
        transcript = item.get("text", "")

        if ok:
            text_prompt = (
                RO.TEXT_INSTRUCTION
                + "\nTitle: " + (title if title else "(none)")
                + "\nTranscript: " + (transcript if transcript else "(none)")
            )
            res, st = encode_tokpos(frames, text_prompt, processor, model, device)
            stats.append(st)
            for L in LAYERS:
                for s in SPANS:
                    acc[L][s].append(res[L][s])
        else:
            zero_guard += 1
            for L in LAYERS:
                for s in SPANS:
                    acc[L][s].append(torch.zeros(d, dtype=torch.float32))

        ids.append(vid)
        labels.append(item["label"])
        if (n + 1) % 25 == 0:
            print("  [%s] %d/%d (zero guards %d)" % (split_name, n + 1, len(items), zero_guard),
                  flush=True)

    labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
    packed = {L: {s: torch.stack(acc[L][s], 0).float() for s in SPANS} for L in LAYERS}
    return ids, packed, labels_t, d, zero_guard, stats


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
    ap.add_argument("--out_suffix", default="tp")
    args = ap.parse_args()
    print(args, flush=True)

    if "tp" not in args.out_suffix:
        raise RuntimeError("out_suffix must contain 'tp' (clobber guard)")

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

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        outname = RO.SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "%s.jsonl" % split)
        items = RO.read_gt(gt_path)
        print("split '%s' (%d items) -> %s" % (split, len(items), outname), flush=True)
        ids, packed, labels_t, d, zg, stats = process_split(
            items, split, args, processor, model, device)

        obj = {"ids": [ids], "labels": labels_t,
               "spans": {str(L): {s: packed[L][s] for s in SPANS} for L in LAYERS},
               "meta": {"layers": LAYERS, "spans": SPANS, "n_seg": N_SEG,
                        "zero_guard": zg, "d": d,
                        "freeze": "idea-stage/R10_TOKPOS_FREEZE.md 2.2"}}
        op = os.path.join(out_dir, "%s_%s-%s.pt" % (outname, args.out_model_base_tag,
                                                    args.out_suffix))
        torch.save(obj, op)
        arr = np.array(stats) if stats else np.zeros((1, 3))
        print("saved %s  N=%d zero=%d  span median: total=%.1f v_end=%.1f hdr=%.1f "
              "(text span len median %.1f)"
              % (op, len(ids), zg, np.median(arr[:, 0]), np.median(arr[:, 1]),
                 np.median(arr[:, 2]), np.median(arr[:, 2] - arr[:, 1])), flush=True)
        with open(os.path.join(out_dir, "TPSTATS_%s_%s.json" % (args.dataset, outname)),
                  "w") as f:
            json.dump({"n": len(stats),
                       "median_total": float(np.median(arr[:, 0])),
                       "median_v_end": float(np.median(arr[:, 1])),
                       "median_hdr": float(np.median(arr[:, 2])),
                       "median_textspan": float(np.median(arr[:, 2] - arr[:, 1])),
                       "min_textspan": float(arr[:, 2].min() - arr[:, 1].max())}, f, indent=1)


if __name__ == "__main__":
    main()
