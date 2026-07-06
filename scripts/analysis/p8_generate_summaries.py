#!/usr/bin/env python
"""P8 — MLLM ≤60-word evidence-dense summaries + single-chunk CLIP text caches.

Stage 1 (Qwen2.5-VL-7B, text-only, greedy): condense each video's Title+Transcript (the `text`
field of data/gt/<DS>/<split>.jsonl) into ≤60 words preserving WHO is targeted / WHAT could be
hateful/offensive / topic. Store data/Summaries/<DS>/<split>.jsonl {id,label,orig_text,summary}.

Stage 2 (CLIP text tower): build drop-in text-channel caches (img_feats / ids / labels copied
VERBATIM from the floor whole-video cache; only text_feats replaced), single-chunk encoded
(≤75 content tokens, ONE forward, pooler_output — the compression, NOT chunk-mean):
  <split>_p8sum_HF.pt     B  : text = summary                     (--model p8sum_HF)
  <split>_p8trunc_HF.pt   C  : text = first-70-token raw (no MLLM) (--model p8trunc_HF)  [rent test]
  <split>_p8concat_HF.pt  D  : text = [l2n(raw chunk-mean) | l2n(summary)] / sqrt(2), dim 1536

Label-free preprocessing (labels never read for summarization) → no leakage. Never mutates the
floor caches; only writes data/Summaries/** and the p8* cache tags.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

ROOT = "/data/jehc223/RGCL"
MODEL_TAG = "openai_clip-vit-large-patch14-336_HF"
SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen", "test": "test_seen"}
MAX_INPUT_CHARS = 6000  # cap the Qwen input (HateMM transcripts reach 80k chars)

SUM_SYS = ("You compress short-video transcripts. You output only the condensed text, no "
           "commentary, labels, or preamble.")
SUM_INSTR = """Condense the following short-video TITLE + TRANSCRIPT into AT MOST 60 words,
preserving WHO is targeted, WHAT is said or shown that could be hateful or offensive, and the
overall topic. Output ONLY the condensed text.

TITLE + TRANSCRIPT:
"""


def read_gt(ds, split):
    recs = []
    with open(os.path.join(ROOT, "data/gt", ds, split + ".jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs.append({"id": str(r["id"]),
                             "text": "" if r.get("text") is None else str(r["text"]),
                             "label": int(r["label"])})
    return recs


@torch.no_grad()
def summarize(text, processor, model, device, max_new_tokens=110):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SUM_SYS}]},
        {"role": "user", "content": [{"type": "text", "text": SUM_INSTR + text[:MAX_INPUT_CHARS]}]},
    ]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=None, videos=None, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


@torch.no_grad()
def encode_single(text, tokenizer, text_model, device, max_content=75):
    """Single-chunk CLIP text: ≤max_content content tokens + BOS/EOS, one forward, pooler."""
    text = text or ""
    content = tokenizer(text, add_special_tokens=False)["input_ids"]
    truncated = len(content) > max_content
    content = content[:max_content]
    seq = []
    if tokenizer.bos_token_id is not None:
        seq.append(tokenizer.bos_token_id)
    seq += content
    if tokenizer.eos_token_id is not None:
        seq.append(tokenizer.eos_token_id)
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    out = text_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    return out.pooler_output.detach().cpu().float()[0], truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh,HateMM")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    device = torch.device(args.device)

    # ---------- stage 1: Qwen summaries ----------
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("Loading MLLM:", args.model, flush=True)
    qmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    qmodel.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    for ds in datasets:
        for split in ("train", "val", "test"):
            recs = read_gt(ds, split)
            if args.limit:
                recs = recs[: args.limit]
            outdir = os.path.join(ROOT, "data/Summaries", ds)
            os.makedirs(outdir, exist_ok=True)
            n_empty = 0
            rows = []
            for n, r in enumerate(recs):
                if not r["text"].strip():
                    summary = ""
                    n_empty += 1
                else:
                    summary = summarize(r["text"], processor, qmodel, device)
                rows.append(dict(id=r["id"], label=r["label"], orig_text=r["text"],
                                 summary=summary))
                if (n + 1) % 100 == 0:
                    print("  [{}/{}] {}/{} empty_src={}".format(
                        ds, split, n + 1, len(recs), n_empty), flush=True)
            with open(os.path.join(outdir, split + ".jsonl"), "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("[{}/{}] summaries={} empty_src={}".format(ds, split, len(rows), n_empty),
                  flush=True)

    del qmodel
    torch.cuda.empty_cache()

    # ---------- stage 2: single-chunk CLIP text caches ----------
    from transformers import CLIPTokenizer, CLIPTextModel
    print("Loading CLIP text tower:", args.clip_model, flush=True)
    tok = CLIPTokenizer.from_pretrained(args.clip_model)
    tmodel = CLIPTextModel.from_pretrained(args.clip_model).to(device).eval()

    for ds in datasets:
        ds_dir = os.path.join(ROOT, "data/CLIP_Embedding", ds)
        for split in ("train", "val", "test"):
            outname = SPLIT_TO_OUTNAME[split]
            floor_p = os.path.join(ds_dir, "{}_{}.pt".format(outname, MODEL_TAG))
            floor = torch.load(floor_p, map_location="cpu")
            floor_ids = [i for sub in floor["ids"] for i in sub]
            floor_text = floor["text_feats"].float()

            sums = {}
            for line in open(os.path.join(ROOT, "data/Summaries", ds, split + ".jsonl")):
                o = json.loads(line)
                sums[o["id"]] = o

            B, C, ntrunc_b, ntrunc_c, n_missing = [], [], 0, 0, 0
            for vid in floor_ids:
                s = sums.get(vid)
                if s is None:            # only in a LIMIT smoke; full run covers all ids
                    s = {"summary": "", "orig_text": ""}
                    n_missing += 1
                b, tb = encode_single(s["summary"], tok, tmodel, device, max_content=75)
                c, tc = encode_single(s["orig_text"], tok, tmodel, device, max_content=70)
                B.append(b); C.append(c); ntrunc_b += tb; ntrunc_c += tc
            if n_missing:
                print("[{}/{}] WARN {} floor ids missing a summary (smoke/limit?) -> empty".format(
                    ds, split, n_missing), flush=True)
            B = torch.stack(B, 0); C = torch.stack(C, 0)
            # D = concat[l2n(raw chunk-mean) | l2n(summary single-chunk)] / sqrt(2)
            D = torch.cat([F.normalize(floor_text, dim=1), F.normalize(B, dim=1)], dim=1) / (2 ** 0.5)

            for tag, text_feats in (("p8sum", B), ("p8trunc", C), ("p8concat", D)):
                d = dict(floor)                       # shallow copy: img/ids/labels shared
                d["text_feats"] = text_feats.float().contiguous()
                assert torch.equal(d["img_feats"], floor["img_feats"])
                assert d["ids"] == floor["ids"] and torch.equal(d["labels"], floor["labels"])
                op = os.path.join(ds_dir, "{}_{}_HF.pt".format(outname, tag))
                torch.save(d, op)
            print("[{}/{}] N={} textdim(B={},D={}) trunc(B={},C={}) -> p8sum/p8trunc/p8concat".format(
                ds, split, len(floor_ids), B.shape[1], D.shape[1], ntrunc_b, ntrunc_c), flush=True)


if __name__ == "__main__":
    main()
