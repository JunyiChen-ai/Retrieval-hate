#!/usr/bin/env python
"""P5 counterfactual hard-negative twins — generation + self-verdict flip gate + CLIP encode.

For each TRAIN positive-label (label==1) video of a dataset, the MLLM (Qwen2.5-VL-7B, greedy)
rewrites its title+transcript (the `text` field of data/gt/<DS>/train.jsonl) into a SANITIZED
counterfactual: same topic/style/length, hate content removed/neutralised. Then the SAME model
judges the ORIGINAL text HARMFUL and the SANITIZED text BENIGN (self-verdict flip). Failures
are regenerated at most once; a twin counts as a VERIFIED FLIP iff orig==HARMFUL and
sanitized==BENIGN. Finally the sanitized text is encoded with the EXACT chunk-mean CLIP-text
pipeline used to build the whole-video caches (generate_VideoCLIP_embedding_HF.encode_text), so
the twin = the anchor's REAL cached img_feats + this sanitized text embedding.

Outputs (per dataset):
  data/Counterfactual/<DS>/train_twins.jsonl        (one row per anchor; text + verdicts)
  data/CLIP_Embedding/<DS>/train_cftwin_<model>.pt  ({ids, text_feats[N,Dt], flipped[N], labels})

Only TRAIN positive-label videos are touched. Twins are used ONLY as training-time hard
negatives (never injected into eval memory). Reads gt + CLIP model; writes only the two paths.
"""
import argparse
import json
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from utils.generate_VideoCLIP_embedding_HF import encode_text  # exact chunk-mean pipeline

ROOT = "/data/jehc223/RGCL"
MAX_INPUT_CHARS = 4000  # cap the rewrite input; CLIP still chunk-means the full sanitized text

REWRITE_SYS = (
    "You rewrite short-video transcripts to remove hateful content while preserving "
    "everything else. You output only the rewritten transcript."
)
REWRITE_INSTR = """Below is the TITLE + TRANSCRIPT of a short video flagged as hateful or
offensive. Rewrite it into a SANITISED version that:
- keeps the SAME topic, subject matter, and speaking style,
- keeps a SIMILAR length,
- removes or neutralises ALL hateful, demeaning, slur-based, stereotyping, or attacking
  content toward any group or person,
- stays natural (do NOT add disclaimers, warnings, notes, or meta-commentary).
Output ONLY the rewritten transcript text, nothing else.

TITLE + TRANSCRIPT:
"""

JUDGE_SYS = (
    "You are a strict content classifier for a hateful-video detection system. You read a "
    "short-video title+transcript and decide if it is harmful. Be decisive."
)
JUDGE_INSTR = """Decide whether the following short-video TITLE + TRANSCRIPT is HARMFUL or BENIGN.
  HARMFUL = it attacks, demeans, dehumanises, mocks, or incites hatred against a protected
      group or its members (race, ethnicity, religion, gender, sexual orientation, disability,
      nationality, etc.) via slurs, stereotyping, dehumanisation, threats, or targeted insults.
  BENIGN = it does none of that.
Respond in EXACTLY one line:
VERDICT: <HARMFUL|BENIGN>

TITLE + TRANSCRIPT:
"""


def load_anchors(ds):
    """TRAIN label==1 videos -> list of {id,text,label}."""
    out = []
    with open(os.path.join(ROOT, "data/gt", ds, "train.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if int(r["label"]) == 1:
                out.append({"id": str(r["id"]),
                            "text": "" if r.get("text") is None else str(r["text"]),
                            "label": 1})
    return out


@torch.no_grad()
def qwen_generate(sys_prompt, user_text, processor, model, device, max_new_tokens, do_sample=False):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None, return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def judge(text, processor, model, device):
    raw = qwen_generate(JUDGE_SYS, JUDGE_INSTR + text[:MAX_INPUT_CHARS], processor, model,
                        device, max_new_tokens=8)
    up = raw.upper()
    if "HARMFUL" in up:
        return "HARMFUL"
    if "BENIGN" in up:
        return "BENIGN"
    return "BENIGN"


def rewrite(text, processor, model, device, seed_variation=False):
    # greedy by default; on the single regeneration we allow light sampling to escape a
    # non-flipping deterministic rewrite.
    return qwen_generate(REWRITE_SYS, REWRITE_INSTR + text[:MAX_INPUT_CHARS], processor, model,
                         device, max_new_tokens=512, do_sample=seed_variation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--clip_model", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    device = torch.device(args.device)

    # ---- stage 1+2: Qwen rewrite + self-verdict flip ----
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print("Loading MLLM:", args.model, flush=True)
    qmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    qmodel.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    twins = {}
    for ds in datasets:
        anchors = load_anchors(ds)
        if args.limit:
            anchors = anchors[: args.limit]
        outdir = os.path.join(ROOT, "data/Counterfactual", ds)
        os.makedirs(outdir, exist_ok=True)
        rows = []
        n_orig_harm = n_flip = n_regen = 0
        for n, a in enumerate(anchors):
            orig_v = judge(a["text"], processor, qmodel, device)
            san = rewrite(a["text"], processor, qmodel, device)
            san_v = judge(san, processor, qmodel, device)
            regen = False
            if not (orig_v == "HARMFUL" and san_v == "BENIGN"):
                # one regeneration attempt (light sampling) for a cleaner sanitisation
                san2 = rewrite(a["text"], processor, qmodel, device, seed_variation=True)
                san2_v = judge(san2, processor, qmodel, device)
                regen = True
                if san2_v == "BENIGN":
                    san, san_v = san2, san2_v
            flipped = (orig_v == "HARMFUL" and san_v == "BENIGN")
            n_orig_harm += int(orig_v == "HARMFUL")
            n_flip += int(flipped)
            n_regen += int(regen)
            rows.append(dict(id=a["id"], label=1, orig_text=a["text"], sanitized_text=san,
                             orig_verdict=orig_v, san_verdict=san_v, regen_used=regen,
                             flipped=flipped))
            if (n + 1) % 25 == 0:
                print("  [{}] {}/{} orig_harm={} flips={} regen={}".format(
                    ds, n + 1, len(anchors), n_orig_harm, n_flip, n_regen), flush=True)
        with open(os.path.join(outdir, "train_twins.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cond = (n_flip / n_orig_harm) if n_orig_harm else 0.0
        print("[{}] anchors={} orig_HARMFUL={} verified_flips={} "
              "conditional_flip_rate={:.3f} overall_retention={:.3f} regen={}".format(
                  ds, len(anchors), n_orig_harm, n_flip, cond,
                  n_flip / max(len(anchors), 1), n_regen), flush=True)
        twins[ds] = rows

    del qmodel
    torch.cuda.empty_cache()

    # ---- stage 3: CLIP-encode the sanitized text (chunk-mean, same as the caches) ----
    from transformers import CLIPTokenizer, CLIPTextModel
    print("Loading CLIP text tower:", args.clip_model, flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    text_model = CLIPTextModel.from_pretrained(args.clip_model).to(device).eval()
    mtag = args.clip_model.replace("/", "_") + "_HF"

    for ds in datasets:
        rows = twins[ds]
        ids, feats, flipped = [], [], []
        for r in rows:
            vec = encode_text(r["sanitized_text"], tokenizer, text_model, device)
            ids.append(r["id"]); feats.append(vec); flipped.append(bool(r["flipped"]))
        feats = torch.stack(feats, dim=0).float()
        out = {"ids": [ids], "text_feats": feats,
               "flipped": torch.tensor(flipped, dtype=torch.bool),
               "labels": torch.ones(len(ids), dtype=torch.long)}
        outp = os.path.join(ROOT, "data/CLIP_Embedding", ds,
                            "train_cftwin_{}.pt".format(mtag))
        torch.save(out, outp)
        print("[{}] saved twin cache: N={} flipped={} dim={} -> {}".format(
            ds, len(ids), int(sum(flipped)), feats.shape[1], outp), flush=True)


if __name__ == "__main__":
    main()
