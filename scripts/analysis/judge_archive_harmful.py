#!/usr/bin/env python
"""Label-FREE coarse HARMFUL/BENIGN verdict over each video's structured archive.

P1 (research-wiki/EXP_p1_zerolabel_recal.md): zero-label drift recalibration via MLLM
prior estimation. This is step 1 — the MLLM reads ONLY the archive a vision-language model
wrote after watching the video (target_groups / mechanism / explicitness / modality cues /
neutral_summary) and returns exactly one of {HARMFUL, BENIGN}. The dataset LABEL is NEVER
shown to the model (unlike scripts/analysis/judge_memory_archive.py, which audits an
(archive, label) pair). Per-video accuracy ~0.7 is fine: this verdict is only an ingredient
for an AGGREGATE prior estimate (adjusted classify-and-count) downstream.

Greedy decoding (temperature 0), text-only (no video frames). The gold label is copied into
the output JSON for BOOKKEEPING ONLY (later TPR/FPR estimation on the train era and final
metric computation) — it is not part of the prompt.

We judge every id in ALL random-split files (train / dev_seen / test_seen) of a version so
that every TEMPORAL-split id (a re-partition of the same video universe) has a verdict.

Output JSON (--out): {dataset: {version: {id: {verdict, label, split, raw}}}}
Reads only data/Archive/**; writes only --out. Does not touch src/ or feature caches.
"""
import argparse
import json
import os
import re

import torch

ROOT = "/data/jehc223/RGCL"
ARCHIVE_TAG = "Qwen2.5-VL-7B-Instruct_archive"

SYSTEM_PROMPT = (
    "You are a careful content classifier for a hateful-video detection system. You read a "
    "structured description and decide whether the described video is harmful. Be decisive."
)

INSTRUCTIONS = """A vision-language model watched a short video and wrote the structured
DESCRIPTION below. Based ONLY on that description, decide whether the video is HARMFUL or
BENIGN.

  HARMFUL = it attacks, demeans, dehumanises, mocks, or incites hatred/violence against a
      protected group or its members (by race, ethnicity, religion, gender, sexual
      orientation, disability, nationality, immigration status, etc.), e.g. via slurs,
      stereotyping, dehumanisation, threats, or targeted insults.
  BENIGN = it does none of that (ordinary content: everyday life, food, pets, scenery,
      sports, commentary, humour that targets no protected group, etc.).

Judge what the description actually depicts. If it names a target group AND a hateful
mechanism (slur / stereotyping / dehumanisation / threat / targeted insult), or the summary
plainly describes an attack on a protected group, answer HARMFUL. If the description depicts
ordinary non-targeting content, or names no target and no hateful mechanism, answer BENIGN.
When the description is vague, pick the more likely label rather than refusing.

Respond in EXACTLY one line and nothing else:
VERDICT: <HARMFUL|BENIGN>
"""

# Synthetic probes for --diag (FABRICATED, never real entries, carry no gold). Expected
# verdicts: harmful_clear -> HARMFUL, benign_clear -> BENIGN, vague -> either (documented).
DIAG_PROBES = [
    ("probe_harmful_clear", dict(archive=dict(
        target_groups=["a religious group"], mechanism=["slur", "dehumanization"],
        explicitness="explicit",
        modality_cues={"speech": "repeated slurs calling them vermin", "visual": "", "on_screen_text": ""},
        neutral_summary="A speaker repeatedly uses slurs and calls a religious group vermin."))),
    ("probe_benign_clear", dict(archive=dict(
        target_groups=[], mechanism=[], explicitness="none",
        modality_cues={"speech": "", "visual": "a cat on a sofa", "on_screen_text": ""},
        neutral_summary="A cat sleeps on a sofa while soft music plays."))),
    ("probe_vague", dict(archive=dict(
        target_groups=[], mechanism=["coded_language"], explicitness="none",
        modality_cues={"speech": "", "visual": "", "on_screen_text": ""},
        neutral_summary="A person talks to the camera; audio unclear."))),
]


def archive_block(rec):
    """Human-readable rendering of the structured archive for the prompt. NO label."""
    a = rec.get("archive") or {}
    mc = a.get("modality_cues") or {}
    lines = [
        "target_groups: {}".format(a.get("target_groups") or []),
        "mechanism: {}".format(a.get("mechanism") or []),
        "explicitness: {}".format(a.get("explicitness")),
        "visual_cue: {}".format((mc.get("visual") or "").strip() or "(none)"),
        "speech_cue: {}".format((mc.get("speech") or "").strip() or "(none)"),
        "on_screen_text_cue: {}".format((mc.get("on_screen_text") or "").strip() or "(none)"),
        "summary: {}".format((a.get("neutral_summary") or "").strip() or "(none)"),
    ]
    return "\n".join(lines)


def build_prompt(rec):
    return INSTRUCTIONS + "\nDESCRIPTION:\n" + archive_block(rec) + "\n"


VERDICT_RE = re.compile(r"VERDICT\s*:\s*(HARMFUL|BENIGN)", re.I)


def parse_verdict(text):
    m = VERDICT_RE.search(text or "")
    if m:
        return m.group(1).upper()
    up = (text or "").upper()
    if "HARMFUL" in up:
        return "HARMFUL"
    if "BENIGN" in up:
        return "BENIGN"
    return "BENIGN"  # default to negative class (never inflates the harmful count)


def load_records(ds, version):
    """id -> archive record over ALL random-split files of a version (last write wins)."""
    recs = {}
    for sp in ("train", "dev_seen", "test_seen"):
        if version == "v1":
            p = os.path.join(ROOT, "data/Archive", ds, "{}_{}.jsonl".format(sp, ARCHIVE_TAG))
        else:
            p = os.path.join(ROOT, "data/Archive", ds, version,
                             "{}_{}.jsonl".format(sp, ARCHIVE_TAG))
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                r["_split"] = sp
                recs[r["id"]] = r
    return recs


@torch.no_grad()
def judge_one(prompt, processor, model, device, max_new_tokens):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None, return_tensors="pt")
    inputs = inputs.to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--versions", default="v2")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: judge first N per (ds,ver)")
    ap.add_argument("--diag", action="store_true",
                    help="judge only the synthetic DIAG_PROBES and print raw output")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    device = torch.device(args.device)
    print("Loading judge model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    if args.diag:
        for name, rec in DIAG_PROBES:
            raw = judge_one(build_prompt(rec), processor, model, device, args.max_new_tokens)
            print("[diag] {:22s} -> {:8s} | raw={!r}".format(
                name, parse_verdict(raw), raw), flush=True)
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    out = {}
    for ds in datasets:
        out[ds] = {}
        for ver in versions:
            recs = load_records(ds, ver)
            ids = list(recs.keys())
            if args.limit:
                ids = ids[: args.limit]
            print("[judge] {} {}: {} entries".format(ds, ver, len(ids)), flush=True)
            vmap = {}
            counts = {"HARMFUL": 0, "BENIGN": 0}
            for n, vid in enumerate(ids):
                raw = judge_one(build_prompt(recs[vid]), processor, model, device,
                                args.max_new_tokens)
                verdict = parse_verdict(raw)
                counts[verdict] += 1
                vmap[vid] = dict(verdict=verdict,
                                 label=int(recs[vid].get("label", 0)),
                                 split=recs[vid].get("_split"), raw=raw)
                if (n + 1) % 100 == 0:
                    print("  {}/{}  running counts={}".format(n + 1, len(ids), counts),
                          flush=True)
            out[ds][ver] = vmap
            print("[judge] {} {} DONE counts={}".format(ds, ver, counts), flush=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1, ensure_ascii=False)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
