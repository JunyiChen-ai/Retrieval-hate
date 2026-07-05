#!/usr/bin/env python
"""Semantic vote for the automatic memory-repair experiment (EXP_auto_memory_repair.md).

Qwen2.5-VL-7B-Instruct reads each TRAIN memory entry's structured archive JSON + its
dataset label and returns exactly one of {SUPPORT, CONTRADICT, UNSURE} with a one-line
reason. This is the seed-independent semantic vote: the verdict depends only on
(archive, label), so it is computed ONCE per (dataset, archive-version) and reused across
all seeds by scripts/analysis/auto_memory_repair.py.

Text-only judging job (no video frames), greedy decoding (temperature 0). Needs a GPU only
because the 7B model is loaded; run via scripts/slurm/judge_memory_archive.sbatch.

CONTRADICT is requested ONLY for a clear label/description mismatch (description plainly
benign but label hateful, or plainly hateful but label benign) — this is the automatic
analogue of the manual W2 noise-id audit.

Output JSON (--out): {dataset: {version: {id: {verdict, reason, label, raw}}}}
Reads only data/Archive/**; writes only the --out json. Does not touch src/ or caches.
"""
import argparse
import json
import os
import re
import sys

import torch

ROOT = "/data/jehc223/RGCL"
ARCHIVE_TAG = "Qwen2.5-VL-7B-Instruct_archive"

SYSTEM_PROMPT = (
    "You are a careful data auditor for a hateful-video detection system. You are strict, "
    "literal, and only flag a CLEAR contradiction."
)

INSTRUCTIONS = """You audit one entry of a memory bank used to detect hateful videos.

Each entry has a dataset LABEL and a structured DESCRIPTION that a vision-language model
wrote after watching the video.
  LABEL "hateful" (1) = the video attacks, demeans, dehumanises, or incites against a
      protected group (by race, ethnicity, religion, gender, sexual orientation, disability,
      nationality, etc.).
  LABEL "benign" (0) = it does none of that.

Compare the LABEL to what the DESCRIPTION actually shows and pick ONE verdict:
  - CONTRADICT -> the description AFFIRMATIVELY depicts content that clearly points the
      OPPOSITE way from the label. Two cases:
        (i)  label=hateful, but the description depicts ordinary, harmless, non-targeting
             content (food, pets, scenery, an everyday activity), naming no target group and
             no hateful mechanism, or it explicitly states there is nothing harmful; OR
        (ii) label=benign, but the description shows an explicit slur or an unmistakable
             attack on a protected group.
      Do NOT hedge when the description is concretely benign yet the label is hateful — that
      is exactly case (i). Choose CONTRADICT.
  - SUPPORT   -> the description is consistent with the label (for hateful: it names a target
      or hateful mechanism, or shows demeaning content; for benign: nothing hateful).
  - UNSURE    -> the description is essentially empty, or so vague/coded that you genuinely
      cannot tell what the video shows. Many hateful videos are subtle, so a hateful label
      with a truly vague description is UNSURE, not CONTRADICT — reserve CONTRADICT for
      descriptions that CONCRETELY depict the opposite of the label.

Respond in EXACTLY two lines and nothing else:
VERDICT: <SUPPORT|CONTRADICT|UNSURE>
REASON: <one short line>
"""

# Synthetic probes for the --diag calibration check. These are FABRICATED entries used only
# to verify the model follows the rubric (they are not real memory entries, carry no gt, and
# never enter any deletion set). Expected verdicts documented in the wiki.
DIAG_PROBES = [
    ("probe_hateful_but_benign", dict(label=1, archive=dict(
        target_groups=[], mechanism=[], explicitness="none", modality_cues={},
        neutral_summary="A cat sleeps on a sofa while soft music plays."))),   # -> CONTRADICT
    ("probe_benign_but_slur", dict(label=0, archive=dict(
        target_groups=["a religious group"], mechanism=["slur", "dehumanization"],
        explicitness="explicit", modality_cues={"speech": "repeated slurs calling them vermin"},
        neutral_summary="A speaker repeatedly uses slurs and calls a religious group vermin."))),  # -> CONTRADICT
    ("probe_hateful_consistent", dict(label=1, archive=dict(
        target_groups=["women"], mechanism=["dehumanization"], explicitness="explicit",
        modality_cues={"speech": "calls women property"},
        neutral_summary="A man says women are property and should obey men."))),  # -> SUPPORT
    ("probe_vague_hateful", dict(label=1, archive=dict(
        target_groups=[], mechanism=["coded_language"], explicitness="none", modality_cues={},
        neutral_summary="A person talks to the camera; audio unclear."))),   # -> UNSURE
]


def archive_block(rec):
    """Human-readable rendering of the structured archive for the prompt."""
    a = rec.get("archive") or {}
    mc = a.get("modality_cues") or {}
    lab = int(rec.get("label", 0))
    lines = [
        "LABEL: {} ({})".format("hateful" if lab == 1 else "benign", lab),
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
    return INSTRUCTIONS + "\nEntry:\n" + archive_block(rec) + "\n"


VERDICT_RE = re.compile(r"VERDICT\s*:\s*(SUPPORT|CONTRADICT|UNSURE)", re.I)
REASON_RE = re.compile(r"REASON\s*:\s*(.+)", re.I)


def parse_verdict(text):
    m = VERDICT_RE.search(text or "")
    if m:
        verdict = m.group(1).upper()
    else:
        # fall back: look for a bare keyword; default UNSURE (never auto-deletes)
        up = (text or "").upper()
        if "CONTRADICT" in up:
            verdict = "CONTRADICT"
        elif "SUPPORT" in up:
            verdict = "SUPPORT"
        else:
            verdict = "UNSURE"
    rm = REASON_RE.search(text or "")
    reason = rm.group(1).strip() if rm else ""
    return verdict, reason[:300]


def load_train_records(ds, version):
    """id -> archive record for the train split (last occurrence wins, matching the demo)."""
    if version == "v1":
        p = os.path.join(ROOT, "data/Archive", ds, "train_{}.jsonl".format(ARCHIVE_TAG))
    else:
        p = os.path.join(ROOT, "data/Archive", ds, version,
                         "train_{}.jsonl".format(ARCHIVE_TAG))
    recs = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
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
    ap.add_argument("--versions", default="v1,v2")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=96)
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
        diag = {}
        for name, rec in DIAG_PROBES:
            raw = judge_one(build_prompt(rec), processor, model, device, args.max_new_tokens)
            verdict, reason = parse_verdict(raw)
            print("[diag] {:28s} -> {:10s} | {}".format(name, verdict, reason), flush=True)
            diag[name] = dict(verdict=verdict, reason=reason, raw=raw)
        with open(args.out, "w") as f:
            json.dump({"diag": diag}, f, indent=1, ensure_ascii=False)
        print("wrote", args.out, flush=True)
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    out = {}
    for ds in datasets:
        out[ds] = {}
        for ver in versions:
            recs = load_train_records(ds, ver)
            ids = list(recs.keys())
            if args.limit:
                ids = ids[: args.limit]
            print("[judge] {} {}: {} entries".format(ds, ver, len(ids)), flush=True)
            vmap = {}
            counts = {"SUPPORT": 0, "CONTRADICT": 0, "UNSURE": 0}
            for n, vid in enumerate(ids):
                raw = judge_one(build_prompt(recs[vid]), processor, model, device,
                                args.max_new_tokens)
                verdict, reason = parse_verdict(raw)
                counts[verdict] += 1
                vmap[vid] = dict(verdict=verdict, reason=reason,
                                 label=int(recs[vid].get("label", 0)), raw=raw)
                if (n + 1) % 50 == 0:
                    print("  {}/{}  running counts={}".format(n + 1, len(ids), counts),
                          flush=True)
            out[ds][ver] = vmap
            print("[judge] {} {} DONE counts={}".format(ds, ver, counts), flush=True)
            # incremental dump so a crash late in the run keeps earlier verdicts
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1, ensure_ascii=False)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
