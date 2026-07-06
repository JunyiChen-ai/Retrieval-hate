#!/usr/bin/env python
"""P2 — MLLM neighbor-comparability judge (GPU, text-only).

For each deduplicated (query, neighbor) pair emitted by
`p2_rerank_eval.py --mode collect` (pairs_<ds>.jsonl), Qwen2.5-VL-7B-Instruct
reads the query's and one neighbor's **v2** structured archive card (NO labels
shown) and returns exactly one of {COMPARABLE, INCOMPARABLE, UNSURE}: is this
retrieved precedent a fair basis for voting on this query?

Criterion (frozen, pre-registered in research-wiki/EXP_p2_neighbor_rerank.md):
COMPARABLE if the two share at least one of (a) target group, (b) attack/harm
mechanism, (c) the evidence modality carrying the salient content; INCOMPARABLE
only when clearly none match; UNSURE if the archives are too thin to tell.

The verdict depends only on the two archives (seed-independent), so each unique
pair is judged once and reused across all heads. Greedy (temperature 0),
text-only. Resume-by-key append JSONL: scripts/analysis/p2_out/verdicts_<ds>.jsonl.
Reads only data/Archive/<ds>/v2/**; writes only the verdicts JSONL.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = "/data/jehc223/RGCL"
ARCHIVE_TAG = "Qwen2.5-VL-7B-Instruct_archive"
OUT_DIR = os.path.join(ROOT, "scripts/analysis/p2_out")
VERDICTS = ("COMPARABLE", "INCOMPARABLE", "UNSURE")

sys.path.insert(0, os.path.join(ROOT, "src"))
from utils.generate_video_archive_HF import _extract_json_candidate  # noqa: E402

SYSTEM_PROMPT = (
    "You compare two short structured descriptions of short videos for a "
    "hateful-video research project. You decide only whether the two are "
    "COMPARABLE cases -- similar enough that one is a fair precedent for judging "
    "the other. You never decide whether either video is hateful. You always "
    "answer with a single valid JSON object and nothing else."
)

INSTRUCTIONS = """A retrieval system pulled a PRECEDENT video from memory as one of the \
nearest neighbours of a QUERY video, to help classify the query. Judge whether the precedent \
is COMPARABLE to the query -- i.e. similar enough in what it is ABOUT that its moderation label \
is a fair vote for the query.

Each video is described by a structured archive card (written by a vision-language model that \
watched it):
  - target_groups : the group(s) the content is about / directed at (empty if none)
  - mechanism     : the harm / rhetorical mechanism (e.g. slur, stereotype, dehumanization,
                    mockery; empty if none)
  - modality      : which channel carries the salient content -- visual / speech / on-screen text
  - explicitness  : how explicit the content is
  - summary       : a neutral one-line summary

Decision (judge ABOUTNESS, not severity, and NOT whether either is hateful):
  - COMPARABLE   -> the two share AT LEAST ONE of: (a) the same or an overlapping target group;
                    (b) the same attack / harm mechanism; (c) the same evidence modality carrying
                    the salient content. If any one clearly matches, answer COMPARABLE.
  - INCOMPARABLE -> ONLY when the two clearly match on NONE of (a)/(b)/(c): they are about
                    different groups, use different mechanisms, and rely on different modalities
                    -- so the precedent's label would be an unfair vote for the query.
  - UNSURE       -> at least one card is essentially empty or too vague to tell what the video is
                    about, so you cannot judge the match.

Prefer COMPARABLE when a match is plausible; reserve INCOMPARABLE for clearly unrelated cases.

Return ONE JSON object with exactly these fields and nothing else (no markdown fences):
{"verdict": "COMPARABLE" | "INCOMPARABLE" | "UNSURE", "reason": "one short line citing the \
matching or mismatching field(s)"}
"""


def load_v2_records(ds):
    """id -> v2 archive record, over train + dev_seen + test_seen (last wins)."""
    recs = {}
    for sp in ["train", "dev_seen", "test_seen"]:
        p = os.path.join(ROOT, "data/Archive", ds, "v2",
                         "{}_{}.jsonl".format(sp, ARCHIVE_TAG))
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["id"]] = r
    return recs


def card(rec, max_summary=400, max_cue=200):
    a = (rec or {}).get("archive") or {}
    mc = a.get("modality_cues") or {}
    trim = lambda s, n: ((s or "").strip())[:n]  # noqa: E731
    tg = a.get("target_groups") or []
    mech = a.get("mechanism") or []
    mods = [k for k in ("visual", "speech", "on_screen_text") if trim(mc.get(k), 1)]
    lines = [
        "  target_groups: {}".format(", ".join(tg) if tg else "(none stated)"),
        "  mechanism: {}".format(", ".join(mech) if mech else "(none stated)"),
        "  modality_with_content: {}".format(", ".join(mods) if mods else "(none stated)"),
        "  explicitness: {}".format(a.get("explicitness") or "unknown"),
    ]
    for k, name in [("visual", "visual"), ("speech", "speech"),
                    ("on_screen_text", "on-screen text")]:
        if trim(mc.get(k), 1):
            lines.append("  {}: {}".format(name, trim(mc.get(k), max_cue)))
    lines.append("  summary: {}".format(trim(a.get("neutral_summary"), max_summary) or "(none)"))
    return "\n".join(lines)


def build_prompt(qrec, nrec):
    return (INSTRUCTIONS
            + "\n[QUERY video card]\n" + card(qrec)
            + "\n\n[PRECEDENT video card]\n" + card(nrec) + "\n")


def parse_verdict(raw):
    """(verdict, reason, fallback_bool). Strict JSON first, then bare-word, else UNSURE."""
    cand = _extract_json_candidate(raw)
    if cand is not None:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                v = str(obj.get("verdict") or "").strip().upper()
                if v in VERDICTS:
                    return v, str(obj.get("reason") or "")[:300], False
        except Exception:  # noqa: BLE001
            pass
    up = (raw or "").upper()
    for v in ("INCOMPARABLE", "COMPARABLE", "UNSURE"):  # INCOMPARABLE first (substring)
        if v in up:
            return v, "", True
    return "UNSURE", "", True


@torch.no_grad()
def judge_one(prompt, processor, model, device, max_new_tokens):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=None,
                       return_tensors="pt").to(device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def load_pairs(ds, cache_dir):
    p = os.path.join(cache_dir, "pairs_{}.jsonl".format(ds))
    pairs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                pairs.append((o["query_id"], o["neighbor_id"]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC")
    ap.add_argument("--cache_dir", default=OUT_DIR)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N pairs per ds")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    device = torch.device(args.device)
    print("Loading judge model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        recs = load_v2_records(ds)
        pairs = load_pairs(ds, args.cache_dir)
        if args.limit:
            pairs = pairs[:args.limit]
        out_path = os.path.join(args.cache_dir, "verdicts_{}.jsonl".format(ds))
        done = set()
        if os.path.exists(out_path):
            for line in open(out_path):
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add((r["query_id"], r["neighbor_id"]))
                    except Exception:  # noqa: BLE001
                        pass
        todo = [p for p in pairs if p not in done]
        print("[{}] {} pairs, {} done, {} to do -> {}".format(
            ds, len(pairs), len(pairs) - len(todo), len(todo), out_path), flush=True)
        counts = {v: 0 for v in VERDICTS}
        n_fb = 0
        with open(out_path, "a") as fout:
            for n, (qid, nid) in enumerate(todo):
                t0 = time.time()
                prompt = build_prompt(recs.get(qid), recs.get(nid))
                try:
                    raw = judge_one(prompt, processor, model, device,
                                    args.max_new_tokens)
                    verdict, reason, fb = parse_verdict(raw)
                except Exception as e:  # noqa: BLE001
                    raw, verdict, reason, fb = repr(e), "UNSURE", "", True
                counts[verdict] += 1
                n_fb += int(fb)
                fout.write(json.dumps(dict(
                    query_id=qid, neighbor_id=nid, verdict=verdict,
                    reason=reason, fallback=fb, wall_s=round(time.time() - t0, 2),
                    raw=raw), ensure_ascii=False) + "\n")
                fout.flush()
                if (n + 1) % 100 == 0:
                    print("  {}/{}  counts={} fb={}".format(
                        n + 1, len(todo), counts, n_fb), flush=True)
        print("[{}] DONE counts={} fallback={}".format(ds, counts, n_fb), flush=True)


if __name__ == "__main__":
    main()
