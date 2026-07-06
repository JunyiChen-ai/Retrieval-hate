#!/usr/bin/env python
"""P2b — configurable neighbour-comparability judge (GPU, text-only).

Same task as P2 (COMPARABLE / INCOMPARABLE / UNSURE, label-blind), but with the three
tunable levers from P2's failure analysis, iterated on the TRAIN benchmark:
  --model      HF id (7B Qwen2.5-VL-*  OR  32B Qwen2.5-*-Instruct text LLM)
  --model_type {auto,vl,causal}   backend (auto: 'VL' in id -> vl)
  --evidence   {archive, archive_transcript}   v2 card only  OR  card + capped title/transcript
  --prompt     {orig, flip}   orig = P2 wording; flip = INCOMPARABLE is burden-of-proof

Reads a pairs JSONL (each line: query_id, neighbor_id, + optional pass-through fields such as
correct_vote for the train benchmark) and writes verdicts JSONL (resume-by-key, UNSURE
fallback). Greedy. --dataset selects which v2 archives / gt transcripts to load.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = "/data/jehc223/RGCL"
ARCHIVE_TAG = "Qwen2.5-VL-7B-Instruct_archive"
VERDICTS = ("COMPARABLE", "INCOMPARABLE", "UNSURE")
MAX_TXT = 300

sys.path.insert(0, os.path.join(ROOT, "src"))
from utils.generate_video_archive_HF import _extract_json_candidate  # noqa: E402

SYSTEM_PROMPT = (
    "You compare two short structured descriptions of short videos for a hateful-video "
    "research project. You decide only whether the two are COMPARABLE cases -- similar enough "
    "that one is a fair precedent for judging the other. You never decide whether either video "
    "is hateful. You always answer with a single valid JSON object and nothing else.")

CRITERIA = (
    "Each video is described by a structured archive card (a vision-language model watched it):\n"
    "  - target_groups : the group(s) the content is about / directed at (empty if none)\n"
    "  - mechanism     : the harm / rhetorical mechanism (slur, stereotype, dehumanization,\n"
    "                    mockery, ...; empty if none)\n"
    "  - modality      : which channel carries the salient content -- visual / speech / on-screen text\n"
    "  - explicitness  : how explicit the content is\n"
    "  - summary       : a neutral one-line summary\n")

# orig = P2 wording (COMPARABLE if >=1 match; INCOMPARABLE only when clearly none).
PROMPT_ORIG = (
    "A retrieval system pulled a PRECEDENT video from memory as one of the nearest neighbours of "
    "a QUERY video, to help classify the query. Judge whether the precedent is COMPARABLE to the "
    "query -- i.e. similar enough in what it is ABOUT that its moderation label is a fair vote "
    "for the query.\n\n" + CRITERIA +
    "\nDecision (judge ABOUTNESS, not severity, and NOT whether either is hateful):\n"
    "  - COMPARABLE   -> the two share AT LEAST ONE of: (a) the same or an overlapping target "
    "group; (b) the same attack / harm mechanism; (c) the same evidence modality carrying the "
    "salient content. If any one clearly matches, answer COMPARABLE.\n"
    "  - INCOMPARABLE -> ONLY when the two clearly match on NONE of (a)/(b)/(c).\n"
    "  - UNSURE       -> at least one card is essentially empty or too vague to tell.\n"
    "Prefer COMPARABLE when a match is plausible; reserve INCOMPARABLE for clearly unrelated cases.\n")

# flip = INCOMPARABLE is burden-of-proof; default COMPARABLE (fixes the over-flag ratchet).
PROMPT_FLIP = (
    "A retrieval system already selected the PRECEDENT video as one of the QUERY video's NEAREST "
    "neighbours, so by construction they are ALREADY topically close. Your job is the narrow one "
    "of catching the RARE precedent that slipped into the neighbour list despite being about a "
    "fundamentally DIFFERENT thing, so that its moderation label would be a misleading vote for "
    "the query.\n\n" + CRITERIA +
    "\nDecision (judge ABOUTNESS, not severity, and NOT whether either is hateful):\n"
    "  - Default to COMPARABLE. Answer COMPARABLE whenever the two plausibly share ANY of: "
    "(a) an overlapping target group; (b) a harm / rhetorical mechanism; (c) the salient "
    "modality -- OR when you are simply unsure they differ.\n"
    "  - Answer INCOMPARABLE ONLY if you can POSITIVELY state that they differ on ALL THREE "
    "(different target group AND different mechanism AND different salient modality), so the "
    "precedent is genuinely about something else. The burden of proof is on INCOMPARABLE; when in "
    "doubt, choose COMPARABLE.\n"
    "  - UNSURE -> only if a card is essentially empty.\n")

PROMPTS = {"orig": PROMPT_ORIG, "flip": PROMPT_FLIP}

TAIL = ('\nReturn ONE JSON object with exactly these fields and nothing else (no markdown '
        'fences):\n{"verdict": "COMPARABLE" | "INCOMPARABLE" | "UNSURE", "reason": "one short '
        'line citing the matching or mismatching field(s)"}\n')


def load_v2_records(ds):
    recs = {}
    for sp in ["train", "dev_seen", "test_seen"]:
        p = os.path.join(ROOT, "data/Archive", ds, "v2",
                         "{}_{}.jsonl".format(sp, ARCHIVE_TAG))
        for line in open(p):
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["id"]] = r
    return recs


def load_texts(ds):
    texts = {}
    for sp in ["train", "val", "test"]:
        p = os.path.join(ROOT, "data/gt", ds, "{}.jsonl".format(sp))
        if not os.path.exists(p):
            continue
        for line in open(p):
            o = json.loads(line)
            texts[str(o["id"])] = "" if o.get("text") is None else str(o["text"])
    return texts


def card(rec, texts, vid, evidence, max_summary=400, max_cue=200):
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
    if evidence == "archive_transcript":
        t = (texts.get(vid, "") or "").strip().replace("\n", " ")
        lines.append("  title/transcript: {}".format(t[:MAX_TXT] if t else "(none)"))
    return "\n".join(lines)


def build_prompt(qrec, nrec, texts, qid, nid, prompt, evidence):
    return (PROMPTS[prompt]
            + "\n[QUERY video card]\n" + card(qrec, texts, qid, evidence)
            + "\n\n[PRECEDENT video card]\n" + card(nrec, texts, nid, evidence)
            + "\n" + TAIL)


def parse_verdict(raw):
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
    for v in ("INCOMPARABLE", "COMPARABLE", "UNSURE"):
        if v in up:
            return v, "", True
    return "UNSURE", "", True


class VLBackend:
    def __init__(self, model_id):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="auto")
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def gen(self, prompt, max_new_tokens):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=None, videos=None,
                           return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(new, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()


class CausalBackend:
    def __init__(self, model_id, quant="none"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                  device_map="auto")
        if quant == "bnb4":
            # 4-bit nf4 load (bitsandbytes) — loads a bf16 checkpoint under the
            # INSTALLED stack (no autoawq/auto_gptq), fits 72B on 1xA100-80G.
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
            kw.pop("torch_dtype", None)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()

    @torch.no_grad()
    def gen(self, prompt, max_new_tokens):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[:, inputs["input_ids"].shape[1]:]
        return self.tok.batch_decode(new, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pairs_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model_type", default="auto", choices=["auto", "vl", "causal"])
    ap.add_argument("--evidence", default="archive",
                    choices=["archive", "archive_transcript"])
    ap.add_argument("--prompt", default="orig", choices=["orig", "flip"])
    ap.add_argument("--quant", default="none", choices=["none", "bnb4"],
                    help="bnb4 = 4-bit nf4 (bitsandbytes) for the 72B causal tier")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mt = args.model_type
    if mt == "auto":
        mt = "vl" if "VL" in args.model else "causal"
    print("[judge_p2b] model={} type={} evidence={} prompt={}".format(
        args.model, mt, args.evidence, args.prompt), flush=True)
    backend = (VLBackend(args.model) if mt == "vl"
               else CausalBackend(args.model, quant=args.quant))

    recs = load_v2_records(args.dataset)
    texts = load_texts(args.dataset)
    pairs = []
    for line in open(args.pairs_file):
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[:args.limit]

    done = set()
    if os.path.exists(args.out_file):
        for line in open(args.out_file):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done.add((r["query_id"], r["neighbor_id"]))
                except Exception:  # noqa: BLE001
                    pass
    todo = [p for p in pairs if (p["query_id"], p["neighbor_id"]) not in done]
    print("[{}] {} pairs, {} done, {} todo -> {}".format(
        args.dataset, len(pairs), len(pairs) - len(todo), len(todo), args.out_file),
        flush=True)

    counts = {v: 0 for v in VERDICTS}
    n_fb = 0
    with open(args.out_file, "a") as fout:
        for n, p in enumerate(todo):
            qid, nid = p["query_id"], p["neighbor_id"]
            t0 = time.time()
            prompt = build_prompt(recs.get(qid), recs.get(nid), texts, qid, nid,
                                  args.prompt, args.evidence)
            try:
                raw = backend.gen(prompt, args.max_new_tokens)
                verdict, reason, fb = parse_verdict(raw)
            except Exception as e:  # noqa: BLE001
                raw, verdict, reason, fb = repr(e), "UNSURE", "", True
            counts[verdict] += 1
            n_fb += int(fb)
            rec = dict(query_id=qid, neighbor_id=nid, verdict=verdict, reason=reason,
                       fallback=fb, wall_s=round(time.time() - t0, 2))
            for k in ("query_label", "neighbor_label", "correct_vote"):
                if k in p:
                    rec[k] = p[k]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (n + 1) % 200 == 0:
                print("  {}/{} counts={} fb={}".format(n + 1, len(todo), counts, n_fb),
                      flush=True)
    print("[{}] DONE counts={} fallback={}".format(args.dataset, counts, n_fb), flush=True)


if __name__ == "__main__":
    main()
