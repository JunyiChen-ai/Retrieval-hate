#!/usr/bin/env python
"""Role 3 step 2 (GPU): Qwen2.5-VL-7B arbitration of gate-deferred samples.

For every sample the margin gate deferred (gate_margin.py output), run frozen
Qwen2.5-VL-7B-Instruct with:
  - 16 evenly-sampled video frames (omitted in the text-only variant),
  - the video's title+transcript,
  - the video's OWN archive entry (auto-generated, label-free),
  - the top-5 retrieved neighbours' archive entries + their ground-truth
    moderation labels as in-context precedent evidence,
and ask for a strict-JSON verdict {verdict, key_evidence, cited_neighbor}.
The 3-way verdict binarizes as hateful/offensive -> 1 (harmful), normal -> 0.

Zero training; the classifier checkpoints and memory are untouched. Frame
decoding and JSON extraction are read-only imports from
src/utils/generate_video_archive_HF.py.

Resume-by-id append-mode JSONL, one output file per run spec:
  scripts/role3/out/arb_<ds>_<variant>_<split>_<mode>.jsonl

Usage (inside SLURM):
  python scripts/role3/arbitrate_qwen.py \
      --runs MHC:base:test:frames,MHC:base:val:frames --limit 0
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.generate_video_archive_HF import (  # noqa: E402  (read-only import)
    load_video_frames, _extract_json_candidate)

OUT_DIR = os.path.join(ROOT, "scripts/role3/out")
RATE_UNION = "0.30"  # nested deferral sets; the 30% set covers all rates
MAX_TEXT_CHARS = 3500
VERDICTS = ("hateful", "offensive", "normal")

# Fine-grained (Hateful/Offensive/Normal) gt of the train memory entries, used
# as precedent labels in prompt v2 (same provenance as the binary labels the
# kNN vote already uses; never applied to val/test queries). v1 stays
# byte-identical to the smoke-gated original.
FINE_ANN = {
    "MHC": os.path.join(
        ROOT, "data/_src_Multihateclip/English/annotation(new).json"),
    "MHC_zh": os.path.join(
        ROOT, "data/_src_Multihateclip/Chinese/annotation(new).json"),
}
_FINE_CACHE = {}


def fine_labels(ds):
    if ds not in _FINE_CACHE:
        _FINE_CACHE[ds] = {
            str(x["Video_ID"]): str(x.get("Label") or "").upper()
            for x in json.load(open(FINE_ANN[ds]))}
    return _FINE_CACHE[ds]

SYSTEM_PROMPT = (
    "You are a senior content-moderation arbiter supporting an academic "
    "research project on hateful-video detection. A fast retrieval-based "
    "classifier escalates to you ONLY the videos it is uncertain about. You "
    "study the video evidence and the retrieved precedent cases, then deliver "
    "a careful final verdict. You never produce hateful content yourself; you "
    "only classify what is observably present. You always respond with a "
    "single valid JSON object and nothing else."
)

POLICY_BLOCK = (
    "Moderation policy (MultiHateClip labelling guidelines):\n"
    "- \"hateful\": attacks, demeans, dehumanizes, or incites hatred or "
    "violence against a group or its members based on protected attributes "
    "(race, ethnicity, nationality, religion, gender, sexual orientation, "
    "disability, etc.).\n"
    "- \"offensive\": rude, insulting, mocking or degrading content that "
    "hurts individuals or groups but does not rise to a hateful attack on a "
    "protected group; includes sexual harassment, crude slurs used "
    "non-ideologically, and demeaning jokes.\n"
    "- \"normal\": neither hateful nor offensive (benign content about "
    "sensitive topics is still normal).\n"
    "For the final decision, hateful and offensive both count as HARMFUL; "
    "normal counts as NOT harmful. Choose the single best of the three."
)

# v2: recalibrated to the dataset's actual bar. The v1 smoke run over-flagged
# (10/10 borderline samples judged harmful vs gt 2/10): a generic
# platform-safety standard is stricter than MultiHateClip's annotators, whose
# Normal class absorbs crude/edgy content (58% of the EN val-deferred queue is
# gt-Normal). v2 states that bar explicitly and defers to precedents for
# calibration; prompt selection between v1/v2 happens on VAL only.
POLICY_BLOCK_V2 = (
    "Moderation policy (MultiHateClip labelling guidelines, as its annotators "
    "actually applied them):\n"
    "- \"hateful\": expresses or endorses hatred, discrimination, demeaning "
    "stereotypes or violence toward a group or its members defined by "
    "protected attributes (gender, sexual orientation, race, ethnicity, "
    "nationality, religion, disability). The content must attack the group as "
    "such - merely mentioning or depicting it does not count.\n"
    "- \"offensive\": clearly insults, degrades, harasses or mocks an "
    "identifiable person or group, or is graphically shocking, but without "
    "the group-directed hatred above.\n"
    "- \"normal\": everything else. IMPORTANT - in this dataset the normal "
    "class is BROAD: crude humour, profanity, sexual innuendo, frank talk "
    "about sex or bodies, insult-comedy skits, rude pranks, edgy sarcasm, "
    "provocative opinions and reporting on offensive events are all labelled "
    "normal unless they demean an identifiable target. If the evidence is "
    "only a rude topic or crude language with no demeaned target, the answer "
    "is normal.\n"
    "Calibration: this video was escalated because the classifier is "
    "uncertain - escalation is NOT evidence of harm; many escalated videos "
    "are normal. Judge by the dataset's bar as shown by the precedents (their "
    "labels are ground truth), not by a general platform-safety standard, "
    "which is stricter than this dataset. Precedents labelled NORMAL show how "
    "far edgy content may go while staying normal.\n"
    "For the final decision, hateful and offensive both count as HARMFUL; "
    "normal counts as NOT harmful. Choose the single best of the three."
)

POLICY_BLOCKS = {"v1": POLICY_BLOCK, "v2": POLICY_BLOCK_V2}


def card_lines(card, indent="  "):
    if card is None:
        return [indent + "(no archive entry available)"]
    lines = [
        indent + "targets: {}; mechanism: {}; explicitness: {}".format(
            ", ".join(card["target_groups"]) or "none",
            ", ".join(card["mechanism"]) or "none",
            card["explicitness"]),
    ]
    for key, name in [("visual", "visual"), ("speech", "speech"),
                      ("on_screen_text", "on-screen text")]:
        if card.get(key):
            lines.append(indent + "{}: {}".format(name, card[key]))
    lines.append(indent + "summary: {}".format(card["summary"] or "(none)"))
    return lines


def build_user_prompt(sample, text_only, pv="v1", fine=None):
    L = []
    L.append(
        "A retrieval-based hate-video classifier is UNCERTAIN about the "
        "video below (its nearest-neighbour vote sits near the decision "
        "boundary). You are the arbiter: decide the final verdict.")
    L.append("")
    L.append(POLICY_BLOCKS[pv])
    L.append("")
    if text_only:
        L.append("No frames are attached; judge from the text evidence below.")
    else:
        L.append("The video's frames are attached (evenly sampled).")
    L.append("")
    txt = (sample["text"] or "").strip() or "(none)"
    if len(txt) > MAX_TEXT_CHARS:
        txt = txt[:MAX_TEXT_CHARS] + " ...[truncated]"
    L.append("[TITLE + TRANSCRIPT] (may be auto-generated and noisy)")
    L.append('"""')
    L.append(txt)
    L.append('"""')
    L.append("")
    L.append("[ARCHIVE ENTRY for this video] "
             "(auto-generated structured analysis; may contain errors)")
    L.extend(card_lines(sample["own_card"]))
    L.append("")
    L.append("[RETRIEVED PRECEDENTS from the moderation memory] "
             "(most similar first; label = dataset ground-truth)")
    for r, nb in enumerate(sample["neighbors"], 1):
        if pv == "v2" and fine and nb["id"] in fine:
            fl = fine[nb["id"]]
            lab = ("NORMAL (not harmful)" if fl == "NORMAL"
                   else "{} (harmful)".format(fl))
        else:
            lab = ("HARMFUL (hateful or offensive)" if nb["label"] == 1
                   else "NORMAL (not harmful)")
        L.append("{}. precedent id={} similarity={:.3f} label={}".format(
            r, nb["id"], nb["sim"], lab))
        L.extend(card_lines(nb["card"], indent="   "))
    L.append("")
    L.append(
        "Weigh the video's own content first (frames, title, transcript, "
        "archive entry); use the precedents as calibration evidence for where "
        "this dataset draws the hateful/offensive/normal boundaries. Do not "
        "simply copy the majority precedent label.")
    L.append("")
    L.append(
        "Return ONE JSON object with exactly these fields:\n"
        "{\"verdict\": \"hateful\" | \"offensive\" | \"normal\", "
        "\"key_evidence\": \"one or two sentences citing the decisive "
        "evidence\", \"cited_neighbor\": \"id of the single most relevant "
        "precedent, or 'none'\"}\n"
        "Output ONLY the JSON object: no markdown fences, no commentary.")
    return "\n".join(L)


WORD_MAP = {"hateful": "hateful", "offensive": "offensive",
            "normal": "normal", "仇恨": "hateful", "正常": "normal"}
_WORD_RE = None


def _word_fallback(raw):
    """Bare-verdict fallback for adapters SFT'd to one-word answers (v3).

    Only consulted AFTER strict-JSON parsing failed; counted separately via
    the 'word-fallback' parse_error prefix so the JSON rate stays reportable.
    """
    global _WORD_RE
    if _WORD_RE is None:
        import re
        _WORD_RE = re.compile(r"(hateful|offensive|normal|仇恨|正常)",
                              re.IGNORECASE)
    m = _WORD_RE.search((raw or "").strip()[:200])
    return WORD_MAP[m.group(1).lower()] if m else None


def parse_verdict(raw):
    """raw generation -> (verdict|None, key_evidence, cited_neighbor, error).

    Strict JSON first; if that fails, a bare-verdict word fallback (error
    field then carries the 'word-fallback' prefix for separate accounting).
    """
    cand = _extract_json_candidate(raw)
    err, obj = None, None
    if cand is None:
        err = "no JSON object found"
    else:
        try:
            obj = json.loads(cand)
        except Exception as e:  # noqa: BLE001
            err = "json.loads: {}".format(e)
    if err is None and not isinstance(obj, dict):
        err, obj = "not a dict", None
    if obj is not None:
        v = str(obj.get("verdict") or "").strip().lower()
        if v in VERDICTS:
            return (v, str(obj.get("key_evidence") or ""),
                    str(obj.get("cited_neighbor") or ""), None)
        err = "bad verdict: %r" % v
    w = _word_fallback(raw)
    if w is not None:
        return w, "", "", "word-fallback ({})".format(err)
    return None, "", "", err


@torch.no_grad()
def generate(model, processor, device, frames, user_prompt, max_new_tokens):
    content = []
    if frames is not None:
        content.append({"type": "video", "video": frames})
    content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=None,
        videos=[frames] if frames is not None else None,
        return_tensors="pt")
    inputs = inputs.to(device)
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    return raw.strip()


def run_one(spec, model, processor, device, args):
    ds, variant, split, mode = spec.split(":")
    assert mode in ("frames", "textonly"), mode
    pv = args.prompt_version
    fine = fine_labels(ds) if pv in ("v2",) else None
    gate_path = os.path.join(OUT_DIR, "gate_{}_{}.json".format(ds, variant))
    gate = json.load(open(gate_path))
    deferred = [s for s in gate["samples"]
                if s["split"] == split and s["defer"][RATE_UNION]]
    if args.limit > 0:
        deferred = deferred[:args.limit]

    tag = args.tag or pv
    suffix = "" if tag == "v1" else "_p{}".format(tag)
    out_path = os.path.join(
        OUT_DIR, "arb_{}_{}_{}_{}{}.jsonl".format(
            ds, variant, split, mode, suffix))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [s for s in deferred if s["id"] not in done]
    print("[{}] {} deferred (rate {}), {} done, {} to do -> {}".format(
        spec, len(deferred), RATE_UNION, len(deferred) - len(todo),
        len(todo), out_path), flush=True)

    video_root = os.path.join(ROOT, "data/video", ds, "All")
    n_json = n_fail = 0
    with open(out_path, "a") as fout:
        for n, s in enumerate(todo):
            t0 = time.time()
            frames = None
            frame_ok = None
            if mode == "frames":
                vp = os.path.join(video_root, "{}.mp4".format(s["id"]))
                frames, frame_ok = load_video_frames(vp, args.num_frames)
                if not frame_ok:
                    frames = None  # fall back to text-only for this sample
            prompt = build_user_prompt(s, text_only=(frames is None),
                                       pv=pv, fine=fine)
            rec = dict(
                id=s["id"], dataset=ds, variant=variant, split=split,
                mode=mode, prompt_version=pv, tag=tag,
                adapter=(args.adapter or None), frame_ok=frame_ok,
                label=s["label"],
                pred_knn=s["pred_knn"], vote=s["vote"], margin=s["margin"],
                prompt_chars=len(prompt), raw_output=None, verdict=None,
                verdict_bin=None, key_evidence="", cited_neighbor="",
                parse_error=None, wall_s=None)
            try:
                raw = generate(model, processor, device, frames, prompt,
                               args.max_new_tokens)
                rec["raw_output"] = raw
                v, ev, cited, err = parse_verdict(raw)
                rec.update(verdict=v, key_evidence=ev, cited_neighbor=cited,
                           parse_error=err)
                if v is not None:
                    rec["verdict_bin"] = int(v in ("hateful", "offensive"))
                    n_json += 1
                else:
                    n_fail += 1
            except Exception as e:  # noqa: BLE001
                rec["parse_error"] = "generation failed: {}".format(repr(e))
                n_fail += 1
            rec["wall_s"] = round(time.time() - t0, 2)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print("  [{}] {}/{} id={} verdict={} bin={} knn={} gt={} {:.1f}s".format(
                spec, n + 1, len(todo), s["id"], rec["verdict"],
                rec["verdict_bin"], s["pred_knn"], s["label"], rec["wall_s"]),
                flush=True)
    print("[{}] done: json_ok={} fail={}".format(spec, n_json, n_fail),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma list of ds:variant:split:mode "
                         "(e.g. MHC:base:test:frames)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0 process only first N deferred items (smoke)")
    ap.add_argument("--prompt_version", default="v1",
                    choices=sorted(POLICY_BLOCKS.keys()),
                    help="v1 = original (byte-stable); v2 = dataset-bar "
                         "recalibration + fine precedent labels; selection "
                         "between versions happens on VAL only")
    ap.add_argument("--adapter", default="",
                    help="optional peft LoRA adapter dir (task-calibrated "
                         "arbiter, e.g. logging/lora/MHC); merged into the "
                         "base weights at load time (v3)")
    ap.add_argument("--tag", default="",
                    help="output-file tag (defaults to --prompt_version); "
                         "v3 = adapter arbiter runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    device = torch.device(args.device)
    print("Loading {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=None)
    if args.adapter:
        # mirror generate_VideoMLLM_embedding_lora_HF.py: attach + merge
        if not os.path.isdir(args.adapter):
            raise SystemExit("--adapter '{}' is not a directory".format(
                args.adapter))
        from peft import PeftModel
        print("Attaching LoRA adapter from: {}".format(args.adapter),
              flush=True)
        model = PeftModel.from_pretrained(model, args.adapter)
        print("Merging LoRA adapter into base weights ...", flush=True)
        model = model.merge_and_unload()
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels)

    for spec in [r.strip() for r in args.runs.split(",") if r.strip()]:
        run_one(spec, model, processor, device, args)


if __name__ == "__main__":
    main()
