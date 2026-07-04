import argparse
import json
import os
import re

import numpy as np
import torch
from PIL import Image

# DESIGN_iter3 §4.1 (MLLM Role 1, E0b): generate a structured, human-auditable
# archive entry per video with frozen Qwen2.5-VL-7B-Instruct, then encode each
# archive text with the CLIP text encoder as a future retrieval key.
#
# Archive schema (ALL fields in English, ZH videos use English as pivot):
#   target_groups   : list[str]  groups targeted/disparaged (empty if none)
#   mechanism       : list[str]  delivery mechanism from a fixed vocabulary
#                     (slur / insult / dehumanization / stereotyping / mockery /
#                      coded_language / incitement / threat / misinformation /
#                      sexual_harassment / glorification_of_violence / other)
#   modality_cues   : {visual, speech, on_screen_text} one evidence sentence each
#   explicitness    : "explicit" | "implicit" | "none"
#   neutral_summary : 2-3 neutral factual sentences describing the video
#
# Outputs:
#   (generate) data/Archive/<DS>/<outname>_Qwen2.5-VL-7B-Instruct_archive.jsonl
#              one record per video: {id, split, label, raw_output, archive,
#              parse_ok, schema_ok, refusal, error}; append-mode, resume by id.
#   (encode)   data/CLIP_Embedding/<DS>/<outname>_archive_openai_clip-vit-large-patch14-336_HF.pt
#              {"ids": [ids], "text_feats": [N,768], "labels": LongTensor[N]}
#              following the existing cache contract (ids = ONE sublist, order =
#              gt order; zero-vector guard for refusals / missing entries).
#
# The frame sampler (decord + PyAV fallback) is reused verbatim from
# generate_VideoMLLM_embedding_HF.py / generate_VideoCLIP_embedding_HF.py.

SPLIT_TO_OUTNAME = {
    "train": "train",
    "val": "dev_seen",
    "test": "test_seen",
}

MECHANISM_VOCAB = [
    "slur",
    "insult",
    "dehumanization",
    "stereotyping",
    "mockery",
    "coded_language",
    "incitement",
    "threat",
    "misinformation",
    "sexual_harassment",
    "glorification_of_violence",
    "other",
]

# ----------------------------------------------------------------------------
# Prompt (neutral forensic content-analysis framing; strict JSON output)
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a multimodal content analyst supporting an academic content-moderation "
    "research project. Researchers study how harmful content spreads on social media; "
    "your job is to produce a neutral, factual, structured analysis of each video so "
    "that it can be catalogued. You never endorse or produce hateful content yourself; "
    "you only describe and categorise what is observably present. Many videos in this "
    "archive are completely benign. You always respond with a single valid JSON object "
    "and nothing else."
)

USER_PROMPT_TEMPLATE = (
    "Analyse this social-media video for a content-moderation research archive. The "
    "video may or may not contain hateful or offensive material; describe what is "
    "present without endorsing it.\n"
    "\n"
    "Video title and transcript (may be auto-generated and noisy):\n"
    "\"\"\"\n"
    "{text}\n"
    "\"\"\"\n"
    "\n"
    "Return ONE JSON object with exactly these fields:\n"
    "- \"target_groups\": list of strings naming the specific group(s) targeted or "
    "disparaged (e.g. by race, ethnicity, religion, gender, sexual orientation, "
    "nationality, disability); use an empty list [] if no group is targeted.\n"
    "- \"mechanism\": list of strings describing how any attack or disparagement is "
    "delivered; choose only from [\"slur\", \"insult\", \"dehumanization\", "
    "\"stereotyping\", \"mockery\", \"coded_language\", \"incitement\", \"threat\", "
    "\"misinformation\", \"sexual_harassment\", \"glorification_of_violence\", "
    "\"other\"]; use an empty list [] if none.\n"
    "- \"modality_cues\": object with keys \"visual\", \"speech\", \"on_screen_text\". "
    "For each key give one factual sentence describing the concrete evidence in that "
    "channel: symbols, gestures, people and imagery for \"visual\"; spoken words from "
    "the transcript for \"speech\"; captions, overlaid text or signs for "
    "\"on_screen_text\". Use an empty string \"\" if that channel carries no relevant "
    "cue.\n"
    "- \"explicitness\": exactly one of \"explicit\" (hostility is overt), "
    "\"implicit\" (hostility is conveyed indirectly via sarcasm, coded references or "
    "dog-whistles), or \"none\" (no hostility present).\n"
    "- \"neutral_summary\": 2-3 sentences of neutral, factual description of what the "
    "video shows and says, in plain English, without judgement words.\n"
    "\n"
    "Rules:\n"
    "- Respond in English only, even if the video or transcript is in another "
    "language.\n"
    "- Output ONLY the JSON object: no markdown fences, no commentary before or "
    "after.\n"
    "- Ground every field in observable evidence from the frames, title or "
    "transcript; do not speculate beyond what is shown."
)

# ----------------------------------------------------------------------------
# Prompt v2 (AUDIT_archive_faithfulness.md fixes; same forensic style + JSON):
#   defect 1 -- target_groups recall (v1 full-run non-empty rate: ZH 1.4%,
#     EN 6.5%; audited attacks on 娘炮/泼妇/小日子/homosexuals all left target
#     empty): v2 makes naming the attacked group MANDATORY, states that slurs
#     identify their own target group, and counts title-only attacks.
#   defect 2 -- spurious mechanism on benign content (9/60 audited
#     hallucinations were benign medical/nursery/greeting videos given
#     stereotyping/slur; v1 mechanism non-empty on gt-Normal: ZH 87.7%,
#     EN 79.2%): v2 forbids filling mechanism without concrete quotable
#     evidence and adds an explicitness/mechanism/target consistency rule.
# v1 stays byte-identical (default); v2 writes to <dir>/<dataset>/v2/.
# ----------------------------------------------------------------------------
USER_PROMPT_TEMPLATE_V2 = (
    "Analyse this social-media video for a content-moderation research archive. The "
    "video may or may not contain hateful or offensive material; describe what is "
    "present without endorsing it.\n"
    "\n"
    "Video title and transcript (may be auto-generated and noisy):\n"
    "\"\"\"\n"
    "{text}\n"
    "\"\"\"\n"
    "\n"
    "Return ONE JSON object with exactly these fields:\n"
    "- \"target_groups\": list of strings naming the specific group(s) of people "
    "the video targets, disparages or mocks. If the content attacks, demeans or "
    "mocks ANY group -- by race, ethnicity, nationality or region, religion, "
    "gender, sexual orientation or gender expression, disability, occupation, or "
    "any other group identity -- you MUST name that group here; never leave this "
    "list empty when a group is attacked. A derogatory group label or slur "
    "identifies its own target: name the group the term refers to (e.g. a slur "
    "for effeminate men targets \"effeminate men\"; a derogatory word for women "
    "targets \"women\"; a mocking nickname for a nationality targets that "
    "nationality; a slur for gay people targets \"gay people\"). Attacks that "
    "appear only in the title still count. Use the empty list [] ONLY if no group "
    "of people is targeted (benign content, or an insult aimed at one specific "
    "individual rather than at a group).\n"
    "- \"mechanism\": list of strings describing how any attack or disparagement "
    "is delivered; choose only from [\"slur\", \"insult\", \"dehumanization\", "
    "\"stereotyping\", \"mockery\", \"coded_language\", \"incitement\", \"threat\", "
    "\"misinformation\", \"sexual_harassment\", \"glorification_of_violence\", "
    "\"other\"]; use an empty list [] if none. Include a mechanism ONLY when you "
    "can point to concrete evidence of an actual attack or disparagement in the "
    "title, transcript, speech, imagery or on-screen text -- evidence you could "
    "quote. Benign content about sensitive topics (medical or sex education, "
    "product reviews, greetings, children's songs, news reporting, neutral "
    "discussion of a group) is NOT an attack: use [] for it. Never fill in a "
    "mechanism without sufficient evidence.\n"
    "- \"modality_cues\": object with keys \"visual\", \"speech\", \"on_screen_text\". "
    "For each key give one factual sentence describing the concrete evidence in that "
    "channel: symbols, gestures, people and imagery for \"visual\"; spoken words from "
    "the transcript for \"speech\"; captions, overlaid text or signs for "
    "\"on_screen_text\". Use an empty string \"\" if that channel carries no relevant "
    "cue.\n"
    "- \"explicitness\": exactly one of \"explicit\" (hostility is overt), "
    "\"implicit\" (hostility is conveyed indirectly via sarcasm, coded references or "
    "dog-whistles), or \"none\" (no hostility present).\n"
    "- \"neutral_summary\": 2-3 sentences of neutral, factual description of what the "
    "video shows and says, in plain English, without judgement words.\n"
    "\n"
    "Rules:\n"
    "- Respond in English only, even if the video or transcript is in another "
    "language.\n"
    "- Output ONLY the JSON object: no markdown fences, no commentary before or "
    "after.\n"
    "- Ground every field in observable evidence from the frames, title or "
    "transcript; do not speculate beyond what is shown.\n"
    "- The title is part of the content: hostility or slurs appearing only in the "
    "title must still be reflected in \"target_groups\", \"mechanism\" and "
    "\"explicitness\".\n"
    "- Consistency: if \"mechanism\" is non-empty and the hostility is aimed at "
    "people as a group, \"target_groups\" must name that group; if you cannot name "
    "any targeted group or individual, re-check whether a mechanism is really "
    "present. If \"explicitness\" is \"none\", then \"mechanism\" and "
    "\"target_groups\" must both be [].")

PROMPT_TEMPLATES = {
    "v1": USER_PROMPT_TEMPLATE,
    "v2": USER_PROMPT_TEMPLATE_V2,
}

MAX_TEXT_CHARS = 6000  # cap the title+transcript blob fed into the prompt


def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(
        description=(
            "Generate structured video archives (Qwen2.5-VL) + CLIP text-encoder "
            "retrieval keys for the RGCL pipeline."
        )
    )
    arg_parser.add_argument("--dataset", type=str, default="MHC")
    arg_parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["generate", "encode", "reparse", "all"],
        help=(
            "generate = MLLM archives (GPU); encode = CLIP text keys from JSONL; "
            "reparse = CPU re-parse of failed records with the repair parser "
            "(appends salvaged records to the JSONL, run encode afterwards)."
        ),
    )
    arg_parser.add_argument(
        "--archive_dir",
        type=str,
        default="./data/Archive",
        help="Directory for the per-split archive JSONL files.",
    )
    arg_parser.add_argument(
        "--EXP_FOLDER",
        type=str,
        default="./data/CLIP_Embedding",
        help="Output dir for the CLIP-encoded archive .pt caches.",
    )
    arg_parser.add_argument("--gt_dir", type=str, default="./data/gt")
    arg_parser.add_argument("--video_dir", type=str, default="./data/video")
    arg_parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    arg_parser.add_argument(
        "--out_model_tag",
        type=str,
        default="Qwen2.5-VL-7B-Instruct",
        help="Tag used in the archive JSONL filename.",
    )
    arg_parser.add_argument(
        "--clip_model", type=str, default="openai/clip-vit-large-patch14-336"
    )
    arg_parser.add_argument("--num_frames", type=int, default=8)
    arg_parser.add_argument(
        "--max_pixels",
        type=int,
        default=360 * 420,
        help="max_pixels per frame for the Qwen vision preprocessor.",
    )
    arg_parser.add_argument("--max_new_tokens", type=int, default=600)
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    arg_parser.add_argument("--splits", type=str, default="train,val,test")
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N items of each split (smoke test).",
    )
    arg_parser.add_argument(
        "--prompt_version",
        type=str,
        default="v1",
        choices=sorted(PROMPT_TEMPLATES.keys()),
        help=(
            "Prompt revision. v1 (default) = original prompt, original output "
            "paths (bit-compatible with all existing runs). v2 = target-recall "
            "+ mechanism-evidence revision; JSONL goes to "
            "<archive_dir>/<dataset>/v2/ and the encoded .pt to "
            "<EXP_FOLDER>/<dataset>/v2/ so v1 artefacts are never touched."
        ),
    )
    arg_parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Re-generate entries whose previous record has parse_ok=false.",
    )
    arg_parser.add_argument(
        "--selftest",
        action="store_true",
        help=(
            "CPU-only: render the prompt on sample texts and unit-test the JSON "
            "parser; loads NO model weights."
        ),
    )
    args = arg_parser.parse_args(args_list)
    return args


def read_gt(gt_path):
    items = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "id": str(obj["id"]),
                    "text": "" if obj.get("text") is None else str(obj["text"]),
                    "label": obj["label"],
                }
            )
    return items


# ----------------------------------------------------------------------------
# Frame sampler (reused verbatim from generate_VideoMLLM_embedding_HF.py)
# ----------------------------------------------------------------------------
def _sample_frame_indices(num_total, num_frames):
    if num_total <= 0:
        return None
    idx = np.linspace(0, num_total - 1, num_frames)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, num_total - 1)
    return idx.tolist()


def _decode_with_decord(video_path, num_frames):
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("native")
    vr = VideoReader(video_path, ctx=cpu(0))
    num_total = len(vr)
    indices = _sample_frame_indices(num_total, num_frames)
    if indices is None:
        return None
    batch = vr.get_batch(indices).asnumpy()  # [num_frames, H, W, 3] RGB
    frames = [Image.fromarray(batch[i]).convert("RGB") for i in range(batch.shape[0])]
    return frames


def _decode_with_pyav(video_path, num_frames):
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    num_total = stream.frames
    decoded = []
    if num_total and num_total > 0:
        target = set(_sample_frame_indices(num_total, num_frames))
        for i, frame in enumerate(container.decode(video=0)):
            if i in target:
                decoded.append((i, frame.to_image().convert("RGB")))
            if len(decoded) >= len(target) and i >= max(target):
                break
        container.close()
        if decoded:
            indices = _sample_frame_indices(num_total, num_frames)
            lookup = {i: img for i, img in decoded}
            avail = sorted(lookup.keys())
            frames = []
            for idx in indices:
                if idx in lookup:
                    frames.append(lookup[idx])
                else:
                    nearest = min(avail, key=lambda a: abs(a - idx))
                    frames.append(lookup[nearest])
            return frames
        return None
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_image().convert("RGB"))
    container.close()
    if not all_frames:
        return None
    indices = _sample_frame_indices(len(all_frames), num_frames)
    return [all_frames[i] for i in indices]


def load_video_frames(video_path, num_frames):
    if not os.path.exists(video_path):
        print("[WARN] missing video file: {}".format(video_path))
        return None, False

    frames = None
    try:
        frames = _decode_with_decord(video_path, num_frames)
    except Exception as e:  # noqa: BLE001
        print("[WARN] decord failed for {} ({}); trying PyAV.".format(video_path, repr(e)))
        frames = None

    if frames is None:
        try:
            frames = _decode_with_pyav(video_path, num_frames)
        except Exception as e:  # noqa: BLE001
            print("[WARN] PyAV failed for {} ({}).".format(video_path, repr(e)))
            frames = None

    if not frames:
        print("[WARN] no decodable frames for {}.".format(video_path))
        return None, False
    return frames, True


# ----------------------------------------------------------------------------
# Prompt building / JSON parsing / schema validation
# ----------------------------------------------------------------------------
def build_user_prompt(text, prompt_version="v1"):
    text = (text or "").strip()
    if not text:
        text = "(none)"
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + " ...[truncated]"
    return PROMPT_TEMPLATES[prompt_version].format(text=text)


def build_messages(frames, user_prompt):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


REFUSAL_PATTERNS = [
    "i'm sorry",
    "i am sorry",
    "i cannot assist",
    "i can't assist",
    "i cannot help",
    "i can't help",
    "i cannot fulfill",
    "i can't fulfill",
    "i cannot provide",
    "i can't provide",
    "i won't be able",
    "unable to assist",
    "as an ai",
]


def looks_like_refusal(raw):
    head = (raw or "").lower()[:400]
    return any(p in head for p in REFUSAL_PATTERNS)


def _extract_json_candidate(raw):
    """Pull the first balanced {...} block out of the raw generation."""
    if raw is None:
        return None
    text = raw.strip()
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_candidates(raw):
    """Yield progressively more aggressive repairs of a malformed generation.

    Observed failure modes (11/1596 in the full run, all non-refusals):
      * the model forgets the closing brace of the outer object (often after
        flattening top-level fields into "modality_cues") -> append '}'s;
      * output ends inside an unclosed markdown fence -> allow fence w/o close.
    """
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*(?:```|$)", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        return
    body = text[start:]
    last = body.rfind("}")
    if last >= 0:
        yield body[: last + 1]
    for extra in range(1, 4):
        if last >= 0:
            yield body[: last + 1] + "}" * extra
        yield body + "}" * extra


def _as_str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        v = [v] if v.strip() else []
    if not isinstance(v, list):
        return None
    out = []
    for x in v:
        if x is None:
            continue
        out.append(str(x).strip())
    return [x for x in out if x]


def validate_archive(obj):
    """Coerce a parsed dict into the fixed schema.

    Returns (archive_dict, schema_ok). archive_dict always has all five fields;
    schema_ok is False when any field needed repair beyond trivial coercion.
    """
    schema_ok = isinstance(obj, dict)
    obj = dict(obj) if isinstance(obj, dict) else {}

    # Hoist top-level fields the model sometimes flattens into modality_cues.
    _cues = obj.get("modality_cues")
    if isinstance(_cues, dict):
        for k in ("target_groups", "mechanism", "explicitness", "neutral_summary"):
            if k not in obj and k in _cues:
                obj[k] = _cues[k]
                schema_ok = False

    targets = _as_str_list(obj.get("target_groups"))
    if targets is None:
        targets, schema_ok = [], False

    mech = _as_str_list(obj.get("mechanism"))
    if mech is None:
        mech, schema_ok = [], False
    else:
        norm = []
        for m in mech:
            key = m.lower().strip().replace(" ", "_").replace("-", "_")
            if key in MECHANISM_VOCAB:
                norm.append(key)
            else:
                norm.append("other")
                schema_ok = False
        # dedupe, keep order
        seen = set()
        mech = [m for m in norm if not (m in seen or seen.add(m))]

    cues_in = obj.get("modality_cues")
    cues_in = cues_in if isinstance(cues_in, dict) else {}
    if not isinstance(obj.get("modality_cues"), dict):
        schema_ok = False
    cues = {}
    for k in ("visual", "speech", "on_screen_text"):
        v = cues_in.get(k)
        cues[k] = str(v).strip() if v is not None else ""

    expl = str(obj.get("explicitness") or "").lower().strip()
    if expl not in ("explicit", "implicit", "none"):
        schema_ok = False
        expl = expl if expl else "unknown"

    summary = obj.get("neutral_summary")
    summary = str(summary).strip() if summary is not None else ""
    if not summary:
        schema_ok = False

    archive = {
        "target_groups": targets,
        "mechanism": mech,
        "modality_cues": cues,
        "explicitness": expl,
        "neutral_summary": summary,
    }
    return archive, schema_ok


def parse_archive(raw):
    """raw generation -> (archive|None, parse_ok, schema_ok, refusal, error).

    error is None for clean parses and "repaired: ..." for outputs that only
    parsed after brace repair (still counted as parse_ok, schema_ok=False).
    """
    candidate = _extract_json_candidate(raw)
    obj = None
    first_err = "no JSON object found"
    if candidate is not None:
        try:
            obj = json.loads(candidate)
        except Exception as e:  # noqa: BLE001
            first_err = "json.loads: {}".format(e)
            obj = None
    repaired = False
    if not isinstance(obj, dict):
        for cand in _repair_candidates(raw):
            try:
                o = json.loads(cand)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(o, dict):
                obj = o
                repaired = True
                break
    if not isinstance(obj, dict):
        return None, False, False, looks_like_refusal(raw), first_err
    archive, schema_ok = validate_archive(obj)
    if repaired:
        schema_ok = False
        return archive, True, schema_ok, False, "repaired: {}".format(first_err)
    return archive, True, schema_ok, False, None


def archive_to_text(record):
    """Serialise one JSONL record into the flat text that gets CLIP-encoded.

    Falls back to the raw generation for parse failures (still informative);
    returns None (-> zero vector) for refusals / empty outputs.
    """
    if record.get("parse_ok") and record.get("archive"):
        a = record["archive"]
        cues = a.get("modality_cues") or {}
        parts = [
            "Targets: " + (", ".join(a.get("target_groups") or []) or "none"),
            "Mechanism: " + (", ".join(a.get("mechanism") or []) or "none"),
            "Visual cues: " + (cues.get("visual") or "none"),
            "Speech cues: " + (cues.get("speech") or "none"),
            "On-screen text: " + (cues.get("on_screen_text") or "none"),
            "Explicitness: " + (a.get("explicitness") or "unknown"),
            "Summary: " + (a.get("neutral_summary") or ""),
        ]
        return ". ".join(parts)
    raw = (record.get("raw_output") or "").strip()
    if not raw or record.get("refusal"):
        return None
    return raw


# ----------------------------------------------------------------------------
# Stage: generate (GPU)
# ----------------------------------------------------------------------------
def version_subdir(args):
    """'' for v1 (legacy layout, untouched); '<version>' for v2+."""
    return "" if args.prompt_version == "v1" else args.prompt_version


def jsonl_path(args, outname):
    return os.path.join(
        args.archive_dir,
        args.dataset,
        version_subdir(args),
        "{}_{}_archive.jsonl".format(outname, args.out_model_tag),
    )


def load_done_ids(path, retry_failed):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if retry_failed and not rec.get("parse_ok"):
                continue
            done.add(str(rec["id"]))
    return done


@torch.no_grad()
def generate_archive(frames, user_prompt, processor, model, device, max_new_tokens):
    messages = build_messages(frames, user_prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=None,
        videos=[frames],
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    new_ids = out_ids[:, inputs["input_ids"].shape[1] :]
    raw = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return raw.strip()


def run_generate(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = torch.device(args.device)
    print("Loading Qwen2.5-VL model: {}".format(args.model), flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)

    video_root = os.path.join(args.video_dir, args.dataset, "All")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] split '{}' has no output-name mapping; skipping.".format(split))
            continue
        outname = SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] gt file not found, skipping split '{}': {}".format(split, gt_path))
            continue
        items = read_gt(gt_path)
        if args.limit and args.limit > 0:
            items = items[: args.limit]

        out_path = jsonl_path(args, outname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        done = load_done_ids(out_path, args.retry_failed)
        todo = [it for it in items if it["id"] not in done]
        print(
            "[generate] split '{}': {} items, {} already done, {} to do -> {}".format(
                split, len(items), len(items) - len(todo), len(todo), out_path
            ),
            flush=True,
        )

        n_parse_ok = n_schema_ok = n_refusal = n_video_fail = 0
        with open(out_path, "a") as fout:
            for n, item in enumerate(todo):
                vid = item["id"]
                video_path = os.path.join(video_root, "{}.mp4".format(vid))
                frames, ok = load_video_frames(video_path, args.num_frames)
                record = {
                    "id": vid,
                    "split": outname,
                    "label": item["label"],
                    "raw_output": None,
                    "archive": None,
                    "parse_ok": False,
                    "schema_ok": False,
                    "refusal": False,
                    "error": None,
                }
                if not ok:
                    record["error"] = "video decode failed"
                    n_video_fail += 1
                else:
                    try:
                        raw = generate_archive(
                            frames,
                            build_user_prompt(item["text"], args.prompt_version),
                            processor,
                            model,
                            device,
                            args.max_new_tokens,
                        )
                        record["raw_output"] = raw
                        archive, parse_ok, schema_ok, refusal, error = parse_archive(raw)
                        record.update(
                            archive=archive,
                            parse_ok=parse_ok,
                            schema_ok=schema_ok,
                            refusal=refusal,
                            error=error,
                        )
                    except Exception as e:  # noqa: BLE001
                        record["error"] = "generation failed: {}".format(repr(e))
                n_parse_ok += int(record["parse_ok"])
                n_schema_ok += int(record["schema_ok"])
                n_refusal += int(record["refusal"])
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                if (n + 1) % 10 == 0 or (n + 1) == len(todo):
                    print(
                        "  [{}] {}/{} parse_ok={} schema_ok={} refusal={} video_fail={}".format(
                            split, n + 1, len(todo), n_parse_ok, n_schema_ok,
                            n_refusal, n_video_fail,
                        ),
                        flush=True,
                    )
        print(
            "[generate] split '{}' done: new={} parse_ok={} schema_ok={} "
            "refusal={} video_fail={}".format(
                split, len(todo), n_parse_ok, n_schema_ok, n_refusal, n_video_fail
            ),
            flush=True,
        )


# ----------------------------------------------------------------------------
# Stage: reparse (CPU): salvage previously parse-failed records
# ----------------------------------------------------------------------------
def run_reparse(args):
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in SPLIT_TO_OUTNAME:
            continue
        outname = SPLIT_TO_OUTNAME[split]
        path = jsonl_path(args, outname)
        if not os.path.exists(path):
            continue
        records = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records[str(rec["id"])] = rec  # last record wins
        salvaged, still_failed = 0, 0
        appended = []
        for rec in records.values():
            if rec.get("parse_ok") or not rec.get("raw_output"):
                continue
            archive, parse_ok, schema_ok, refusal, error = parse_archive(
                rec["raw_output"]
            )
            if parse_ok:
                new_rec = dict(rec)
                new_rec.update(
                    archive=archive,
                    parse_ok=True,
                    schema_ok=schema_ok,
                    refusal=refusal,
                    error=error,
                )
                appended.append(new_rec)
                salvaged += 1
            else:
                still_failed += 1
        if appended:
            with open(path, "a") as fout:
                for rec in appended:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            "[reparse] '{}': salvaged={} still_failed={} -> {}".format(
                outname, salvaged, still_failed, path
            ),
            flush=True,
        )


# ----------------------------------------------------------------------------
# Stage: encode (CLIP text encoder over the archive texts)
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_text_clip(text, tokenizer, text_model, device):
    """Chunked CLIP text encoding (77-token windows), mean-pooled pooler_output.

    Mirrors encode_text in generate_VideoCLIP_embedding_HF.py.
    """
    text = text if text is not None else ""
    content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    max_len = getattr(tokenizer, "model_max_length", 77)
    if not max_len or max_len > 77:
        max_len = 77
    content_window = max_len - 2

    if len(content_ids) <= content_window:
        windows = [content_ids] if content_ids else [[]]
    else:
        windows = [
            content_ids[i : i + content_window]
            for i in range(0, len(content_ids), content_window)
        ]

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    pooled = []
    for window in windows:
        ids = []
        if bos is not None:
            ids.append(bos)
        ids.extend(window)
        if eos is not None:
            ids.append(eos)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        out = text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled.append(out.pooler_output.detach().cpu().float())
    pooled = torch.cat(pooled, dim=0)
    return pooled.mean(dim=0)


def run_encode(args):
    from transformers import CLIPTokenizer, CLIPTextModel

    device = torch.device(args.device)
    print("Loading CLIP text encoder: {}".format(args.clip_model), flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    text_model = CLIPTextModel.from_pretrained(args.clip_model)
    text_model.to(device).eval()
    dt = text_model.config.hidden_size

    out_dir = os.path.join(args.EXP_FOLDER, args.dataset, version_subdir(args))
    os.makedirs(out_dir, exist_ok=True)
    clip_tag = args.clip_model.replace("/", "_")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in SPLIT_TO_OUTNAME:
            continue
        outname = SPLIT_TO_OUTNAME[split]
        gt_path = os.path.join(args.gt_dir, args.dataset, "{}.jsonl".format(split))
        if not os.path.exists(gt_path):
            print("[WARN] gt file not found, skipping split '{}'.".format(split))
            continue
        items = read_gt(gt_path)
        if args.limit and args.limit > 0:
            items = items[: args.limit]

        arc_path = jsonl_path(args, outname)
        records = {}
        if os.path.exists(arc_path):
            with open(arc_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    records[str(rec["id"])] = rec  # last record wins (retry runs)
        else:
            print("[WARN] archive JSONL missing for split '{}': {}".format(split, arc_path))

        ids, feats, labels = [], [], []
        zero_guard = 0
        missing = 0
        for item in items:
            vid = item["id"]
            rec = records.get(vid)
            if rec is None:
                missing += 1
                vec = torch.zeros(dt, dtype=torch.float32)
                zero_guard += 1
            else:
                arch_text = archive_to_text(rec)
                if arch_text is None:
                    vec = torch.zeros(dt, dtype=torch.float32)
                    zero_guard += 1
                else:
                    vec = encode_text_clip(arch_text, tokenizer, text_model, device)
            ids.append(vid)
            feats.append(vec)
            labels.append(item["label"])

        feats = torch.stack(feats, dim=0).float()
        if all(
            isinstance(l, int) or (isinstance(l, float) and float(l).is_integer())
            for l in labels
        ):
            labels_t = torch.tensor([int(l) for l in labels], dtype=torch.long)
        else:
            labels_t = torch.tensor([float(l) for l in labels], dtype=torch.float32)

        # CONTRACT: ids is a list containing ONE sublist of all string ids.
        save_obj = {"ids": [ids], "text_feats": feats, "labels": labels_t}
        out_path = os.path.join(
            out_dir, "{}_archive_{}_HF.pt".format(outname, clip_tag)
        )
        torch.save(save_obj, out_path)
        print(
            "[encode] '{}': N={} Dt={} zero-vector={} missing-archive={} -> {}".format(
                outname, len(ids), dt, zero_guard, missing, out_path
            ),
            flush=True,
        )


# ----------------------------------------------------------------------------
# Selftest (CPU, no model weights): prompt rendering + parser unit checks
# ----------------------------------------------------------------------------
SELFTEST_SAMPLES = [
    (
        "EN benign",
        "my queer journey (+hair bc why not) | picrew made my @.nuggts on TikTok "
        "| #queer #nonbinary #picrew . Can't the future just wait...",
    ),
    (
        "EN hostile-looking",
        "The Most INSANE FAKE Couple Ever! . So I get back from a weeklong work "
        "trip ... I'm not even sure this guy is into her at all.",
    ),
    (
        "ZH sample",
        "“瘦”马瘤与老鲇莉，丑逼！ . \U0001f3bc바아지.",
    ),
]

SELFTEST_OUTPUTS = [
    (
        "clean JSON",
        json.dumps(
            {
                "target_groups": ["women"],
                "mechanism": ["mockery", "stereotyping"],
                "modality_cues": {
                    "visual": "A man films a couple in a parking lot.",
                    "speech": "The narrator ridicules the couple as fake.",
                    "on_screen_text": "Caption reads 'FAKE couple'.",
                },
                "explicitness": "implicit",
                "neutral_summary": "A commentary video reacting to a couple's clip.",
            }
        ),
        dict(parse_ok=True, schema_ok=True, refusal=False),
    ),
    (
        "fenced JSON with prose",
        "Sure, here is the analysis:\n```json\n"
        '{"target_groups": [], "mechanism": [], "modality_cues": {"visual": "", '
        '"speech": "A person narrates a cooking recipe.", "on_screen_text": ""}, '
        '"explicitness": "none", "neutral_summary": "A cooking tutorial video."}'
        "\n```\nLet me know if you need more.",
        dict(parse_ok=True, schema_ok=True, refusal=False),
    ),
    (
        "refusal",
        "I'm sorry, but I can't help with analysing this content.",
        dict(parse_ok=False, schema_ok=False, refusal=True),
    ),
    (
        "invalid mechanism + missing summary",
        '{"target_groups": "immigrants", "mechanism": ["hate speech"], '
        '"modality_cues": {"visual": "A crowd."}, "explicitness": "EXPLICIT"}',
        dict(parse_ok=True, schema_ok=False, refusal=False),
    ),
    (
        "missing outer brace + fields flattened into modality_cues",
        "```json\n"
        '{\n  "target_groups": [],\n  "mechanism": [],\n  "modality_cues": {\n'
        '    "visual": "Two people sit on a couch.",\n    "speech": "",\n'
        '    "on_screen_text": "couple positions",\n'
        '    "explicitness": "none",\n'
        '    "neutral_summary": "Two people interact playfully on a couch."\n'
        "}\n```",
        dict(parse_ok=True, schema_ok=False, refusal=False),
    ),
]


def run_selftest(args):
    print("=== selftest: prompt rendering (version={}) ===".format(
        args.prompt_version))
    for name, text in SELFTEST_SAMPLES:
        prompt = build_user_prompt(text, args.prompt_version)
        print("\n--- sample: {} ---".format(name))
        print(prompt[:900])
    print("\n(system prompt)\n" + SYSTEM_PROMPT)
    print("\n(jsonl path for split 'train')\n" + jsonl_path(args, "train"))

    print("\n=== selftest: parser unit checks ===")
    n_fail = 0
    for name, raw, expect in SELFTEST_OUTPUTS:
        archive, parse_ok, schema_ok, refusal, error = parse_archive(raw)
        got = dict(parse_ok=parse_ok, schema_ok=schema_ok, refusal=refusal)
        status = "PASS" if got == expect else "FAIL"
        if status == "FAIL":
            n_fail += 1
        print(
            "[{}] {}: got={} expect={} error={}".format(status, name, got, expect, error)
        )
        if archive is not None:
            rec = {"parse_ok": parse_ok, "archive": archive, "raw_output": raw}
            print("    archive_text: {}".format((archive_to_text(rec) or "")[:200]))
    if n_fail:
        raise SystemExit("selftest FAILED: {} case(s)".format(n_fail))
    print("\nselftest OK (all {} parser cases passed)".format(len(SELFTEST_OUTPUTS)))


def main(args):
    if args.selftest:
        run_selftest(args)
        return
    if args.stage in ("generate", "all"):
        run_generate(args)
    if args.stage == "reparse":
        run_reparse(args)
    if args.stage in ("encode", "all"):
        run_encode(args)


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
