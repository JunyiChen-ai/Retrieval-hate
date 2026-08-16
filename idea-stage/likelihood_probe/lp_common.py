"""LIKELIHOOD PROBE -- shared input construction.

Reads the stance signal by COMPARING LIKELIHOODS of two candidate continuations under a
local Qwen VL model. The model emits no token; nothing is generated, nothing is sampled.

Design frozen in idea-stage/LIKELIHOOD_PROBE_FREEZE.md before any eval item is scored.
"""
import base64  # noqa: F401  (kept so the frame pipeline mirrors run_pilot's JPEG round trip)
import hashlib
import io
import json
import os
import re
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")
CS = os.path.join(ROOT, "idea-stage", "contrast_stance")
MSP = os.path.join(ROOT, "idea-stage", "mask_stance_pilot")
for p in (SP, CS):
    if p not in sys.path:
        sys.path.insert(0, p)

from contrast_prompts import BANK, N_PAIRS  # noqa: E402
from run_contrast import targets_map, lang_of  # noqa: E402
from run_pilot import load_texts, MAX_SIDE, JPEG_Q  # noqa: E402

TITLE_DS = {"MHC", "MHC_zh"}
FRAME_DS = ["HateMM", "MHC", "MHC_zh"]
N_FRAMES = 8
MAX_PIXELS = 360 * 420          # project convention (generate_VideoMLLM_embedding_HF)
CTRL_SEED = 20260817
N_CTRL = 20

# freeze B.8 of the contrast round: eval items already burned as qualitative smoke checks
SMOKE_EVAL = {("MHC", "KDcCiUU8q5E"), ("HateMM", "non_hate_video_32"),
              ("HateMM", "non_hate_video_16")}
# the one eval item DashScope refused in all three previous rounds; excluded from the
# 32-row primary denominator so this round is scored on exactly the same rows.
API_LOST = {("MHC_zh", "BV1m8411z7mV")}

CONT_PREFIX = "Pinned comment:"

CTX = """VIDEO EVIDENCE
{frames_note}
Speech transcript (full automatic transcript{title_note}; it may be in English or Chinese, \
and may be noisy or empty):
<<<
{transcript}
>>>

The person who made and posted this video -- the uploader/publisher, NOT anyone who is merely \
filmed, quoted, sampled or replayed inside it -- pinned one comment under it that sums up their \
own view.

"""


# ------------------------------------------------------------------ items
def eval_items():
    return json.load(open(os.path.join(SP, "sample.json")))["eval"]


def ctrl_items():
    """20 non-hate TRAIN videos with frames, sampled by a fixed seed, 5/5/10 over the three
    frame-bearing datasets. Used only for the template-prior control; never scored for stance."""
    import random
    rng = random.Random(CTRL_SEED)
    out = []
    for ds, k in (("HateMM", 7), ("MHC", 7), ("MHC_zh", 6)):
        rows = [json.loads(l) for l in
                open(os.path.join(ROOT, "data", "gt", ds, "train.jsonl"), encoding="utf-8")]
        cand = sorted([r["id"] for r in rows if r.get("label") == 0
                       and os.path.isdir(os.path.join(ROOT, "data", "lora_frames", ds, r["id"]))])
        for vid in rng.sample(cand, k):
            out.append({"dataset": ds, "id": vid, "group": "PRIOR_CTRL", "split": "train"})
    return out


def ctrl_texts(ds):
    o = {}
    for line in open(os.path.join(ROOT, "data", "gt", ds, "train.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        o[r["id"]] = re.sub(r"<[^>]+>", "", r.get("text", "") or "")
    return o


# ------------------------------------------------------------------ frames
def frames_of(ds, vid, n=N_FRAMES):
    """Same frame directory, same sampler and same 512-px / JPEG-q80 round trip the API
    rounds used; returned as PIL images instead of base64 data URLs."""
    d = os.path.join(ROOT, "data", "lora_frames", ds, vid)
    if not os.path.isdir(d):
        return []
    fs = sorted(os.listdir(d), key=lambda x: int(re.findall(r"\d+", x)[0]))
    if n < len(fs):
        step = len(fs) / n
        fs = [fs[int(i * step)] for i in range(n)]
    out = []
    for f in fs:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        if max(im.size) > MAX_SIDE:
            s = MAX_SIDE / max(im.size)
            im = im.resize((max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_Q)
        buf.seek(0)
        out.append(Image.open(buf).convert("RGB"))
    return out


# ------------------------------------------------------------------ prompt assembly
def context_text(ds, n_frames, transcript):
    frames_note = (f"{n_frames} evenly spaced frames of the video are attached above, "
                   "in temporal order." if n_frames else
                   "No video frames are available for this item; judge from the transcript alone.")
    title_note = ("; its first sentence, before the first ' . ', is the video TITLE"
                  if ds in TITLE_DS else "")
    return CTX.format(frames_note=frames_note, title_note=title_note,
                      transcript=(transcript or "").strip() or "(empty)")


def continuations(lang, variant, pair, tgt):
    """(endorsing, opposing) continuation strings, identical bank as the contrast round C1.1."""
    end_t, opp_t = BANK[(variant, lang)][pair]
    if variant == "v2":
        end_t, opp_t = end_t.format(t=tgt), opp_t.format(t=tgt)
    return end_t, opp_t


def plan(items, texts_fn):
    """Frozen request order: item -> variant(v1,v2) -> pair(0..4). Each entry carries the two
    continuations whose log-likelihoods are compared."""
    TG = targets_map()
    cache = {}
    out = []
    for it in items:
        ds, vid = it["dataset"], it["id"]
        if ds not in cache:
            cache[ds] = texts_fn(ds)
        tx = (cache[ds].get(vid) or "").strip()
        lang = lang_of(ds, tx)
        tgt = TG.get((ds, vid))
        for variant in ("v1", "v2"):
            if variant == "v2" and not tgt:
                continue
            for p in range(N_PAIRS):
                end_t, opp_t = continuations(lang, variant, p, tgt)
                out.append({"dataset": ds, "id": vid, "group": it["group"],
                            "variant": variant, "pair": p, "lang": lang,
                            "target": tgt if variant == "v2" else None,
                            "transcript": tx,
                            "endorse_text": end_t, "oppose_text": opp_t})
    return out


def sha_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                          .encode()).hexdigest()[:16]
