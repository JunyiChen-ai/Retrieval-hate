"""Revision 5 step 1 (README section 17.1): soft first-token answers on the query tree.

One yes/no question per queryable node of data/vlm_tree/<Corpus>/manifest_words.jsonl, under a PER-DATASET definition
of the harmful content (the user's 2026-09-29 ruling: each dataset may carry its own platform-style definition, so the
prompt is tuned per dataset). Instead of decoding a 0-3 score, the answer is the first-token probability mass on
"Yes" versus "No" (Huang, Devereux & Wang 2026, arXiv 2608.08315; Probe-VAD 2609.17211; Song & Lee 2608.21244):
    p_yes = P(Yes-tokens) / (P(Yes-tokens) + P(No-tokens))   from the top-k log-probabilities of the first token.
Same model, frames and resolution as the decoded answers (Qwen2.5-VL-7B-Instruct, 4 frames, max_pixels 360*420, vLLM).

Views (revision 5 step 3, README section 17.3): --view both (4 frames + the node's word-timestamp transcript, the
method's input), frames (4 frames only), text (transcript only, no video). The three views of the same node are the
three "annotators" of the triplet / third-order-agreement estimate of the three-state answer model.

    python experiments/20260925_query_paradigm/extract_tree_soft.py --corpus hateclipseg --splits test,val,train --view both
Output: data/vlm_tree/<Corpus>/soft_<view>_<prompt>[.shard<i>of<n>].jsonl, one line per video:
    {"id", "split", "T", "view", "prompt", "nodes": [[a, b, [p_yes, mass_yes, mass_no] or null, top_token], ...]}
    --prompt p5 (five categories, expected levels): soft5_<view>_p5[.shard].jsonl, node answer =
    [[E_1..E_5], [[p0..p3] x 5]] with E_k the expected level 0-3 of category k (k = 1 the per-dataset definition).
Resumes: videos already in the output file are skipped. --limit N (functional check of the parsing only) stops after N videos.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
MAX_PIXELS = 360 * 420
MAX_WORDS = 2000
PROMPT_VERSION = "p1"

# Per-dataset definitions (README section 17.1). Each follows the dataset's own annotation scheme:
#   HateMM (Das et al. 2023): hate videos = attacks on people for a protected attribute; BitChute.
#   HateClipSeg (README of the dataset): five offensive categories, a segment is positive if any applies; BitChute / YouTube.
#   DeHate (Zhang et al. 2025): hate, explicit or implicit, six target attributes; decentralised platforms.
DEFINITIONS = {
    "hatemm": {
        "platform": "BitChute",
        "name": "hateful content",
        "text": (
            "Hateful content is any communication that attacks, dehumanises, demeans, threatens, mocks or incites "
            "hostility against a person or group because of a protected attribute: race, ethnicity, national origin, "
            "religion (e.g. Jews, Muslims, Christians), gender, sexual orientation or gender identity (e.g. LGBTQ+ "
            "people), disability, or immigration status. It includes slurs, dehumanising comparisons, hateful "
            "stereotypes, conspiracy claims about such a group, calls for their exclusion or harm, and hateful symbols "
            "or gestures. Profanity, insults or violence that are NOT aimed at a group for such an attribute are not "
            "hateful content."
        ),
    },
    "hateclipseg": {
        "platform": "BitChute and YouTube",
        "name": "offensive content",
        "text": (
            "Offensive content is content in ANY of these five categories:\n"
            "1. Hateful: attacks, demeans, dehumanises or incites hostility against people for a protected attribute "
            "(race, ethnicity, nationality, religion, gender, sexual orientation, disability, etc.).\n"
            "2. Insulting: insults, mocks, humiliates, harasses or verbally abuses a person or group (slurs, profanity "
            "directed at someone, name-calling, degrading remarks), whether or not a protected attribute is involved.\n"
            "3. Sexual: sexual or sexually explicit content, sexual remarks or nudity.\n"
            "4. Violent: depicts, glorifies, threatens or incites violence, injury, weapons used against people, or gore.\n"
            "5. Self-harm: depicts or encourages self-harm, suicide or dangerous acts.\n"
            "A part is offensive if at least one category applies to it."
        ),
        "hate5": (
            "Hateful content attacks, demeans, dehumanises or incites hostility against people for a protected "
            "attribute (race, ethnicity, nationality, religion, gender, sexual orientation, disability, etc.), by "
            "speech, on-screen text, symbols, gestures or imagery."
        ),
    },
    "dehate": {
        "platform": "decentralised video platforms such as BitChute and Odysee",
        "name": "hateful content",
        "text": (
            "Hateful content is content that attacks, dehumanises, demeans, threatens or promotes hostility, "
            "discrimination or violence against people because of race or ethnicity, gender, religion, sexual "
            "orientation, disability, or immigration status. It may be EXPLICIT (slurs, open calls for harm or "
            "exclusion, dehumanising claims) or IMPLICIT (coded language, dog whistles, memes, symbols, sarcasm, "
            "conspiracy narratives that blame such a group). It can be carried by speech, on-screen text, images or "
            "gestures. Rudeness or violence not aimed at such a group is not hateful content."
        ),
    },
}

SYSTEM_TEMPLATE = (
    "You are a careful content-moderation rater for {platform}. The platform's policy definition of {name}:\n"
    "{text}\n\n"
    "{inputs} Judge ONLY what is present in this part of the video, not what the rest of the video might contain."
)
INPUTS = {
    "both": "You are shown a few frames sampled from ONE part of a video and the speech transcript of that part.",
    "frames": "You are shown a few frames sampled from ONE part of a video.",
    "text": "You are given the speech transcript of ONE part of a video (no frames).",
}
USER_BOTH = "Transcript of this part: \"{asr}\"\n\nDoes this part contain {name} as defined above? Answer Yes or No."
USER_FRAMES = "Does this part contain {name} as defined above? Answer Yes or No."
USER_TEXT = "Transcript of this part: \"{asr}\"\n\nDoes this part contain {name} as defined above? Answer Yes or No."
NO_SPEECH = "(no speech in this part)"

# README 17.1 variant "five categories, soft": the revision-4 five-category rating (extract_tree_answers.py taxonomy),
# category 1 replaced by the per-dataset definition; per category the expected level 0-3 from the first-token
# distribution at that category's position (SamplingParams logprobs at every generated position).
SYSTEM5_TEMPLATE = (
    "You are a careful content-moderation rater for {platform}. {inputs} Rate the harmful content present IN THIS "
    "PART in five categories:\n"
    "1. hate (the platform's policy definition): {text}\n"
    "2. harassment: content that insults, mocks, bullies or threatens a person or group, not for a protected "
    "attribute.\n"
    "3. sexual: sexual or sexually explicit content.\n"
    "4. violence: content that depicts, glorifies or threatens violence, injury or gore.\n"
    "5. self-harm: content that depicts or encourages self-harm, suicide or dangerous acts.\n"
    "Harmful content can be carried by any channel: spoken words, on-screen text, symbols, gestures, or imagery. "
    "Judge ONLY what is present in this part of the video, not what the rest of the video might contain."
)
USER5_SCALE = (
    "Scale for each category: 0 = none; 1 = faint or ambiguous cue; 2 = clear but implicit or coded; "
    "3 = explicit and unambiguous.\n"
    "Answer with exactly five integers separated by spaces, in the order: hate harassment sexual violence "
    "self-harm. No other text."
)
USER5_BOTH = "Transcript of this part: \"{asr}\"\n\n" + USER5_SCALE
USER5_FRAMES = USER5_SCALE
USER5_TEXT = USER5_BOTH
N_CAT5, MAX_TOKENS5 = 5, 16


def digit_dists(logprobs_per_pos):
    """The generated positions whose top token is a digit 0-3: for each of the first N_CAT5 of them, the
    probability vector over 0..3 (renormalised over the digit mass in the top-k) and its expectation."""
    out = []
    for top in logprobs_per_pos:
        best = max(top.values(), key=lambda lp: lp.logprob) if top else None
        tok = (best.decoded_token or "").strip() if best else ""
        if tok in ("0", "1", "2", "3"):
            pv = [0.0] * 4
            for lp in top.values():
                t = (lp.decoded_token or "").strip()
                if t in ("0", "1", "2", "3"):
                    pv[int(t)] += math.exp(lp.logprob)
            z = sum(pv)
            if z <= 0:
                continue
            pv = [x / z for x in pv]
            out.append([round(sum(i * x for i, x in enumerate(pv)), 6), [round(x, 6) for x in pv]])
        if len(out) == N_CAT5:
            break
    if len(out) == N_CAT5 - 1:                 # the model sometimes stops after four integers: the fifth = 0
        out.append([0.0, [1.0, 0.0, 0.0, 0.0]])
    return out if len(out) == N_CAT5 else None


def yes_no_mass(top):
    """Probability mass on Yes-like and No-like tokens among the top-k first-token log-probabilities."""
    my = mn = 0.0
    for lp in top.values():
        tok = (lp.decoded_token or "").strip().lower()
        if tok.startswith("yes"):
            my += math.exp(lp.logprob)
        elif tok.startswith("no") and not tok.startswith("not") and not tok.startswith("non"):
            mn += math.exp(lp.logprob)
    return my, mn


def smart_size(h, w, factor=28, max_pixels=MAX_PIXELS, min_pixels=56 * 56):
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return hb, wb


def load_frames(fdir, idx):
    ims = [Image.open(os.path.join(fdir, "%06d.jpg" % i)).convert("RGB") for i in idx]
    h, w = smart_size(ims[0].height, ims[0].width)
    return np.stack([np.asarray(im.resize((w, h), Image.BICUBIC)) for im in ims])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=tuple(CORPUS_DIR))
    ap.add_argument("--splits", default="test,val,train")
    ap.add_argument("--view", default="both", choices=("both", "frames", "text"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--videos-per-batch", type=int, default=24)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--prompt", default="p1", choices=("p1", "p5"),
                    help="p1 = one Yes/No question (README 17.1); p5 = five categories, expected level each (variant)")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes videos with index %% n == i")
    ap.add_argument("--manifest", default="manifest_words.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="functional check: stop after this many videos")
    a = ap.parse_args()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    d = CORPUS_DIR[a.corpus]
    base = os.path.join(ROOT, "data", "vlm_tree", d)
    si, sn = (int(x) for x in a.shard.split("/"))
    prefix = ("soft_%s_%s" if a.prompt == "p1" else "soft5_%s_%s") % (a.view, a.prompt)
    out_path = os.path.join(base, "%s%s.jsonl" % (prefix, "" if sn == 1 else ".shard%dof%d" % (si, sn)))
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    splits = a.splits.split(",")
    todo = [json.loads(l) for l in open(os.path.join(base, a.manifest))]
    todo = [r for i, r in enumerate(todo) if i % sn == si]
    todo = sorted([r for r in todo if r["split"] in splits and r["id"] not in done], key=lambda r: splits.index(r["split"]))
    if a.limit:
        todo = todo[:a.limit]
    print("%s %s shard %s: %d videos to do, %d done -> %s" % (a.corpus, a.view, a.shard, len(todo), len(done), out_path),
          flush=True)
    if not todo:
        return
    D = DEFINITIONS[a.corpus]
    system = (SYSTEM_TEMPLATE if a.prompt == "p1" else SYSTEM5_TEMPLATE).format(
        platform=D["platform"], name=D["name"], text=D.get("hate5", D["text"]) if a.prompt == "p5" else D["text"],
        inputs=INPUTS[a.view])
    proc = AutoProcessor.from_pretrained(a.model)
    user_content = ([{"type": "video"}] if a.view != "text" else []) + [{"type": "text", "text": "{USER}"}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content}]
    template = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert template.count("{USER}") == 1 and (("<|video_pad|>" in template) == (a.view != "text"))
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 0, "video": 1 if a.view != "text" else 0}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=1 if a.prompt == "p1" else MAX_TOKENS5, logprobs=a.topk)
    pool = ThreadPoolExecutor(16)
    t0, n_calls, n_fail = time.time(), 0, 0
    with open(out_path, "a") as fh:
        for s in range(0, len(todo), a.videos_per_batch):
            batch = todo[s:s + a.videos_per_batch]
            jobs = []
            for r in batch:
                fdir = os.path.join(ROOT, "data", "frames_1fps", d, r["id"])
                for (na, nb, idx, text) in r["nodes"]:
                    words = text.split()
                    if len(words) > MAX_WORDS:
                        text = " ".join(words[:MAX_WORDS])
                    jobs.append((r["id"], na, nb, fdir, idx, text.strip() or NO_SPEECH))
            inputs = []
            frames = list(pool.map(lambda j: load_frames(j[3], j[4]), jobs)) if a.view != "text" else [None] * len(jobs)
            for j, f in zip(jobs, frames):
                asr = j[5].replace('"', "'")
                ub, uf, ut = ((USER_BOTH, USER_FRAMES, USER_TEXT) if a.prompt == "p1"
                              else (USER5_BOTH, USER5_FRAMES, USER5_TEXT))
                if a.view == "both":
                    user = ub.format(asr=asr, name=D["name"])
                elif a.view == "frames":
                    user = uf.format(name=D["name"])
                else:
                    user = ut.format(asr=asr, name=D["name"])
                item = {"prompt": template.replace("{USER}", user)}
                if f is not None:
                    item["multi_modal_data"] = {"video": f}
                inputs.append(item)
            outs = llm.generate(inputs, sp, use_tqdm=False)
            by_vid = {}
            for j, o in zip(jobs, outs):
                out0 = o.outputs[0]
                if a.prompt == "p1":
                    top = out0.logprobs[0] if out0.logprobs else {}
                    my, mn = yes_no_mass(top)
                    ans = None if my + mn <= 0 else [round(my / (my + mn), 6), round(my, 6), round(mn, 6)]
                else:                        # p5: [[E_1..E_5], [[p0..p3] x 5]]
                    dd = digit_dists(out0.logprobs or [])
                    ans = None if dd is None else [[e for e, _ in dd], [pv for _, pv in dd]]
                n_fail += ans is None
                by_vid.setdefault(j[0], []).append([j[1], j[2], ans, out0.text.strip()])
            for r in batch:
                fh.write(json.dumps({"id": r["id"], "split": r["split"], "T": r["T"], "view": a.view,
                                     "prompt": a.prompt, "nodes": by_vid.get(r["id"], [])}) + "\n")
            fh.flush()
            n_calls += len(jobs)
            el = time.time() - t0
            print("%d/%d videos, %d calls, %.1f calls/s, unparsed so far %d" % (
                min(s + a.videos_per_batch, len(todo)), len(todo), n_calls, n_calls / el, n_fail), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
