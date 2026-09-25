"""Pilot for revision 4 (README section 11; development evidence on test under rule 10): does the VLM separate
harmful from non-harmful short nodes inside a hateful video better when it sees the transcript of the whole video,
with the speech of the asked node marked, instead of the node's transcript only (README section 2.1)?

Sample (fixed seed 0): test videos of one corpus; nodes of length 4-16 s; from positive videos that have both
classes, and from negative videos. Same 4 frames per node as the cached answers, same model, same five
categories and 0-3 scale; only the text part of the question changes.

    python experiments/20260925_query_paradigm/pilot_context_prompt.py build --corpus hatemm      # local (ASR)
    python experiments/20260925_query_paradigm/pilot_context_prompt.py run --corpus hatemm        # lab (vLLM)
    python experiments/20260925_query_paradigm/pilot_context_prompt.py eval --corpus hatemm       # local (GT)
Files: runs/20260925_query_paradigm/pilot_context/<corpus>_{jobs,answers}.jsonl and <corpus>_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "runs", "20260925_query_paradigm", "pilot_context")
MAX_WORDS = 1500

SYSTEM_PROMPT = (
    "You are a careful content-moderation rater. You are shown a few frames sampled from ONE marked part of a "
    "video, and the speech transcript of the WHOLE video in which the speech of the marked part is enclosed in "
    "<<< and >>>. Rate the harmful content present IN THE MARKED PART in five categories:\n"
    "1. hate: content that attacks, dehumanises, demeans, threatens or incites against people for a protected "
    "attribute (race, ethnicity, religion, nationality, gender, sexual orientation, disability, caste, "
    "immigration status, etc.).\n"
    "2. harassment: content that insults, mocks, bullies or threatens a person or group, not for a protected "
    "attribute.\n"
    "3. sexual: sexual or sexually explicit content.\n"
    "4. violence: content that depicts, glorifies or threatens violence, injury or gore.\n"
    "5. self-harm: content that depicts or encourages self-harm, suicide or dangerous acts.\n"
    "Harmful content can be carried by any channel: spoken words, on-screen text, symbols, gestures, or imagery. "
    "Use the rest of the transcript only to understand what the marked part means. Content outside the marked "
    "part does not count: a harmless marked part of a harmful video gets 0."
)
USER_PROMPT = (
    "Transcript of the whole video (marked part in <<< >>>): \"{asr}\"\n\n"
    "Scale for each category: 0 = none; 1 = faint or ambiguous cue; 2 = clear but implicit or coded; "
    "3 = explicit and unambiguous.\n"
    "Answer with exactly five integers separated by spaces, in the order: hate harassment sexual violence "
    "self-harm, for the marked part only. No other text."
)
NO_SPEECH = "(no speech in the marked part)"


def build(corpus):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import hier_evidence_common as hc
    import build_tree_manifest as btm
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    words = btm.load_words(corpus)
    man = {}
    for line in open(os.path.join(ROOT, "data", "vlm_tree", btm.CORPUS_DIR[corpus], "manifest.jsonl")):
        r = json.loads(line)
        if r["split"] == "test":
            man[r["id"]] = r
    rng = np.random.RandomState(0)
    jobs = []
    for v in ids["test"]:
        G = np.asarray(gt["test"][v])
        pos = labels[v] == 1 and 0 < G.mean() < 1
        if not (pos or labels[v] == 0):
            continue
        nodes = [n for n in man[v]["nodes"] if 4 <= n[1] - n[0] < 16]
        k = min(len(nodes), 24 if pos else 12)
        for i in rng.choice(len(nodes), size=k, replace=False):
            a, b, idx, _text = nodes[i]
            w = words.get(v, [])
            before = [t for m, t in w if m < a]
            part = [t for m, t in w if a <= m < b]
            after = [t for m, t in w if m >= b]
            room = max(0, MAX_WORDS - len(part))
            nb = min(len(before), room - min(len(after), room // 2))
            na = min(len(after), room - nb)
            text = " ".join(before[len(before) - nb:] + ["<<<"] + (part or [NO_SPEECH]) + [">>>"] + after[:na])
            jobs.append({"id": v, "a": a, "b": b, "idx": idx, "text": " ".join(text.split()),
                         "harm": int(G[a:b].max() > 0), "pos_video": int(labels[v])})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "%s_jobs.jsonl" % corpus), "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print("%d jobs (%d in positive videos, %d harmful)" % (len(jobs), sum(j["pos_video"] for j in jobs),
                                                          sum(j["harm"] for j in jobs)))


def run(corpus, model, gpu_mem):
    import extract_tree_answers as eta
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    jobs = [json.loads(l) for l in open(os.path.join(OUT, "%s_jobs.jsonl" % corpus))]
    proc = AutoProcessor.from_pretrained(model)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "{USER}"}]}]
    template = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=8192,
              limit_mm_per_prompt={"image": 0, "video": 1}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    d = eta.CORPUS_DIR[corpus]
    frames = list(ThreadPoolExecutor(16).map(
        lambda j: eta.load_frames(os.path.join(ROOT, "data", "frames_1fps", d, j["id"]), j["idx"]), jobs))
    inputs = [{"prompt": template.replace("{USER}", USER_PROMPT.format(asr=j["text"].replace('"', "'"))),
               "multi_modal_data": {"video": f}} for j, f in zip(jobs, frames)]
    outs = llm.generate(inputs, sp, use_tqdm=False)
    with open(os.path.join(OUT, "%s_answers.jsonl" % corpus), "w") as fh:
        for j, o in zip(jobs, outs):
            raw = o.outputs[0].text.strip()
            fh.write(json.dumps({"id": j["id"], "a": j["a"], "b": j["b"], "answer": eta.parse(raw), "raw": raw}) + "\n")
    print("DONE %d answers" % len(jobs), flush=True)


def evaluate(corpus):
    import data as qdata
    cached, _ = qdata.load_answers(corpus)
    jobs = {(j["id"], j["a"], j["b"]): j for j in map(json.loads, open(os.path.join(OUT, "%s_jobs.jsonl" % corpus)))}
    rows = []
    for r in map(json.loads, open(os.path.join(OUT, "%s_answers.jsonl" % corpus))):
        j = jobs[(r["id"], r["a"], r["b"])]
        old = cached[r["id"]].get((r["a"], r["b"]))
        if r["answer"] is None or old is None:
            continue
        rows.append((j["pos_video"], j["harm"], r["answer"][0] >= 2, max(r["answer"]) >= 2, old[0] >= 2, max(old) >= 2))
    R = np.array(rows, dtype=float)
    lg = lambda p: float(np.log(p / (1 - p)))
    res = {"n": len(R)}
    for k, name in ((2, "context_hate"), (3, "context_any"), (4, "cached_hate"), (5, "cached_any")):
        harm = R[(R[:, 0] == 1) & (R[:, 1] == 1), k].mean()
        non = R[(R[:, 0] == 1) & (R[:, 1] == 0), k].mean()
        neg = R[R[:, 0] == 0, k].mean()
        res[name] = {"harm": harm, "pos_nonharm": non, "neg": neg,
                     "logit_harm_minus_nonharm": lg(harm) - lg(non), "logit_nonharm_minus_neg": lg(non) - lg(neg)}
        print("%-13s yes-rate harm %.3f | positive-video non-harm %.3f | negative %.3f | logit harm-nonharm %.2f "
              "nonharm-neg %.2f" % (name, harm, non, neg, lg(harm) - lg(non), lg(non) - lg(neg)))
    print("n = %d (harm %d, pos non-harm %d, negative %d)" % (len(R), ((R[:, 0] == 1) & (R[:, 1] == 1)).sum(),
                                                             ((R[:, 0] == 1) & (R[:, 1] == 0)).sum(), (R[:, 0] == 0).sum()))
    json.dump(res, open(os.path.join(OUT, "%s_eval.json" % corpus), "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "run", "eval"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    a = ap.parse_args()
    {"build": lambda: build(a.corpus), "run": lambda: run(a.corpus, a.model, a.gpu_mem),
     "eval": lambda: evaluate(a.corpus)}[a.mode]()
