"""Pilot for revision 4 (README section 14; development evidence on test under rule 10): does a comparative question
("which of the two halves of this node contains harmful content: A, B, both or neither?") locate harm inside a
hateful video better than the two absolute answers the method already has for the halves?

With the absolute answers, for test nodes whose halves differ (exactly one half contains GT harm) the harmful half
gets the higher maximum level only 28% (HateMM) / 21% (HCS) of the time and the same level 61% / 65% of the time
(pick accuracy, ties counted as half: .584 / .538); the VLM rates a harmless part of a hateful video almost like a
harmful one. Comparing the two halves in one question removes what the halves share (the video, the speaker, the
topic).

Sample (fixed seed 0): internal tree nodes of test videos, both halves queryable (>= 4 s), node <= 64 s. Groups:
    one       exactly one half contains GT harm: all of them, each asked twice (halves in order and swapped)
    both      both halves contain GT harm: 200
    none_pos  positive video, neither half contains GT harm: 200
    none_neg  negative video: 200
Each half shows the same 4 frames and transcript as its cached absolute question (data/vlm_tree/<C>/manifest.jsonl);
same model (Qwen2.5-VL-7B-Instruct), greedy, same frame resolution; two videos per prompt (Part A, Part B).

    python experiments/20260925_query_paradigm/pilot_compare_prompt.py build --corpus hatemm    # local (GT)
    python experiments/20260925_query_paradigm/pilot_compare_prompt.py run --corpus hatemm      # lab (vLLM env)
    python experiments/20260925_query_paradigm/pilot_compare_prompt.py eval --corpus hatemm     # local
Files: runs/20260927_query_paradigm_r4/pilot_compare/<corpus>_{jobs,answers}.jsonl and <corpus>_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "runs", "20260927_query_paradigm_r4", "pilot_compare")
N_OTHER = 200
MAX_NODE = 64

SYSTEM_PROMPT = (
    "You are a careful content-moderation rater. You are shown two consecutive parts of ONE video, Part A and then "
    "Part B: a few frames sampled from each part and the speech transcript of each part. Decide which of the two "
    "parts contains harmful content of any of these five categories:\n"
    "1. hate: content that attacks, dehumanises, demeans, threatens or incites against people for a protected "
    "attribute (race, ethnicity, religion, nationality, gender, sexual orientation, disability, caste, "
    "immigration status, etc.).\n"
    "2. harassment: content that insults, mocks, bullies or threatens a person or group, not for a protected "
    "attribute.\n"
    "3. sexual: sexual or sexually explicit content.\n"
    "4. violence: content that depicts, glorifies or threatens violence, injury or gore.\n"
    "5. self-harm: content that depicts or encourages self-harm, suicide or dangerous acts.\n"
    "Harmful content can be carried by any channel: spoken words, on-screen text, symbols, gestures, or imagery. "
    "Judge each part only by what is visible or audible in that part. A part counts as harmful only if its harmful "
    "content is clear (implicit or coded, or explicit), not a faint or ambiguous cue."
)
USER_PROMPT = (
    "Transcript of Part A: \"{asr_a}\"\n"
    "Transcript of Part B: \"{asr_b}\"\n\n"
    "Which part contains harmful content? Answer with exactly one of: A, B, both, neither. No other text."
)
_ANS = re.compile(r"^\W*(both|neither|a|b)\b", re.IGNORECASE)


def parse(raw):
    m = _ANS.match(raw or "")
    return None if m is None else {"a": "A", "b": "B", "both": "both", "neither": "neither"}[m.group(1).lower()]


def build(corpus):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import hier_evidence_common as hc
    import data as qdata
    import qtree
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    test = set(ids["test"])
    man = {}
    for line in open(os.path.join(ROOT, "data", "vlm_tree", qdata.CORPUS_DIR[corpus], "manifest.jsonl")):
        r = json.loads(line)
        if r["id"] in test:
            man[r["id"]] = {(int(a), int(b)): (idx, text) for a, b, idx, text in r["nodes"]}
    groups = {"one": [], "both": [], "none_pos": [], "none_neg": []}
    for v in ids["test"]:
        T = len(np.asarray(gt["test"][v]))
        y = np.asarray(gt["test"][v])
        tr = qtree.tree(T)
        for n in range(len(tr["a"])):
            L, R = int(tr["left"][n]), int(tr["right"][n])
            if L < 0 or tr["b"][n] - tr["a"][n] > MAX_NODE:
                continue
            kl, kr = (int(tr["a"][L]), int(tr["b"][L])), (int(tr["a"][R]), int(tr["b"][R]))
            if kl not in man[v] or kr not in man[v]:
                continue                                  # a half shorter than 4 s is not queryable
            hl, hr = bool(y[kl[0]:kl[1]].any()), bool(y[kr[0]:kr[1]].any())
            g = ("none_neg" if labels[v] == 0 else "one" if hl != hr else "both" if hl else "none_pos")
            groups[g].append({"id": v, "left": kl, "right": kr, "harm_left": hl, "harm_right": hr, "group": g,
                              "idx_left": man[v][kl][0], "idx_right": man[v][kr][0],
                              "text_left": man[v][kl][1], "text_right": man[v][kr][1]})
    rng = np.random.RandomState(0)
    jobs = []
    for g, items in groups.items():
        if g != "one" and len(items) > N_OTHER:
            items = [items[i] for i in sorted(rng.choice(len(items), N_OTHER, replace=False))]
        for it in items:
            for swap in ((False, True) if g == "one" else (False,)):
                jobs.append(dict(it, swap=swap))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "%s_jobs.jsonl" % corpus), "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(corpus, {g: len(v) for g, v in groups.items()}, "jobs", len(jobs))


def run(corpus, model, gpu_mem):
    import extract_tree_answers as eta
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    jobs = [json.loads(l) for l in open(os.path.join(OUT, "%s_jobs.jsonl" % corpus))]
    proc = AutoProcessor.from_pretrained(model)
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": "Part A:"}, {"type": "video"},
                                         {"type": "text", "text": "Part B:"}, {"type": "video"},
                                         {"type": "text", "text": "{USER}"}]}]
    template = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert template.count("{USER}") == 1 and template.count("<|video_pad|>") == 2
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=8192,
              limit_mm_per_prompt={"image": 0, "video": 2}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    d = eta.CORPUS_DIR[corpus]
    fdir = lambda v: os.path.join(ROOT, "data", "frames_1fps", d, v)  # noqa: E731
    pool = ThreadPoolExecutor(16)
    fl = list(pool.map(lambda j: eta.load_frames(fdir(j["id"]), j["idx_left"]), jobs))
    fr = list(pool.map(lambda j: eta.load_frames(fdir(j["id"]), j["idx_right"]), jobs))

    def text(t):
        w = t.split()
        t = " ".join(w[:eta.MAX_WORDS]) if len(w) > eta.MAX_WORDS else t
        return (t.strip() or eta.NO_SPEECH).replace('"', "'")

    inputs = []
    for j, a, b in zip(jobs, fl, fr):
        (va, ta), (vb, tb) = ((b, j["text_right"]), (a, j["text_left"])) if j["swap"] else \
                             ((a, j["text_left"]), (b, j["text_right"]))
        inputs.append({"prompt": template.replace("{USER}", USER_PROMPT.format(asr_a=text(ta), asr_b=text(tb))),
                       "multi_modal_data": {"video": [va, vb]}})
    outs = llm.generate(inputs, sp, use_tqdm=False)
    with open(os.path.join(OUT, "%s_answers.jsonl" % corpus), "w") as fh:
        for j, o in zip(jobs, outs):
            raw = o.outputs[0].text.strip()
            fh.write(json.dumps({k: j[k] for k in ("id", "left", "right", "group", "swap", "harm_left",
                                                     "harm_right")} | {"answer": parse(raw), "raw": raw}) + "\n")
    print("DONE %d answers" % len(jobs), flush=True)


def evaluate(corpus):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import data as qdata
    answers, _ = qdata.load_answers(corpus)
    rows = [json.loads(l) for l in open(os.path.join(OUT, "%s_answers.jsonl" % corpus))]
    res = {"corpus": corpus, "n_jobs": len(rows), "unparsed": sum(r["answer"] is None for r in rows), "groups": {}}
    for g in ("one", "both", "none_pos", "none_neg"):
        rg = [r for r in rows if r["group"] == g and r["answer"] is not None]
        dist = {k: float(np.mean([r["answer"] == k for r in rg])) for k in ("A", "B", "both", "neither")}
        res["groups"][g] = {"n": len(rg), "answers": dist}
    one = [r for r in rows if r["group"] == "one" and r["answer"] is not None]
    # the side that holds the harm, as shown to the VLM (A = first shown half)
    shown = lambda r: ("A" if r["harm_left"] != r["swap"] else "B")  # noqa: E731
    right = np.array([r["answer"] == shown(r) for r in one])
    wrong = np.array([r["answer"] in ("A", "B") and r["answer"] != shown(r) for r in one])
    tie = ~right & ~wrong
    res["one"] = {"correct": float(right.mean()), "wrong_half": float(wrong.mean()), "both_or_neither": float(tie.mean()),
                  "pick_accuracy_ties_half": float(right.mean() + tie.mean() / 2),
                  "answer_A_rate": float(np.mean([r["answer"] == "A" for r in one]))}
    # the absolute-answer baseline on the same nodes (maximum level over the five categories of each half)
    base = []
    for r in one:
        if r["swap"]:
            continue
        la, lb = answers[r["id"]].get(tuple(r["left"])), answers[r["id"]].get(tuple(r["right"]))
        if la is None or lb is None:
            continue
        mh, mc = (max(la), max(lb)) if r["harm_left"] else (max(lb), max(la))
        base.append(1.0 if mh > mc else 0.5 if mh == mc else 0.0)
    res["one"]["absolute_pick_accuracy_ties_half"] = float(np.mean(base))
    json.dump(res, open(os.path.join(OUT, "%s_eval.json" % corpus), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "run", "eval"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    a = ap.parse_args()
    {"build": lambda: build(a.corpus), "run": lambda: run(a.corpus, a.model, a.gpu_mem),
     "eval": lambda: evaluate(a.corpus)}[a.mode]()


if __name__ == "__main__":
    main()
