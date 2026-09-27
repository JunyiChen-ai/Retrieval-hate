"""Revision-5 premise check (README section 16.4; development evidence on test under rule 10; nothing here trains or
selects): can the VLM POINT to the harmful part of a node, instead of only saying whether the node has harm?

Why: the node answers barely separate the harmful and the harmless parts of a hateful video (HateMM 4-8 s nodes
"yes" .57 / .47, README 10.4), and no model of the answers can learn that from video labels (README 11, 16.4). A
question whose answer is a location inside the node carries within-video information by construction.

Question (same model Qwen2.5-VL-7B-Instruct, greedy, same frames and resolution as the node question; the frames
are given as four numbered images and the node's word-timestamp transcript as numbered lines): "which transcript
lines and which frames contain harmful content". A line is up to 8 consecutive words (a new line also after a pause
> 1 s); its seconds are those its words span. Frame k stands for the k-th quarter of the node (the frames are
sampled at the quarter centres). The pointed seconds are the union.

Sample (seed 0): tree nodes of 16-64 s of test videos. Groups:
    mixed      positive video, part of the node's seconds are GT harmful: up to 800
    full       every second GT harmful: 150
    clean_pos  positive video, no GT harmful second: 150
    neg        negative video: 150
Measures: per mixed node, the ROC of pointed / not pointed against GT seconds (.5 when nothing or everything is
pointed), precision and recall of the pointed seconds; for mixed nodes whose GT harm lies in one half only, whether
more pointed seconds fall in that half (ties count half), next to the same comparison with the two halves' cached
node answers (README 14.3: .584-.620 HateMM, .538-.545 HCS); the rate of pointing at anything per group next to the
cached node answer's "some category >= 2" rate on the same nodes.

    python experiments/20260925_query_paradigm/pilot_point_prompt.py build --corpus hatemm   # local (GT)
    python experiments/20260925_query_paradigm/pilot_point_prompt.py run --corpus hatemm     # lab (vLLM env)
    python experiments/20260925_query_paradigm/pilot_point_prompt.py eval --corpus hatemm    # local
Files: runs/20260928_query_paradigm_r5_analysis/pilot_point/<corpus>_{jobs,answers}.jsonl, <corpus>_eval.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "runs", "20260928_query_paradigm_r5_analysis", "pilot_point")
N_GROUP = {"mixed": 800, "full": 150, "clean_pos": 150, "neg": 150}
MIN_LEN, MAX_LEN = 16, 64
LINE_WORDS, LINE_GAP = 8, 1.0

SYSTEM_PROMPT = (
    "You are a careful content-moderation rater. You are shown four frames sampled in time order from ONE part of "
    "a video (Frame 1 to Frame 4, one from each quarter of the part) and the speech transcript of that part, split "
    "into numbered lines in time order. Harmful content means any of these five categories:\n"
    "1. hate: content that attacks, dehumanises, demeans, threatens or incites against people for a protected "
    "attribute (race, ethnicity, religion, nationality, gender, sexual orientation, disability, caste, "
    "immigration status, etc.).\n"
    "2. harassment: content that insults, mocks, bullies or threatens a person or group, not for a protected "
    "attribute.\n"
    "3. sexual: sexual or sexually explicit content.\n"
    "4. violence: content that depicts, glorifies or threatens violence, injury or gore.\n"
    "5. self-harm: content that depicts or encourages self-harm, suicide or dangerous acts.\n"
    "Harmful content can be carried by any channel: spoken words, on-screen text, symbols, gestures, or imagery. "
    "Point only to lines and frames that themselves contain clear harmful content (implicit or coded, or "
    "explicit); do not point to harmless lines or frames around them."
)
USER_PROMPT = (
    "Transcript lines of this part:\n{lines}\n\n"
    "Which transcript lines and which frames contain harmful content? Answer with JSON only, in the form "
    "{{\"lines\": [line numbers], \"frames\": [frame numbers]}}; use empty lists when none do."
)
NO_SPEECH = "(no speech in this part)"
_LIST = {k: re.compile(r'"%s"\s*:\s*\[([^\]]*)\]' % k) for k in ("lines", "frames")}


def parse(raw):
    out = {}
    for k, rx in _LIST.items():
        m = rx.search(raw or "")
        if m is None:
            return None
        out[k] = sorted({int(x) for x in re.findall(r"\d+", m.group(1))})
    return out


def lines_of(words, a, b):
    """Words (start, end, text) whose midpoint lies in [a, b), grouped into lines: (text, first second, last
    second + 1) clipped to [a, b)."""
    ws = [w for w in words if a <= (w[0] + w[1]) / 2 < b]
    lines, cur = [], []
    for w in ws:
        if cur and (len(cur) >= LINE_WORDS or w[0] - cur[-1][1] > LINE_GAP):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        s0 = max(a, int(math.floor(ln[0][0])))
        s1 = min(b, max(s0 + 1, int(math.ceil(ln[-1][1]))))
        out.append(("".join(w[2] for w in ln).strip(), s0, s1))
    return out


def build(corpus):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import hier_evidence_common as hc
    import data as qdata
    import qtree
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    test = set(ids["test"])
    d = qdata.CORPUS_DIR[corpus]
    man, words = {}, {}
    for line in open(os.path.join(ROOT, "data", "vlm_tree", d, "manifest_words.jsonl")):
        r = json.loads(line)
        if r["id"] in test:
            man[r["id"]] = {(int(a), int(b)): idx for a, b, idx, _t in r["nodes"]}
    for line in open(os.path.join(ROOT, "data", "ASR_words", d, "words.jsonl")):
        r = json.loads(line)
        if r["id"] in test:
            words[r["id"]] = r["words"]
    groups = {g: [] for g in N_GROUP}
    for v in ids["test"]:
        y = np.asarray(gt["test"][v])
        tr = qtree.tree(len(y))
        for n in range(len(tr["a"])):
            a, b = int(tr["a"][n]), int(tr["b"][n])
            if not MIN_LEN <= b - a <= MAX_LEN or (a, b) not in man[v]:
                continue
            cov = float(y[a:b].mean())
            g = "neg" if labels[v] == 0 else "clean_pos" if cov == 0 else "full" if cov == 1 else "mixed"
            m = a + (b - a) // 2
            groups[g].append({"id": v, "a": a, "b": b, "m": m, "group": g, "idx": man[v][(a, b)],
                              "lines": lines_of(words.get(v, []), a, b), "gt": y[a:b].astype(int).tolist()})
    rng = np.random.RandomState(0)
    jobs = []
    for g, items in groups.items():
        if len(items) > N_GROUP[g]:
            items = [items[i] for i in sorted(rng.choice(len(items), N_GROUP[g], replace=False))]
        jobs += items
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "%s_jobs.jsonl" % corpus), "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(corpus, {g: len(v) for g, v in groups.items()}, "jobs", len(jobs))


def run(corpus, model, gpu_mem):
    import extract_tree_answers as eta
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    jobs = [json.loads(l) for l in open(os.path.join(OUT, "%s_jobs.jsonl" % corpus))]
    proc = AutoProcessor.from_pretrained(model)
    content = []
    for k in range(4):
        content += [{"type": "text", "text": "Frame %d:" % (k + 1)}, {"type": "image"}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content + [{"type": "text", "text": "{USER}"}]}]
    template = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert template.count("{USER}") == 1 and template.count("<|image_pad|>") == 4
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=8192,
              limit_mm_per_prompt={"image": 4, "video": 0}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=96)
    d = eta.CORPUS_DIR[corpus]
    pool = ThreadPoolExecutor(16)
    frames = list(pool.map(lambda j: eta.load_frames(os.path.join(ROOT, "data", "frames_1fps", d, j["id"]),
                                                     j["idx"]), jobs))

    def lines(j):
        if not j["lines"]:
            return NO_SPEECH
        return "\n".join("[%d] %s" % (i + 1, t.replace('"', "'")) for i, (t, _s, _e) in enumerate(j["lines"]))

    inputs = [{"prompt": template.replace("{USER}", USER_PROMPT.format(lines=lines(j))),
               "multi_modal_data": {"image": [Image.fromarray(f) for f in fr]}} for j, fr in zip(jobs, frames)]
    outs = llm.generate(inputs, sp, use_tqdm=False)
    with open(os.path.join(OUT, "%s_answers.jsonl" % corpus), "w") as fh:
        for j, o in zip(jobs, outs):
            raw = o.outputs[0].text.strip()
            fh.write(json.dumps({k: j[k] for k in ("id", "a", "b", "m", "group")} | {"answer": parse(raw),
                                                                                      "raw": raw}) + "\n")
    print("DONE %d answers" % len(jobs), flush=True)


def pointed(job, ans):
    """(b - a,) 0/1 array of the pointed seconds of one node."""
    a, b = job["a"], job["b"]
    p = np.zeros(b - a, dtype=int)
    for k in ans["lines"]:
        if 1 <= k <= len(job["lines"]):
            _t, s0, s1 = job["lines"][k - 1]
            p[s0 - a:s1 - a] = 1
    q = (b - a) / 4.0
    for k in ans["frames"]:
        if 1 <= k <= 4:
            p[int(round((k - 1) * q)):int(round(k * q))] = 1
    return p


def evaluate(corpus):
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import data as qdata
    jobs = {(j["id"], j["a"], j["b"]): j for j in map(json.loads, open(os.path.join(OUT, "%s_jobs.jsonl" % corpus)))}
    rows = [json.loads(l) for l in open(os.path.join(OUT, "%s_answers.jsonl" % corpus))]
    cached, _ = qdata.load_answers(corpus, "words")
    res = {"corpus": corpus, "n": len(rows), "unparsed": sum(r["answer"] is None for r in rows), "groups": {}}
    ok = [(jobs[(r["id"], r["a"], r["b"])], r["answer"]) for r in rows if r["answer"] is not None]
    for g in N_GROUP:
        rg = [(j, x) for j, x in ok if j["group"] == g]
        yes_c = [max(cached[j["id"]][(j["a"], j["b"])]) >= 2 for j, _x in rg
                 if cached[j["id"]].get((j["a"], j["b"])) is not None]
        res["groups"][g] = {"n": len(rg), "points_any": float(np.mean([pointed(j, x).any() for j, x in rg])),
                            "points_line": float(np.mean([bool(x["lines"]) for j, x in rg])),
                            "points_frame": float(np.mean([bool(x["frames"]) for j, x in rg])),
                            "mean_pointed_share": float(np.mean([pointed(j, x).mean() for j, x in rg])),
                            "cached_answer_yes": float(np.mean(yes_c))}
    mixed = [(j, x) for j, x in ok if j["group"] == "mixed"]
    aucs, prec, rec, base_prec, halves, halves_base = [], [], [], [], [], []
    for j, x in mixed:
        y, p = np.asarray(j["gt"]), pointed(j, x)
        aucs.append(roc_auc_score(y, p) if 0 < p.mean() < 1 else 0.5)
        base_prec.append(float(y.mean()))
        if p.any():
            prec.append(float(y[p == 1].mean()))
        rec.append(float(p[y == 1].mean()))
        k = j["m"] - j["a"]
        hl, hr = bool(y[:k].any()), bool(y[k:].any())
        if hl != hr:
            nh, nc = (p[:k].sum(), p[k:].sum()) if hl else (p[k:].sum(), p[:k].sum())
            halves.append(1.0 if nh > nc else 0.5 if nh == nc else 0.0)
            ol, orr = cached[j["id"]].get((j["a"], j["m"])), cached[j["id"]].get((j["m"], j["b"]))
            if ol is not None and orr is not None:
                mh, mc = (max(ol), max(orr)) if hl else (max(orr), max(ol))
                halves_base.append(1.0 if mh > mc else 0.5 if mh == mc else 0.0)
    res["mixed"] = {"n": len(mixed), "node_roc_mean": float(np.mean(aucs)),
                    "node_roc_mean_when_pointed": float(np.mean([u for u, (j, x) in zip(aucs, mixed)
                                                                 if 0 < pointed(j, x).mean() < 1] or [np.nan])),
                    "precision_pointed": float(np.mean(prec)) if prec else None, "n_pointed": len(prec),
                    "precision_by_chance": float(np.mean(base_prec)), "recall": float(np.mean(rec)),
                    "one_half_n": len(halves), "one_half_pick_ties_half": float(np.mean(halves)),
                    "one_half_cached_answers_ties_half": float(np.mean(halves_base)) if halves_base else None}
    json.dump(res, open(os.path.join(OUT, "%s_eval.json" % corpus), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "run", "eval"))
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    a = ap.parse_args()
    {"build": lambda: build(a.corpus), "run": lambda: run(a.corpus, a.model, a.gpu_mem),
     "eval": lambda: evaluate(a.corpus)}[a.mode]()


if __name__ == "__main__":
    main()
