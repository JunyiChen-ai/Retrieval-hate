"""Revision-5 premise check (README section 16; development evidence on test under rule 10; no training or
selection): can the VLM say HOW MUCH of a video is harmful?

Error analysis (fraction_analysis.py): pooled AP is decided by how each video's score level matches the share of
its seconds that are hateful (GT fraction). Giving every video its GT fraction and keeping the model's ranking inside
the video lifts pooled AP from .669 to .85 (HCS) and from .687 to .92 (HateMM); the model's per-video mean ranks the
positive videos by fraction with Spearman only .23-.44, and the existing node answers ("is there harmful content in
this part", 0-3 per category) even less (.17-.35 with every node answered): the VLM says "yes" about as often for a
node that is 10% hateful as for one that is 100% hateful.

Question here (one call per test video): the whole video, F frames sampled uniformly + the whole word-level
transcript; "what share of the video's running time contains clearly harmful content" (0, 10, 25, 50, 75, 90, 100
percent). Two versions: F = 4 (the cost of one node question) and F = 16. Same model (Qwen2.5-VL-7B-Instruct), greedy,
same frame resolution as the node questions.

    python experiments/20260925_query_paradigm/pilot_fraction_prompt.py build --corpus hatemm   # local
    python experiments/20260925_query_paradigm/pilot_fraction_prompt.py run --corpus hatemm     # lab (vLLM env)
    python experiments/20260925_query_paradigm/pilot_fraction_prompt.py eval --corpus hatemm    # local
Files: runs/20260928_query_paradigm_r5_analysis/pilot_fraction/<corpus>_{jobs,answers}.jsonl, <corpus>_eval.json.
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
OUT = os.path.join(ROOT, "runs", "20260928_query_paradigm_r5_analysis", "pilot_fraction")
FRAMES = (4, 16)
LEVELS = (0, 10, 25, 50, 75, 90, 100)

SYSTEM_PROMPT = (
    "You are a careful content-moderation rater. You are shown frames sampled uniformly over a WHOLE video, in time "
    "order, and the speech transcript of the whole video. Harmful content means any of these five categories:\n"
    "1. hate: content that attacks, dehumanises, demeans, threatens or incites against people for a protected "
    "attribute (race, ethnicity, religion, nationality, gender, sexual orientation, disability, caste, "
    "immigration status, etc.).\n"
    "2. harassment: content that insults, mocks, bullies or threatens a person or group, not for a protected "
    "attribute.\n"
    "3. sexual: sexual or sexually explicit content.\n"
    "4. violence: content that depicts, glorifies or threatens violence, injury or gore.\n"
    "5. self-harm: content that depicts or encourages self-harm, suicide or dangerous acts.\n"
    "Harmful content can be carried by any channel: spoken words, on-screen text, symbols, gestures, or imagery. "
    "Estimate how much of the video's running time contains clear harmful content (implicit or coded, or explicit)."
)
USER_PROMPT = (
    "Transcript of the whole video: \"{asr}\"\n\n"
    "What share of this video's running time contains harmful content? Answer with exactly one number from: "
    "0, 10, 25, 50, 75, 90, 100 (percent). No other text."
)
_NUM = re.compile(r"\d+")


def parse(raw):
    m = _NUM.search(raw or "")
    if m is None:
        return None
    x = int(m.group(0))
    return min(LEVELS, key=lambda lv: abs(lv - x)) if 0 <= x <= 100 else None


def build(corpus):
    sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import hier_evidence_common as hc
    import data as qdata
    labels, ids, gt, _ = hc.load_fixed_cohort(corpus)
    d = qdata.CORPUS_DIR[corpus]
    words = {}
    for line in open(os.path.join(ROOT, "data", "ASR_words", d, "words.jsonl")):
        r = json.loads(line)
        words[r["id"]] = "".join(t for s, e, t in r["words"]).strip()
    jobs = []
    for v in ids["test"]:
        T = len(np.asarray(gt["test"][v]))
        n_frames = len([f for f in os.listdir(os.path.join(ROOT, "data", "frames_1fps", d, v)) if f.endswith(".jpg")])
        idx = {F: [min(int(T * (i + 0.5) / F), n_frames - 1) for i in range(F)] for F in FRAMES}
        jobs.append({"id": v, "label": int(labels[v]), "T": T, "fraction": float(np.mean(gt["test"][v])),
                     "idx": {str(F): idx[F] for F in FRAMES}, "text": words.get(v, "")})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "%s_jobs.jsonl" % corpus), "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j) + "\n")
    print(corpus, "jobs", len(jobs))


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
    assert template.count("{USER}") == 1 and "<|video_pad|>" in template
    llm = LLM(model=model, gpu_memory_utilization=gpu_mem, max_model_len=12288,
              limit_mm_per_prompt={"image": 0, "video": 1}, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    d = eta.CORPUS_DIR[corpus]
    pool = ThreadPoolExecutor(16)

    def text(t):
        w = t.split()
        t = " ".join(w[:eta.MAX_WORDS]) if len(w) > eta.MAX_WORDS else t
        return (t.strip() or eta.NO_SPEECH).replace('"', "'")

    with open(os.path.join(OUT, "%s_answers.jsonl" % corpus), "w") as fh:
        for F in FRAMES:
            frames = list(pool.map(lambda j: eta.load_frames(os.path.join(ROOT, "data", "frames_1fps", d, j["id"]),
                                                             j["idx"][str(F)]), jobs))
            inputs = [{"prompt": template.replace("{USER}", USER_PROMPT.format(asr=text(j["text"]))),
                       "multi_modal_data": {"video": f}} for j, f in zip(jobs, frames)]
            outs = llm.generate(inputs, sp, use_tqdm=False)
            for j, o in zip(jobs, outs):
                raw = o.outputs[0].text.strip()
                fh.write(json.dumps({"id": j["id"], "F": F, "label": j["label"], "fraction": j["fraction"],
                                     "answer": parse(raw), "raw": raw}) + "\n")
    print("DONE %d answers" % (len(jobs) * len(FRAMES)), flush=True)


def evaluate(corpus):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rows = [json.loads(l) for l in open(os.path.join(OUT, "%s_answers.jsonl" % corpus))]
    res = {"corpus": corpus, "by_frames": {}}
    for F in FRAMES:
        r = [x for x in rows if x["F"] == F]
        ok = [x for x in r if x["answer"] is not None]
        pos = [x for x in ok if x["label"] == 1 and x["fraction"] > 0]
        out = {"n": len(r), "unparsed": len(r) - len(ok),
               "spearman_fraction_positives": float(spearmanr([x["fraction"] for x in pos],
                                                              [x["answer"] for x in pos]).correlation),
               "video_auc": float(roc_auc_score([x["label"] for x in ok], [x["answer"] for x in ok])),
               "answer_dist_pos": {str(lv): float(np.mean([x["answer"] == lv for x in pos])) for lv in LEVELS},
               "answer_dist_neg": {str(lv): float(np.mean([x["answer"] == lv for x in ok if x["label"] == 0]))
                                   for lv in LEVELS},
               "mean_answer_by_fraction_quartile": {}}
        qs = np.percentile([x["fraction"] for x in pos], [25, 50, 75])
        for k, (lo, hi) in enumerate(zip([-1] + list(qs), list(qs) + [2])):
            g = [x["answer"] for x in pos if lo < x["fraction"] <= hi]
            out["mean_answer_by_fraction_quartile"]["q%d" % (k + 1)] = [float(np.mean(g)) if g else None, len(g)]
        res["by_frames"][str(F)] = out
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
