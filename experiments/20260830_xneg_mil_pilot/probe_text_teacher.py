#!/usr/bin/env python3
"""Step-5 recon probe: can an EXTERNAL hate-speech text classifier order the
seconds of hateful test videos? Per second t: ASR text overlapping [t-2, t+3)
-> cardiffnlp/twitter-roberta-base-hate-latest hate probability; no speech ->
score 0. Also a 'has_speech' trivial arm. Within-hate macro ROC on test.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402

ASR = {"hatemm": "hatemm_all", "mhclip_en": "mhclip_en_all",
       "mhclip_zh": "mhclip_zh_all", "hateclipseg": "hateclipseg_all"}
OUT_DIR = os.path.join(REPO, "runs", "20260830_xneg_mil_pilot")
DEVICE = "cpu"
MODEL = "cardiffnlp/twitter-roberta-base-hate-latest"
CORPORA = ("hatemm", "mhclip_en", "hateclipseg")  # EN-language classifier


def load_asr(corpus):
    table = {}
    with open(os.path.join(REPO, "results/reproduction/asr", ASR[corpus],
                           "timestamped_chunks.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            table[d["video_id"]] = d.get("chunks") or []
    return table


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL, torch_dtype=torch.float32).to(DEVICE).eval()
    hate_idx = [i for i, l in mdl.config.id2label.items()
                if "hate" in l.lower() and "non" not in l.lower()]
    assert hate_idx, mdl.config.id2label
    hate_idx = hate_idx[0]

    lines = ["# External text-classifier ordering probe (TEST hate videos)", "",
             "| corpus | arm | within-ROC macro (n) | speech coverage |",
             "|---|---|---:|---:|"]
    for corpus in CORPORA:
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        asr = load_asr(corpus)
        text_scores, speech_scores = {}, {}
        cache = {}
        cov_n = cov_d = 0
        for v in sorted(hate_ids):
            if v not in gt:
                continue
            T = len(gt[v])
            chunks = asr.get(v, [])
            ts_arr = np.zeros(T)
            sp_arr = np.zeros(T)
            for t in range(T):
                parts = [c["text"] for c in chunks
                         if c.get("start") is not None and c.get("end") is not None
                         and c["end"] > t - 2 and c["start"] < t + 3]
                txt = " ".join(p.strip() for p in parts).strip()[:400]
                sp_arr[t] = 1.0 if txt else 0.0
                if not txt:
                    continue
                if txt not in cache:
                    with torch.no_grad():
                        enc = tok(txt, return_tensors="pt", truncation=True,
                                  max_length=128).to(DEVICE)
                        p = torch.softmax(mdl(**enc).logits, -1)[0, hate_idx]
                    cache[txt] = float(p)
                ts_arr[t] = cache[txt]
            cov_n += int(sp_arr.sum()); cov_d += T
            text_scores[v] = ts_arr
            speech_scores[v] = sp_arr
        for arm, sc in (("text_hate_prob", text_scores),
                        ("has_speech", speech_scores)):
            m = evaluate_scores(sc, {v: gt[v] for v in sc}, hate_ids)
            macro = m["per_video"]["macro_auc"]
            lines.append("| %s | %s | %s (%d) | %.2f |" % (
                corpus, arm, ("%.4f" % macro) if macro is not None else "n/a",
                m["per_video"]["n_videos_both_classes"], cov_n / max(1, cov_d)))
        print(lines[-2]); print(lines[-1], flush=True)
    with open(os.path.join(OUT_DIR, "probe_text_teacher.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
