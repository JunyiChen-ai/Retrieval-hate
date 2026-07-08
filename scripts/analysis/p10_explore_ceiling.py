#!/usr/bin/env python
"""P10 EXPLORATORY ceiling probe (CPU-only, NOT for promotion).

Pure re-aggregation of already-landed HateMM-calibration scores to ask a single
question: within the existing scorer pool, can any combination push the HateMM
calibration wv-AUC toward ~0.616 (the level whose calib->test extrapolation would
project test 0.60)? Touches NO test, NO HateClipSeg, NO GPU, submits NO SLURM.

Every recipe reuses the *exact* pre-registered functions from p10_aggregate_b.py
(fuse_vec, flat_lexicon, hits, load_* ) so the fuse/lexicon math is bit-identical
to the registered A-fuse / A-lex / fuselex7b recipes. Only the model source and
(for the sensitivity rows) the fixed 0.5/0.5 blend weight change.

Outputs are written with an explicit `p10-xplor-*` tag so they can never be
confused with the pre-registered score files, and are used only for this
EXPLORATORY appendix.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p10_aggregate_b import (  # noqa: E402  (reuse registered recipes verbatim)
    MD, AD, K30, K4, load_jsonl_scores, load_asr, flat_lexicon, fuse_vec)
import re  # noqa: E402


def build_lex():
    lex = flat_lexicon()
    lex_res = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in lex if " " not in t]
    lex_phr = [t for t in lex if " " in t]

    def hits(text):
        t = (text or "").lower()
        n = sum(1 for r in lex_res if r.search(t))
        n += sum(1 for p in lex_phr if p in t)
        return n
    return hits


def fuse_vec_w(s30, s4, w30):
    """Weight-generalised A-fuse: w30*K30[k] + (1-w30)*K4[map(k)] (same linear
    blend as fuse_vec; fuse_vec == fuse_vec_w with w30=0.5). Not a new mechanism,
    just a coefficient on the identical coarse x fine blend."""
    return [w30 * s30[k] + (1.0 - w30) * s4[min(K4 - 1, k * K4 // K30)]
            for k in range(K30)]


def write(D, name):
    p = os.path.join(MD, "train_segscoreK30_{}.jsonl".format(name))
    with open(p, "w") as f:
        for vid, sc in D.items():
            f.write(json.dumps(dict(id=vid, scores=sc, video_ok=True)) + "\n")
    print("wrote {} ({} videos) -> eval tag segscoreK30_{}".format(p, len(D), name))


def load_model(prefix):
    s30 = load_jsonl_scores(os.path.join(MD, "train_segscoreK30_{}.jsonl".format(prefix)))
    s4 = load_jsonl_scores(os.path.join(MD, "train_segscoreK4_{}.jsonl".format(prefix)))
    return s30, s4


def main():
    asr = load_asr(os.path.join(AD, "train_asrK30_whisper-large-v3.jsonl"), K30)
    hits = build_lex()

    models = {"32b": "p10-p6-32b", "72b": "p10-p6-72b-bnb4"}

    # rows 1-2: fuse x lex  (A-fuse base + additive ASR hate-lexicon boost),
    # exact fuselex7b recipe but on the 32B / 72B scores.
    for short, pre in models.items():
        s30, s4 = load_model(pre)
        out = {}
        for vid, sc in s30.items():
            if len(sc) < K30:
                continue
            s4v = s4.get(vid)
            base = fuse_vec(sc, s4v) if (s4v and len(s4v) >= K4) else list(sc)
            wt = asr.get(vid, [""] * K30)
            out[vid] = [base[k] + min(hits(wt[k]), 3) for k in range(K30)]
        write(out, "p10-xplor-{}-fuselex".format(short))

    # row 3: 72B fuse blend-weight sensitivity (same linear mechanism, no lex).
    #   default A-fuse = K4:K30 = 1:1  (w30=0.5)
    #   variant "w12"  = K4:K30 = 1:2  (w30=2/3, fine-heavy)
    #   variant "w21"  = K4:K30 = 2:1  (w30=1/3, coarse-heavy)
    s30, s4 = load_model("p10-p6-72b-bnb4")
    for tag, w30 in [("w12", 2.0 / 3.0), ("w21", 1.0 / 3.0)]:
        out = {}
        for vid, sc in s30.items():
            if len(sc) < K30:
                continue
            s4v = s4.get(vid)
            out[vid] = fuse_vec_w(sc, s4v, w30) if (s4v and len(s4v) >= K4) else list(sc)
        write(out, "p10-xplor-72b-fuse-{}".format(tag))


if __name__ == "__main__":
    main()
