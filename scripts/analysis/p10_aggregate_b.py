#!/usr/bin/env python
"""P10-b — CPU aggregation for the scale-ladder second calibration round.

Two modes (pure re-aggregation of already-scored windows, no re-scoring):

  fuselex7b : the pre-registered 7B multiple-comparison-extension row.
              stacks the round-1 winners A-fuse (coarse K4 x fine K30) and
              A-lex (ASR hate-lexicon boost) on the SAME 7B scores:
                  s[k] = 0.5*K30_7b[k] + 0.5*K4_7b[map(k)] + min(hits(asr[k]), 3)
              -> data/MLLM_scores/HateMM/train_segscoreK30_p10-fuselex.jsonl

  fuse      : A-fuse aggregation for a stronger scorer (32B/72B). Fuses the
              SAME model's K30 (fine) and K4 (coarse) scores, matching the 7B
              A-fuse recipe exactly (no cross-model channel):
                  s[k] = 0.5*K30_m[k] + 0.5*K4_m[map(k)]
              --prefix names the model tag, e.g. p10-p6-32b or p10-p6-72b-bnb4.
              reads train_segscoreK30_<prefix>.jsonl + train_segscoreK4_<prefix>.jsonl
              -> data/MLLM_scores/HateMM/train_segscoreK30_<prefix>-fuse.jsonl

The "anchor aggregation" candidate needs no file: it IS the raw K30 score file
(evaluated directly by p10_eval_hatemm.py --scores_tag segscoreK30_<prefix>).
"""
import argparse
import json
import os
import re

ROOT = "/data/jehc223/RGCL"
MD = os.path.join(ROOT, "data/MLLM_scores/HateMM")
AD = os.path.join(ROOT, "data/ASR/HateMM")
LEX = "/data/jehc223/HateClipSeg/lexicons.json"
K30, K4 = 30, 4


def load_jsonl_scores(p, key="scores"):
    S = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            S[str(r["id"])] = r.get(key) or []
    return S


def load_asr(p, K):
    out = {}
    for line in open(p):
        line = line.strip()
        if line:
            o = json.loads(line)
            wt = [(t or "") for t in (o.get("window_text") or [])]
            out[str(o["id"])] = (wt + [""] * K)[:K]
    return out


def flat_lexicon():
    txt = open(LEX).read()
    try:
        clean = re.sub(r"//[^\n]*", "", txt)
        clean = re.sub(r",(\s*[}\]])", r"\1", clean)
        d = json.loads(clean)
        terms = set()

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for t in x:
                    if isinstance(t, str):
                        terms.add(t.lower())
        walk(d)
        if terms:
            return terms
    except Exception:  # noqa: BLE001
        pass
    cats = set(re.findall(r'"([a-z_]+)"\s*:', txt))
    return {m.lower() for m in re.findall(r'"([^"]+)"', txt)} - cats


def fuse_vec(s30, s4):
    """0.5*K30[k] + 0.5*K4[map(k)] (map = k*K4//K30, capped)."""
    return [0.5 * s30[k] + 0.5 * s4[min(K4 - 1, k * K4 // K30)] for k in range(K30)]


def write(D, name):
    p = os.path.join(MD, "train_segscoreK30_{}.jsonl".format(name))
    with open(p, "w") as f:
        for vid, sc in D.items():
            f.write(json.dumps(dict(id=vid, scores=sc, video_ok=True)) + "\n")
    print("wrote {} ({} videos)".format(p, len(D)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fuselex7b", "fuse"])
    ap.add_argument("--prefix", default="", help="model tag for --mode fuse, e.g. p10-p6-32b")
    args = ap.parse_args()

    if args.mode == "fuselex7b":
        s30 = load_jsonl_scores(os.path.join(MD, "train_segscoreK30_qwen.jsonl"))
        s4 = load_jsonl_scores(os.path.join(MD, "train_segscoreK4_qwen.jsonl"))
        asr = load_asr(os.path.join(AD, "train_asrK30_whisper-large-v3.jsonl"), K30)
        lex = flat_lexicon()
        lex_res = [re.compile(r"\b" + re.escape(t) + r"\b", re.I)
                   for t in lex if " " not in t]
        lex_phr = [t for t in lex if " " in t]

        def hits(text):
            t = (text or "").lower()
            n = sum(1 for r in lex_res if r.search(t))
            n += sum(1 for p in lex_phr if p in t)
            return n

        out = {}
        for vid, sc in s30.items():
            if len(sc) < K30:
                continue
            s4v = s4.get(vid)
            base = fuse_vec(sc, s4v) if (s4v and len(s4v) >= K4) else list(sc)
            wt = asr.get(vid, [""] * K30)
            out[vid] = [base[k] + min(hits(wt[k]), 3) for k in range(K30)]
        write(out, "p10-fuselex")
        return

    # mode == fuse: A-fuse for a stronger scorer (needs its own K30 + K4 files)
    pre = args.prefix
    p30 = os.path.join(MD, "train_segscoreK30_{}.jsonl".format(pre))
    p4 = os.path.join(MD, "train_segscoreK4_{}.jsonl".format(pre))
    if not (os.path.exists(p30) and os.path.exists(p4)):
        raise SystemExit("missing scores for prefix {}: {} / {}".format(pre, p30, p4))
    s30 = load_jsonl_scores(p30)
    s4 = load_jsonl_scores(p4)
    out = {}
    for vid, sc in s30.items():
        if len(sc) < K30:
            continue
        s4v = s4.get(vid)
        out[vid] = fuse_vec(sc, s4v) if (s4v and len(s4v) >= K4) else list(sc)
    write(out, "{}-fuse".format(pre))


if __name__ == "__main__":
    main()
