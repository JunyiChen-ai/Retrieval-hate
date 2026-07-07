#!/usr/bin/env python
"""P10 — CPU aggregation variants of the anchor K=30 scores (no re-scoring).

Produces window-vector variant score files (same format as the MLLM scorer) from
the landed anchor scores + ASR window text + K=4 scores + a hate lexicon:
  gate  : zero windows with no ASR speech (localization is speech-borne).
  lex   : additive hate-lexicon boost (score + min(term_hits, 3)).
  fuse  : 0.5*K30 + 0.5*K4(mapped) — coarse x fine blend.
Writes data/MLLM_scores/HateMM/train_segscoreK30_p10-<v>.jsonl (train = calib set).
"""
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
    """Tolerant load: strip // comments + trailing commas; fall back to regex
    extraction of quoted leaf terms if JSON still won't parse."""
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
    # fallback: every quoted string that is not a bare category key
    cats = set(re.findall(r'"([a-z_]+)"\s*:', txt))
    return {m.lower() for m in re.findall(r'"([^"]+)"', txt)} - cats


def main():
    s30 = load_jsonl_scores(os.path.join(MD, "train_segscoreK30_qwen.jsonl"))
    s4 = load_jsonl_scores(os.path.join(MD, "train_segscoreK4_qwen.jsonl"))
    asr = load_asr(os.path.join(AD, "train_asrK30_whisper-large-v3.jsonl"), K30)
    lex = flat_lexicon()
    lex_res = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in lex if " " not in t]
    lex_phr = [t for t in lex if " " in t]

    def hits(text):
        t = (text or "").lower()
        n = sum(1 for r in lex_res if r.search(t))
        n += sum(1 for p in lex_phr if p in t)
        return n

    gate, lexv, fuse = {}, {}, {}
    for vid, sc in s30.items():
        if len(sc) < K30:
            continue
        wt = asr.get(vid, [""] * K30)
        gate[vid] = [sc[k] if (wt[k] and wt[k].strip()) else 0 for k in range(K30)]
        lexv[vid] = [sc[k] + min(hits(wt[k]), 3) for k in range(K30)]
        s4v = s4.get(vid)
        if s4v and len(s4v) >= K4:
            fuse[vid] = [0.5 * sc[k] + 0.5 * s4v[min(K4 - 1, k * K4 // K30)]
                         for k in range(K30)]
        else:
            fuse[vid] = list(sc)

    for name, D in [("gate", gate), ("lex", lexv), ("fuse", fuse)]:
        p = os.path.join(MD, "train_segscoreK30_p10-{}.jsonl".format(name))
        with open(p, "w") as f:
            for vid, sc in D.items():
                f.write(json.dumps(dict(id=vid, scores=sc, video_ok=True)) + "\n")
        print("wrote {} ({} videos)".format(p, len(D)))


if __name__ == "__main__":
    main()
