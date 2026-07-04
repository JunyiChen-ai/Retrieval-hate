"""QC for segment-aligned ASR jsonl (EXP_mm_segment_keys phase 1).

Reads data/ASR/<DS>/<split>_asrK<K>_<tag>.jsonl and prints:
  * coverage: videos with audio, with any text, per-window text rate
  * timestamp sanity: chunks outside [0, duration], word vs chunk fallback rate
  * language sanity: CJK character fraction (MHC_zh should be high, MHC low)
  * repetition-hallucination flag: videos whose text has a >=6-fold repeated
    5-gram (whisper music/silence pathology)
  * a few sample windows for eyeballing

Usage: python scripts/analysis/asr_qc.py --dataset MHC [--split train] [--samples 3]
"""

import argparse
import json
import os
import re
from collections import Counter


def cjk_frac(text):
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    alnum = sum(1 for ch in text if not ch.isspace())
    return cjk / max(alnum, 1)


def has_repetition(text, n=5, times=6):
    toks = re.findall(r"\w+|[一-鿿]", text.lower())
    if len(toks) < n * times:
        return False
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return grams.most_common(1)[0][1] >= times if grams else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MHC")
    ap.add_argument("--split", default="train")
    ap.add_argument("--asr_dir", default="./data/ASR")
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--tag", default="whisper-large-v3")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    path = os.path.join(args.asr_dir, args.dataset, "{}_asrK{}_{}.jsonl".format(
        args.split, args.num_subclips, args.tag))
    recs = [json.loads(l) for l in open(path) if l.strip()]
    n = len(recs)
    K = args.num_subclips
    print("== ASR QC {} {} ({} videos) == {}".format(
        args.dataset, args.split, n, path))
    if n == 0:
        return

    audio_ok = sum(1 for r in recs if r.get("audio_ok"))
    any_text = sum(1 for r in recs if any((t or "").strip()
                                          for t in r["window_text"]))
    win_text = sum(1 for r in recs for t in r["window_text"]
                   if (t or "").strip())
    fallback = sum(1 for r in recs if r.get("timestamps") == "chunk")
    bad_ts = 0
    for r in recs:
        d = r["duration"]
        for s, e, _ in r["chunks"]:
            if s < -0.01 or e > d + 0.01 or e < s:
                bad_ts += 1
    all_text = " ".join(t for r in recs for t in r["window_text"])
    rep = [r["id"] for r in recs
           if has_repetition(" ".join(r["window_text"]))]
    empty_w = [0] * (K + 1)
    for r in recs:
        empty_w[sum(1 for t in r["window_text"] if (t or "").strip())] += 1

    print("audio_ok           : {}/{} ({:.1f}%)".format(
        audio_ok, n, 100.0 * audio_ok / n))
    print("videos w/ any text : {}/{} ({:.1f}%)".format(
        any_text, n, 100.0 * any_text / n))
    print("windows w/ text    : {}/{} ({:.1f}%)".format(
        win_text, n * K, 100.0 * win_text / (n * K)))
    print("windows-with-text histogram (0..{} windows): {}".format(K, empty_w))
    print("chunk-level fallback (word-ts crash): {}/{} ({:.1f}%)".format(
        fallback, n, 100.0 * fallback / n))
    print("chunks outside [0,duration]: {}".format(bad_ts))
    print("CJK char fraction  : {:.3f} (expect high for MHC_zh, ~0 for MHC)".format(
        cjk_frac(all_text)))
    print("repetition-flagged videos: {}/{} ({:.1f}%) e.g. {}".format(
        len(rep), n, 100.0 * len(rep) / n, rep[:5]))

    for r in recs[: args.samples]:
        print("\n-- {} (dur {:.1f}s, ts={}) --".format(
            r["id"], r["duration"], r.get("timestamps")))
        for k, t in enumerate(r["window_text"]):
            b = r["window_bounds"][k]
            print("  w{} [{:5.1f},{:5.1f}]: {}".format(
                k, b[0], b[1], (t or "")[:90]))


if __name__ == "__main__":
    main()
