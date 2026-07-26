#!/usr/bin/env python
"""ERRPAT MHC-ZH: is the thin-transcript cluster a TRANSCRIPTION failure (fixable by a better
ASR channel) or a SPEECH-ABSENCE fact (not fixable by any transcript)?

Test: for every test item, compare the deployed transcript length against our independent
whisper-large-v3 K4 ASR of the same audio. If Whisper also finds little speech, the text
channel is thin because the video is quiet -- no transcript upgrade exists.
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
TAX = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))
PER = TAX["per_item"]
OUT = ROOT / "scripts/analysis/errpat_zh_asr_ceiling_OUT.json"
EM = re.compile(r'<em class="keyword">(.*?)</em>')
PUNCT = set("，。？！,.?!、；：“”‘’…·—《》()（）【】 \n\t🎼😊")

ann = {r["Video_ID"]: r for r in
       json.load(open(ROOT / "data/_src_Multihateclip/Chinese/annotation(new).json"))}
gt = [json.loads(l) for l in open(ROOT / "data/gt/MHC_zh/test.jsonl")]
asr = {r["id"]: r for r in (json.loads(l) for l in
                            open(ROOT / "data/ASR/MHC_zh/test_seen_asrK4_whisper-large-v3.jsonl"))}


def norm(t):
    return "".join(c for c in EM.sub(r"\1", t) if c not in PUNCT)


rows = []
for r in gt:
    a = ann.get(r["id"], {})
    dep_tr = norm(a.get("Transcript") or "")
    w = norm("".join(c[2] for c in asr[r["id"]]["chunks"]))
    rows.append(dict(id=r["id"], gold=r["label"], errc=PER[r["id"]]["n_seeds_wrong"],
                     label3=PER[r["id"]]["cov"]["label3"],
                     dep_transcript_chars=len(dep_tr), whisper_chars=len(w),
                     duration_s=PER[r["id"]]["cov"]["duration_s"],
                     whisper_gain_chars=len(w) - len(dep_tr)))

core = [x for x in rows if x["errc"] == 3]
thin = [x for x in rows if 31 <= x["dep_transcript_chars"] < 76]
thin_core = [x for x in thin if x["errc"] == 3]

res = {
    "definitions": {
        "core_error": "wrong at final epoch in all 3 seeds (re-mint proxy)",
        "thin_transcript": "deployed transcript 31-76 chars (2nd quartile of transcript length)",
    },
    "all_test": {
        "n": len(rows),
        "median_dep_transcript_chars": float(np.median([x["dep_transcript_chars"] for x in rows])),
        "median_whisper_chars": float(np.median([x["whisper_chars"] for x in rows])),
        "n_whisper_longer_by_ge_50": sum(1 for x in rows if x["whisper_gain_chars"] >= 50),
        "n_whisper_longer_by_ge_20": sum(1 for x in rows if x["whisper_gain_chars"] >= 20),
        "n_whisper_shorter": sum(1 for x in rows if x["whisper_gain_chars"] < 0),
    },
    "core_errors": {
        "n": len(core),
        "median_dep_transcript_chars": float(np.median([x["dep_transcript_chars"] for x in core])),
        "median_whisper_chars": float(np.median([x["whisper_chars"] for x in core])),
        "n_whisper_longer_by_ge_50": sum(1 for x in core if x["whisper_gain_chars"] >= 50),
        "n_whisper_longer_by_ge_20": sum(1 for x in core if x["whisper_gain_chars"] >= 20),
        "ids_whisper_adds_ge_50": [x["id"] for x in core if x["whisper_gain_chars"] >= 50],
    },
    "thin_transcript_band": {
        "n": len(thin), "n_core_errors": len(thin_core),
        "core_error_rate_per_seed": round(
            sum(x["errc"] for x in thin) / (3 * len(thin)), 4),
        "n_whisper_longer_by_ge_50": sum(1 for x in thin if x["whisper_gain_chars"] >= 50),
        "n_whisper_longer_by_ge_20": sum(1 for x in thin if x["whisper_gain_chars"] >= 20),
        "thin_core_whisper_gain_chars": {x["id"]: x["whisper_gain_chars"] for x in thin_core},
    },
    "CEILING_asr_rechannel": {
        "argument": "an upgraded/alternative ZH transcript channel can only help where a "
                    "DIFFERENT transcript exists. Ceiling = core errors where our independent "
                    "whisper-large-v3 run recovers materially more speech than the deployed "
                    "transcript (>=50 chars).",
        "n_core_errors_with_material_new_speech": sum(
            1 for x in core if x["whisper_gain_chars"] >= 50),
        "acc_ceiling_if_ALL_of_those_flip": round(
            sum(1 for x in core if x["whisper_gain_chars"] >= 50) / len(rows), 4),
        "note": "this is a gold-cheat oracle: it assumes every such item flips to correct.",
    },
    "per_item": rows,
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
r = dict(res)
r.pop("per_item")
print(json.dumps(r, indent=1, ensure_ascii=False))
print(f"\nwrote {OUT}")
