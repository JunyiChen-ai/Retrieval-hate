#!/usr/bin/env python
"""ERRPAT MHC-ZH: quantify the two mechanisms the exemplars expose.

M1  ASR-CORRUPTED TRANSCRIPT. The deployed ZH text = Title + ' . ' + a whisper-class ASR
    transcript. Several core errors carry visibly broken transcripts (wrong-language output,
    repetition loops). Score it objectively and test enrichment among errors.

M2  TOPIC-vs-STANCE. The Bilibili <em class="keyword"> markup is the SEARCH TERM the clip was
    harvested by -- very often the slur itself. A Normal-labelled clip that *discusses* the slur
    carries it in its text and retrieves hate neighbours. Test FP enrichment.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
TAX = json.load(open(ROOT / "scripts/analysis/errpat_zh_taxonomy_OUT.json"))
PER = TAX["per_item"]
OUT = ROOT / "scripts/analysis/errpat_zh_mech_OUT.json"
EM = re.compile(r'<em class="keyword">(.*?)</em>')
rng = np.random.default_rng(20260726)

ann = {r["Video_ID"]: r for r in
       json.load(open(ROOT / "data/_src_Multihateclip/Chinese/annotation(new).json"))}
gt = [json.loads(l) for l in open(ROOT / "data/gt/MHC_zh/test.jsonl")]


def is_cjk(c):
    return "一" <= c <= "鿿"


def is_hangul(c):
    return "가" <= c <= "힯" or "ᄀ" <= c <= "ᇿ"


def is_kana(c):
    return "぀" <= c <= "ヿ"


def is_latin(c):
    return ("a" <= c.lower() <= "z")


def garbage_score(t):
    """Foreign-script share + repetition-loop share of a supposedly-Chinese transcript."""
    letters = [c for c in t if not c.isspace() and not unicodedata.category(c).startswith("P")
               and not c.isdigit() and unicodedata.category(c) not in ("So", "Sk", "Sm", "Sc")]
    n = len(letters)
    if n == 0:
        return {"n_letters": 0, "foreign_share": 0.0, "repeat_share": 0.0, "garbage": 0.0}
    foreign = sum(1 for c in letters if is_hangul(c) or is_kana(c) or is_latin(c))
    # repetition loop: share of 3-grams that occur >= 3 times
    g = [t[i:i + 3] for i in range(len(t) - 2)]
    cnt = Counter(g)
    rep = sum(v for k, v in cnt.items() if v >= 3)
    rep_share = rep / max(len(g), 1)
    fs = foreign / n
    return {"n_letters": n, "foreign_share": round(fs, 4),
            "repeat_share": round(rep_share, 4), "garbage": round(max(fs, rep_share), 4)}


rows = []
for r in gt:
    a = ann.get(r["id"], {})
    tr = (a.get("Transcript") or "").strip()
    ti = EM.sub(r"\1", (a.get("Title") or "").strip())
    gsc = garbage_score(tr)
    em_terms = EM.findall(r["text"])
    rows.append(dict(id=r["id"], gold=r["label"], errc=PER[r["id"]]["n_seeds_wrong"],
                     label3=PER[r["id"]]["cov"]["label3"],
                     title=ti, title_chars=len(ti), tr_chars=len(tr),
                     em_terms=em_terms, has_em=bool(em_terms),
                     title_share_of_text=round(len(ti) / max(len(ti) + len(tr), 1), 4),
                     **{f"g_{k}": v for k, v in gsc.items()}))

errc = np.array([x["errc"] for x in rows])
gold = np.array([x["gold"] for x in rows])
core = errc == 3
G = np.array([x["g_garbage"] for x in rows])
res = {}


def perm_p(mask, n_draw=20000):
    """P(as many core errors land in `mask` by chance)."""
    obs = int(np.sum(core & mask))
    k = int(core.sum())
    null = np.array([int(mask[rng.choice(len(rows), size=k, replace=False)].sum())
                     for _ in range(n_draw)])
    return {"observed": obs, "expected": round(float(null.mean()), 2),
            "null_p95": int(np.percentile(null, 95)),
            "p_one_sided": round(float(np.mean(null >= obs)), 4)}


# ------------------------------------------------------------------ M1
res["M1_asr_corruption"] = {
    "definition": "garbage = max(foreign-script share, share of 3-grams repeating >=3x) of the "
                  "deployed ZH transcript",
    "garbage_quartiles": [round(float(x), 4) for x in np.percentile(G, [25, 50, 75])],
    "by_band": [],
    "high_garbage_ge_0.15": {
        "n": int(np.sum(G >= 0.15)),
        "n_core_errors": int(np.sum(core & (G >= 0.15))),
        "err_rate_per_seed": round(float(errc[G >= 0.15].sum() / (3 * max(int(np.sum(G >= 0.15)), 1))), 4),
        "perm": perm_p(G >= 0.15),
    },
    "low_garbage_lt_0.15": {
        "n": int(np.sum(G < 0.15)),
        "n_core_errors": int(np.sum(core & (G < 0.15))),
        "err_rate_per_seed": round(float(errc[G < 0.15].sum() / (3 * int(np.sum(G < 0.15)))), 4),
    },
    "worst_offenders_among_core": sorted(
        [{"id": x["id"], "gold": x["gold"], "garbage": x["g_garbage"],
          "foreign_share": x["g_foreign_share"], "repeat_share": x["g_repeat_share"],
          "tr_chars": x["tr_chars"], "title": x["title"]}
         for x in rows if x["errc"] == 3 and x["g_garbage"] >= 0.15],
        key=lambda d: -d["garbage"]),
}
q = np.percentile(G, [25, 50, 75])
for lo, hi, tag in ((0, q[0], "Q1"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"), (q[2], 1.01, "Q4")):
    sel = (G >= lo) & (G < hi)
    res["M1_asr_corruption"]["by_band"].append({
        "band": tag, "garbage_range": [round(float(lo), 4), round(float(hi), 4)],
        "n": int(sel.sum()), "n_core": int(np.sum(core & sel)),
        "err_rate_per_seed": round(float(errc[sel].sum() / (3 * max(int(sel.sum()), 1))), 4)})

# ------------------------------------------------------------------ M2
hem = np.array([x["has_em"] for x in rows])
neg = gold == 0
pos = gold == 1
res["M2_topic_vs_stance"] = {
    "definition": "<em class='keyword'> = the Bilibili search term the clip was harvested by "
                  "(usually the slur). Present in the deployed text verbatim.",
    "counts": {"n_with_em": int(hem.sum()), "n_without": int((~hem).sum())},
    "negatives_only": {
        "n_with_em": int(np.sum(neg & hem)), "n_without_em": int(np.sum(neg & ~hem)),
        "FP_rate_per_seed_with_em": round(float(errc[neg & hem].sum() / (3 * max(int(np.sum(neg & hem)), 1))), 4),
        "FP_rate_per_seed_without_em": round(float(errc[neg & ~hem].sum() / (3 * max(int(np.sum(neg & ~hem)), 1))), 4),
        "n_core_FP_with_em": int(np.sum(core & neg & hem)),
        "n_core_FP_without_em": int(np.sum(core & neg & ~hem)),
    },
    "positives_only": {
        "n_with_em": int(np.sum(pos & hem)), "n_without_em": int(np.sum(pos & ~hem)),
        "FN_rate_per_seed_with_em": round(float(errc[pos & hem].sum() / (3 * max(int(np.sum(pos & hem)), 1))), 4),
        "FN_rate_per_seed_without_em": round(float(errc[pos & ~hem].sum() / (3 * max(int(np.sum(pos & ~hem)), 1))), 4),
    },
    "perm_core_FP_in_em_negatives": perm_p(neg & hem),
    "core_FP_em_terms": [{"id": x["id"], "em": x["em_terms"], "title": x["title"]}
                         for x in rows if x["errc"] == 3 and x["gold"] == 0 and x["has_em"]],
    "core_FN_em_terms": [{"id": x["id"], "em": x["em_terms"], "title": x["title"]}
                         for x in rows if x["errc"] == 3 and x["gold"] == 1 and x["has_em"]],
}

# ------------------------------------------------------------------ title/transcript balance
TS = np.array([x["title_share_of_text"] for x in rows])
res["M3_title_share"] = {
    "note": "title is short and clean; transcript is long and ASR-noisy. share = |title|/(|title|+|transcript|)",
    "median_title_share": round(float(np.median(TS)), 4),
    "core_errors_median_title_share": round(float(np.median(TS[core])), 4),
    "correct_median_title_share": round(float(np.median(TS[errc == 0])), 4),
    "title_chars_median": float(np.median([x["title_chars"] for x in rows])),
    "transcript_chars_median": float(np.median([x["tr_chars"] for x in rows])),
}

with open(OUT, "w") as f:
    json.dump({"summary": res, "per_item": rows}, f, indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print(f"\nwrote {OUT}")
