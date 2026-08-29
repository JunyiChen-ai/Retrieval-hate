"""CONTRAST STANCE PILOT -- frozen scorer (rules: MASK_STANCE_PILOT_FREEZE.md Appendix B.5-B.7,
plus the baseline-binarisation addendum B.10 R2).

Written and committed BEFORE the eval batch outputs existed. Nothing here may be edited after
the eval batch is fetched.

  python score_contrast.py --tag c1
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SP = os.path.join(ROOT, "idea-stage", "stance_pilot")
MSP = os.path.join(ROOT, "idea-stage", "mask_stance_pilot")
sys.path.insert(0, os.path.join(ROOT, "idea-stage"))

FRAME_DS = ["HateMM", "MHC", "MHC_zh"]
TEXT_DS = ["ImpliHateVid"]
ALL_DS = FRAME_DS + TEXT_DS
M1_BAR, M2_BAR = 0.70, 0.15
N_SET = {"quotes_mentions", "condemns", "reports"}

# freeze B.8: eval items burned as mandatory qualitative smoke checks
SMOKE_EVAL = {("MHC", "KDcCiUU8q5E"), ("HateMM", "non_hate_video_32"),
              ("HateMM", "non_hate_video_16")}

GOLD = {"S_FP": "OPPOSE", "S_FN": "ENDORSE", "CTRL_HATE": "ENDORSE", "CTRL_NONHATE": None}


# ------------------------------------------------------------------ hand-coded voice form (B.7.3)
def gold_voice():
    from voice_field_analysis import GOLD_VOICE
    return {k: v[0] for k, v in GOLD_VOICE.items()}


# ------------------------------------------------------------------ lexical overlap (B.7.6)
STOP = set("""a an the and or but if of to in on for with at by from as is are was were be been
being it its this that these those i you he she they we them him her his their our your my me us
do does did done have has had not no so too very just about over under again then than there here
what which who whom when where why how all any both each few more most other some such only own
same s t can will dont don't should now up out off down into out""".split())
WORD = re.compile(r"[a-z']+")
CJK = re.compile(r"[一-鿿]")


def toks(s, lang):
    s = s or ""
    if lang == "zh":
        c = "".join(CJK.findall(s))
        return set(c[i:i + 2] for i in range(len(c) - 1))
    return set(w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 1)


def ov(template, transcript, lang):
    T = toks(template, lang)
    if not T:
        return 0.0
    S = toks(transcript, lang)
    return len(T & S) / len(T)


# ------------------------------------------------------------------ loading
def load_sample():
    return {(x["dataset"], x["id"]): x for x in
            json.load(open(os.path.join(SP, "sample.json")))["eval"]}


def load_transcripts(items):
    sys.path.insert(0, SP)
    from run_pilot import load_texts
    c = {}
    for (ds, vid) in items:
        if ds not in c:
            c[ds] = load_texts(ds)
    return {(ds, vid): (c[ds].get(vid) or "").strip() for (ds, vid) in items}


def read_5way(path, sample):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        k = (r["dataset"], r["id"])
        if k not in sample:
            continue
        out[k] = ((r.get("parsed") or {}).get("stance"))
    return out


def binarise_5way(s):
    """freeze B.10 R2(b): ENDORSE iff `endorses`; all four others -> OPPOSE."""
    if s is None:
        return None
    return "ENDORSE" if s == "endorses" else "OPPOSE"


def collapse(votes):
    """majority of the parsed votes; ties -> 'TIE'; no parsed vote -> None."""
    v = [x for x in votes if x]
    if not v:
        return None
    c = Counter(v)
    if c["ENDORSE"] == c["OPPOSE"]:
        return "TIE"
    return "ENDORSE" if c["ENDORSE"] > c["OPPOSE"] else "OPPOSE"


def load(tag):
    sample = load_sample()
    rows = [json.loads(l) for l in open(os.path.join(HERE, f"pred_{tag}.jsonl"), encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["id"], r["variant"])].append(r)
    tx = load_transcripts(sample.keys())
    r1 = read_5way(os.path.join(SP, "pred_strong.jsonl"), sample)
    mk = read_5way(os.path.join(MSP, "pred_m1.jsonl"), sample)
    gv = gold_voice()

    items = {}
    for (ds, vid, variant), vs in by.items():
        vs = sorted(vs, key=lambda x: x["pair"])
        k = (ds, vid)
        it = items.setdefault(k, {**sample[k], "dataset": ds, "id": vid,
                                  "r1_5way": r1.get(k), "mask_5way": mk.get(k),
                                  "r1_bin": binarise_5way(r1.get(k)),
                                  "mask_bin": binarise_5way(mk.get(k)),
                                  "voice": gv.get(k, "UNCODED"),
                                  "in_smoke": k in SMOKE_EVAL,
                                  "lang": vs[0]["lang"], "target": None})
        lang = vs[0]["lang"]
        t = tx[k]
        dv = []
        for r in vs:
            dv.append(round(ov(r["endorse_text"], t, lang) - ov(r["oppose_text"], t, lang), 4))
        it[variant] = {
            "votes": [r["vote_side"] for r in vs],
            "slots": [r["vote_slot"] for r in vs],
            "a_is_endorse": [r["a_is_endorse"] for r in vs],
            "call": collapse([r["vote_side"] for r in vs]),
            "n_parsed": sum(1 for r in vs if r["vote_side"]),
            "ov_d": dv, "ov_D": round(sum(dv) / max(1, len(dv)), 4),
            "usage_in": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in vs),
            "usage_out": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in vs)}
        if variant == "v2":
            it["target"] = vs[0]["target"]
    return list(items.values())


# ------------------------------------------------------------------ scoring helpers
def ok(it, variant="v1"):
    g = GOLD[it["group"]]
    if g is None or variant not in it:
        return None
    return it[variant]["call"] == g          # TIE / None both score wrong (freeze B.4)


def ok_key(it, key):
    g = GOLD[it["group"]]
    if g is None or it.get(key) is None:
        return None
    return it[key] == g


def acc(sub, fn):
    v = [fn(x) for x in sub]
    v = [x for x in v if x is not None]
    return (sum(v) / len(v)) if v else None, len(v)


def binom_p(k, n, p=0.5):
    """two-sided exact binomial p against chance p."""
    if n == 0:
        return None
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return round(min(1.0, sum(q for q in probs if q <= probs[k] + 1e-12)), 4)


def block(rows, name, variant="v1", exclude_smoke=True):
    S = [r for r in rows if r["group"] in ("S_FP", "S_FN")
         and not (exclude_smoke and r["in_smoke"])]
    ch = [r for r in rows if r["group"] == "CTRL_HATE"]
    cn = [r for r in rows if r["group"] == "CTRL_NONHATE"]
    R = {"view": name, "variant": variant, "exclude_smoke": exclude_smoke,
         "n_items": len(rows), "n_S": len(S)}

    a, n = acc(S, lambda r: ok(r, variant))
    R["M1_acc_S"], R["M1_n"] = a, n
    R["M1_pass"] = bool(a is not None and a >= M1_BAR)
    R["M1_p_vs_chance_0.5"] = binom_p(int(round((a or 0) * n)), n) if n else None
    for cell in ("S_FP", "S_FN"):
        sub = [r for r in S if r["group"] == cell]
        R[f"acc_{cell}"], R[f"n_{cell}"] = acc(sub, lambda r: ok(r, variant))

    cv = [r for r in ch if variant in r and r[variant]["call"]]
    R["M2_ctrl_hate_said_OPPOSE"] = (sum(1 for r in cv if r[variant]["call"] == "OPPOSE")
                                     / len(cv)) if cv else None
    R["M2_n"] = len(cv)
    R["M2_pass"] = bool(R["M2_ctrl_hate_said_OPPOSE"] is not None
                        and R["M2_ctrl_hate_said_OPPOSE"] <= M2_BAR)
    R["VERDICT"] = "PASS" if (R["M1_pass"] and R["M2_pass"]) else "FAIL"

    nv = [r for r in cn if variant in r and r[variant]["call"]]
    R["ctrl_nonhate_said_ENDORSE"] = (sum(1 for r in nv if r[variant]["call"] == "ENDORSE")
                                      / len(nv)) if nv else None
    R["ctrl_nonhate_n"] = len(nv)

    # ---- baselines on EXACTLY these rows (freeze B.10 R2)
    for key, lab in (("r1_bin", "r1_binary"), ("mask_bin", "mask_binary")):
        R[f"{lab}_acc_S"], R[f"{lab}_n"] = acc(S, lambda r, k=key: ok_key(r, k))
    for key, lab in (("r1_5way", "r1_5way"), ("mask_5way", "mask_5way")):
        def f5(r, k=key):
            s = r.get(k)
            if s is None:
                return None
            if r["group"] == "S_FP":
                return s in N_SET
            if r["group"] == "S_FN":
                return s == "endorses"
            return None
        R[f"{lab}_acc_S"], R[f"{lab}_n"] = acc(S, f5)

    # ---- strata
    R["by_voice"] = {}
    for lab in ("OWN", "NOT_OWN", "UNDET", "UNCODED"):
        sub = [r for r in S if r["voice"] == lab]
        a2, n2 = acc(sub, lambda r: ok(r, variant))
        R["by_voice"][lab] = {"n": n2, "acc": a2,
                              "acc_r1_binary": acc(sub, lambda r: ok_key(r, "r1_bin"))[0],
                              "cells": dict(Counter(r["group"] for r in sub))}
    R["by_dataset"] = {}
    for ds in sorted(set(r["dataset"] for r in S)):
        sub = [r for r in S if r["dataset"] == ds]
        a2, n2 = acc(sub, lambda r: ok(r, variant))
        R["by_dataset"][ds] = {"n": n2, "acc": a2}
    R["call_hist"] = {g: dict(Counter((r[variant]["call"] if variant in r else "NO_V2")
                                      for r in rows if r["group"] == g))
                      for g in ("S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE")}

    # ---- vote patterns / per-pair / position bias
    pair_acc, pair_end = {}, {}
    for p in range(5):
        hit = tot = end = tote = 0
        for r in S:
            if variant not in r:
                continue
            v = r[variant]["votes"][p] if p < len(r[variant]["votes"]) else None
            if not v:
                continue
            tot += 1
            hit += int(v == GOLD[r["group"]])
        for r in rows:
            if variant not in r:
                continue
            v = r[variant]["votes"][p] if p < len(r[variant]["votes"]) else None
            if not v:
                continue
            tote += 1
            end += int(v == "ENDORSE")
        pair_acc[p] = {"n": tot, "acc_on_S": round(hit / tot, 3) if tot else None}
        pair_end[p] = {"n": tote, "endorse_rate_all_items": round(end / tote, 3) if tote else None}
    R["per_pair_acc_on_S"] = pair_acc
    R["per_pair_endorse_rate"] = pair_end
    split = Counter()
    for r in S:
        if variant not in r:
            continue
        c = Counter(x for x in r[variant]["votes"] if x)
        split[f"{max(c.values(), default=0)}-{min(c.values(), default=0) if len(c) > 1 else 0}"] += 1
    R["vote_split_hist_on_S"] = dict(split)
    R["acc_by_split"] = {}
    for s in split:
        sub = []
        for r in S:
            if variant not in r:
                continue
            c = Counter(x for x in r[variant]["votes"] if x)
            k = f"{max(c.values(), default=0)}-{min(c.values(), default=0) if len(c) > 1 else 0}"
            if k == s:
                sub.append(r)
        R["acc_by_split"][s] = {"n": len(sub), "acc": acc(sub, lambda r: ok(r, variant))[0]}
    na = nt = 0
    for r in rows:
        if variant not in r:
            continue
        for sl in r[variant]["slots"]:
            if sl:
                nt += 1
                na += int(sl == "A")
    R["position_bias_slotA_rate"] = round(na / nt, 4) if nt else None
    R["position_bias_n"] = nt

    # ---- lexical overlap (freeze B.7.6)
    aligned = notal = zero = 0
    al_rows, na_rows = [], []
    for r in S:
        if variant not in r or not r[variant]["call"] or r[variant]["call"] == "TIE":
            continue
        D = r[variant]["ov_D"]
        if abs(D) < 1e-9:
            zero += 1
            continue
        sgn = 1 if D > 0 else -1
        vote = 1 if r[variant]["call"] == "ENDORSE" else -1
        if sgn == vote:
            aligned += 1
            al_rows.append(r)
        else:
            notal += 1
            na_rows.append(r)
    pair_agree = tot_p = 0
    for r in rows:
        if variant not in r:
            continue
        for d, v in zip(r[variant]["ov_d"], r[variant]["votes"]):
            if not v or abs(d) < 1e-9:
                continue
            tot_p += 1
            pair_agree += int((d > 0) == (v == "ENDORSE"))
    R["overlap"] = {
        "n_scored": aligned + notal, "n_zero_diff": zero,
        "aligned_share": round(aligned / max(1, aligned + notal), 3),
        "acc_aligned": acc(al_rows, lambda r: ok(r, variant))[0],
        "acc_not_aligned": acc(na_rows, lambda r: ok(r, variant))[0],
        "per_pair_vote_follows_overlap": round(pair_agree / max(1, tot_p), 3),
        "per_pair_n": tot_p,
        "mean_D_by_group": {g: round(sum(r[variant]["ov_D"] for r in rows
                                         if r["group"] == g and variant in r)
                                     / max(1, sum(1 for r in rows if r["group"] == g
                                                  and variant in r)), 4)
                            for g in ("S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE")}}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows = load(a.tag)
    out = {"tag": a.tag, "n_items": len(rows)}

    views = [("A_frames_primary", FRAME_DS), ("B_all99", ALL_DS), ("C_textonly", TEXT_DS)]
    for name, ds in views:
        sub = [r for r in rows if r["dataset"] in ds]
        out[name] = block(sub, name, "v1", True)
        out[name + "__incl_smoke_items"] = block(sub, name, "v1", False)
    out["A_variant2"] = block([r for r in rows if r["dataset"] in FRAME_DS and "v2" in r],
                              "A_variant2", "v2", True)
    # variant1 restricted to the same target-bearing rows, for the paired v1-vs-v2 read
    out["A_variant1_on_v2_rows"] = block([r for r in rows if r["dataset"] in FRAME_DS
                                          and "v2" in r], "A_variant1_on_v2_rows", "v1", True)
    out["VERDICT"] = out["A_frames_primary"]["VERDICT"]

    tin = sum(r[v]["usage_in"] for r in rows for v in ("v1", "v2") if v in r)
    tout = sum(r[v]["usage_out"] for r in rows for v in ("v1", "v2") if v in r)
    ncalls = sum(len(r[v]["votes"]) for r in rows for v in ("v1", "v2") if v in r)
    out["cost"] = {"n_calls": ncalls, "input_tokens": tin, "output_tokens": tout,
                   "cny_at_listprice_no_batch_discount": round(tin / 1000 * 0.002
                                                               + tout / 1000 * 0.008, 3)}
    out["losses"] = [{"id": r["id"], "ds": r["dataset"], "grp": r["group"],
                      "v1_call": (r.get("v1") or {}).get("call"),
                      "n_parsed": (r.get("v1") or {}).get("n_parsed")}
                     for r in rows if not (r.get("v1") or {}).get("call")
                     or (r.get("v1") or {}).get("n_parsed", 0) < 5]
    out["per_item"] = sorted([{
        "id": r["id"], "ds": r["dataset"], "grp": r["group"], "voice": r["voice"],
        "smoke": r["in_smoke"], "lang": r["lang"], "target": r["target"],
        "v1": (r.get("v1") or {}).get("call"),
        "v1_votes": (r.get("v1") or {}).get("votes"),
        "v2": (r.get("v2") or {}).get("call"),
        "ov_D": (r.get("v1") or {}).get("ov_D"),
        "r1_5way": r["r1_5way"], "mask_5way": r["mask_5way"],
        "gold": GOLD[r["group"]], "ok_v1": ok(r, "v1")} for r in rows],
        key=lambda x: (x["grp"], x["ds"], x["id"]))
    json.dump(out, open(os.path.join(HERE, f"score_{a.tag}.json"), "w"),
              indent=1, ensure_ascii=False)

    for name in ("A_frames_primary", "A_frames_primary__incl_smoke_items",
                 "B_all99", "C_textonly", "A_variant2", "A_variant1_on_v2_rows"):
        v = out[name]
        print("=" * 80)
        print(name, "variant", v["variant"], "n_items", v["n_items"])
        print(f"  M1 acc_S = {v['M1_acc_S']} (n={v['M1_n']}) bar 0.70 -> {v['M1_pass']}"
              f"   p vs 0.5 = {v['M1_p_vs_chance_0.5']}")
        print(f"     S_FP {v['acc_S_FP']} (n={v['n_S_FP']})   S_FN {v['acc_S_FN']} (n={v['n_S_FN']})")
        print(f"     same rows: r1_binary={v['r1_binary_acc_S']} mask_binary={v['mask_binary_acc_S']}"
              f" | r1_5way={v['r1_5way_acc_S']} mask_5way={v['mask_5way_acc_S']}")
        print(f"  M2 CTRL_HATE->OPPOSE = {v['M2_ctrl_hate_said_OPPOSE']} (n={v['M2_n']})"
              f" bar 0.15 -> {v['M2_pass']}    CTRL_NONHATE->ENDORSE ="
              f" {v['ctrl_nonhate_said_ENDORSE']}")
        print("  VERDICT:", v["VERDICT"])
        print("  by voice :", json.dumps(v["by_voice"]))
        print("  by ds    :", json.dumps(v["by_dataset"]))
        print("  calls    :", json.dumps(v["call_hist"]))
        print("  per-pair acc/S :", json.dumps(v["per_pair_acc_on_S"]))
        print("  per-pair endors:", json.dumps(v["per_pair_endorse_rate"]))
        print("  splits   :", json.dumps(v["vote_split_hist_on_S"]),
              json.dumps(v["acc_by_split"]))
        print("  posbias  :", v["position_bias_slotA_rate"], "n", v["position_bias_n"])
        print("  overlap  :", json.dumps(v["overlap"]))
    print("\ncost:", json.dumps(out["cost"]))
    print("losses:", json.dumps(out["losses"]))
    print("\nPRIMARY VERDICT (view A, variant 1, smoke items excluded):", out["VERDICT"])


if __name__ == "__main__":
    main()
