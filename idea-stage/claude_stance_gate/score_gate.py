"""CLAUDE_STANCE_GATE -- frozen scorer.

Written BEFORE any annotator output was read. Implements CLAUDE_STANCE_GATE_FREEZE.md
sections 3 / 4 verbatim.

  python score_gate.py
"""
import json
import os
import sys
from collections import Counter, defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "idea-stage"))

RATERS = ["r7k", "m3q", "z9x"]
N_BATCH = 6

# freeze section 3
GOLD = {"S_FP": "DISTANCED", "S_FN": "ENDORSE",
        "CTRL_HATE": "ENDORSE", "CTRL_NONHATE": "DISTANCED"}
SCORED_GROUPS_M1 = ("S_FP", "S_FN")

FRAME_DS = {"HateMM", "MHC", "MHC_zh"}
SMOKE = {("MHC", "KDcCiUU8q5E"), ("HateMM", "non_hate_video_32"),
         ("HateMM", "non_hate_video_16")}
# qwen never scored this item (DashScope moderation refusal); excluded so the 32-row
# denominator is identical to the qwen rounds (freeze 4.1)
QWEN_MISSING = {("MHC_zh", "BV1m8411z7mV")}

VOICE_OWN = {"uploader"}
VOICE_NOT_OWN = {"on_screen_speaker", "quoted_third_party", "archival", "caption",
                 "archival_source", "caption_overlay"}

M1_BAR_PASS, M1_BAR_WEAK = 0.70, 0.563


def load_manifest():
    m = json.load(open(os.path.join(HERE, "manifest.json")))["items"]
    return {x["item"]: x for x in m}


def load_rater(tag):
    out = {}
    for b in range(1, N_BATCH + 1):
        p = os.path.join(HERE, f"annot_{tag}", f"batch_{b}.jsonl")
        if not os.path.exists(p):
            print(f"  !! missing {p}")
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            b_ = (r.get("binary") or "").strip().upper()
            if b_ not in ("ENDORSE", "DISTANCED"):
                b_ = None
            out[r["item"]] = {"binary": b_, "voice": (r.get("voice") or "").strip().lower(),
                              "why": r.get("why", "")}
    return out


def majority(votes):
    v = [x for x in votes if x]
    if not v:
        return None
    c = Counter(v)
    if len(c) > 1 and c.most_common(1)[0][1] == c.most_common(2)[1][1]:
        return "TIE"
    return c.most_common(1)[0][0]


def acc(rows, fn):
    v = [fn(r) for r in rows]
    v = [x for x in v if x is not None]
    return (round(sum(v) / len(v), 4) if v else None), len(v)


def binom_p(k, n, p=0.5):
    if n == 0:
        return None
    pr = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return round(min(1.0, sum(q for q in pr if q <= pr[k] + 1e-12)), 4)


def fleiss_kappa(rows, key="votes"):
    """3 raters, 2 categories. rows: list of vote-lists; rows with <2 valid votes dropped."""
    cats = ["ENDORSE", "DISTANCED"]
    tab = []
    for votes in rows:
        v = [x for x in votes if x in cats]
        if len(v) < 2:
            continue
        tab.append([sum(1 for x in v if x == c) for c in cats])
    if not tab:
        return None, 0
    N = len(tab)
    n = sum(tab[0])
    if any(sum(t) != n for t in tab):     # ragged -> per-row n
        Pi = []
        for t in tab:
            ni = sum(t)
            Pi.append((sum(x * x for x in t) - ni) / (ni * (ni - 1)))
        Pbar = sum(Pi) / N
        tot = sum(sum(t) for t in tab)
        pj = [sum(t[j] for t in tab) / tot for j in range(len(cats))]
    else:
        Pi = [(sum(x * x for x in t) - n) / (n * (n - 1)) for t in tab]
        Pbar = sum(Pi) / N
        pj = [sum(t[j] for t in tab) / (N * n) for j in range(len(cats))]
    Pe = sum(p * p for p in pj)
    if abs(1 - Pe) < 1e-12:
        return None, N
    return round((Pbar - Pe) / (1 - Pe), 4), N


def main():
    man = load_manifest()
    raters = {t: load_rater(t) for t in RATERS}
    for t in RATERS:
        print(f"rater {t}: {len(raters[t])} lines, "
              f"{sum(1 for v in raters[t].values() if v['binary'] is None)} unparsed")

    sys.path.insert(0, os.path.join(ROOT, "idea-stage"))
    from voice_field_analysis import GOLD_VOICE
    gv = {k: v[0] for k, v in GOLD_VOICE.items()}

    rows = []
    for name, meta in sorted(man.items()):
        key = (meta["dataset"], meta["id"])
        votes = [raters[t].get(name, {}).get("binary") for t in RATERS]
        vvotes = [raters[t].get(name, {}).get("voice") for t in RATERS]
        rows.append({
            "item": name, "dataset": meta["dataset"], "id": meta["id"],
            "group": meta["group"], "n_frames": meta["n_frames"],
            "gold": GOLD[meta["group"]],
            "votes": votes, "maj": majority(votes),
            "voice_votes": vvotes,
            "voice_folded": [("OWN" if x in VOICE_OWN else
                              "NOT_OWN" if x in VOICE_NOT_OWN else
                              "NONE" if x == "none" else None) for x in vvotes],
            "gold_voice": gv.get(key),
            "in_smoke": key in SMOKE, "qwen_missing": key in QWEN_MISSING,
            "why": {t: raters[t].get(name, {}).get("why", "") for t in RATERS},
        })

    def ok(r, vote_key="maj"):
        v = r[vote_key] if vote_key == "maj" else r[vote_key]
        if v is None or v == "TIE":
            return False        # freeze 4.1: TIE / all-missing count wrong
        return v == r["gold"]

    out = {"n_items": len(rows)}

    # ---------------- primary M1
    S32 = [r for r in rows if r["group"] in SCORED_GROUPS_M1 and r["dataset"] in FRAME_DS
           and not r["in_smoke"] and not r["qwen_missing"]]
    a, n = acc(S32, ok)
    out["M1_primary"] = {
        "n": n, "acc": a,
        "p_vs_chance_0.5": binom_p(int(round((a or 0) * n)), n),
        "verdict": ("PASS" if a >= M1_BAR_PASS else
                    "WEAK" if a >= M1_BAR_WEAK else "FAIL") if a is not None else "NO_DATA"}
    out["VERDICT"] = out["M1_primary"]["verdict"]

    # per rater on the same 32 rows
    out["M1_per_rater"] = {}
    for i, t in enumerate(RATERS):
        ar, nr = acc(S32, lambda r, i=i: (r["votes"][i] == r["gold"])
                     if r["votes"][i] else False)
        out["M1_per_rater"][t] = {"n": nr, "acc": ar}

    # 33-row sensitivity
    S33 = [r for r in rows if r["group"] in SCORED_GROUPS_M1 and r["dataset"] in FRAME_DS
           and not r["in_smoke"]]
    a33, n33 = acc(S33, ok)
    out["M1_sensitivity_33rows"] = {"n": n33, "acc": a33}

    # ---------------- strata
    def blk(sub):
        a_, n_ = acc(sub, ok)
        return {"n": n_, "acc": a_,
                "calls": dict(Counter(r["maj"] or "NONE" for r in sub))}

    out["by_group"] = {g: blk([r for r in rows if r["group"] == g])
                       for g in ("S_FP", "S_FN", "CTRL_HATE", "CTRL_NONHATE")}
    out["by_group_S32"] = {g: blk([r for r in S32 if r["group"] == g])
                           for g in SCORED_GROUPS_M1}
    out["by_frames"] = {"with_frames": blk([r for r in rows if r["n_frames"]]),
                        "no_frames": blk([r for r in rows if not r["n_frames"]])}
    out["by_frames_S_all49"] = {
        "with_frames": blk([r for r in rows if r["n_frames"]
                            and r["group"] in SCORED_GROUPS_M1 and not r["in_smoke"]]),
        "no_frames": blk([r for r in rows if not r["n_frames"]
                          and r["group"] in SCORED_GROUPS_M1])}
    out["by_dataset_S"] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds and r["group"] in SCORED_GROUPS_M1
               and not r["in_smoke"]]
        out["by_dataset_S"][ds] = blk(sub)
    out["all99"] = blk(rows)
    out["all99_excl_ctrl_nonhate"] = blk([r for r in rows if r["group"] != "CTRL_NONHATE"])

    # ---------------- agreement
    def pairwise(sub):
        o = {}
        for i in range(3):
            for j in range(i + 1, 3):
                pair = [(r["votes"][i], r["votes"][j]) for r in sub
                        if r["votes"][i] and r["votes"][j]]
                o[f"{RATERS[i]}-{RATERS[j]}"] = {
                    "n": len(pair),
                    "agree": round(sum(1 for a_, b_ in pair if a_ == b_) / len(pair), 4)
                    if pair else None}
        return o

    k99, nk99 = fleiss_kappa([r["votes"] for r in rows])
    k32, nk32 = fleiss_kappa([r["votes"] for r in S32])
    out["agreement"] = {
        "all99": {"pairwise": pairwise(rows), "fleiss_kappa": k99, "kappa_n": nk99},
        "S32": {"pairwise": pairwise(S32), "fleiss_kappa": k32, "kappa_n": nk32},
        "unanimous_rate_all99": round(sum(1 for r in rows
                                          if len({v for v in r["votes"] if v}) == 1)
                                      / len(rows), 4),
        "acc_by_split_S32": {}}
    for r in S32:
        c = Counter(v for v in r["votes"] if v)
        k = f"{max(c.values(), default=0)}-{min(c.values()) if len(c) > 1 else 0}"
        d = out["agreement"]["acc_by_split_S32"].setdefault(k, {"n": 0, "hit": 0})
        d["n"] += 1
        d["hit"] += int(ok(r))
    for k, d in out["agreement"]["acc_by_split_S32"].items():
        d["acc"] = round(d["hit"] / d["n"], 4)

    # ---------------- voice vs GOLD_VOICE (freeze 4.3.5)
    vrows = [r for r in rows if r["gold_voice"] in ("OWN", "NOT_OWN")]
    hit = 0
    vdet = []
    for r in vrows:
        mv = majority(r["voice_folded"])
        r["voice_maj"] = mv
        good = (mv == r["gold_voice"])
        hit += int(good)
        vdet.append({"item": r["item"], "gold": r["gold_voice"], "maj": mv,
                     "votes": r["voice_folded"], "ok": good})
    out["voice_vs_gold"] = {
        "n": len(vrows), "agree": round(hit / len(vrows), 4) if vrows else None,
        "maj_hist": dict(Counter(r.get("voice_maj") or "NONE_OR_TIE" for r in vrows)),
        "gold_hist": dict(Counter(r["gold_voice"] for r in vrows)),
        "raw_voice_hist_all99": dict(Counter(v for r in rows for v in r["voice_votes"] if v))}

    out["per_item"] = sorted(
        [{"item": r["item"], "ds": r["dataset"], "id": r["id"], "grp": r["group"],
          "frames": r["n_frames"], "gold": r["gold"], "votes": r["votes"], "maj": r["maj"],
          "ok": ok(r), "in_M1_32": r in S32, "voice_votes": r["voice_votes"],
          "gold_voice": r["gold_voice"], "why": r["why"]} for r in rows],
        key=lambda x: (x["grp"], x["ds"], x["id"]))

    json.dump(out, open(os.path.join(HERE, "score.json"), "w"), indent=1, ensure_ascii=False)

    P = print
    P("=" * 78)
    P(f"M1 PRIMARY (32 rows, majority vote) = {out['M1_primary']['acc']} "
      f"(n={out['M1_primary']['n']})  p vs 0.5 = {out['M1_primary']['p_vs_chance_0.5']}")
    P(f"VERDICT = {out['VERDICT']}   [PASS >=0.70 | WEAK >=0.563 | FAIL <0.563]")
    P(f"per rater: " + "  ".join(f"{t}={out['M1_per_rater'][t]['acc']}" for t in RATERS))
    P(f"33-row sensitivity: {out['M1_sensitivity_33rows']}")
    P("-" * 78)
    P("S_FP/S_FN on the 32: " + json.dumps(out["by_group_S32"]))
    P("all groups (99): " + json.dumps(out["by_group"]))
    P("frames strata (99): " + json.dumps(out["by_frames"]))
    P("frames strata (S only): " + json.dumps(out["by_frames_S_all49"]))
    P("by dataset (S only): " + json.dumps(out["by_dataset_S"]))
    P("-" * 78)
    P("agreement: " + json.dumps(out["agreement"], ensure_ascii=False))
    P("-" * 78)
    P("voice vs GOLD_VOICE: " + json.dumps(out["voice_vs_gold"], ensure_ascii=False))


if __name__ == "__main__":
    main()
