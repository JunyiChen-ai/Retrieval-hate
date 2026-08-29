"""LIKELIHOOD PROBE -- frozen scorer.

Written and committed BEFORE any eval item was scored by any arm. Nothing here may be
edited after the first eval jsonl exists.

  python score_likelihood.py --arms A1 A2 B1 C1   ->  score_lp.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lp_common import (API_LOST, CS, FRAME_DS, MSP, ROOT, SMOKE_EVAL, SP,  # noqa: E402
                       eval_items)
sys.path.insert(0, os.path.join(ROOT, "idea-stage"))

GOLD = {"S_FP": "OPPOSE", "S_FN": "ENDORSE", "CTRL_HATE": "ENDORSE", "CTRL_NONHATE": None}
BAR_SIGNAL, BAR_FAIL = 0.70, 0.563          # frozen decision line
BASE_DELTA_BAR = 0.10                       # frozen line for the tuning-contrast claim
N_PAIRS = 5


# ------------------------------------------------------------------ helpers
def binom_p(k, n, p=0.5):
    if not n:
        return None
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return round(min(1.0, sum(q for q in probs if q <= probs[k] + 1e-12)), 4)


def collapse(votes):
    v = [x for x in votes if x]
    if not v:
        return None
    c = Counter(v)
    if c["ENDORSE"] == c["OPPOSE"]:
        return "TIE"
    return "ENDORSE" if c["ENDORSE"] > c["OPPOSE"] else "OPPOSE"


def read_arm(arm, which):
    p = os.path.join(HERE, f"lp_{arm}_{which}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def margin(r, field):
    if r.get("error") or not r.get("endorse") or not r.get("oppose"):
        return None
    return r["endorse"][field] - r["oppose"][field]


def prior_table(ctrl_rows, field):
    """mean endorsing-minus-opposing margin per (variant, pair, lang) over the 20 control
    videos, plus the share of control videos on which the endorsing side wins."""
    acc = defaultdict(list)
    for r in ctrl_rows:
        m = margin(r, field)
        if m is not None:
            acc[(r["variant"], r["pair"], r["lang"])].append(m)
    return {k: {"mean": sum(v) / len(v), "n": len(v),
                "endorse_win_rate": sum(1 for x in v if x > 0) / len(v)}
            for k, v in acc.items()}


def item_calls(rows, field, prior=None, variant="v1"):
    """(dataset,id) -> {'call','votes','margins'} using majority over the 5 pairs."""
    by = defaultdict(dict)
    for r in rows:
        if r["variant"] != variant:
            continue
        m = margin(r, field)
        if m is None:
            continue
        if prior is not None:
            m -= prior.get((r["variant"], r["pair"], r["lang"]), {"mean": 0.0})["mean"]
        by[(r["dataset"], r["id"])][r["pair"]] = m
    out = {}
    for k, d in by.items():
        ms = [d[p] for p in sorted(d)]
        votes = ["ENDORSE" if x > 0 else "OPPOSE" for x in ms]
        out[k] = {"call": collapse(votes), "votes": votes,
                  "margins": [round(x, 4) for x in ms],
                  "mean_margin": round(sum(ms) / len(ms), 4)}
    return out


# ------------------------------------------------------------------ baselines
def read_5way(path, keys):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        k = (r["dataset"], r["id"])
        if k in keys:
            out[k] = (r.get("parsed") or {}).get("stance")
    return out


def bin5(s):
    return None if s is None else ("ENDORSE" if s == "endorses" else "OPPOSE")


def contrast_calls(keys, variant="v1"):
    p = os.path.join(CS, "pred_c1.jsonl")
    if not os.path.exists(p):
        return {}
    by = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["variant"] == variant and (r["dataset"], r["id"]) in keys:
            by[(r["dataset"], r["id"])].append(r["vote_side"])
    return {k: collapse(v) for k, v in by.items()}


def gold_voice():
    from voice_field_analysis import GOLD_VOICE
    return {k: v[0] for k, v in GOLD_VOICE.items()}


# ------------------------------------------------------------------ views
def views():
    ev = eval_items()
    S = [x for x in ev if x["group"] in ("S_FP", "S_FN")]
    A33 = [x for x in S if x["dataset"] in FRAME_DS
           and (x["dataset"], x["id"]) not in SMOKE_EVAL]
    A32 = [x for x in A33 if (x["dataset"], x["id"]) not in API_LOST]
    C = [x for x in S if x["dataset"] == "ImpliHateVid"]
    return {"A32_primary": A32, "A33_all_local": A33, "C_textonly": C}, ev


def acc_of(rows, calls):
    hit = n = 0
    per = {}
    for it in rows:
        k = (it["dataset"], it["id"])
        g = GOLD[it["group"]]
        c = (calls.get(k) or {}).get("call") if isinstance(calls.get(k), dict) else calls.get(k)
        per[f"{k[0]}::{k[1]}"] = c
        if g is None:
            continue
        n += 1
        hit += int(c == g)
    return (hit / n if n else None), n, hit, per


def block(rows, calls, ev, name):
    a, n, hit, per = acc_of(rows, calls)
    R = {"view": name, "n": n, "hit": hit, "acc": a,
         "p_vs_chance_0.5": binom_p(hit, n) if n else None,
         "verdict": ("SIGNAL" if (a is not None and a >= BAR_SIGNAL)
                     else "FAIL" if (a is not None and a < BAR_FAIL) else "WEAK"),
         "per_item": per}
    for cell in ("S_FP", "S_FN"):
        sub = [x for x in rows if x["group"] == cell]
        a2, n2, h2, _ = acc_of(sub, calls)
        R[f"acc_{cell}"], R[f"n_{cell}"] = a2, n2
    R["by_dataset"] = {}
    for ds in sorted(set(x["dataset"] for x in rows)):
        sub = [x for x in rows if x["dataset"] == ds]
        a2, n2, _, _ = acc_of(sub, calls)
        R["by_dataset"][ds] = {"n": n2, "acc": a2}
    try:
        gv = gold_voice()
        R["by_voice"] = {}
        for lab in ("OWN", "NOT_OWN", "UNDET"):
            sub = [x for x in rows if gv.get((x["dataset"], x["id"])) == lab]
            a2, n2, _, _ = acc_of(sub, calls)
            R["by_voice"][lab] = {"n": n2, "acc": a2}
    except Exception as e:                                        # pragma: no cover
        R["by_voice"] = {"error": str(e)[:80]}
    R["call_hist"] = dict(Counter(v for v in per.values()))
    return R


def ctrl_block(ev, calls):
    out = {}
    for g in ("CTRL_HATE", "CTRL_NONHATE"):
        sub = [x for x in ev if x["group"] == g and x["dataset"] in FRAME_DS]
        c = [(calls.get((x["dataset"], x["id"])) or {}).get("call") for x in sub]
        c = [x for x in c if x]
        out[g] = {"n": len(c),
                  "endorse_rate": (sum(1 for x in c if x == "ENDORSE") / len(c)) if c else None}
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "B1", "C1"])
    ap.add_argument("--out", default=os.path.join(HERE, "score_lp.json"))
    a = ap.parse_args()

    V, ev = views()
    keys = {(x["dataset"], x["id"]) for x in ev}
    RES = {"bars": {"signal": BAR_SIGNAL, "fail": BAR_FAIL, "base_delta": BASE_DELTA_BAR},
           "view_sizes": {k: len(v) for k, v in V.items()}, "arms": {}}

    # ---- prior-round baselines on exactly these rows
    r1 = read_5way(os.path.join(SP, "pred_strong.jsonl"), keys)
    mk = read_5way(os.path.join(MSP, "pred_m1.jsonl"), keys)
    cc = contrast_calls(keys)
    RES["baselines"] = {}
    for lab, calls in (("round1_binarised", {k: bin5(v) for k, v in r1.items()}),
                       ("round2_masked_binarised", {k: bin5(v) for k, v in mk.items()}),
                       ("round3_contrast_v1", cc)):
        RES["baselines"][lab] = {vn: acc_of(rows, calls)[:3] for vn, rows in V.items()}

    for arm in a.arms:
        ev_rows, ct_rows = read_arm(arm, "eval"), read_arm(arm, "ctrl")
        if not ev_rows:
            RES["arms"][arm] = {"missing": True}
            continue
        A = {"n_eval_rows": len(ev_rows), "n_ctrl_rows": len(ct_rows),
             "n_errors": sum(1 for r in ev_rows if r.get("error")),
             "model": ev_rows[0]["model"], "fmt": ev_rows[0]["fmt"]}
        for field, tag in (("mean_lp", "mean"), ("sum_lp", "sum")):
            pri = prior_table(ct_rows, field) if ct_rows else None
            A[f"template_prior_{tag}"] = ({f"{k[0]}|p{k[1]}|{k[2]}": {
                "mean_margin": round(v["mean"], 4), "endorse_win_rate": round(v["endorse_win_rate"], 3),
                "n": v["n"]} for k, v in sorted(pri.items())} if pri else None)
            for corr in (False, True):
                if corr and not pri:
                    continue
                calls = item_calls(ev_rows, field, prior=(pri if corr else None))
                tagc = f"{tag}{'_priorcorr' if corr else '_raw'}"
                A[tagc] = {vn: block(rows, calls, ev, vn) for vn, rows in V.items()}
                A[tagc]["controls"] = ctrl_block(ev, calls)
            # v2 (target-named) secondary, raw only
            c2 = item_calls(ev_rows, field, prior=None, variant="v2")
            A[f"{tag}_raw_v2"] = {vn: acc_of(rows, c2)[:3] for vn, rows in V.items()}
        RES["arms"][arm] = A

    # ---- tuning contrast
    def acc32(arm, tagc="mean_raw"):
        d = RES["arms"].get(arm, {})
        return (d.get(tagc) or {}).get("A32_primary", {}).get("acc")
    RES["tuning_contrast"] = {
        "C1_minus_B1_same_generation": (None if acc32("C1") is None or acc32("B1") is None
                                        else round(acc32("C1") - acc32("B1"), 4)),
        "C1_minus_A1": (None if acc32("C1") is None or acc32("A1") is None
                        else round(acc32("C1") - acc32("A1"), 4)),
        "bar": BASE_DELTA_BAR}
    json.dump(RES, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("wrote", a.out)
    for arm in a.arms:
        d = RES["arms"].get(arm, {})
        if d.get("missing"):
            print(f"{arm}: MISSING")
            continue
        for tagc in ("mean_raw", "mean_priorcorr"):
            b = (d.get(tagc) or {}).get("A32_primary")
            if b:
                print(f"{arm:>3} {tagc:>15}  A32 acc={b['acc']:.4f} ({b['hit']}/{b['n']}) "
                      f"S_FP={b['acc_S_FP']} S_FN={b['acc_S_FN']} {b['verdict']}")
    print("tuning contrast:", RES["tuning_contrast"])


if __name__ == "__main__":
    main()
