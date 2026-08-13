"""ARBITER step 3 -- frozen fusion grid, val-only selection, single test readout.

Grid (frozen, ARBITER_FREEZE.md sections 2 and 4):
  w    in {0.1, 0.2, 0.3, 0.4}   band = |p_head - 0.5| < w
  rule in {a, b, c}

  p_mllm = confidence            if hateful
         = 1 - confidence        if not hateful

  in band, with a usable judgement:
    a  hard replace   p_final = 1.0 if hateful else 0.0
    b  average        p_final = (p_head + p_mllm) / 2
    c  agree-only     agree -> p_final = clip(0.5 + 2 * (p_head - 0.5), 0, 1)
                      disagree -> p_final = 0.5
  otherwise (out of band, moderation-refused, or unparsed): p_final = p_head

Decision: pred = 1 iff p_final >= 0.5.  This `>= 0.5` convention is imported unchanged
from idea-stage/desc_channel/analyze_arms.py and applies to rule c's 0.5 as well.

Selection: per seed, argmax over the 12 combos of VAL macro-F1; ties broken by smaller w
then rule order a < b < c.  The selected combo is then evaluated once on TEST.

Verdict: GO iff mean over 3 seeds of (test macro-F1 fused - test macro-F1 baseline)
>= +0.005 AND all 3 seeds strictly positive.  Otherwise KILL.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = [0.1, 0.2, 0.3, 0.4]
RULES = ["a", "b", "c"]
SEEDS = ["0", "1", "2"]
THRESH = 0.005


def macro_f1(y, p):
    out = []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(out))


def load_judgements():
    j = {}
    p = os.path.join(HERE, "judgements.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                j[r["id"]] = r
    return j


def p_mllm(rec):
    """-> float or None if there is no usable judgement."""
    if not rec or rec.get("parse") != "ok" or not rec.get("judgement"):
        return None
    g = rec["judgement"]
    c = float(g["confidence"])
    return c if g["hateful"] else 1.0 - c


def fuse(ids, ph, jud, w, rule):
    out = np.array(ph, dtype=float).copy()
    used = np.zeros(len(ids), dtype=bool)
    for i, (vid, p) in enumerate(zip(ids, ph)):
        if abs(p - 0.5) >= w:
            continue
        pm = p_mllm(jud.get(vid))
        if pm is None:
            continue
        used[i] = True
        if rule == "a":
            out[i] = 1.0 if jud[vid]["judgement"]["hateful"] else 0.0
        elif rule == "b":
            out[i] = (p + pm) / 2.0
        else:
            agree = (pm >= 0.5) == (p >= 0.5)
            out[i] = min(1.0, max(0.0, 0.5 + 2.0 * (p - 0.5))) if agree else 0.5
    return out, used


def main():
    hp = json.load(open(os.path.join(HERE, "head_probs.json")))
    jud = load_judgements()
    res = {"grid": {}, "selected": {}, "baseline": {}, "diag": {}, "n_judgements": len(jud)}

    # ---------------------------------------------------------------- grid on val
    for seed in SEEDS:
        sp = hp["seeds"][seed]["splits"]
        vids = sp["val"]["ids"]
        vy = np.asarray(sp["val"]["labels"])
        vp = sp["val"]["prob"]
        g = {}
        for w in WS:
            for rule in RULES:
                f, used = fuse(vids, vp, jud, w, rule)
                g["w%.1f_%s" % (w, rule)] = {
                    "val_macro_f1": macro_f1(vy, (f >= 0.5).astype(int)),
                    "n_used": int(used.sum())}
        res["grid"][seed] = g
        best = max([(w, r) for w in WS for r in RULES],
                   key=lambda k: (g["w%.1f_%s" % k]["val_macro_f1"], -k[0],
                                  -RULES.index(k[1])))
        res["selected"][seed] = {"w": best[0], "rule": best[1],
                                 "val_macro_f1": g["w%.1f_%s" % best]["val_macro_f1"],
                                 "val_baseline_macro_f1": macro_f1(
                                     vy, (np.asarray(vp) >= 0.5).astype(int)),
                                 "n_used_val": g["w%.1f_%s" % best]["n_used"]}

    # ---------------------------------------------------------------- single test readout
    deltas = []
    for seed in SEEDS:
        sp = hp["seeds"][seed]["splits"]
        tids = sp["test"]["ids"]
        ty = np.asarray(sp["test"]["labels"])
        tp = sp["test"]["prob"]
        w = res["selected"][seed]["w"]
        rule = res["selected"][seed]["rule"]
        base = macro_f1(ty, (np.asarray(tp) >= 0.5).astype(int))
        f, used = fuse(tids, tp, jud, w, rule)
        fused = macro_f1(ty, (f >= 0.5).astype(int))
        res["baseline"][seed] = base
        res["selected"][seed].update({"test_baseline_macro_f1": base,
                                      "test_fused_macro_f1": fused,
                                      "test_delta": fused - base,
                                      "n_used_test": int(used.sum())})
        deltas.append(fused - base)

    mean_d = float(np.mean(deltas))
    res["verdict"] = {
        "per_seed_delta": deltas, "mean_delta": mean_d,
        "clause1_mean_ge_0.005": bool(mean_d >= THRESH),
        "clause2_3of3_positive": bool(all(x > 0 for x in deltas)),
        "verdict": "GO" if (mean_d >= THRESH and all(x > 0 for x in deltas)) else "KILL"}

    # ------------------------------------------------- descriptive, not part of the verdict
    for seed in SEEDS:
        sp = hp["seeds"][seed]["splits"]
        for split in ("val", "test"):
            ids = sp[split]["ids"]
            y = np.asarray(sp[split]["labels"])
            ph = np.asarray(sp[split]["prob"])
            for w in WS:
                sel = [i for i in range(len(ids))
                       if abs(ph[i] - 0.5) < w and p_mllm(jud.get(ids[i])) is not None]
                nb = int((np.abs(ph - 0.5) < w).sum())
                if not sel:
                    res["diag"]["%s_%s_w%.1f" % (seed, split, w)] = {
                        "n_band": nb, "n_judged": 0}
                    continue
                hw = np.asarray([(ph[i] >= 0.5) != bool(y[i]) for i in sel])
                mw = np.asarray([bool(jud[ids[i]]["judgement"]["hateful"]) != bool(y[i])
                                 for i in sel])
                res["diag"]["%s_%s_w%.1f" % (seed, split, w)] = {
                    "n_band": nb, "n_judged": len(sel),
                    "head_acc": float((~hw).mean()), "mllm_acc": float((~mw).mean()),
                    "both_wrong": int((hw & mw).sum()),
                    "head_wrong_only": int((hw & ~mw).sum()),
                    "mllm_wrong_only": int((~hw & mw).sum()),
                    "both_right": int((~hw & ~mw).sum()),
                    "err_jaccard": float((hw & mw).sum() / max(1, (hw | mw).sum()))}

    json.dump(res, open(os.path.join(HERE, "fuse_results.json"), "w"), indent=1)

    print("selection (val only)")
    for seed in SEEDS:
        s = res["selected"][seed]
        print("  seed %s: w=%.1f rule=%s  val %.4f (base %.4f, n_used %d)"
              % (seed, s["w"], s["rule"], s["val_macro_f1"],
                 s["val_baseline_macro_f1"], s["n_used_val"]))
    print("\ntest readout")
    for seed in SEEDS:
        s = res["selected"][seed]
        print("  seed %s: base %.4f -> fused %.4f  delta %+.4f (n_used %d)"
              % (seed, s["test_baseline_macro_f1"], s["test_fused_macro_f1"],
                 s["test_delta"], s["n_used_test"]))
    print("\nmean delta %+.4f  per-seed %s"
          % (mean_d, " ".join("%+.4f" % x for x in deltas)))
    print("FROZEN VERDICT: %s" % res["verdict"]["verdict"])
    print("wrote fuse_results.json")


if __name__ == "__main__":
    main()
