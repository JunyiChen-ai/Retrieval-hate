#!/usr/bin/env python
"""CAT close-out Leg D -- per-item read-out audit.

Frozen design: idea-stage/CAT_CLOSEOUT_FREEZE.md section 5.  DESCRIPTIVE ONLY:
no bar, no gate, no verdict.  Zero new runs, zero GPU.

Source: the per-item head logits already dumped by R10-COMBO, arms A0 and CAT,
each seed read at its OWN P1 epoch selected from its OWN dev curve.

BELT D1: macro-F1 recomputed from the dumped logits must agree with the
trainlog-logged macro-F1 to 1e-4 for every run used.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_cc import curves, select_p1, BELT_TOL  # noqa: E402

FIX_HI = 2.0 / 3.0   # frozen in the freeze: A0 wrong in >= 2/3 of seeds
FIX_LO = 1.0 / 3.0   # and CAT wrong in <= 1/3 of seeds
TOP_N = 10


def err_rates(logdir, ds, arms, seeds):
    """-> ids, labels, {arm: per-item error rate over seeds}, belt worst"""
    ids_ref = None
    rates, worst = {}, 0.0
    y_ref = None
    for arm in arms:
        acc = None
        for s in seeds:
            tl = os.path.join(logdir, "%s_%s_s%d.trainlog" % (ds, arm, s))
            sj = os.path.join(logdir, "%s_%s_s%d.scores.jsonl" % (ds, arm, s))
            cur, sc, w = curves(sj, tl)
            worst = max(worst, w)
            e1 = select_p1(cur["dev"])
            ids, y, z = sc["test"][e1]
            if ids_ref is None:
                ids_ref, y_ref = ids, y
            assert ids == ids_ref, "test id order differs (%s s%d)" % (arm, s)
            pred = (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)
            e = (pred != y).astype(float)
            acc = e if acc is None else acc + e
        rates[arm] = acc / len(seeds)
    return ids_ref, y_ref, rates, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ocr", default="")
    ap.add_argument("--cc", default="", help="Leg A -cc test cache for span statistics")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    ids, y, rates, worst = err_rates(a.logdir, a.dataset, ["A0", "CAT"], seeds)
    print("BELT D1 max |macroF1(dumped) - macroF1(trainlog)| = %.2e" % worst)
    if worst > BELT_TOL:
        raise SystemExit("HALT: BELT D1 failed")

    gt = {json.loads(l)["id"]: json.loads(l) for l in open(a.gt)}
    ocr = {}
    if a.ocr and os.path.exists(a.ocr):
        for l in open(a.ocr):
            d = json.loads(l)
            ocr[str(d.get("id", d.get("video_id", "")))] = d

    span = {}
    if a.cc and os.path.exists(a.cc):
        import torch
        cc = torch.load(a.cc, map_location="cpu", weights_only=False)
        span = cc["stats"]
        norms = {}
        for s in ("A0", "TXT"):
            v = cc["spans"]["28"][s].float()
            norms[s] = v.norm(dim=1).numpy()
        span_norms = {i: {s: float(norms[s][k]) for s in norms}
                      for k, i in enumerate(cc["ids"][0])}
    else:
        span_norms = {}

    dA, dC = rates["A0"], rates["CAT"]
    fixed = [(ids[i], dA[i] - dC[i]) for i in range(len(ids))
             if dA[i] >= FIX_HI and dC[i] <= FIX_LO]
    broken = [(ids[i], dC[i] - dA[i]) for i in range(len(ids))
              if dC[i] >= FIX_HI and dA[i] <= FIX_LO]
    fixed.sort(key=lambda t: -t[1])
    broken.sort(key=lambda t: -t[1])

    def tlen(i):
        t = gt[i].get("text") or ""
        return len(t.strip())

    def ocr_len(i):
        d = ocr.get(i)
        if not d:
            return None
        for k in ("ocr_text", "text", "video_text", "ocr"):
            if k in d and isinstance(d[k], str):
                return len(d[k].strip())
        return None

    def describe(sel):
        rows = []
        for i, gap in sel[:TOP_N]:
            r = {"id": i, "label": int(gt[i]["label"]), "gap": float(gap),
                 "a0_err_rate": float(dA[ids.index(i)]),
                 "cat_err_rate": float(dC[ids.index(i)]),
                 "transcript_chars": tlen(i), "ocr_chars": ocr_len(i),
                 "transcript": (gt[i].get("text") or "")[:600]}
            if i in span:
                r["span"] = span[i]
                r["span"]["A0_len"] = span[i]["T"] - span[i]["hdr"]
                r["span"]["TXT_len"] = span[i]["hdr"] - span[i]["v_end"]
            if i in span_norms:
                r["l2_norms"] = span_norms[i]
            rows.append(r)
        return rows

    rest = [i for i in ids if i not in {x for x, _ in fixed} | {x for x, _ in broken}]

    def dist(idlist):
        L = np.array([tlen(i) for i in idlist], dtype=float)
        if not len(L):
            return {}
        return {"n": len(L), "empty_rate": float((L == 0).mean()),
                "median_chars": float(np.median(L)), "mean_chars": float(L.mean()),
                "q25": float(np.percentile(L, 25)), "q75": float(np.percentile(L, 75))}

    res = {"what": "CAT close-out Leg D per-item read-out audit (descriptive)",
           "freeze": "idea-stage/CAT_CLOSEOUT_FREEZE.md 5", "dataset": a.dataset,
           "seeds": seeds, "n_test": len(ids), "belt_D1_max_abs_diff": worst,
           "thresholds": {"wrong_in_at_least": FIX_HI, "wrong_in_at_most": FIX_LO},
           "n_fixed": len(fixed), "n_broken": len(broken),
           "mean_err_rate": {"A0": float(dA.mean()), "CAT": float(dC.mean())},
           "transcript_len": {"FIXED": dist([x for x, _ in fixed]),
                              "BROKEN": dist([x for x, _ in broken]),
                              "REST": dist(rest)},
           "label_mix": {"FIXED": [int(gt[x]["label"]) for x, _ in fixed],
                         "BROKEN": [int(gt[x]["label"]) for x, _ in broken]},
           "FIXED": describe(fixed), "BROKEN": describe(broken)}

    if span:
        A0len = np.array([span[i]["T"] - span[i]["hdr"] for i in ids if i in span])
        TXTlen = np.array([span[i]["hdr"] - span[i]["v_end"] for i in ids if i in span])
        T = np.array([span[i]["T"] for i in ids if i in span])
        res["mechanical_audit"] = {
            "n_items": int(len(A0len)),
            "A0_span_len": {"min": int(A0len.min()), "max": int(A0len.max()),
                            "median": float(np.median(A0len)),
                            "all_equal_3": bool((A0len == 3).all())},
            "TXT_span_len": {"min": int(TXTlen.min()), "max": int(TXTlen.max()),
                             "median": float(np.median(TXTlen))},
            "total_tokens": {"min": int(T.min()), "max": int(T.max()),
                             "median": float(np.median(T))},
            "n_degenerate_v_end_zero": int(sum(1 for i in ids
                                               if i in span and span[i]["v_end"] == 0))}
        if span_norms:
            nA = np.array([span_norms[i]["A0"] for i in ids if i in span_norms])
            nT = np.array([span_norms[i]["TXT"] for i in ids if i in span_norms])
            res["mechanical_audit"]["l2_norms"] = {
                "A0_min": float(nA.min()), "A0_max": float(nA.max()),
                "TXT_min": float(nT.min()), "TXT_max": float(nT.max()),
                "note": "the frozen extractor L2-normalises every span, so per-item "
                        "norms are 1.0 by construction (0.0 only for zero-guard rows); "
                        "pre-normalisation magnitude is not stored and the head never "
                        "sees it."}

    json.dump(res, open(a.out, "w"), indent=1)
    print("%s: n_test=%d FIXED=%d BROKEN=%d  mean err A0=%.3f CAT=%.3f"
          % (a.dataset, len(ids), len(fixed), len(broken), dA.mean(), dC.mean()))
    print("transcript len:", json.dumps(res["transcript_len"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
