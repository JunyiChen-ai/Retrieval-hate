"""STANCE PILOT -- deterministic sample selection.

Frozen 2026-08-11 (see idea-stage/STANCE_PILOT_FREEZE.md). Seed 20260811.

EVAL sample (99 items):
  * all 49 primary-S bucket errors of the round-4 comparator (idea-stage/r5_buckets.json)
  * 50 matched controls, allocated across datasets in proportion to the S counts
    (HateMM 8 / MHC 16 / MHC_zh 12 / ImpliHateVid 14), half correctly-predicted hate
    (label==1, not in err_ids) and half correctly-predicted non-hate (label==0,
    not in err_ids), sampled with numpy default_rng(20260811) over the id list
    sorted lexicographically.

SMOKE sample (5 items): drawn from OUTSIDE the eval sample so that prompt
iteration on the smoke items cannot contaminate the frozen metrics.
  * 2 secondary-S errors (primary bucket D / M, secondary S) -- stance-like but
    not part of the 49
  * 1 X-bucket error
  * 2 controls drawn from the control pool AFTER the eval controls are removed
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260811
DATASETS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]
N_CTRL = {"HateMM": 8, "MHC": 16, "MHC_zh": 12, "ImpliHateVid": 14}
SMOKE_SEC_S = ["hate_video_295", "lzKJ_AWegCc"]      # secondary-S, primary D / M
SMOKE_X = ["NH_350"]                                  # X bucket, ImpliHateVid
# smoke round 2 (added 2026-08-11 after V1.0 smoke: the round-1 smoke set contained no
# non-endorsing error and no non-hate control, so it could not detect a prompt biased
# towards "endorses"). Explicit ids, appended after the rng draws so `eval` is unchanged.
SMOKE_EXTRA_ERR = ["NH_180"]                          # secondary-S, FP side, ImpliHateVid
SMOKE_EXTRA_CTRL_NONHATE = [("MHC", "sVA-q76vNBo"), ("ImpliHateVid", "NH_836")]


def labels(ds):
    o = {}
    for line in open(os.path.join(ROOT, "data", "gt", ds, "test.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        o[r["id"]] = int(r["label"])
    return o


def main():
    A = json.load(open(os.path.join(ROOT, "idea-stage", "r5_phase_a.json")))["A2_error_attribution"]
    B = json.load(open(os.path.join(ROOT, "idea-stage", "r5_buckets.json")))
    rng = np.random.default_rng(SEED)

    eval_items, smoke_items = [], []
    ctrl_pool_left = {}
    for ds in DATASETS:
        lab = labels(ds)
        err = set(A[ds]["err_ids"])
        fp = set(A[ds]["fp_ids"])
        # --- errors: primary bucket S
        for i in sorted(k for k, v in B[ds].items() if v == "S"):
            eval_items.append({"dataset": ds, "id": i, "group": "S_FP" if i in fp else "S_FN",
                               "label": lab[i], "pred_correct": False})
        # --- controls
        correct = [i for i in sorted(lab) if i not in err]
        pos = [i for i in correct if lab[i] == 1]
        neg = [i for i in correct if lab[i] == 0]
        k = N_CTRL[ds] // 2
        pick_p = list(rng.choice(pos, size=k, replace=False))
        pick_n = list(rng.choice(neg, size=k, replace=False))
        for i in pick_p:
            eval_items.append({"dataset": ds, "id": str(i), "group": "CTRL_HATE",
                               "label": 1, "pred_correct": True})
        for i in pick_n:
            eval_items.append({"dataset": ds, "id": str(i), "group": "CTRL_NONHATE",
                               "label": 0, "pred_correct": True})
        ctrl_pool_left[ds] = ([i for i in pos if i not in set(map(str, pick_p))],
                              [i for i in neg if i not in set(map(str, pick_n))])

    # --- smoke: outside the eval sample
    for i in SMOKE_SEC_S + SMOKE_X + SMOKE_EXTRA_ERR:
        ds = next(d for d in DATASETS if i in B[d])
        lab = labels(ds)
        fp = set(A[ds]["fp_ids"])
        smoke_items.append({"dataset": ds, "id": i,
                            "group": "SMOKE_ERR_" + ("FP" if i in fp else "FN"),
                            "label": lab[i], "pred_correct": False})
    for ds, want in (("HateMM", 1), ("MHC_zh", 1)):
        pos, neg = ctrl_pool_left[ds]
        smoke_items.append({"dataset": ds, "id": str(rng.choice(pos, size=1)[0]),
                            "group": "SMOKE_CTRL_HATE", "label": 1, "pred_correct": True})

    for ds, i in SMOKE_EXTRA_CTRL_NONHATE:
        smoke_items.append({"dataset": ds, "id": i, "group": "SMOKE_CTRL_NONHATE",
                            "label": 0, "pred_correct": True})

    ids = [x["id"] for x in eval_items]
    assert len(ids) == len(set(ids)) == 99, len(ids)
    assert not (set(ids) & {x["id"] for x in smoke_items})
    out = {"seed": SEED, "eval": eval_items, "smoke": smoke_items}
    p = os.path.join(ROOT, "idea-stage", "stance_pilot", "sample.json")
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    from collections import Counter
    print(Counter((x["dataset"], x["group"]) for x in eval_items))
    print("eval", len(eval_items), "smoke", len(smoke_items), "->", p)


if __name__ == "__main__":
    main()
