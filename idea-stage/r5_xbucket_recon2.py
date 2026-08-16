"""X-bucket recon addendum: what would the two non-oracle exits actually do?

E1  global threshold move   (does the boundary concentration of X buy anything?)
E2  member-vote override    (disagreement-aware combination, applied to EVERY item)
Both are measured on the whole test set, not only on X.  Zero GPU, zero API.
"""
import json
import os
import numpy as np
from sklearn.metrics import f1_score
import r5_xbucket_recon as R

ROOT = R.ROOT
OUT = {}
for ds in R.BEST:
    key = R.BEST[ds]
    blk = R.PIL["datasets"][ds]
    seeds = blk["per_seed"]
    y = np.array(seeds[0]["scores"]["y"], dtype=int)
    base = R.load_split(ds, R.CLIP, "test")
    ids = base["ids"]
    idx = {i: k for k, i in enumerate(ids)}
    n = len(ids)
    bmap = R.BUCK[ds]
    bucket = np.array(["-"] * n, dtype=object)
    for vid, b in bmap.items():
        bucket[idx[vid]] = b
    Xmask = bucket == "X"

    preds, thrs, svs, members = [], [], [], []
    for s in seeds:
        sv = np.array(s["scores"][key], dtype=float)
        t = R.recover_threshold(y, sv, s["methods"][key]["test_macro_f1"])
        preds.append((sv >= t).astype(int))
        thrs.append(t)
        svs.append(sv)
        tags, M, _ = R.recover_members(ds, s)
        members.append(np.stack([M[tg] for tg in tags], 1))
    tags = [t for t, _ in R.CELLS[ds]]
    D = len(tags)
    base_f1 = float(np.mean([f1_score(y, q, average="macro") for q in preds]))

    # ---- E1: best achievable by moving the single global threshold on TEST (oracle)
    best_f1, best_shift, gained, lost = [], [], [], []
    for sv, t, p in zip(svs, thrs, preds):
        cands = np.unique(sv)
        grid = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2,
                               [cands[-1] + 1e-9]])
        f1s = np.array([f1_score(y, (sv >= g).astype(int), average="macro") for g in grid])
        j = int(np.argmax(f1s))
        best_f1.append(float(f1s[j]))
        best_shift.append(float((grid[j] - t) / np.std(sv)))
        q = (sv >= grid[j]).astype(int)
        gained.append(int(((q == y) & (p != y) & Xmask).sum()))
        lost.append(int(((q != y) & (p == y)).sum()))
    e1 = {"base_macro_f1": base_f1,
          "oracle_global_threshold_macro_f1": float(np.mean(best_f1)),
          "delta": float(np.mean(best_f1)) - base_f1,
          "shift_in_sd_units": [round(x, 3) for x in best_shift],
          "X_items_recovered_mean": float(np.mean(gained)),
          "previously_correct_items_broken_mean": float(np.mean(lost))}

    # ---- E2: member-vote override applied to every test item
    e2 = {}
    for rule, need in (("majority", D / 2), ("unanimous", D - 0.5)):
        f1s, flipped, fixedX, broke = [], [], [], []
        for m, p in zip(members, preds):
            mp = (m >= 0).astype(int)
            nvote = mp.sum(1)
            vote_pred = (nvote > D / 2).astype(int)
            fire = (nvote > need) | (nvote < D - need)   # members lean one way strongly
            fire = fire & (vote_pred != p)
            q = p.copy()
            q[fire] = vote_pred[fire]
            f1s.append(f1_score(y, q, average="macro"))
            flipped.append(int(fire.sum()))
            fixedX.append(int(((q == y) & (p != y) & Xmask).sum()))
            broke.append(int(((q != y) & (p == y)).sum()))
        e2[rule] = {"macro_f1": float(np.mean(f1s)),
                    "delta": float(np.mean(f1s)) - base_f1,
                    "items_flipped_mean": float(np.mean(flipped)),
                    "X_fixed_mean": float(np.mean(fixedX)),
                    "correct_broken_mean": float(np.mean(broke))}
    OUT[ds] = {"E1_global_threshold": e1, "E2_member_override": e2}
    print(ds, json.dumps(OUT[ds]), flush=True)

with open(os.path.join(ROOT, "idea-stage", "r5_xbucket_recon2.json"), "w") as f:
    json.dump(OUT, f, indent=1, ensure_ascii=False)
