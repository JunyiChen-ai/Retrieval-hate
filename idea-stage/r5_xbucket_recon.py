"""Round-5 X-bucket reconnaissance (zero GPU, zero API).

Reads only frozen artefacts:
  idea-stage/r4_pilot1.json      per-seed test scores of every round-4 ensemble comparator
  idea-stage/r5_buckets.json     manual bucket attribution of the error set
  idea-stage/r5_phase_a.json     error ids / thresholds context
  data/CLIP_Embedding/<ds>/*.pt  frozen encoder features (train + test)

Five hypotheses about the 37 "X" (ordinary ranking error) items:
  H1 boundary        distance to the decision threshold
  H2 coverage        cosine distance to the nearest training neighbours
  H3 member split    per-encoder logits -> does any member already get it right
  H4 label purity    top-20 train-neighbour label purity
  H5 seed stability  3-seed agreement of the error

Per-encoder test logits are NOT stored in r4_pilot1.json.  They are recovered
exactly from the stored ensemble scores (algebraic inversion, see recover_members).
"""
import json
import os
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = {}

BEST = {"HateMM": "mlp", "MHC": "mean_logit", "MHC_zh": "logistic",
        "ImpliHateVid": "logistic"}
CLIP = "openai_clip-vit-large-patch14-336_HF"
QWEN = "Qwen2.5-VL-7B-Instruct_HF"
LORA = {"HateMM": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
        "MHC": "Qwen2.5-VL-7B-Instruct-LoRA_HF",
        "MHC_zh": "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"}
CELLS = {
    "HateMM": [("CLIP", CLIP), ("QWEN", QWEN), ("LORA", LORA["HateMM"])],
    "MHC": [("CLIP", CLIP), ("QWEN", QWEN), ("LORA", LORA["MHC"])],
    "MHC_zh": [("CLIP", CLIP), ("QWEN", QWEN), ("LORA", LORA["MHC_zh"])],
    "ImpliHateVid": [("CLIP", CLIP), ("QWEN", QWEN)],
}
SPLIT_FILE = {"train": "train", "val": "dev_seen", "test": "test_seen"}
PIL = json.load(open(os.path.join(ROOT, "idea-stage", "r4_pilot1.json")))
BUCK = json.load(open(os.path.join(ROOT, "idea-stage", "r5_buckets.json")))


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_split(ds, tag, split):
    p = os.path.join(ROOT, "data", "CLIP_Embedding", ds,
                     f"{SPLIT_FILE[split]}_{tag}.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    ids = list(d["ids"][0]) if isinstance(d["ids"][0], list) else list(d["ids"])
    return {"ids": ids, "img": d["img_feats"].float().numpy(),
            "txt": d["text_feats"].float().numpy(),
            "y": torch.as_tensor(d["labels"]).view(-1).numpy().astype(int)}


# --------------------------------------------------------------------- threshold
def recover_threshold(y, sv, target):
    cands = np.unique(sv)
    grid = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2,
                           [cands[-1] + 1e-9]])
    hits = [t for t in grid
            if abs(f1_score(y, (sv >= t).astype(int), average="macro") - target) < 1e-9]
    assert hits, "threshold recovery failed"
    return float(hits[len(hits) // 2])


# --------------------------------------------------------------------- members
def solve_pair(s, q):
    """x+y = s, sigmoid(x)+sigmoid(y) = q  ->  (s/2+t, s/2-t), t>=0.

    g(t)=sig(s/2+t)+sig(s/2-t) is strictly monotone in t for s != 0, so the
    unordered pair is unique.  Bisection.
    """
    lo, hi = 0.0, 1.0
    g = lambda t: sig(s / 2 + t) + sig(s / 2 - t)
    g0 = g(0.0)
    if abs(q - g0) < 1e-12:
        return 0.0
    dec = s > 0            # g decreasing in t when s>0
    for _ in range(200):
        if (g(hi) - q) * (g0 - q) <= 0:
            break
        hi *= 2
        if hi > 1e4:
            break
    for _ in range(200):
        mid = (lo + hi) / 2
        gm = g(mid)
        if (gm > q) == dec:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def recover_members(ds, seed_row):
    """Return (tags, M) where M[k] = test logit vector of encoder tags[k]."""
    tags = [t for t, _ in CELLS[ds]]
    S = seed_row["scores"]
    ml = np.array(S["mean_logit"], dtype=float)
    mp = np.array(S["mean_prob"], dtype=float)
    single = np.array(S["single"], dtype=float)
    ref = seed_row["ref"]
    D = len(tags)
    total = ml * D
    if D == 2:
        other = total - single
        M = {ref: single, [t for t in tags if t != ref][0]: other}
        resid = float(np.max(np.abs((sig(single) + sig(other)) / 2 - mp)))
        return tags, M, {"mode": "exact2", "mean_prob_resid": resid}
    # D == 3
    s = total - single
    q = mp * D - sig(single)
    tvec = np.array([solve_pair(si, qi) for si, qi in zip(s, q)])
    A = s / 2 + tvec
    B = s / 2 - tvec
    others = [t for t in tags if t != ref]
    # order A/B using the `weighted` comparator: weighted = w_ref*r + w_a*x + w_b*y
    W = np.array(S["weighted"], dtype=float)
    best = None
    for wref in np.linspace(0.0, 1.0, 201):
        rest = 1.0 - wref
        for frac in np.linspace(0.0, 1.0, 201):
            wa, wb = rest * frac, rest * (1 - frac)
            d = wa - wb
            if abs(d) < 1e-9:
                continue
            pred_sigt = (W - wref * single - rest * s / 2) / d
            err = float(np.mean((np.abs(pred_sigt) - tvec) ** 2))
            if best is None or err < best[0]:
                best = (err, wref, wa, wb, np.sign(pred_sigt))
    err, wref, wa, wb, sgn = best
    # refine the ordering with the `logistic` comparator, which is EXACTLY affine in
    # the member logits (sklearn decision_function).  Coordinate descent: fit the
    # affine map, then re-choose each item's sign, until the assignment is stable.
    lg = np.array(S["logistic"], dtype=float)
    for _ in range(50):
        x = s / 2 + sgn * tvec
        yy = s / 2 - sgn * tvec
        X = np.stack([single, x, yy, np.ones_like(s)], 1)
        coef, *_ = np.linalg.lstsq(X, lg, rcond=None)
        cr, cx, cy, cb = coef
        base_part = cr * single + (cx + cy) * s / 2 + cb
        d = cx - cy
        if abs(d) < 1e-12:
            break
        new = np.sign((lg - base_part) / d)
        new[new == 0] = 1
        if np.array_equal(new, sgn):
            break
        sgn = new
    x = s / 2 + sgn * tvec
    y = s / 2 - sgn * tvec
    M = {ref: single, others[0]: x, others[1]: y}
    # verification: mean_logit and mean_prob must be reproduced
    stack = np.stack([M[t] for t in tags], 1)
    v1 = float(np.max(np.abs(stack.mean(1) - ml)))
    v2 = float(np.max(np.abs(sig(stack).mean(1) - mp)))
    # independent check: `logistic` must be an affine function of the members
    Xa = np.concatenate([stack, np.ones((len(ml), 1))], 1)
    coef, *_ = np.linalg.lstsq(Xa, lg, rcond=None)
    v3 = float(np.max(np.abs(Xa @ coef - lg)))
    # independent check: `weighted` is a convex combination of the members
    wcoef, *_ = np.linalg.lstsq(stack, W, rcond=None)
    v4 = float(np.max(np.abs(stack @ wcoef - W)))
    return tags, M, {"mode": "solved3", "order_fit_mse": err, "w_ref_grid": wref,
                     "mean_logit_resid": v1, "mean_prob_resid": v2,
                     "logistic_affine_maxresid": v3,
                     "weighted_linear_maxresid": v4,
                     "weighted_recovered_w": wcoef.tolist(),
                     "logistic_coef": coef.tolist()}


# --------------------------------------------------------------------- features
def feat_matrix(d):
    a = d["img"] / (np.linalg.norm(d["img"], axis=1, keepdims=True) + 1e-9)
    b = d["txt"] / (np.linalg.norm(d["txt"], axis=1, keepdims=True) + 1e-9)
    f = np.concatenate([a, b], 1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


def quant(v, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v = np.asarray(v, dtype=float)
    return {str(q): float(np.quantile(v, q)) for q in qs}


# --------------------------------------------------------------------- main
def run():
    res = {}
    for ds in BEST:
        key = BEST[ds]
        blk = PIL["datasets"][ds]
        seeds = blk["per_seed"]
        y = np.array(seeds[0]["scores"]["y"], dtype=int)
        base = load_split(ds, CLIP, "test")
        ids = base["ids"]
        assert np.array_equal(base["y"], y)
        idx = {i: k for k, i in enumerate(ids)}
        n = len(ids)

        # per-seed thresholds + predictions of the round-4 best comparator
        preds, thrs, svs = [], [], []
        for s in seeds:
            sv = np.array(s["scores"][key], dtype=float)
            t = recover_threshold(y, sv, s["methods"][key]["test_macro_f1"])
            preds.append((sv >= t).astype(int))
            thrs.append(t)
            svs.append(sv)
        pred = (np.mean(preds, axis=0) >= 0.5).astype(int)
        err_mask = pred != y
        correct_mask = ~err_mask

        bmap = BUCK[ds]
        bucket = np.array(["-"] * n, dtype=object)
        for vid, b in bmap.items():
            bucket[idx[vid]] = b
        Xmask = bucket == "X"
        Omask = np.isin(bucket, ["S", "O", "M", "A", "D"])

        # ---------------- H1 boundary
        # scale-free: fraction of test items lying between the item and threshold
        h1 = {}
        per_item_gap = np.zeros(n)
        per_item_rankgap = np.zeros(n)
        for sv, t in zip(svs, thrs):
            sd = np.std(sv)
            per_item_gap += np.abs(sv - t) / sd
            r = np.array([np.mean((sv <= v) if v >= t else (sv >= v)) for v in sv])
            # rank gap = share of test items strictly between score and threshold
            rg = np.array([np.mean(((sv > min(v, t)) & (sv < max(v, t)))) for v in sv])
            per_item_rankgap += rg
        per_item_gap /= len(svs)
        per_item_rankgap /= len(svs)
        for nm, m in (("X_err", Xmask), ("nonX_err", Omask), ("correct", correct_mask)):
            h1[nm] = {"n": int(m.sum()),
                      "sd_units": quant(per_item_gap[m]),
                      "rank_gap": quant(per_item_rankgap[m]),
                      "mean_sd_units": float(per_item_gap[m].mean()),
                      "mean_rank_gap": float(per_item_rankgap[m].mean())}
        # class-matched correct control
        for lab in (0, 1):
            m = correct_mask & (y == lab)
            h1[f"correct_y{lab}"] = {"n": int(m.sum()),
                                     "mean_sd_units": float(per_item_gap[m].mean()),
                                     "mean_rank_gap": float(per_item_rankgap[m].mean())}
        h1["X_by_class"] = {str(l): {"n": int((Xmask & (y == l)).sum()),
                                     "mean_sd_units": float(per_item_gap[Xmask & (y == l)].mean())
                                     if (Xmask & (y == l)).sum() else None}
                            for l in (0, 1)}
        # where do the X items sit in the |gap| distribution of ALL test items
        allrank = np.array([np.mean(per_item_gap <= v) for v in per_item_gap])
        h1["X_gap_percentile_within_test"] = quant(allrank[Xmask])
        h1["correct_gap_percentile_within_test"] = quant(allrank[correct_mask])

        # ---------------- H2/H4 retrieval geometry
        h2, h4 = {}, {}
        for tag, mt in CELLS[ds]:
            tr = load_split(ds, mt, "train")
            te = load_split(ds, mt, "test")
            assert te["ids"] == ids
            Ftr, Fte = feat_matrix(tr), feat_matrix(te)
            simm = Fte @ Ftr.T
            order = np.argsort(-simm, axis=1)
            top1 = simm[np.arange(n), order[:, 0]]
            top5 = np.take_along_axis(simm, order[:, :5], 1).mean(1)
            ytr = tr["y"]
            nb20 = ytr[order[:, :20]]
            purity = (nb20 == y[:, None]).mean(1)
            nb20_pos = nb20.mean(1)
            for nm, m in (("X_err", Xmask), ("nonX_err", Omask), ("correct", correct_mask)):
                h2.setdefault(tag, {})[nm] = {
                    "n": int(m.sum()),
                    "top1_cos_mean": float(top1[m].mean()),
                    "top1_cos_q": quant(top1[m]),
                    "top5_cos_mean": float(top5[m].mean()),
                }
                h4.setdefault(tag, {})[nm] = {
                    "top20_gold_purity_mean": float(purity[m].mean()),
                    "top20_gold_purity_q": quant(purity[m]),
                    "frac_purity_below_0.5": float((purity[m] < 0.5).mean()),
                }
            for lab in (0, 1):
                m = correct_mask & (y == lab)
                h2[tag][f"correct_y{lab}"] = {"n": int(m.sum()),
                                              "top1_cos_mean": float(top1[m].mean()),
                                              "top5_cos_mean": float(top5[m].mean())}
                h4[tag][f"correct_y{lab}"] = {
                    "top20_gold_purity_mean": float(purity[m].mean())}
            for lab in (0, 1):
                m = Xmask & (y == lab)
                if m.sum():
                    h2[tag][f"X_y{lab}"] = {"n": int(m.sum()),
                                            "top1_cos_mean": float(top1[m].mean()),
                                            "top5_cos_mean": float(top5[m].mean())}
                    h4[tag][f"X_y{lab}"] = {
                        "top20_gold_purity_mean": float(purity[m].mean())}
            del simm, order

        # ---------------- H3 member split + H5 seed stability
        members = []
        recon_info = []
        for s in seeds:
            tags, M, info = recover_members(ds, s)
            members.append(np.stack([M[t] for t in tags], 1))
            recon_info.append(info)
        tags = [t for t, _ in CELLS[ds]]
        h3 = {"reconstruction": recon_info[0], "tags": tags}
        # member decision at its own zero logit (prob 0.5)
        mem_pred = [(m >= 0).astype(int) for m in members]          # per seed (n, D)
        mem_correct = [(mp_ == y[:, None]) for mp_ in mem_pred]
        # majority over seeds per member
        mem_corr_maj = (np.mean([mc.astype(float) for mc in mem_correct], 0) >= 0.5)
        n_right = mem_corr_maj.sum(1)
        for nm, m in (("X_err", Xmask), ("nonX_err", Omask), ("correct", correct_mask)):
            h3[nm] = {
                "n": int(m.sum()),
                "frac_at_least_one_member_correct": float((n_right[m] >= 1).mean()),
                "frac_majority_of_members_correct": float(
                    (n_right[m] >= (len(tags) + 1) // 2 + (1 if len(tags) % 2 == 0 else 0)).mean()),
                "mean_members_correct": float(n_right[m].mean()),
                "hist_members_correct": {str(k): int((n_right[m] == k).sum())
                                         for k in range(len(tags) + 1)},
            }
        h3["per_member_accuracy_on_X"] = {
            t: float(mem_corr_maj[Xmask][:, j].mean()) for j, t in enumerate(tags)}
        h3["per_member_test_accuracy"] = {
            t: float(mem_corr_maj[:, j].mean()) for j, t in enumerate(tags)}
        # oracle: how many X items would a per-item member-selection rescue
        h3["oracle_X_rescued_by_any_member"] = int((n_right[Xmask] >= 1).sum())
        h3["oracle_X_rescued_by_member_majority"] = int(
            (n_right[Xmask] > len(tags) / 2).sum())
        # macro-F1 value of the oracle repair of the rescuable X items
        def repair(sel_idx):
            vals = []
            for q in preds:
                z = q.copy()
                z[sel_idx] = y[sel_idx]
                vals.append(f1_score(y, z, average="macro"))
            return float(np.mean(vals))
        base_f1 = float(np.mean([f1_score(y, q, average="macro") for q in preds]))
        Xidx = np.where(Xmask)[0]
        h3["base_macro_f1"] = base_f1
        h3["repair_all_X"] = repair(Xidx)
        h3["repair_X_with_a_correct_member"] = repair(
            Xidx[n_right[Xidx] >= 1]) if len(Xidx) else base_f1
        h3["repair_X_member_majority"] = repair(
            Xidx[n_right[Xidx] > len(tags) / 2]) if len(Xidx) else base_f1
        # cost control: if a per-item member selector fires everywhere, what does it
        # break?  count correct items where a member majority disagrees with the gold.
        h3["correct_items_where_member_majority_wrong"] = int(
            (n_right[correct_mask] < len(tags) / 2).sum())

        # ---------------- H5
        per_seed_wrong = np.stack([(p != y) for p in preds], 0)     # (S, n)
        nwrong = per_seed_wrong.sum(0)
        h5 = {}
        for nm, m in (("X_err", Xmask), ("nonX_err", Omask), ("correct", correct_mask)):
            h5[nm] = {"n": int(m.sum()),
                      "hist_seeds_wrong": {str(k): int((nwrong[m] == k).sum())
                                           for k in range(len(preds) + 1)},
                      "frac_all_seeds_wrong": float((nwrong[m] == len(preds)).mean())}
        res[ds] = {"n_test": n, "comparator": key, "thresholds": thrs,
                   "n_err": int(err_mask.sum()), "n_X": int(Xmask.sum()),
                   "X_ids": [ids[i] for i in np.where(Xmask)[0]],
                   "X_gold": {ids[i]: int(y[i]) for i in np.where(Xmask)[0]},
                   "H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5,
                   "per_item": {
                       ids[i]: {"gold": int(y[i]), "bucket": bucket[i],
                                "gap_sd": float(per_item_gap[i]),
                                "rank_gap": float(per_item_rankgap[i]),
                                "members_correct": int(n_right[i]),
                                "seeds_wrong": int(nwrong[i])}
                       for i in np.where(Xmask)[0]}}
        print(ds, "done", flush=True)
    return res


if __name__ == "__main__":
    OUT = run()
    with open(os.path.join(ROOT, "idea-stage", "r5_xbucket_recon.json"), "w") as f:
        json.dump(OUT, f, indent=1, ensure_ascii=False)
    print("written")
