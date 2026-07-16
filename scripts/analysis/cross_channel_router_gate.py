#!/usr/bin/env python
"""Mechanism-guided per-item cross-channel ROUTER — $0 screening gate (ZERO GPU, ZERO test-touch).

Predicts, per video, which RGCL arm to trust (frozen-CLIP encoder vs frozen-Qwen2.5-VL-7B encoder),
from decision-level meta-features on the channel-disagreement subset, and measures whether routing
beats committing to the globally-better single channel. Design/bars are pre-declared in
refine-logs/ROUTER_GATE_RECORD.md (quoted verbatim from the mandate); this script only computes.

Deployed channel = enc3s e29 head (fusion_mode='align') -> embed=mlp[:-2](x) -> top-20 rank-weighted
signed-cosine kNN vote (majority_voting='arithmetic', use_sim=True, topk=20, memory=train), decision
1{vote>=0}. Reproduces the deployed dev (Val_Retrieval) acc of all 12 arms bit-exact (asserted).

Run: conda activate HateVideo; OMP_NUM_THREADS=4 python scripts/analysis/cross_channel_router_gate.py
"""
import os, sys, json, hashlib, argparse
import numpy as np
import torch
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from model.classifier import classifier_hateClipper  # noqa: E402
from easydict import EasyDict  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

faiss.omp_set_num_threads(4)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT_DIR = os.path.join(REPO, "refine-logs", "router_ckpt_snapshot")
CACHE = os.path.join(REPO, "data", "CLIP_Embedding")
MODEL = {"CLIP": "openai_clip-vit-large-patch14-336_HF", "Qwen": "Qwen2.5-VL-7B-Instruct_HF"}
TOPK = 20
# deployed dev (Val_Retrieval) acc anchors = ckpt filename suffix = trainlog e29 val acc
ANCHOR = {("HateMM", "CLIP"): [.7944, .8131, .8224], ("HateMM", "Qwen"): [.8505, .8224, .8505],
          ("MHC", "CLIP"): [.7375, .7500, .7000], ("MHC", "Qwen"): [.7625, .7875, .7750]}
SEEDS = [0, 1, 2]
BAR = 0.020
RNG = 20260717


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cache(ds, split, enc):
    d = torch.load(os.path.join(CACHE, ds, f"{split}_{MODEL[enc]}.pt"), map_location="cpu")
    ids = d["ids"]
    ids = ids[0] if (isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list)) else ids
    return list(ids), d["img_feats"].float(), d["text_feats"].float(), d["labels"].long().numpy()


def build_head(sd):
    a = EasyDict(dataset="X")
    m = classifier_hateClipper(sd["img_proj.0.weight"].shape[1], sd["text_proj.0.weight"].shape[1],
                               num_layers=3, proj_dim=1024, map_dim=1024, fusion_mode="align",
                               dropout=[0.2, 0.4, 0.1], batch_norm=False, args=a)
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def embed(m, img, txt):
    _, e = m(img, txt, return_embed=True)
    return e.cpu().numpy().astype("float32")


def _weighted_signed_vote(nb_lab, nb_sim):
    """arithmetic rank-weighted signed-cosine vote over top-k neighbours (metrics.py use_sim path)."""
    k = len(nb_lab)
    w = np.arange(1, TOPK + 1)[::-1].astype("float64")[:k]
    lm = (nb_lab.astype("float64") * 2 - 1) * nb_sim
    return float(np.sum(lm * w) / np.sum(w))


def knn_channel(train_e, train_lab, query_e, exclude_self):
    """Return per-query dict of vote + neighbour stats. exclude_self => LOO (query is train)."""
    tr = train_e.copy()
    q = query_e.copy()
    faiss.normalize_L2(tr)
    faiss.normalize_L2(q)
    ix = faiss.IndexFlatIP(tr.shape[1])
    ix.add(tr)
    kk = TOPK + (1 if exclude_self else 0)
    D, I = ix.search(q, kk)
    out = []
    for i in range(len(q)):
        idx, sim = I[i], D[i]
        if exclude_self:
            keep = idx != i
            idx, sim = idx[keep][:TOPK], sim[keep][:TOPK]
        else:
            idx, sim = idx[:TOPK], sim[:TOPK]
        lab = train_lab[idx].astype("float64")
        vote = _weighted_signed_vote(lab, sim)
        phate = float(lab.mean())
        agree = np.sign(lab * 2 - 1) == np.sign(vote if vote != 0 else 1)
        sm_ag = float(sim[agree].mean()) if agree.any() else 0.0
        sm_dis = float(sim[~agree].mean()) if (~agree).any() else 0.0
        p = np.clip([phate, 1 - phate], 1e-9, 1)
        out.append(dict(vote=vote, phate=phate, purity=max(phate, 1 - phate),
                        entropy=float(-(p * np.log(p)).sum()), topsim=float(sim[0]),
                        meansim=float(sim.mean()), simmargin=sm_ag - sm_dis))
    return out


def raw_modality_vote(train_feat, train_lab, query_feat, exclude_self):
    """signed-cosine top-20 vote over RAW frozen single-modality features (no head)."""
    tr = train_feat.numpy().astype("float32").copy()
    q = query_feat.numpy().astype("float32").copy()
    faiss.normalize_L2(tr)
    faiss.normalize_L2(q)
    ix = faiss.IndexFlatIP(tr.shape[1])
    ix.add(tr)
    kk = TOPK + (1 if exclude_self else 0)
    D, I = ix.search(q, kk)
    votes = np.zeros(len(q))
    for i in range(len(q)):
        idx, sim = I[i], D[i]
        if exclude_self:
            keep = idx != i
            idx, sim = idx[keep][:TOPK], sim[keep][:TOPK]
        else:
            idx, sim = idx[:TOPK], sim[:TOPK]
        votes[i] = _weighted_signed_vote(train_lab[idx], sim)
    return votes


def empty_text_indicator(txt):
    """1 if the raw text feature is a near-duplicate of the most-common (degenerate/empty) text vec."""
    x = txt.numpy().astype("float64")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.clip(n, 1e-9, None)
    # most common vector = the one with the most near-duplicates
    K = min(len(xn), 400)
    G = xn @ xn[:K].T  # [N,K] cosine to first-K anchors is enough to find the degenerate cluster
    dupcount = (G > 0.9999).sum(0)
    anchor = xn[:K][int(np.argmax(dupcount))]
    cos_to_anchor = xn @ anchor
    return (cos_to_anchor > 0.9999).astype("float64"), np.linalg.norm(x, axis=1)


def channel_features(ds, seed, split):
    """Per-item meta-features for one (dataset, seed, split) for BOTH channels + fused votes+labels."""
    exclude = (split == "train")
    feats = {}
    votes = {}
    ids_ref = None
    labels_ref = None
    order = {}
    for enc in ["CLIP", "Qwen"]:
        ids, img, txt, lab = load_cache(ds, "train" if split == "train" else "dev_seen", enc)
        tr_ids, tr_img, tr_txt, tr_lab = load_cache(ds, "train", enc)
        sd = torch.load(os.path.join(CKPT_DIR, f"{ds}_{enc}_s{seed}_e29.pt"), map_location="cpu")
        m = build_head(sd)
        tr_e = embed(m, tr_img, tr_txt)
        q_e = tr_e if split == "train" else embed(m, img, txt)
        st = knn_channel(tr_e, tr_lab, q_e, exclude_self=exclude)
        vimg = raw_modality_vote(tr_img, tr_lab, tr_img if split == "train" else img, exclude)
        vtxt = raw_modality_vote(tr_txt, tr_lab, tr_txt if split == "train" else txt, exclude)
        empt, tnorm = empty_text_indicator(txt)
        F = {}
        for i, s in enumerate(st):
            F[ids[i]] = dict(**{f"{k}_{enc}": v for k, v in s.items()},
                             **{f"vimg_{enc}": float(vimg[i]), f"vtxt_{enc}": float(vtxt[i]),
                                f"empty_{enc}": float(empt[i]), f"tnorm_{enc}": float(tnorm[i])})
        feats[enc] = F
        votes[enc] = {ids[i]: st[i]["vote"] for i in range(len(ids))}
        order[enc] = ids
        if ids_ref is None:
            ids_ref, labels_ref = ids, {ids[i]: int(lab[i]) for i in range(len(ids))}
    # align by common ids (defensive; splits share the same set)
    common = [i for i in ids_ref if i in feats["Qwen"]]
    rows, meta = [], []
    for vid in common:
        fc, fq = feats["CLIP"][vid], feats["Qwen"][vid]
        vC, vQ = votes["CLIP"][vid], votes["Qwen"][vid]
        predC, predQ = int(vC >= 0), int(vQ >= 0)
        row = {**fc, **fq}
        row["vote_diff"] = vC - vQ
        row["absmargin_diff"] = abs(vC) - abs(vQ)
        row["agree"] = float(predC == predQ)
        row["fuse_img_agree_CLIP"] = float(np.sign(vC) == np.sign(fc["vimg_CLIP"] or 1))
        row["fuse_img_agree_Qwen"] = float(np.sign(vQ) == np.sign(fq["vimg_Qwen"] or 1))
        rows.append(row)
        y = labels_ref[vid]
        meta.append(dict(id=vid, label=y, predC=predC, predQ=predQ, voteC=vC, voteQ=vQ,
                         disagree=int(predC != predQ), qwen_correct=int(predQ == y),
                         clip_correct=int(predC == y)))
    keys = sorted(rows[0].keys())
    X = np.array([[r[k] for k in keys] for r in rows], dtype="float64")
    return X, keys, meta


def acc_of(meta, chan):
    key = "predQ" if chan == "Qwen" else "predC"
    return float(np.mean([m[key] == m["label"] for m in meta]))


def routed_acc(dev_meta, dev_route):
    """dev_route: dict id-> chosen channel ('CLIP'/'Qwen') for disagreement items."""
    corr = 0
    for m in dev_meta:
        if m["disagree"] and m["id"] in dev_route:
            ch = dev_route[m["id"]]
            pred = m["predQ"] if ch == "Qwen" else m["predC"]
        else:
            pred = m["predQ"]  # == predC on agreement
        corr += int(pred == m["label"])
    return corr / len(dev_meta)


def fit_router(Xtr, ytr, model="gbm"):
    if model == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                              l2_regularization=1.0, min_samples_leaf=5,
                                              random_state=RNG)
        clf.fit(Xtr, ytr)
        return clf, None
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RNG)
    clf.fit(sc.transform(Xtr), ytr)
    return clf, sc


def predict_router(clf, sc, X):
    return clf.predict(sc.transform(X)) if sc is not None else clf.predict(X)


def run_dataset(ds, model="gbm", verbose=True):
    res = {"seeds": []}
    per_seed_gain = []
    boot_diffs_per_seed = []  # for bootstrap: list over seeds of per-item (routed_correct - best_correct)
    for seed in SEEDS:
        Xtr, keys, tr_meta = channel_features(ds, seed, "train")
        Xdv, _, dv_meta = channel_features(ds, seed, "dev")
        ktr = keys
        # disagreement masks
        tr_dis = np.array([m["disagree"] for m in tr_meta], dtype=bool)
        dv_dis = np.array([m["disagree"] for m in dv_meta], dtype=bool)
        ytr = np.array([m["qwen_correct"] for m in tr_meta])  # on disagreement: 1 iff Qwen correct
        # machinery: dev acc anchors
        accC, accQ = acc_of(dv_meta, "CLIP"), acc_of(dv_meta, "Qwen")
        best_chan = "Qwen" if accQ >= accC else "CLIP"
        best_acc = max(accC, accQ)
        # fit on train-disagreement, predict dev-disagreement
        Xtr_d, ytr_d = Xtr[tr_dis], ytr[tr_dis]
        route = {}
        if dv_dis.sum() > 0 and len(np.unique(ytr_d)) == 2:
            clf, sc = fit_router(Xtr_d, ytr_d, model)
            pred = predict_router(clf, sc, Xdv[dv_dis])
            dv_ids_dis = [m["id"] for m in dv_meta if m["disagree"]]
            for vid, p in zip(dv_ids_dis, pred):
                route[vid] = "Qwen" if p == 1 else "CLIP"
        r_acc = routed_acc(dv_meta, route) if route else best_acc
        gain = r_acc - best_acc
        per_seed_gain.append(gain)
        # per-item routed-correct vs best-single-correct (for bootstrap)
        bkey = "predQ" if best_chan == "Qwen" else "predC"
        diffs = []
        for m in dv_meta:
            if m["disagree"] and m["id"] in route:
                ch = route[m["id"]]
                rp = m["predQ"] if ch == "Qwen" else m["predC"]
            else:
                rp = m["predQ"]
            rc = int(rp == m["label"])
            bc = int(m[bkey] == m["label"])
            diffs.append(rc - bc)
        boot_diffs_per_seed.append(np.array(diffs))
        # disagreement base rate
        dis_ids = [m for m in dv_meta if m["disagree"]]
        base_q = float(np.mean([m["qwen_correct"] for m in dis_ids])) if dis_ids else float("nan")
        res["seeds"].append(dict(seed=seed, dev_acc_CLIP=round(accC, 4), dev_acc_Qwen=round(accQ, 4),
                                 best_chan=best_chan, best_acc=round(best_acc, 4),
                                 routed_acc=round(r_acc, 4), gain=round(gain, 4),
                                 n_dev=len(dv_meta), n_dev_disagree=int(dv_dis.sum()),
                                 n_train_disagree=int(tr_dis.sum()),
                                 dev_disagree_qwen_correct_rate=round(base_q, 4)))
        if verbose:
            print(f"[{ds} s{seed}] CLIP {accC:.4f} Qwen {accQ:.4f} best={best_chan} "
                  f"routed {r_acc:.4f} gain {gain:+.4f} | dev_dis={int(dv_dis.sum())} "
                  f"tr_dis={int(tr_dis.sum())} base_q={base_q:.3f}")
    mean_gain = float(np.mean(per_seed_gain))
    res["mean_gain"] = round(mean_gain, 4)
    res["per_seed_gain"] = [round(g, 4) for g in per_seed_gain]
    # bootstrap 1000: resample dev items within seed, mean over seeds of mean(diff)
    rng = np.random.default_rng(RNG)
    boot = np.zeros(1000)
    for b in range(1000):
        sm = []
        for d in boot_diffs_per_seed:
            idx = rng.integers(0, len(d), len(d))
            sm.append(d[idx].mean())
        boot[b] = np.mean(sm)
    res["boot_ci95"] = [round(float(np.percentile(boot, 2.5)), 4),
                        round(float(np.percentile(boot, 97.5)), 4)]
    res["boot_ci_low"] = round(float(np.percentile(boot, 2.5)), 4)
    res["keys_n"] = len(ktr)
    return res, boot_diffs_per_seed


def permutation_null_devcv(ds, model="gbm", nperm=100):
    """K-R3 on the FAVORABLE dev-CV router: shuffle dev-disagreement y, redo OOF CV routing, recompute
    3-seed-mean dev gain. Tests whether the dev-CV router's signal is real vs overfitting noise."""
    cache = []
    for seed in SEEDS:
        Xdv, _, dv_meta = channel_features(ds, seed, "dev")
        dv_dis = np.array([m["disagree"] for m in dv_meta], dtype=bool)
        y = np.array([m["qwen_correct"] for m in dv_meta])[dv_dis]
        accC, accQ = acc_of(dv_meta, "CLIP"), acc_of(dv_meta, "Qwen")
        cache.append((Xdv[dv_dis], y, [m["id"] for m in dv_meta if m["disagree"]], dv_meta,
                      max(accC, accQ)))
    rng = np.random.default_rng(RNG + 7)
    null = np.zeros(nperm)
    for p in range(nperm):
        gains = []
        for (Xd, y, vids, dv_meta, best_acc) in cache:
            yp = y.copy()
            rng.shuffle(yp)
            route = {}
            if len(np.unique(yp)) == 2 and len(yp) >= 6:
                k = min(5, int(np.min(np.bincount(yp))))
                if k >= 2:
                    skf = StratifiedKFold(k, shuffle=True, random_state=RNG + p)
                    oof = np.zeros(len(yp))
                    for tri, tei in skf.split(Xd, yp):
                        clf, sc = fit_router(Xd[tri], yp[tri], model)
                        oof[tei] = predict_router(clf, sc, Xd[tei])
                    for vid, q in zip(vids, oof):
                        route[vid] = "Qwen" if q == 1 else "CLIP"
            gains.append(routed_acc(dv_meta, route) - best_acc)
        null[p] = np.mean(gains)
    return null


def oracle_calibration(ds, model="gbm"):
    """K-R2: plant true y as a feature; router must recover accZA>=0.99. Uses the DEV-disagreement
    subset (where y=1{Qwen correct} is non-degenerate; the TRAIN-disagreement target is degenerate
    because the CLIP head memorises train, LOO acc ~0.998 -> Qwen wrong on ~all train disagreements)."""
    accs = []
    for seed in SEEDS:
        Xdv, keys, dv_meta = channel_features(ds, seed, "dev")
        dv_dis = np.array([m["disagree"] for m in dv_meta], dtype=bool)
        y = np.array([m["qwen_correct"] for m in dv_meta])[dv_dis]
        X = Xdv[dv_dis]
        Xp = np.column_stack([X, y.astype("float64")])  # planted oracle feature
        if len(np.unique(y)) < 2 or len(y) < 8:
            continue
        k = min(5, int(np.min(np.bincount(y))))
        if k < 2:
            continue
        skf = StratifiedKFold(k, shuffle=True, random_state=RNG)
        preds = np.zeros(len(y))
        for tri, tei in skf.split(Xp, y):
            clf, sc = fit_router(Xp[tri], y[tri], model)
            preds[tei] = predict_router(clf, sc, Xp[tei])
        accs.append(float(np.mean(preds == y)))
    return float(np.mean(accs)) if accs else float("nan")


def oracle_headroom(ds):
    """Label-oracle routing CEILING (ruled first, B5-style): route every dev-disagreement item to the
    channel that IS correct. Returns 3-seed mean gain of the PERFECT router over best-single-channel —
    the maximum accuracy a per-item router could ever add. If this ceiling < BAR, no router can pass."""
    gains, per = [], []
    for seed in SEEDS:
        Xdv, _, dv_meta = channel_features(ds, seed, "dev")
        accC, accQ = acc_of(dv_meta, "CLIP"), acc_of(dv_meta, "Qwen")
        best = max(accC, accQ)
        corr = 0
        for m in dv_meta:
            if m["disagree"]:
                corr += 1 if (m["qwen_correct"] or m["clip_correct"]) else 0  # perfect pick
            else:
                corr += int(m["predQ"] == m["label"])
        oacc = corr / len(dv_meta)
        gains.append(oacc - best)
        per.append(dict(seed=seed, oracle_acc=round(oacc, 4), best=round(best, 4),
                        gain=round(oacc - best, 4), n_dev_disagree=int(sum(m["disagree"] for m in dv_meta))))
    return dict(mean_gain=round(float(np.mean(gains)), 4),
                per_seed=[round(g, 4) for g in gains], detail=per)


def dev_cv_router(ds, model="gbm"):
    """Maximally-FAVORABLE realizable router (supplementary, post-hoc): fit WITHIN dev via stratified
    k-fold OOF on the dev-disagreement subset (in-distribution supervision, uses dev labels via CV),
    then route. Upper-bounds what any train-fit router could realise; if this fails the bar too, the
    kill is at the realizable ceiling, not merely a train->dev transfer failure."""
    per, boot_diffs = [], []
    for seed in SEEDS:
        Xdv, _, dv_meta = channel_features(ds, seed, "dev")
        dv_dis = np.array([m["disagree"] for m in dv_meta], dtype=bool)
        accC, accQ = acc_of(dv_meta, "CLIP"), acc_of(dv_meta, "Qwen")
        best_chan = "Qwen" if accQ >= accC else "CLIP"
        best = max(accC, accQ)
        y = np.array([m["qwen_correct"] for m in dv_meta])[dv_dis]
        Xd = Xdv[dv_dis]
        route = {}
        dis_ids = [m["id"] for m in dv_meta if m["disagree"]]
        if len(np.unique(y)) == 2 and len(y) >= 6:
            k = min(5, int(np.min(np.bincount(y))))
            if k >= 2:
                skf = StratifiedKFold(k, shuffle=True, random_state=RNG)
                oof = np.zeros(len(y))
                for tri, tei in skf.split(Xd, y):
                    clf, sc = fit_router(Xd[tri], y[tri], model)
                    oof[tei] = predict_router(clf, sc, Xd[tei])
                for vid, p in zip(dis_ids, oof):
                    route[vid] = "Qwen" if p == 1 else "CLIP"
        r_acc = routed_acc(dv_meta, route)
        per.append(round(r_acc - best, 4))
        bkey = "predQ" if best_chan == "Qwen" else "predC"
        diffs = []
        for m in dv_meta:
            if m["disagree"] and m["id"] in route:
                rp = m["predQ"] if route[m["id"]] == "Qwen" else m["predC"]
            else:
                rp = m["predQ"]
            diffs.append(int(rp == m["label"]) - int(m[bkey] == m["label"]))
        boot_diffs.append(np.array(diffs))
    rng = np.random.default_rng(RNG + 3)
    boot = np.array([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in boot_diffs])
                     for _ in range(1000)])
    return dict(mean_gain=round(float(np.mean(per)), 4), per_seed=per,
                boot_ci95=[round(float(np.percentile(boot, 2.5)), 4),
                           round(float(np.percentile(boot, 97.5)), 4)],
                boot_ci_low=round(float(np.percentile(boot, 2.5)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "refine-logs", "ROUTER_GATE_OUT.json"))
    args = ap.parse_args()
    OUT = {"config": dict(topk=TOPK, bar=BAR, rng=RNG, fusion="align", vote="arithmetic_signed_cosine",
                          protocol="final-epoch e29"), "script_sha256": sha(os.path.abspath(__file__))}
    # ckpt/cache provenance shas
    OUT["ckpt_sha"] = {f: sha(os.path.join(CKPT_DIR, f)) for f in sorted(os.listdir(CKPT_DIR))
                       if f.endswith(".pt")}
    # machinery validation (assert bit-exact anchors)
    val = {}
    for ds in ["HateMM", "MHC"]:
        for enc in ["CLIP", "Qwen"]:
            for seed in SEEDS:
                ids, img, txt, lab = load_cache(ds, "dev_seen", enc)
                tids, timg, ttxt, tlab = load_cache(ds, "train", enc)
                sd = torch.load(os.path.join(CKPT_DIR, f"{ds}_{enc}_s{seed}_e29.pt"), map_location="cpu")
                m = build_head(sd)
                st = knn_channel(embed(m, timg, ttxt), tlab, embed(m, img, txt), False)
                acc = float(np.mean([(s["vote"] >= 0) == lab[i] for i, s in enumerate(st)]))
                val[f"{ds}_{enc}_s{seed}"] = dict(acc=round(acc, 4), anchor=ANCHOR[(ds, enc)][seed],
                                                  match=abs(acc - ANCHOR[(ds, enc)][seed]) < 1.1e-4)
    OUT["machinery_validation"] = val
    OUT["machinery_all_match"] = all(v["match"] for v in val.values())

    # (i) label-oracle routing headroom = the CEILING, ruled first (B5-style)
    OUT["oracle_headroom"] = {ds: oracle_headroom(ds) for ds in ["MHC", "HateMM"]}
    for ds in ["MHC", "HateMM"]:
        h = OUT["oracle_headroom"][ds]
        print(f"[oracle-headroom {ds}] perfect-router 3-seed mean gain {h['mean_gain']:+.4f} "
              f"per-seed {h['per_seed']}")

    for model in ["gbm", "linear"]:
        print(f"\n===== ROUTER MODEL = {model} =====")
        OUT.setdefault("router", {})[model] = {}
        # (ii) pre-registered PRIMARY read: train-fit -> dev judged read
        for ds in ["MHC", "HateMM"]:
            r, _ = run_dataset(ds, model)
            OUT["router"][model][ds] = r
            print(f"  [primary train->dev] {ds}: 3-seed mean gain {r['mean_gain']:+.4f} "
                  f"CI95 {r['boot_ci95']} per-seed {r['per_seed_gain']}")
        # (iii) FAVORABLE realizable ceiling: dev-CV router
        OUT["router"][model]["devcv"] = {ds: dev_cv_router(ds, model) for ds in ["MHC", "HateMM"]}
        for ds in ["MHC", "HateMM"]:
            d = OUT["router"][model]["devcv"][ds]
            print(f"  [dev-CV favorable] {ds}: 3-seed mean gain {d['mean_gain']:+.4f} "
                  f"CI95 {d['boot_ci95']} per-seed {d['per_seed']}")
        # K-R2 machinery calibration (dev-disagreement, planted y)
        OUT["router"][model]["oracle_calib_accZA"] = {
            ds: round(oracle_calibration(ds, model), 4) for ds in ["MHC", "HateMM"]}
        # K-R3 permutation null on the FAVORABLE dev-CV router (primary dataset = MHC)
        null = permutation_null_devcv("MHC", model, 100)
        obs = OUT["router"][model]["devcv"]["MHC"]["mean_gain"]
        OUT["router"][model]["perm_null_MHC_devcv"] = dict(
            null_mean=round(float(null.mean()), 4), null_p95=round(float(np.percentile(null, 95)), 4),
            observed=obs, exceeds_p95=bool(obs > np.percentile(null, 95)),
            p_value=round(float((np.sum(null >= obs) + 1) / (len(null) + 1)), 4))
        print(f"  [{model}] MHC oracle-calib accZA={OUT['router'][model]['oracle_calib_accZA']['MHC']} "
              f"| dev-CV perm-null p95={OUT['router'][model]['perm_null_MHC_devcv']['null_p95']} "
              f"obs={obs} p={OUT['router'][model]['perm_null_MHC_devcv']['p_value']}")

    # ---- mechanical kill/pass (pre-declared; PRIMARY router = nonlinear gbm, train->dev) ----
    prim = OUT["router"]["gbm"]
    mhc_gain = prim["MHC"]["mean_gain"]                 # pre-registered primary read
    mhc_cilow = prim["MHC"]["boot_ci_low"]
    hatemm_gain = prim["HateMM"]["mean_gain"]
    headroom = OUT["oracle_headroom"]["MHC"]["mean_gain"]
    devcv_gain = prim["devcv"]["MHC"]["mean_gain"]
    devcv_cilow = prim["devcv"]["MHC"]["boot_ci_low"]
    accZA = prim["oracle_calib_accZA"]["MHC"]
    perm = prim["perm_null_MHC_devcv"]
    KR2_valid = accZA >= 0.99
    KR1_kill = (mhc_gain < BAR) or (mhc_cilow <= 0)
    KR3_survive = perm["exceeds_p95"]
    devcv_kill = (devcv_gain < BAR) or (devcv_cilow <= 0)   # realizable-ceiling confirmation
    hatemm_ok = hatemm_gain >= -1e-9
    OUT["mechanical"] = dict(
        BAR=BAR, mhc_primary_gain=mhc_gain, mhc_primary_ci_low=mhc_cilow,
        oracle_headroom_MHC=headroom, devcv_gain_MHC=devcv_gain, devcv_ci_low_MHC=devcv_cilow,
        hatemm_primary_gain=hatemm_gain, K_R2_oracle_accZA=accZA, K_R2_valid=KR2_valid,
        K_R1_kill=bool(KR1_kill), devcv_ceiling_kill=bool(devcv_kill),
        K_R3_devcv_survives_null=bool(KR3_survive), hatemm_sanity_ok=bool(hatemm_ok))
    if not KR2_valid:
        OUT["label"] = "MACHINERY_INVALID"
    elif KR1_kill and devcv_kill:
        OUT["label"] = "KILL"                              # dead at both the read AND the realizable ceiling
    elif KR1_kill and not devcv_kill:
        OUT["label"] = "KILL_TRANSFER"                     # dev-CV shows signal but train->dev fails
    elif (mhc_gain >= BAR) and (mhc_cilow > 0) and KR3_survive and hatemm_ok:
        OUT["label"] = "PASS"
    else:
        OUT["label"] = "AMBIGUOUS"
    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=2)
    print(f"\nMACHINERY_ALL_MATCH={OUT['machinery_all_match']}  NON-BINDING LABEL={OUT['label']}")
    print(f"MHC primary gain={mhc_gain:+.4f} (bar +{BAR}) CI-low={mhc_cilow} | oracle-headroom "
          f"{headroom:+.4f} | dev-CV {devcv_gain:+.4f} CI-low {devcv_cilow} | HateMM {hatemm_gain:+.4f}")
    print(f"K-R1_kill={KR1_kill} K-R2_valid={KR2_valid}(accZA={accZA}) devcv_ceiling_kill={devcv_kill} "
          f"K-R3_devcv_survive={KR3_survive}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
