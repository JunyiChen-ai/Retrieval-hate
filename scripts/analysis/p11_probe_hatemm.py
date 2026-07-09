#!/usr/bin/env python
"""P11 probe gate (CPU) — HateMM gold-span localization: MLLM vs MIL-proxy.

Pre-registration: research-wiki/EXP_p11_weaksup_localization.md §4.
Compares, on the HateMM calibration set (train hateful, both-class videos), the
per-video within-video AUC of three gold-span predictors:
  1. MLLM arm-B teacher : 72B A-fuse per-segment density (existing asset)
  2. MIL-proxy          : 5-fold CV top-k MIL LINEAR head on HateMM K=4 subclip
                          CLIP features, trained on video labels (arm-A analogue)
  3. memory (context)   : cross-dataset (MHC-video) consensus-kNN vote

GATE (binding): wv-AUC(MLLM A-fuse) - wv-AUC(MIL-proxy) >= +0.03 AND the paired
bootstrap 95% CI on the per-video delta excludes 0. Estimators mirror
p10_eval_hatemm.py / p6 bit-for-bit (1-fps second-midpoint, both-class videos).
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

ROOT = "/data/jehc223/RGCL"
sys.path.insert(0, os.path.join(ROOT, "src"))
from utils.consensus import _knn_vote, _l2n  # noqa: E402

SPANS = os.path.join(ROOT, "data/gt/HateMM/hate_spans.json")
MDIR = os.path.join(ROOT, "data/MLLM_scores/HateMM")
EMB = os.path.join(ROOT, "data/CLIP_Embedding")
CE = "openai_clip-vit-large-patch14-336_HF"
TOPK_MEM = 10
SEED = 0


# ----------------------------- shared estimators --------------------------- #
def load_spans():
    d = json.load(open(SPANS))
    return {k: v for k, v in d.items() if v.get("spans")}


def sec_labels(v, K):
    D = float(v["duration"])
    spans = v["spans"]
    labs, qs = [], []
    for t in range(int(np.floor(D))):
        mid = t + 0.5
        labs.append(int(any(s <= mid < e for s, e in spans)))
        qs.append(min(K - 1, int(mid * K / D)))
    return np.array(labs), np.array(qs)


def wv_auc_by_vid(spans, scores, K):
    """scores: {vid: [K] window scores}. -> {vid: within-video AUC} over
    both-class videos present in scores (const scores -> 0.5)."""
    out = {}
    for vid, v in spans.items():
        if vid not in scores:
            continue
        row = np.asarray(scores[vid], dtype=np.float64)
        if len(row) < K:
            continue
        lab, qs = sec_labels(v, K)
        if len(lab) == 0 or lab.sum() == 0 or lab.sum() == len(lab):
            continue
        sc = row[qs]
        out[vid] = 0.5 if np.allclose(sc, sc[0]) else float(roc_auc_score(lab, sc))
    return out


def _rank01(a):
    from scipy.stats import rankdata
    if len(a) <= 1:
        return np.zeros_like(a, dtype=np.float64)
    return (rankdata(a, method="average") - 1.0) / (len(a) - 1)


def wv_auc_fuse_by_vid(spans, scA, KA, scB, KB):
    """Per-video rank-average fusion of two window-score sources (identical
    operator for MLLM and MIL) evaluated at the 1-fps second level."""
    out = {}
    for vid, v in spans.items():
        if vid not in scA or vid not in scB:
            continue
        rowA = np.asarray(scA[vid], dtype=np.float64)
        rowB = np.asarray(scB[vid], dtype=np.float64)
        if len(rowA) < KA or len(rowB) < KB:
            continue
        lab, qsA = sec_labels(v, KA)
        _, qsB = sec_labels(v, KB)
        if len(lab) == 0 or lab.sum() == 0 or lab.sum() == len(lab):
            continue
        fused = 0.5 * _rank01(rowA[qsA]) + 0.5 * _rank01(rowB[qsB])
        out[vid] = 0.5 if np.allclose(fused, fused[0]) else float(
            roc_auc_score(lab, fused))
    return out


def boot_ci(a, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    a = np.asarray(a)
    b = np.array([a[rng.randint(0, len(a), len(a))].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarize(name, d):
    a = np.array(list(d.values()))
    lo, hi = boot_ci(a, seed=SEED)
    gt = int((a > 0.5).sum()); lt = int((a < 0.5).sum())
    p = binomtest(gt, gt + lt, 0.5, alternative="greater").pvalue if gt + lt else 1.0
    print("  {:22s} n={:3d}  wv-AUC {:.4f}  CI[{:.4f},{:.4f}]  (>.5 {}/{}, p={:.2g})"
          .format(name, len(a), a.mean(), lo, hi, gt, gt + lt, p))
    return dict(n=len(a), wv=float(a.mean()), ci=[lo, hi], sign_p=float(p))


def paired(name, dcfg, dbase):
    common = sorted(set(dcfg) & set(dbase))
    delta = np.array([dcfg[v] - dbase[v] for v in common])
    lo, hi = boot_ci(delta, seed=SEED)
    gt = int((delta > 1e-9).sum()); lt = int((delta < -1e-9).sum())
    p = binomtest(gt, gt + lt, 0.5, alternative="greater").pvalue if gt + lt else 1.0
    print("  PAIRED {:26s} n={:3d}  Delta {:+.4f}  CI[{:+.4f},{:+.4f}]  "
          "excl0={}  sign-p {:.2g}  [{:.4f} vs {:.4f}]".format(
              name, len(common), delta.mean(), lo, hi, lo > 0, p,
              np.mean([dcfg[v] for v in common]), np.mean([dbase[v] for v in common])))
    return dict(n=len(common), delta=float(delta.mean()), ci=[lo, hi],
                excl0=bool(lo > 0), sign_p=float(p))


# ------------------------------- signal 1: MLLM ---------------------------- #
def load_mllm(tag):
    """train_<tag>.jsonl -> {vid: [K] scores}."""
    scores = {}
    p = os.path.join(MDIR, "train_{}.jsonl".format(tag))
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            scores[str(r["id"])] = np.asarray(r.get("scores") or [], dtype=np.float64)
    return scores


# ------------------------- signal 2: MIL-proxy (CV) ------------------------ #
def mil_proxy_cv(kpool, K=4, n_folds=5, epochs=400, lr=1e-2, seed=SEED):
    """5-fold-by-video top-k MIL linear head on HateMM K-window subclip CLIP
    feats. Returns {vid: [K] per-window logit} from held-out folds only (no
    leakage). kpool: MIL top-k (video logit = mean of top-k segment logits).
    kpool is clipped to K."""
    kpool = min(kpool, K)
    fn = "train_subclipK{}_{}.pt".format(K, CE)
    d = torch.load(os.path.join(EMB, "HateMM", fn), map_location="cpu")
    vids = list(d["video_ids"])
    V = len(vids)
    X = _l2n(d["subclip_img_feats"].float()).reshape(V, K, -1)  # [V,K,1024]
    par = d["subclip_parent"].numpy()
    # per-video label = the video-level label (subclip labels inherit it)
    ylab = np.zeros(V, dtype=np.float32)
    sl = d["labels"].numpy()
    for i in range(len(par)):
        ylab[par[i]] = sl[i]
    ylab_t = torch.tensor(ylab)

    rng = np.random.RandomState(seed)
    # stratified fold ids by label
    fold = np.zeros(V, dtype=np.int64)
    for c in (0, 1):
        idx = np.where(ylab == c)[0]
        idx = idx[rng.permutation(len(idx))]
        for j, vi in enumerate(idx):
            fold[vi] = j % n_folds

    Dfeat = X.shape[-1]
    out_logits = np.zeros((V, K), dtype=np.float64)
    torch.manual_seed(seed)
    for f in range(n_folds):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        Xtr = X[tr]                       # [ntr,4,D]
        ytr = ylab_t[tr]
        head = torch.nn.Linear(Dfeat, 1)
        torch.nn.init.zeros_(head.bias)
        torch.nn.init.normal_(head.weight, std=0.01)
        opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            s = head(Xtr).squeeze(-1)                    # [ntr,4]
            topk = torch.topk(s, k=kpool, dim=1).values  # [ntr,k]
            vlogit = topk.mean(1)                        # [ntr]
            loss = bce(vlogit, ytr)
            loss.backward()
            opt.step()
        with torch.no_grad():
            out_logits[te] = head(X[te]).squeeze(-1).numpy()
    return {vids[i]: out_logits[i] for i in range(V)}


# ------------------------- signal 3: memory (context) ---------------------- #
def memory_scores(mem_kind="mhc_video"):
    """Cross-dataset consensus-kNN vote on HateMM K=4 windows. {vid: [4]}."""
    d = torch.load(os.path.join(EMB, "HateMM", "train_subclipK4_{}.pt".format(CE)),
                   map_location="cpu")
    vids = list(d["video_ids"])
    V = len(vids)
    query = _l2n(d["subclip_img_feats"].float())         # [V*4, D]
    if mem_kind == "mhc_video":
        w = torch.load(os.path.join(EMB, "MHC", "train_{}.pt".format(CE)),
                       map_location="cpu")
        feats, labs = w["img_feats"].float(), w["labels"].numpy().astype(np.int64)
    else:
        raise ValueError(mem_kind)
    keep = (feats.abs().sum(1) != 0).numpy()
    memory = _l2n(feats[torch.as_tensor(keep)])
    labs = labs[keep]
    own = np.full(query.shape[0], -1, dtype=np.int64)
    vote = _knn_vote(query, memory, labs, own, topk=TOPK_MEM).reshape(V, 4)
    return {vids[i]: vote[i] for i in range(V)}


# --------------------------------- main ------------------------------------ #
def main():
    spans = load_spans()
    print("HateMM hateful spans:", len(spans))
    res = {}

    print("\n[1] MLLM signals (per-video within-video AUC on gold spans):")
    sc_afuse = load_mllm("segscoreK30_p10-p6-72b-bnb4-fuse")   # raw window dicts
    sc_raw30 = load_mllm("segscoreK30_p10-p6-72b-bnb4")
    sc_72k4 = load_mllm("segscoreK4_p10-p6-72b-bnb4")
    d_afuse = wv_auc_by_vid(spans, sc_afuse, K=30)
    res["mllm_72b_afuse_K30"] = summarize("MLLM 72B A-fuse K30", d_afuse)
    res["mllm_72b_rawK30"] = summarize(
        "MLLM 72B raw-K30", wv_auc_by_vid(spans, sc_raw30, K=30))
    res["mllm_7b_K30"] = summarize(
        "MLLM 7B K30 (anchor)", wv_auc_by_vid(spans, load_mllm("segscoreK30_qwen"), K=30))
    d_72k4 = wv_auc_by_vid(spans, sc_72k4, K=4)
    res["mllm_72b_K4"] = summarize("MLLM 72B K4 (matched)", d_72k4)
    res["mllm_7b_K4"] = summarize(
        "MLLM 7B K4 (matched)", wv_auc_by_vid(spans, load_mllm("segscoreK4_qwen"), K=4))
    d_raw30 = wv_auc_by_vid(spans, sc_raw30, K=30)

    print("\n[2a] MIL-proxy K=4 (5-fold CV top-k MIL linear head, HateMM K4 feats):")
    milwin4, mil4 = {}, {}
    for k in (1, 2, 4):
        w = mil_proxy_cv(kpool=k, K=4)
        milwin4[k] = w
        mil4[k] = wv_auc_by_vid(spans, w, K=4)
        res["mil_proxy_K4_k{}".format(k)] = summarize("MIL-proxy K4 k={}".format(k), mil4[k])
    best_k4 = max(mil4, key=lambda k: np.mean(list(mil4[k].values())))
    d_mil4 = mil4[best_k4]
    print("  -> strongest K4 MIL-proxy = k={} (wv {:.4f})".format(
        best_k4, np.mean(list(d_mil4.values()))))

    k30_cache = os.path.join(EMB, "HateMM", "train_subclipK30_{}.pt".format(CE))
    have_k30 = os.path.exists(k30_cache)
    milwin30, mil30, best_k30 = {}, {}, None
    if have_k30:
        print("\n[2b] MIL-proxy K=30 (matched to MLLM K30; cache present):")
        for k in (2, 3, 5):
            w = mil_proxy_cv(kpool=k, K=30)
            milwin30[k] = w
            mil30[k] = wv_auc_by_vid(spans, w, K=30)
            res["mil_proxy_K30_k{}".format(k)] = summarize(
                "MIL-proxy K30 k={}".format(k), mil30[k])
        best_k30 = max(mil30, key=lambda k: np.mean(list(mil30[k].values())))
        print("  -> strongest K30 MIL-proxy = k={} (wv {:.4f})".format(
            best_k30, np.mean(list(mil30[best_k30].values()))))
    else:
        print("\n[2b] MIL-proxy K=30 SKIPPED (cache absent; run "
              "p11_hatemm_subclipK30.sbatch)")

    print("\n[3] memory (context, NOT gating): cross-dataset MHC-video kNN vote")
    res["memory_mhc_video_K4"] = summarize(
        "memory MHC-video K4", wv_auc_by_vid(spans, memory_scores("mhc_video"), K=4))

    print("\n[GATE] pre-registered binding gate (letter): MLLM A-fuse(K30) - MIL(K4):")
    res["gate_afuse_vs_milK4"] = paired("MLLM A-fuse - MIL K4", d_afuse, d_mil4)
    res["robust_72bK4_vs_milK4"] = paired("MLLM 72B K4 - MIL K4", d_72k4, d_mil4)

    passed = None
    if have_k30:
        print("\n[MATCHED-GRANULARITY resolution (pre-reg §4/§5 ambiguity remedy)]")
        # (i) raw K30 vs raw K30 -- crispest apples-to-apples (BINDING)
        res["matched_raw_K30"] = paired("MLLM raw-K30 - MIL K30", d_raw30, mil30[best_k30])
        # (ii) A-fuse vs A-fuse -- identical rank-fusion operator for both labellers
        d_mllm_af = wv_auc_fuse_by_vid(spans, sc_raw30, 30, sc_72k4, 4)
        d_mil_af = wv_auc_fuse_by_vid(spans, milwin30[best_k30], 30, milwin4[best_k4], 4)
        res["mllm_afuse_rankfuse"] = summarize("MLLM A-fuse (rankfuse)", d_mllm_af)
        res["mil_afuse_rankfuse"] = summarize("MIL  A-fuse (rankfuse)", d_mil_af)
        res["matched_afuse"] = paired("MLLM Afuse - MIL Afuse", d_mllm_af, d_mil_af)
        # BINDING = best-vs-best at matched granularity + identical fusion operator
        # (gives the MLLM its strongest usable config; the fair "is the MLLM a
        #  better weak-labeller than MIL" test). raw-K30 = supporting confirmation.
        gA = res["matched_afuse"]
        gR = res["matched_raw_K30"]
        passed = (gA["delta"] >= 0.03) and gA["excl0"]

    gl = res["gate_afuse_vs_milK4"]
    print("\n==================== P11 PROBE VERDICT ====================")
    print("  [letter] MLLM A-fuse(K30) - MIL(K4)  = {:+.4f}  CI excl0 {}  "
          "(GRANULARITY-CONFOUNDED)".format(gl["delta"], gl["excl0"]))
    print("  [matched-K4] MLLM 72B K4 - MIL K4    = {:+.4f}  CI excl0 {}".format(
        res["robust_72bK4_vs_milK4"]["delta"], res["robust_72bK4_vs_milK4"]["excl0"]))
    if have_k30:
        print("  [matched-K30 raw]  MLLM raw-K30 - MIL K30  = {:+.4f}  CI excl0 {}".format(
            res["matched_raw_K30"]["delta"], res["matched_raw_K30"]["excl0"]))
        print("  [matched A-fuse]   MLLM Afuse - MIL Afuse   = {:+.4f}  CI excl0 {}  "
              "<== BINDING".format(res["matched_afuse"]["delta"],
                                   res["matched_afuse"]["excl0"]))
        print("  ==> PROBE {}".format(
            "PASS -> proceed to training" if passed
            else "FAIL -> KILL P11 (no training)"))
        res["verdict"] = {"passed": bool(passed),
                          "binding": "matched_afuse", "bar": 0.03,
                          "matched_afuse_delta": res["matched_afuse"]["delta"],
                          "matched_afuse_excl0": res["matched_afuse"]["excl0"],
                          "matched_raw_K30_delta": res["matched_raw_K30"]["delta"],
                          "matched_raw_K30_excl0": res["matched_raw_K30"]["excl0"]}
    else:
        print("  ==> matched-K30 cache missing; run the sbatch then re-run this probe.")
        res["verdict"] = {"passed": None, "note": "K30 cache missing"}
    outp = os.path.join(ROOT, "scripts/analysis/loc_out/p11_probe_hatemm.json")
    json.dump(res, open(outp, "w"), indent=1)
    print("wrote", outp)


if __name__ == "__main__":
    main()
