#!/usr/bin/env python3
"""W2-B Stage-P probe — sub-clip SET-matching on banked CLIP caches. CPU, ZERO training, ZERO test touch.

Design spec : refine-logs/W2B_PROBE_DESIGN.md   (r1, B1-B3 + N1-N5 folded)
Recon       : refine-logs/W2B_FORENSIC_RECON.md
Review      : refine-logs/W2B_PREREG_REVIEW.md  (APPROVED-WITH-AMENDMENTS; §8 conditional authorization)
Machinery   : adapted verbatim from scripts/analysis/s2s_probe.py (r3): vote reuse, rank-only credit
              rule (B3), NEG_INF filter (NB-a), same-perm null 0..99, bootstrap, near-dup, ASYM.

WHAT THIS IS (B1). The memory is V VIDEO-LEVEL sets: g = subclip_img_feats.view(V, K, D), one K-element
set per video. Retrieval + vote are VIDEO-TO-VIDEO (query video's K-set vs each memory video's K-set),
LOO holds out the WHOLE query set (diagonal exclusion at the VIDEO index). There is NO flat [V*K, D]
sub-clip retrieval bank anywhere -- that would leak sibling sub-clips. The len(memory)==851/629 (or
744/549 train-only) VIDEO-COUNT assert would itself trip on a flat bank. Parent-exclusion is subsumed by
video-level LOO by construction.

ARMS (per dataset, memory = train u dev_seen for the K4 PRIMARY, LOO):
  POOLED           cos(mean_k g_k^Q, mean_k g_k^M)                visual-isolated null (primary pair)
  SET (primary)    MeanMaxSim(Q,M) = mean_q max_m cos(ghat^Q_q, ghat^M_m)   late interaction
  SET-Chamfer      0.5[MMS(Q->M)+MMS(M->Q)]                       single sensitivity
  ASYM             max_m cos(ghat^Q_pooled, ghat^M_m)             pooled-query x set-memory (S2S C2 fold)
  PIPELINE-ANCHOR  pooled cosine over BANKED whole-video img_feats  internal reference
  WITH-TEXT        POOLED/SET visual + video-level cos(text,text)   text-channel sensitivity (K4)
  RANK-ONLY (A2)   POOLED/SET retrieval, sim neutralised to 1.0    sim-scale de-confound
  ORACLE (A4)      per-query oracle-sub-clip MaxSim (K4 PRIMARY, video-level gold ONLY)  kill-switch ceiling
  FANO (N2)        +/-1 gold-label-agreement key                  machine-validity calibration
SENSITIVITY (train-only, NEVER survival-determining -- B2):
  K30  granularity contrast: HateMM K4-train-only vs K30-train-only (744)   breadth-modifier ONLY
  _mm  modality contrast:    MHC-EN K4-visual-train-only vs _mm multimodal (549), has_text mask + eps guard

GATES (mechanical arithmetic only; independent verdict reviewer rules -- house rule):
  Fano >= 0.99 both primary datasets ; oracle Delta < +0.04 on EVERY primary dataset (K4 arm, B3) -> DEAD ;
  HateMM K4 raw Delta acc >= +0.05 AND Delta mF1 >= +0.05 corroborated by RANK-ONLY (A2) ;
  MHC-EN K4 survival Delta acc >= +0.03 AND Delta mF1 >= +0.03 ;
  observed Delta > 95th pct perm null (0..99, same perm both arms) ; bootstrap 5th pct > 0 ;
  near-dup-excluded advantage survives. K4 PRIMARY is the SOLE survival-determining arm (B2); K30/_mm
  only modulate a negative's BREADTH.

FAIL CLOSED (N4): never constructs/opens any test_seen file; asserts memory VIDEO-count == the exact
train u dev_seen count (primary) or train-only count (sensitivity). No silent excepts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# src is at <repo>/src locally and /root/src on Modal -- both = parents[2]/src of this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from utils.metrics import compute_metrics_retrieval  # the REAL vote (do NOT reimplement)

# ------- pre-registered constants -------
TOPK = 20
MODEL_TAG = "openai_clip-vit-large-patch14-336_HF"
EXPECTED_MEM_PRIMARY = {"HateMM": 851, "MHC": 629}      # train u dev_seen (K4 primary, N4/B1 guard)
EXPECTED_MEM_TRAINONLY = {"HateMM": 744, "MHC": 549}    # train-only sensitivity arms
NEAR_DUP_THRESH = 0.995                                  # A3 binding flag (pooled OR MMS)
NEAR_DUP_REPORT = [0.98, 0.99, 0.995]                    # A3 distribution
ORACLE_BAR = 0.04                                        # kill-switch (K4 primary only, B3)
RAW_BAR = 0.05                                           # HateMM anchor raw bar
MHC_SURVIVAL_BAR = 0.03                                  # MHC-EN survival bar
FANO_BAR = 0.99
NULL_SEEDS = list(range(100))                            # N1 seed set 0..99
TIEBREAK_EPS = 1e-9
NEG_INF = -1e30
EPS = 1e-12


def _log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Paths (N4: train + dev_seen ONLY; test_seen is NEVER constructed).
# ---------------------------------------------------------------------------
def _subclip_path(data_root, dataset, split, tag):
    assert split in ("train", "dev_seen"), \
        "N4 GUARD: probe may only load train/dev_seen, never '{}'".format(split)
    return os.path.join(data_root, "CLIP_Embedding", dataset,
                        "{}_{}_{}.pt".format(split, tag, MODEL_TAG))


def _pooled_path(data_root, dataset, split):
    assert split in ("train", "dev_seen"), \
        "N4 GUARD: probe may only load train/dev_seen, never '{}'".format(split)
    return os.path.join(data_root, "CLIP_Embedding", dataset, "{}_{}.pt".format(split, MODEL_TAG))


# ---------------------------------------------------------------------------
# Loader (B1: video-level view(V,K,D); contiguous-parent + video-count guards).
# ---------------------------------------------------------------------------
def load_memory(data_root, dataset, tag, splits, expected_mem, want_pooled=False, want_mm=False):
    ids, g_list, lab_list = [], [], []
    txt_list, has_list, pimg_list, ptxt_list = [], [], [], []
    for split in splits:                                # NEVER test_seen (N4)
        fp = _subclip_path(data_root, dataset, split, tag)
        if not os.path.exists(fp):
            raise RuntimeError("missing sub-clip cache: {}".format(fp))
        d = torch.load(fp, map_location="cpu", weights_only=False)
        vids = [str(x) for x in d["video_ids"]]
        V = len(vids)
        K = int(d["num_subclips"])
        rows = int(d["subclip_img_feats"].shape[0])
        # B1-a: contiguous K-block parent AND V*K == rows (else HALT -- non-video-level layout).
        exp_par = torch.repeat_interleave(torch.arange(V), K)
        if rows != V * K or not torch.equal(d["subclip_parent"], exp_par):
            raise RuntimeError("B1-a GUARD: non-contiguous parent or V*K!=rows for {}/{} "
                               "(V={} K={} rows={})".format(dataset, split, V, K, rows))
        g = d["subclip_img_feats"].float().view(V, K, -1)
        lab = d["labels"].long().view(V, K)
        if not bool((lab == lab[:, :1]).all()):
            raise RuntimeError("label not constant within a video: {}/{}".format(dataset, split))
        ids.extend(vids)
        g_list.append(g)
        lab_list.append(lab[:, 0])                       # per-video label = first sub-clip (MIL)
        if want_mm:
            txt_list.append(d["subclip_txt_feats"].float().view(V, K, -1))
            has_list.append(d["subclip_txt_has_text"].bool().view(V, K))
        if want_pooled:
            p = torch.load(_pooled_path(data_root, dataset, split),
                           map_location="cpu", weights_only=False)
            pids = [str(x) for x in p["ids"][0]]
            pindex = {i: k for k, i in enumerate(pids)}
            order = [pindex[i] for i in vids]            # align pooled rows to sub-clip video order
            pimg_list.append(p["img_feats"][order].float())
            ptxt_list.append(p["text_feats"][order].float())

    g = torch.cat(g_list, 0)                             # [V, K, D]
    labels = torch.cat(lab_list, 0).numpy()
    V = int(g.shape[0])
    if V != expected_mem:                                # B1/N4 VIDEO-COUNT guard (flat-bank / test leak)
        raise RuntimeError("N4/B1 VIDEO-COUNT GUARD: memory V={} != expected {} for {}/{} "
                           "(flat sub-clip bank or stray test row?)".format(V, expected_mem, dataset, tag))
    guard = (g.norm(dim=-1) < 1e-6).all(dim=1).numpy()   # per-video: all-K-sub-clips-zero
    out = {"ids": ids, "g": g, "labels": labels, "guard": guard, "V": V, "K": int(g.shape[1])}
    if want_mm:
        out["txt"] = torch.cat(txt_list, 0)
        out["has_text"] = torch.cat(has_list, 0)
    if want_pooled:
        out["pimg"] = torch.cat(pimg_list, 0)
        out["ptxt"] = torch.cat(ptxt_list, 0)
    _log("[{}] tag={} splits={} memory V={} (video-level) K={} D={} zero-guard={}".format(
        dataset, tag, list(splits), V, out["K"], int(g.shape[2]), int(guard.sum())))
    return out


# ---------------------------------------------------------------------------
# Score matrices (float; deterministic index tie-break baked in).
# ---------------------------------------------------------------------------
def _l2norm(x, dim=-1, eps=EPS):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _tiebreak(S):
    N = S.shape[1]
    return S - (np.arange(N, dtype=np.float64)[None, :] * TIEBREAK_EPS)


def build_matrices(mem, want_pooled=False):
    g = mem["g"]                                         # [V, K, D]
    V, K, D = g.shape
    ghat = _l2norm(g, dim=-1)                            # zero-guard rows -> ~0 vectors
    G = ghat.reshape(V * K, D)
    Sff = (G @ G.t()).reshape(V, K, V, K).numpy().astype(np.float32)   # cos(sub-clip, sub-clip)
    mms = Sff.max(axis=3).mean(axis=1).astype(np.float64)             # MeanMaxSim [V, V]
    pooled = _l2norm(g.mean(dim=1), dim=-1)                            # [V, D]
    spool = (pooled @ pooled.t()).numpy().astype(np.float64)          # POOLED [V, V]
    # ASYM = pooled-query x set-memory (|Q|=1 reduction of MeanMaxSim) on the SAME frozen vectors.
    asym = (pooled @ G.t()).reshape(V, V, K).max(dim=2).values.numpy().astype(np.float64)  # [V, V]
    maxframe = Sff.max(axis=(1, 3)).astype(np.float64)               # single-sub-clip max cos [V, V]
    M = {"C": Sff, "mms": mms, "spool": spool, "asym": asym, "maxframe": maxframe, "V": V, "K": K}
    if want_pooled and "pimg" in mem:
        imn = _l2norm(mem["pimg"], dim=-1)
        M["spipe"] = (imn @ imn.t()).numpy().astype(np.float64)      # PIPELINE-ANCHOR
        txn = _l2norm(mem["ptxt"], dim=-1)
        M["stext"] = (txn @ txn.t()).numpy().astype(np.float64)      # video-level text (WITH-TEXT)
    return M


def build_mm_matrices(mem):
    """(N3-ii) Multimodal sub-clip MeanMaxSim: per-pair score = cos(img)+cos(txt) where BOTH sub-clips
    carry ASR (has_text), else the text term is 0. Text vectors eps-normalized so a zero-norm empty-text
    row can never make cos(0,.) a 0/0 NaN; the has_text mask additionally zeroes the term semantically."""
    g = mem["g"]
    txt = mem["txt"]
    has = mem["has_text"]
    V, K, D = g.shape
    ghat = _l2norm(g, dim=-1).reshape(V * K, D)
    that = _l2norm(txt, dim=-1).reshape(V * K, txt.shape[-1])          # eps-guarded (zero rows -> ~0)
    hasf = has.reshape(V * K).float()
    Sv = (ghat @ ghat.t()).reshape(V, K, V, K)
    St = (that @ that.t()).reshape(V, K, V, K)
    hmask = (hasf[:, None] * hasf[None, :]).reshape(V, K, V, K)        # text term only if BOTH have ASR
    Smm = (Sv + St * hmask).numpy().astype(np.float32)
    Svn = Sv.numpy().astype(np.float32)
    mms_mm = Smm.max(axis=3).mean(axis=1).astype(np.float64)
    mms_vis = Svn.max(axis=3).mean(axis=1).astype(np.float64)
    pooled = _l2norm(g.mean(dim=1), dim=-1)
    spool = (pooled @ pooled.t()).numpy().astype(np.float64)
    return {"mms_mm": mms_mm, "mms_vis": mms_vis, "spool": spool, "V": V, "K": K,
            "text_coverage": float(hasf.mean().item())}


# ---------------------------------------------------------------------------
# Retrieval + REAL vote (verbatim from s2s_probe.py, incl. NB-a NEG_INF filter / N3-i).
# ---------------------------------------------------------------------------
def run_vote(S, labels, k=TOPK, rank_only=False, exclude=None):
    """Top-k VIDEO-LEVEL LOO retrieval by score matrix S [V,V], then the pipeline's real vote.
    Diagonal (the query's own video) is excluded -> the whole query set is held out (B1-b)."""
    N = S.shape[0]
    St = _tiebreak(S).copy()
    np.fill_diagonal(St, NEG_INF)                        # video-level LOO self-exclusion (B1-b)
    if exclude is not None:
        St[exclude] = NEG_INF
    logging_dict = {}
    for i in range(N):
        row = St[i]
        topk_idx = np.argpartition(-row, k)[:k]
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]  # exact order (desc, tie by idx)
        # N3-i (NB-a): never let an excluded/self NEG_INF entry enter the vote as a neighbour.
        topk_idx = topk_idx[row[topk_idx] > (NEG_INF / 2)]
        if topk_idx.size == 0:
            raise RuntimeError("degenerate retrieval: no finite neighbours for query {}".format(i))
        sims = (np.ones(topk_idx.size, dtype=np.float64) if rank_only
                else row[topk_idx].astype(np.float64))
        logging_dict[i] = {
            "retrieved_label": [int(labels[j]) for j in topk_idx],
            "retrieved_scores": list(sims),
        }
    acc, roc, pre, rec, f1, votes, _lab, macro = compute_metrics_retrieval(
        logging_dict, torch.tensor(labels), majority_voting="arithmetic",
        topk=k, use_sim=True)
    return float(acc), float(macro["macro_f1"]), float(roc), np.asarray(votes, dtype=np.float64)


def _preds_from_votes(votes):
    return (1.0 / (1.0 + np.exp(-votes)) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Oracle ceiling (A4) -- per-query sub-clip selection, video-level gold ONLY, K4 PRIMARY arm (B3).
# ---------------------------------------------------------------------------
def _single_query_vote_margins(C, labels, k=TOPK):
    N, K = C.shape[0], C.shape[1]
    weight = np.arange(1, k + 1, dtype=np.float64)[::-1]
    wsum = weight.sum()
    lab_signed = (labels.astype(np.float64) * 2.0 - 1.0)
    Vm = np.zeros((N, K), dtype=np.float64)
    for t in range(K):
        St = C[:, t, :, :].max(axis=2).astype(np.float64)   # s_t(i,j) = max_m C[i,t,j,m]  [N,N]
        St = _tiebreak(St)
        np.fill_diagonal(St, NEG_INF)
        for i in range(N):
            row = St[i]
            idx = np.argpartition(-row, k)[:k]
            idx = idx[np.argsort(-row[idx])]
            labelmap = lab_signed[idx] * row[idx]
            Vm[i, t] = float((labelmap * weight).sum() / wsum)
    return Vm


def oracle_ceiling(C, labels, k=TOPK):
    N, K = C.shape[0], C.shape[1]
    Vm = _single_query_vote_margins(C, labels, k)
    signed = (labels.astype(np.float64) * 2.0 - 1.0)[:, None] * Vm     # [N, K]
    tstar = np.argmax(signed, axis=1)                                  # smallest index on ties
    Sorc = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        Sorc[i] = C[i, tstar[i], :, :].max(axis=1)                     # s_{t*(i)}(i, .)
    acc, mf1, roc, votes = run_vote(Sorc, labels, k)
    return acc, mf1, votes, tstar.tolist()


# ---------------------------------------------------------------------------
# Fano (N2) -- +/-1 gold-label-agreement machine-validity arm.
# ---------------------------------------------------------------------------
def fano(labels, k=TOPK):
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    S = same * 2.0 - 1.0                                  # +1 same-label, -1 diff-label
    acc, mf1, roc, votes = run_vote(S, labels, k)
    return acc


# ---------------------------------------------------------------------------
# Near-duplicate audit (A3).
# ---------------------------------------------------------------------------
def near_dup_audit(M, guard):
    N = M["V"]
    tri = np.triu(np.ones((N, N), dtype=bool), k=1)
    valid = tri & ~(guard[:, None] | guard[None, :])
    dist = {}
    for th in NEAR_DUP_REPORT:
        dist["pooled>=%.3f" % th] = int(((M["spool"] >= th) & valid).sum())
        dist["mms>=%.3f" % th] = int(((M["mms"] >= th) & valid).sum())
        dist["maxframe>=%.3f" % th] = int(((M["maxframe"] >= th) & valid).sum())  # N2 single-unit mass
    flag_u = ((M["spool"] >= NEAR_DUP_THRESH) | (M["mms"] >= NEAR_DUP_THRESH)) & valid
    flag = flag_u | flag_u.T
    return flag, dist, int(flag_u.sum())


# ---------------------------------------------------------------------------
# Permutation null (N1) -- same permutation applied to every arm within a seed.
# ---------------------------------------------------------------------------
def permutation_null(M, labels, k=TOPK):
    mms, spool, asym = M["mms"], M["spool"], M["asym"]
    obs_set_acc, obs_set_f1, _, _ = run_vote(mms, labels, k)
    obs_pool_acc, obs_pool_f1, _, _ = run_vote(spool, labels, k)
    obs_asym_acc, obs_asym_f1, _, _ = run_vote(asym, labels, k)
    obs_set_acc_r, obs_set_f1_r, _, _ = run_vote(mms, labels, k, rank_only=True)
    obs_pool_acc_r, obs_pool_f1_r, _, _ = run_vote(spool, labels, k, rank_only=True)
    obs = {"dacc": obs_set_acc - obs_pool_acc, "df1": obs_set_f1 - obs_pool_f1,
           "dacc_rank": obs_set_acc_r - obs_pool_acc_r, "df1_rank": obs_set_f1_r - obs_pool_f1_r,
           "dacc_asym": obs_asym_acc - obs_pool_acc, "df1_asym": obs_asym_f1 - obs_pool_f1,
           "dacc_asym_vs_set": obs_asym_acc - obs_set_acc,
           "df1_asym_vs_set": obs_asym_f1 - obs_set_f1}
    nd = {kk: [] for kk in obs}
    for s in NULL_SEEDS:
        perm = np.random.default_rng(s).permutation(M["V"])
        ix = np.ix_(perm, perm)
        mms_s, spool_s, asym_s = mms[ix], spool[ix], asym[ix]
        sa, sf, _, _ = run_vote(mms_s, labels, k)
        pa, pf, _, _ = run_vote(spool_s, labels, k)
        aa, af, _, _ = run_vote(asym_s, labels, k)
        sar, sfr, _, _ = run_vote(mms_s, labels, k, rank_only=True)
        par, pfr, _, _ = run_vote(spool_s, labels, k, rank_only=True)
        nd["dacc"].append(sa - pa); nd["df1"].append(sf - pf)
        nd["dacc_rank"].append(sar - par); nd["df1_rank"].append(sfr - pfr)
        nd["dacc_asym"].append(aa - pa); nd["df1_asym"].append(af - pf)
        nd["dacc_asym_vs_set"].append(aa - sa); nd["df1_asym_vs_set"].append(af - sf)
    p95 = {kk: float(np.percentile(np.array(vv), 95)) for kk, vv in nd.items()}
    out = {"n_seeds": len(NULL_SEEDS)}
    for kk in obs:
        out["obs_" + kk] = obs[kk]
        out["null_" + kk + "_p95"] = p95[kk]
        out["obs_" + kk + "_gt_p95"] = bool(obs[kk] > p95[kk])
    return out


def simple_null(sA, sB, labels, k=TOPK, n_seeds=100):
    """Lean same-perm null for a single paired contrast Delta(A - B) (sensitivity arms)."""
    obs_a_acc, obs_a_f1, _, _ = run_vote(sA, labels, k)
    obs_b_acc, obs_b_f1, _, _ = run_vote(sB, labels, k)
    obs = {"dacc": obs_a_acc - obs_b_acc, "df1": obs_a_f1 - obs_b_f1}
    dacc, df1 = [], []
    N = sA.shape[0]
    for s in range(n_seeds):
        perm = np.random.default_rng(s).permutation(N)
        ix = np.ix_(perm, perm)
        aa, af, _, _ = run_vote(sA[ix], labels, k)
        ba, bf, _, _ = run_vote(sB[ix], labels, k)
        dacc.append(aa - ba); df1.append(af - bf)
    return {"obs_dacc": obs["dacc"], "obs_df1": obs["df1"],
            "null_dacc_p95": float(np.percentile(dacc, 95)),
            "null_df1_p95": float(np.percentile(df1, 95)),
            "obs_dacc_gt_p95": bool(obs["dacc"] > float(np.percentile(dacc, 95))),
            "a_acc": obs_a_acc, "a_f1": obs_a_f1, "b_acc": obs_b_acc, "b_f1": obs_b_f1}


def per_frame_null(mem, labels, n_seeds, k=TOPK):
    """Optional secondary null: shuffle individual sub-clip vectors across all videos, destroying set
    structure, recompute MMS -> Delta. Separates 'alignment' from a generic 'richer-key' effect."""
    g = mem["g"]
    V, K, D = g.shape
    ghat = _l2norm(g, dim=-1).reshape(V * K, D)
    pooled = _l2norm(g.mean(dim=1), dim=-1)
    spool = (pooled @ pooled.t()).numpy().astype(np.float64)
    obs_pool_acc, obs_pool_f1, _, _ = run_vote(spool, labels, k)
    dacc, df1 = [], []
    for s in range(n_seeds):
        perm = np.random.default_rng(10_000 + s).permutation(V * K)
        Gs = ghat[perm].reshape(V, K, D)
        Sff = (Gs.reshape(V * K, D) @ Gs.reshape(V * K, D).t()).reshape(V, K, V, K)
        mms = Sff.max(dim=3).values.mean(dim=1).numpy().astype(np.float64)
        sa, sf, _, _ = run_vote(mms, labels, k)
        dacc.append(sa - obs_pool_acc)
        df1.append(sf - obs_pool_f1)
    return {"n_seeds": n_seeds, "dacc_p95": float(np.percentile(dacc, 95)),
            "df1_p95": float(np.percentile(df1, 95)),
            "dacc_mean": float(np.mean(dacc)), "df1_mean": float(np.mean(df1))}


# ---------------------------------------------------------------------------
# Bootstrap (D3).
# ---------------------------------------------------------------------------
def bootstrap_delta(votes_a, votes_b, labels, n_boot, seed=20260714):
    from sklearn.metrics import f1_score
    pa, pb = _preds_from_votes(votes_a), _preds_from_votes(votes_b)
    y = labels.astype(int)
    N = len(y)
    rng = np.random.default_rng(seed)
    dacc, df1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        yy = y[idx]
        dacc[b] = (pa[idx] == yy).mean() - (pb[idx] == yy).mean()
        df1[b] = (f1_score(yy, pa[idx], average="macro", zero_division=0)
                  - f1_score(yy, pb[idx], average="macro", zero_division=0))
    return {"n_boot": n_boot,
            "dacc_p5": float(np.percentile(dacc, 5)), "dacc_p50": float(np.percentile(dacc, 50)),
            "dacc_p95": float(np.percentile(dacc, 95)),
            "df1_p5": float(np.percentile(df1, 5)), "df1_p50": float(np.percentile(df1, 50)),
            "df1_p95": float(np.percentile(df1, 95)),
            "dacc_p5_gt0": bool(np.percentile(dacc, 5) > 0),
            "df1_p5_gt0": bool(np.percentile(df1, 5) > 0)}


# ---------------------------------------------------------------------------
# Synthetic set-matching positive control (planted shared sub-clip => MMS > POOLED).
# ---------------------------------------------------------------------------
def synthetic_set_control():
    rng = np.random.default_rng(7)
    D, K = 1024, 4
    shared = rng.standard_normal(D)
    A = rng.standard_normal((K, D)); A[0] = shared
    B = rng.standard_normal((K, D)); B[0] = shared           # A,B share sub-clip 0, differ elsewhere
    Ah = torch.tensor(A) / torch.tensor(A).norm(dim=-1, keepdim=True)
    Bh = torch.tensor(B) / torch.tensor(B).norm(dim=-1, keepdim=True)
    mms = float((Ah @ Bh.t()).max(dim=1).values.mean().item())
    pa = A.mean(0); pb = B.mean(0)
    pooled = float(np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb)))
    if not (mms > pooled):
        raise RuntimeError("[synthetic set control] MMS {:.4f} !> POOLED {:.4f} -- set-metric bug".format(
            mms, pooled))
    _log("[probe self-test] synthetic shared-segment: MMS {:.4f} > POOLED {:.4f} OK".format(mms, pooled))
    return {"mms": mms, "pooled": pooled, "passed": True}


# ---------------------------------------------------------------------------
# Primary probe (K4, full battery). SURVIVAL-DETERMINING.
# ---------------------------------------------------------------------------
def probe_primary(dataset, data_root, n_boot, n_perframe):
    exp = EXPECTED_MEM_PRIMARY[dataset]
    mem = load_memory(data_root, dataset, "subclipK4", ("train", "dev_seen"), exp,
                      want_pooled=True, want_mm=False)
    labels = mem["labels"]
    M = build_matrices(mem, want_pooled=True)

    arms = {}

    def add(name, S, rank_only=False, exclude=None):
        acc, mf1, roc, votes = run_vote(S, labels, TOPK, rank_only=rank_only, exclude=exclude)
        arms[name] = {"acc": acc, "macro_f1": mf1, "roc": roc, "votes": votes}
        return arms[name]

    a_pool = add("POOLED", M["spool"])
    a_set = add("SET", M["mms"])
    add("SET_CHAMFER", 0.5 * (M["mms"] + M["mms"].T))
    a_asym = add("ASYM", M["asym"])
    if "spipe" in M:
        add("PIPELINE_ANCHOR", M["spipe"])
        add("WITH_TEXT_POOLED", M["spool"] + M["stext"])
        add("WITH_TEXT_SET", M["mms"] + M["stext"])
    a_pool_r = add("POOLED_RANKONLY", M["spool"], rank_only=True)
    a_set_r = add("SET_RANKONLY", M["mms"], rank_only=True)

    d_acc = a_set["acc"] - a_pool["acc"]
    d_f1 = a_set["macro_f1"] - a_pool["macro_f1"]
    d_acc_rank = a_set_r["acc"] - a_pool_r["acc"]
    d_f1_rank = a_set_r["macro_f1"] - a_pool_r["macro_f1"]
    d_asym_set_acc = a_asym["acc"] - a_set["acc"]
    d_asym_set_f1 = a_asym["macro_f1"] - a_set["macro_f1"]

    fano_acc = fano(labels)
    orc_acc, orc_f1, orc_votes, tstar = oracle_ceiling(M["C"], labels)   # B3: K4 primary arm
    d_orc_acc = orc_acc - a_pool["acc"]
    d_orc_f1 = orc_f1 - a_pool["macro_f1"]

    flag, nd_dist, nd_pairs = near_dup_audit(M, mem["guard"])
    a_pool_x = add("POOLED_NEARDUP_EXCL", M["spool"], exclude=flag)
    a_set_x = add("SET_NEARDUP_EXCL", M["mms"], exclude=flag)
    d_acc_x = a_set_x["acc"] - a_pool_x["acc"]
    d_f1_x = a_set_x["macro_f1"] - a_pool_x["macro_f1"]

    null = permutation_null(M, labels)
    boot = bootstrap_delta(a_set["votes"], a_pool["votes"], labels, n_boot)
    boot_rank = bootstrap_delta(a_set_r["votes"], a_pool_r["votes"], labels, n_boot)
    boot_asym_vs_set = bootstrap_delta(a_asym["votes"], a_set["votes"], labels, n_boot)
    boot_asym = bootstrap_delta(a_asym["votes"], a_pool["votes"], labels, n_boot)
    pfn = per_frame_null(mem, labels, n_perframe) if n_perframe > 0 else None

    asym_beats_set = bool(d_asym_set_acc > 0.0 and d_asym_set_f1 > 0.0)
    rank_sign_ok = bool(np.sign(d_acc) == np.sign(d_acc_rank)
                        and np.sign(d_f1) == np.sign(d_f1_rank))
    rank_null_ok = bool(null["obs_dacc_rank_gt_p95"])
    rank_boot_ok = bool(boot_rank["dacc_p5_gt0"])
    rank_corroborates = bool(rank_sign_ok and rank_null_ok and rank_boot_ok)

    arm_summary = {k: {kk: v[kk] for kk in ("acc", "macro_f1", "roc")} for k, v in arms.items()}
    return {
        "dataset": dataset, "arm": "K4_primary", "V": mem["V"], "K": mem["K"],
        "zero_guard": int(mem["guard"].sum()),
        "arms": arm_summary,
        "primary": {"d_acc": d_acc, "d_f1": d_f1,
                    "rankonly_d_acc": d_acc_rank, "rankonly_d_f1": d_f1_rank,
                    "rankonly_sign_ok": rank_sign_ok, "rankonly_null_ok": rank_null_ok,
                    "rankonly_boot_ok": rank_boot_ok, "rankonly_corroborates": rank_corroborates,
                    "rankonly_null_p95_acc": null["null_dacc_rank_p95"],
                    "rankonly_obs_dacc": null["obs_dacc_rank"],
                    "rankonly_boot_dacc_p5": boot_rank["dacc_p5"]},
        "bootstrap_rankonly": boot_rank,
        "c2_asym": {"asym_acc": a_asym["acc"], "asym_macro_f1": a_asym["macro_f1"],
                    "asym_vs_set_d_acc": d_asym_set_acc, "asym_vs_set_d_f1": d_asym_set_f1,
                    "asym_beats_set": asym_beats_set,
                    "asym_vs_set_obs_gt_p95": null["obs_dacc_asym_vs_set_gt_p95"],
                    "asym_vs_set_boot_dacc_p5": boot_asym_vs_set["dacc_p5"],
                    "asym_vs_pool_boot_dacc_p5": boot_asym["dacc_p5"]},
        "fano_acc": fano_acc,
        "oracle": {"acc": orc_acc, "macro_f1": orc_f1, "d_acc": d_orc_acc, "d_f1": d_orc_f1},
        "near_dup": {"threshold": NEAR_DUP_THRESH, "flagged_pairs": nd_pairs, "distribution": nd_dist,
                     "excluded_d_acc": d_acc_x, "excluded_d_f1": d_f1_x},
        "permutation_null": null,
        "bootstrap": boot,
        "per_frame_null": pfn,
    }


# ---------------------------------------------------------------------------
# Sensitivity arms (train-only). BREADTH / MODALITY reports -- NEVER survival-determining (B2).
# ---------------------------------------------------------------------------
def sensitivity_k30(data_root, n_boot):
    exp = EXPECTED_MEM_TRAINONLY["HateMM"]                # 744, train-only
    out = {"arm": "K30_sensitivity_HateMM", "note": "breadth-modifier only (B2); NOT survival-determining"}
    for tag in ("subclipK4", "subclipK30"):
        mem = load_memory(data_root, "HateMM", tag, ("train",), exp, want_pooled=False)
        M = build_matrices(mem, want_pooled=False)
        labels = mem["labels"]
        a_pool_acc, a_pool_f1, _, vp = run_vote(M["spool"], labels)
        a_set_acc, a_set_f1, _, vs = run_vote(M["mms"], labels)
        nul = simple_null(M["mms"], M["spool"], labels)
        boot = bootstrap_delta(vs, vp, labels, n_boot)
        out[tag] = {"V": mem["V"], "K": mem["K"], "zero_guard": int(mem["guard"].sum()),
                    "pool_acc": a_pool_acc, "pool_f1": a_pool_f1,
                    "set_acc": a_set_acc, "set_f1": a_set_f1,
                    "d_acc": a_set_acc - a_pool_acc, "d_f1": a_set_f1 - a_pool_f1,
                    "obs_dacc_gt_null95": nul["obs_dacc_gt_p95"], "boot_dacc_p5": boot["dacc_p5"]}
    return out


def sensitivity_mm(data_root, n_boot):
    exp = EXPECTED_MEM_TRAINONLY["MHC"]                   # 549, train-only
    mem = load_memory(data_root, "MHC", "subclipK4_mm", ("train",), exp, want_pooled=False, want_mm=True)
    mm = build_mm_matrices(mem)
    labels = mem["labels"]
    p_acc, p_f1, _, vp = run_vote(mm["spool"], labels)
    v_acc, v_f1, _, vv = run_vote(mm["mms_vis"], labels)
    m_acc, m_f1, _, vm = run_vote(mm["mms_mm"], labels)
    nul_mm_pool = simple_null(mm["mms_mm"], mm["spool"], labels)
    nul_mm_vis = simple_null(mm["mms_mm"], mm["mms_vis"], labels)
    boot_mm_pool = bootstrap_delta(vm, vp, labels, n_boot)
    boot_mm_vis = bootstrap_delta(vm, vv, labels, n_boot)
    return {"arm": "mm_sensitivity_MHC", "note": "modality contrast only (B2); NOT survival-determining",
            "V": mem["V"], "K": mem["K"], "text_coverage": mm["text_coverage"],
            "pool_acc": p_acc, "pool_f1": p_f1,
            "vis_set_acc": v_acc, "vis_set_f1": v_f1, "mm_set_acc": m_acc, "mm_set_f1": m_f1,
            "d_mm_vs_pool_acc": m_acc - p_acc, "d_mm_vs_pool_f1": m_f1 - p_f1,
            "d_mm_vs_vis_acc": m_acc - v_acc, "d_mm_vs_vis_f1": m_f1 - v_f1,
            "mm_vs_pool_obs_gt_null95": nul_mm_pool["obs_dacc_gt_p95"],
            "mm_vs_vis_obs_gt_null95": nul_mm_vis["obs_dacc_gt_p95"],
            "mm_vs_pool_boot_dacc_p5": boot_mm_pool["dacc_p5"],
            "mm_vs_vis_boot_dacc_p5": boot_mm_vis["dacc_p5"]}


# ---------------------------------------------------------------------------
# Mechanical gate check (NOT the binding verdict -- house rule). K4 PRIMARY is sole survival arm (B2).
# ---------------------------------------------------------------------------
def mechanical_gate_check(primaries, k30, mm):
    by_ds = {r["dataset"]: r for r in primaries}
    checks = []

    def rec(name, value, thr, op, note=""):
        ok = (value >= thr) if op == ">=" else (value < thr) if op == "<" else (value > thr)
        checks.append({"gate": name, "value": value, "threshold": thr, "op": op,
                       "result": "ABOVE" if ok else "BELOW", "note": note})

    for ds, r in by_ds.items():
        rec("Fano[%s]" % ds, r["fano_acc"], FANO_BAR, ">=", "vote machine valid if ABOVE")
    # B3: oracle kill-switch computed on the K4 PRIMARY arm ONLY.
    oracle_all_below = all(r["oracle"]["d_acc"] < ORACLE_BAR for r in primaries)
    for ds, r in by_ds.items():
        rec("OracleDacc_K4primary[%s]" % ds, r["oracle"]["d_acc"], ORACLE_BAR, ">=",
            "headroom if ABOVE (kill-switch fires only if ALL primary datasets BELOW)")
    checks.append({"gate": "OracleKillSwitch(K4-primary,all-datasets)", "value": oracle_all_below,
                   "threshold": "all < %.2f" % ORACLE_BAR,
                   "result": "KILL(DEAD-family)" if oracle_all_below else "SURVIVES",
                   "note": "B3: K4 primary oracle only; DEAD -> don't-pool family prior down"})
    if "HateMM" in by_ds:
        h = by_ds["HateMM"]; hp = h["primary"]
        rec("RawDacc_K4[HateMM]", hp["d_acc"], RAW_BAR, ">=")
        rec("RawDmF1_K4[HateMM]", hp["d_f1"], RAW_BAR, ">=")
        checks.append({"gate": "RankOnlyCorroborates[HateMM] (A2)", "value": hp["rankonly_corroborates"],
                       "threshold": True, "result": "ABOVE" if hp["rankonly_corroborates"] else "BELOW",
                       "note": "sign={} null={} boot={}".format(hp["rankonly_sign_ok"],
                                                                 hp["rankonly_null_ok"],
                                                                 hp["rankonly_boot_ok"])})
        rec("ObsDacc>null95[HateMM]", h["permutation_null"]["obs_dacc"],
            h["permutation_null"]["null_dacc_p95"], ">")
        rec("Bootstrap5th>0[HateMM]", h["bootstrap"]["dacc_p5"], 0.0, ">")
        rec("NearDupExclSurvives[HateMM] (A3)", h["near_dup"]["excluded_d_acc"], 0.0, ">")
    if "MHC" in by_ds:
        m = by_ds["MHC"]; mp = m["primary"]
        rec("SurvivalDacc_K4[MHC-EN]", mp["d_acc"], MHC_SURVIVAL_BAR, ">=")
        rec("SurvivalDmF1_K4[MHC-EN]", mp["d_f1"], MHC_SURVIVAL_BAR, ">=")
    # K4-primary-determined dataset rule (B2): K30/_mm are breadth/modality reports only.
    hate_pass = ("HateMM" in by_ds and by_ds["HateMM"]["primary"]["d_acc"] >= RAW_BAR
                 and by_ds["HateMM"]["primary"]["d_f1"] >= RAW_BAR
                 and by_ds["HateMM"]["primary"]["rankonly_corroborates"])
    mhc_pass = ("MHC" in by_ds and by_ds["MHC"]["primary"]["d_acc"] >= MHC_SURVIVAL_BAR
                and by_ds["MHC"]["primary"]["d_f1"] >= MHC_SURVIVAL_BAR)
    if oracle_all_below:
        outcome = "(a) DEAD-family (oracle dead on every primary dataset)"
    elif hate_pass and mhc_pass:
        outcome = "(b) BOTH -> escalate to Qwen-token S2S"
    elif hate_pass or mhc_pass:
        outcome = "(c) SINGLE (Delta-1-style dataset-specific; no family greenlight)"
    else:
        outcome = "(d) NEGATIVE (neither raw bar; weak-negative family update)"
    checks.append({"gate": "DatasetRule (K4-primary determined, B2)", "value": outcome,
                   "threshold": "a/b/c/d", "result": outcome,
                   "note": "K30/_mm are NON-determining breadth/modality reports"})
    if k30 is not None and "subclipK4" in k30 and "subclipK30" in k30:
        checks.append({"gate": "K30 breadth-modifier[HateMM] (NON-determining, B2)",
                       "value": "K4tr Dacc={:+.4f} / K30tr Dacc={:+.4f}".format(
                           k30["subclipK4"]["d_acc"], k30["subclipK30"]["d_acc"]),
                       "threshold": "reported only", "result": "REPORTED",
                       "note": "characterises a negative's breadth across granularity; cannot rescue K4"})
    if mm is not None:
        checks.append({"gate": "_mm modality report[MHC-EN] (NON-determining, B2)",
                       "value": "mm-vs-pool Dacc={:+.4f} / mm-vs-vis Dacc={:+.4f}".format(
                           mm["d_mm_vs_pool_acc"], mm["d_mm_vs_vis_acc"]),
                       "threshold": "reported only", "result": "REPORTED",
                       "note": "the _mm sliver; cannot rescue K4 primary"})
    return checks


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------
def write_markdown(primaries, k30, mm, checks, out_md):
    L = ["# W2-B Sub-clip Set-Matching Probe — RAW RESULTS (no pass/fail interpretation)\n"]
    L.append("_Executor writes raw numbers only; the independent verdict reviewer renders the binding "
             "ruling (house rule). The mechanical gate arithmetic in `w2b_probe_results.json` is NOT a "
             "verdict. K4 primary is the sole survival-determining arm (B2); K30/_mm are breadth/modality "
             "reports._\n")
    for r in primaries:
        L.append("\n## {} — K4 PRIMARY (memory V={}, K={}, zero-guard={})\n".format(
            r["dataset"], r["V"], r["K"], r["zero_guard"]))
        L.append("| arm | acc | macro_f1 | roc |")
        L.append("|---|---|---|---|")
        for name, a in r["arms"].items():
            L.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(name, a["acc"], a["macro_f1"], a["roc"]))
        p = r["primary"]
        L.append("\n**Primary paired Δ(SET−POOLED):** acc {:+.4f}, macro_f1 {:+.4f}. **Rank-only (A2):** "
                 "acc {:+.4f}, macro_f1 {:+.4f}; obs Δacc {:+.4f} vs rank-only null-95th {:+.4f}, "
                 "rank-only bootstrap-5th {:+.4f} (corroborates={}).".format(
                     p["d_acc"], p["d_f1"], p["rankonly_d_acc"], p["rankonly_d_f1"],
                     p["rankonly_obs_dacc"], p["rankonly_null_p95_acc"], p["rankonly_boot_dacc_p5"],
                     p["rankonly_corroborates"]))
        c2 = r["c2_asym"]
        L.append("\n**ASYM (pooled-query × set-memory):** acc {:.4f}, macro_f1 {:.4f}; Δ(ASYM−SET) acc "
                 "{:+.4f}, mF1 {:+.4f} (beats_set={}).".format(
                     c2["asym_acc"], c2["asym_macro_f1"], c2["asym_vs_set_d_acc"],
                     c2["asym_vs_set_d_f1"], c2["asym_beats_set"]))
        L.append("\n**Fano (±1 gold-label key) acc:** {:.4f}.".format(r["fano_acc"]))
        o = r["oracle"]
        L.append("**Oracle ceiling (A4, K4 primary):** acc {:.4f} (Δ vs POOLED acc {:+.4f}, "
                 "mF1 {:+.4f}).".format(o["acc"], o["d_acc"], o["d_f1"]))
        nd = r["near_dup"]
        L.append("**Near-dup (A3):** flagged pairs (≥{:.3f}) = {}; excluded-retrieval Δ(SET−POOLED) acc "
                 "{:+.4f}, mF1 {:+.4f}. Distribution: {}.".format(
                     nd["threshold"], nd["flagged_pairs"], nd["excluded_d_acc"], nd["excluded_d_f1"],
                     json.dumps(nd["distribution"])))
        nl = r["permutation_null"]
        L.append("**Permutation null (N1, {} seeds):** obs Δacc {:+.4f} vs null-95th {:+.4f}; obs ΔmF1 "
                 "{:+.4f} vs null-95th {:+.4f}.".format(
                     nl["n_seeds"], nl["obs_dacc"], nl["null_dacc_p95"], nl["obs_df1"],
                     nl["null_df1_p95"]))
        b = r["bootstrap"]
        L.append("**Bootstrap ({} resamples):** Δacc [5/50/95]=[{:+.4f}/{:+.4f}/{:+.4f}]; ΔmF1 "
                 "[5/50/95]=[{:+.4f}/{:+.4f}/{:+.4f}].".format(
                     b["n_boot"], b["dacc_p5"], b["dacc_p50"], b["dacc_p95"],
                     b["df1_p5"], b["df1_p50"], b["df1_p95"]))
        if r["per_frame_null"]:
            pf = r["per_frame_null"]
            L.append("**Per-frame null (optional, {} seeds):** Δacc-95th {:+.4f}, ΔmF1-95th {:+.4f}.".format(
                pf["n_seeds"], pf["dacc_p95"], pf["df1_p95"]))
    if k30 is not None:
        L.append("\n## K30 granularity SENSITIVITY — HateMM train-only (breadth-modifier only, B2)\n")
        for tag in ("subclipK4", "subclipK30"):
            if tag in k30:
                s = k30[tag]
                L.append("- **{}** (V={}, K={}): POOLED acc {:.4f}/mF1 {:.4f}; SET acc {:.4f}/mF1 {:.4f}; "
                         "Δ(SET−POOLED) acc {:+.4f}, mF1 {:+.4f}; obs>null95={}, boot5th {:+.4f}.".format(
                             tag, s["V"], s["K"], s["pool_acc"], s["pool_f1"], s["set_acc"], s["set_f1"],
                             s["d_acc"], s["d_f1"], s["obs_dacc_gt_null95"], s["boot_dacc_p5"]))
    if mm is not None:
        L.append("\n## _mm modality SENSITIVITY — MHC-EN train-only (modality report only, B2)\n")
        L.append("- V={}, K={}, text-coverage={:.3f}. POOLED acc {:.4f}/mF1 {:.4f}; VIS-SET acc "
                 "{:.4f}/mF1 {:.4f}; MM-SET acc {:.4f}/mF1 {:.4f}. Δ(MM−POOLED) acc {:+.4f}/mF1 {:+.4f}; "
                 "Δ(MM−VIS) acc {:+.4f}/mF1 {:+.4f}; mm-vs-vis obs>null95={}, boot5th {:+.4f}.".format(
                     mm["V"], mm["K"], mm["text_coverage"], mm["pool_acc"], mm["pool_f1"],
                     mm["vis_set_acc"], mm["vis_set_f1"], mm["mm_set_acc"], mm["mm_set_f1"],
                     mm["d_mm_vs_pool_acc"], mm["d_mm_vs_pool_f1"], mm["d_mm_vs_vis_acc"],
                     mm["d_mm_vs_vis_f1"], mm["mm_vs_vis_obs_gt_null95"], mm["mm_vs_vis_boot_dacc_p5"]))
    L.append("\n## Mechanical gate arithmetic (NOT a verdict — see JSON)\n")
    L.append("| gate | value | threshold | op | result |")
    L.append("|---|---|---|---|---|")
    for c in checks:
        L.append("| {} | {} | {} | {} | {} |".format(
            c["gate"], c["value"], c["threshold"], c.get("op", ""), c["result"]))
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="W2-B Stage-P CPU probe (zero training, zero test touch).")
    ap.add_argument("--data_root", type=str, default=os.path.join(_REPO_ROOT, "data"),
                    help="dir holding CLIP_Embedding/ (local repo data/; /root/data on Modal).")
    ap.add_argument("--datasets", type=str, default="HateMM,MHC",
                    help="K4 PRIMARY datasets (survival-determining).")
    ap.add_argument("--k30_sensitivity", type=int, default=1, help="run HateMM K4-vs-K30 train-only (1/0).")
    ap.add_argument("--mm_sensitivity", type=int, default=1, help="run MHC-EN _mm train-only (1/0).")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perframe_null", type=int, default=100, help="optional per-sub-clip null (0 to skip).")
    ap.add_argument("--out_md", type=str,
                    default=os.path.join(_REPO_ROOT, "refine-logs/W2B_PROBE_RESULTS.md"))
    ap.add_argument("--out_json", type=str,
                    default=os.path.join(_REPO_ROOT, "refine-logs/w2b_probe_results.json"))
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # hard CPU
    torch.manual_seed(20260714)
    np.random.seed(20260714)

    _log("=" * 78)
    _log("[W2-B probe] data_root={} datasets={} topk={} null_seeds={} n_boot={}".format(
        args.data_root, args.datasets, TOPK, len(NULL_SEEDS), args.n_boot))
    _log("[W2-B probe] N4 fail-closed: train+dev_seen ONLY; expected memory {}".format(EXPECTED_MEM_PRIMARY))
    _log("=" * 78)

    synthetic_set_control()

    t0 = time.time()
    primaries = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        _log("[W2-B probe] --- {} K4 PRIMARY ---".format(ds))
        primaries.append(probe_primary(ds, args.data_root, args.n_boot, args.n_perframe_null))
    k30 = sensitivity_k30(args.data_root, args.n_boot) if args.k30_sensitivity else None
    mm = sensitivity_mm(args.data_root, args.n_boot) if args.mm_sensitivity else None
    _log("[W2-B probe] probing done in {:.1f}s".format(time.time() - t0))

    checks = mechanical_gate_check(primaries, k30, mm)
    _log("\n[W2-B probe] MECHANICAL gate arithmetic (pre-registered thresholds; NOT the binding verdict):")
    for c in checks:
        _log("  GATE {:<42} value={} threshold={} {} -> {}".format(
            c["gate"], c["value"], c["threshold"], c.get("op", ""), c["result"]))

    out = {
        "meta": {"topk": TOPK, "null_seeds": len(NULL_SEEDS), "n_boot": args.n_boot,
                 "expected_mem_primary": EXPECTED_MEM_PRIMARY,
                 "expected_mem_trainonly": EXPECTED_MEM_TRAINONLY,
                 "raw_bar": RAW_BAR, "mhc_survival_bar": MHC_SURVIVAL_BAR, "oracle_bar": ORACLE_BAR,
                 "fano_bar": FANO_BAR, "near_dup_threshold": NEAR_DUP_THRESH,
                 "note": "Executor writes RAW numbers. mechanical_gate_check is pre-registered threshold "
                         "arithmetic ONLY, NOT the binding verdict. K4 primary is the sole "
                         "survival-determining arm (B2); K30/_mm are breadth/modality reports."},
        "primaries": primaries, "k30_sensitivity": k30, "mm_sensitivity": mm,
        "mechanical_gate_check": checks,
    }

    def _jsonable(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError("not JSON serializable: {}".format(type(o)))

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=_jsonable)
    write_markdown(primaries, k30, mm, checks, args.out_md)
    _log("[W2-B probe] wrote {} and {}".format(args.out_md, args.out_json))


if __name__ == "__main__":
    main()
