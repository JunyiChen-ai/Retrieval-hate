#!/usr/bin/env python3
"""S2S Stage-P G0-cond probe — CPU, ZERO training, ZERO test touch.

Pre-registration : research-wiki/experiments/exp-s2s-r3.md  (r1 APPROVED-WITH-AMENDMENTS)
Executable spec  : refine-logs/S2S_PROBE_DESIGN.md          (r1 §5-§7)
Review           : refine-logs/S2S_PREREG_REVIEW.md         (A1-A5 + N1-N7)

Builds the memory from TRAIN + VAL only (LOO), retrieves top-20 by each arm's pairwise
score, and reuses the pipeline's REAL vote (src/utils/metrics.py compute_metrics_retrieval,
use_sim=True, arithmetic, topk=20). Reports raw per-arm AUC/acc/mF1 and paired Δ(SET-POOLED)
with NO pass/fail interpretation in the human results file; a machine JSON additionally
carries a `mechanical_gate_check` block (pre-registered threshold arithmetic — explicitly
NOT the binding verdict; §12 resolution 2).

ARMS (per dataset, memory = train u val, LOO)
  POOLED           cos(mean_t g_t^Q, mean_t g_t^M)          visual-isolated null (primary pair)
  SET (primary)    MeanMaxSim(Q,M) = mean_q max_m cos(g^Q_q, g^M_m)   late interaction
  SET-Chamfer      0.5[MMS(Q->M)+MMS(M->Q)]                  single sensitivity
  PIPELINE-ANCHOR  pooled-cosine over the BANKED img_feats   internal reference
  WITH-TEXT        POOLED/SET visual + fixed cos(text,text)  text-channel sensitivity
  RANK-ONLY (A2)   POOLED/SET retrieval, sim neutralised to 1.0   sim-scale de-confound
  ORACLE (A4)      per-query oracle-frame MaxSim (video-level gold ONLY)   kill-switch ceiling
  FANO (N2)        +/-1 gold-label-agreement key             machine-validity calibration

GATES (mechanical arithmetic only; independent verdict reviewer rules — house rule):
  grid/G-decomp/G-recon : from Stage-E gate logs (surfaced here for the reviewer).
  Fano >= 0.99 both datasets ; oracle Delta < +0.04 all datasets -> DEAD ;
  raw HateMM Delta acc >= +0.05 AND Delta mF1 >= +0.05 corroborated by RANK-ONLY (A2) ;
  observed Delta > 95th pct permutation null (N1 seeds 0..99, same perm both arms) ;
  bootstrap 5th pct of paired Delta > 0 (D3) ; near-dup-excluded advantage survives (A3).

FAIL CLOSED (N4): never constructs/opens any test_seen file; asserts memory size == the
exact train u val count (851 HateMM / 629 MHC-EN). No silent excepts.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = "/data/jehc223/RGCL"
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from utils.metrics import compute_metrics_retrieval  # the REAL vote (do NOT reimplement)

# ------- pre-registered constants -------
TOPK = 20
EXPECTED_MEM = {"HateMM": 851, "MHC": 629}     # train u val (N4 fail-closed size guard)
BANK_TAG = "Qwen2.5-VL-7B-Instruct_HF"
NEAR_DUP_THRESH = 0.995                          # A3 binding flag (pooled OR MMS)
NEAR_DUP_REPORT = [0.98, 0.99, 0.995]            # A3 distribution
ORACLE_BAR = 0.04                                # §6.4 kill-switch
RAW_BAR = 0.05                                   # §6.5
FANO_BAR = 0.99                                  # §6.3
NULL_SEEDS = list(range(100))                    # N1 seed set 0..99
TIEBREAK_EPS = 1e-9                              # deterministic tie-break: smaller memory idx wins
NEG_INF = -1e30


def _log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Loading (N4: train + dev_seen ONLY; test_seen is NEVER constructed).
# ---------------------------------------------------------------------------
def _frameset_path(dataset, frameset_dir, outname):
    assert outname in ("train", "dev_seen"), \
        "N4 GUARD: probe may only load train/dev_seen, never '{}'".format(outname)
    return os.path.join(REPO, "data/CLIP_Embedding", dataset, frameset_dir,
                        "{}_frameset.pt".format(outname))


def _bank_path(dataset, outname):
    assert outname in ("train", "dev_seen"), \
        "N4 GUARD: probe may only load train/dev_seen, never '{}'".format(outname)
    return os.path.join(REPO, "data/CLIP_Embedding", dataset,
                        "{}_{}.pt".format(outname, BANK_TAG))


def load_memory(dataset, frameset_dir):
    """Assemble the LOO memory from train u val ONLY. HARD-fails on any test touch or a
    size that is not the exact expected train u val count."""
    ids, g, labels, guard = [], [], [], []
    img, txt = [], []
    for outname in ("train", "dev_seen"):          # NEVER test_seen (N4)
        fp = _frameset_path(dataset, frameset_dir, outname)
        if not os.path.exists(fp):
            raise RuntimeError("missing frame-set cache: {}".format(fp))
        fs = torch.load(fp, map_location="cpu", weights_only=False)
        fs_ids = [str(x) for x in fs["ids"][0]]
        ids.extend(fs_ids)
        g.append(fs["g"].float())                  # [n, T, D]
        labels.append(fs["labels"].long())
        guard.append(fs["zero_guard"].bool())
        # banked img/text feats for the same ids, in frame-set order (arms 4/5).
        bk = torch.load(_bank_path(dataset, outname), map_location="cpu", weights_only=False)
        bk_index = {str(i): k for k, i in enumerate(bk["ids"][0])}
        order = [bk_index[i] for i in fs_ids]
        img.append(bk["img_feats"][order].float())
        txt.append(bk["text_feats"][order].float())

    g = torch.cat(g, 0)                             # [N, T, D]
    labels = torch.cat(labels, 0).numpy()
    guard = torch.cat(guard, 0).numpy()
    img = torch.cat(img, 0)
    txt = torch.cat(txt, 0)
    N = g.shape[0]
    exp = EXPECTED_MEM[dataset]
    if N != exp:
        raise RuntimeError("N4 SIZE GUARD: memory N={} != expected train u val {} for {} "
                           "(a stray test row?)".format(N, exp, dataset))
    _log("[{}] memory N={} (train u val), T={}, D={}, zero-guard rows={}".format(
        dataset, N, g.shape[1], g.shape[2], int(guard.sum())))
    return {"ids": ids, "g": g, "labels": labels, "guard": guard, "img": img, "txt": txt,
            "N": N, "T": int(g.shape[1])}


# ---------------------------------------------------------------------------
# Score matrices (float32; deterministic index tie-break baked in).
# ---------------------------------------------------------------------------
def _l2norm(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _tiebreak(S):
    """Subtract a tiny idx-increasing term over columns so exact-tie neighbours resolve to
    the smaller memory index (deterministic). Negligible vs real cosine gaps."""
    N = S.shape[1]
    return S - (np.arange(N, dtype=np.float64)[None, :] * TIEBREAK_EPS)


def build_matrices(mem):
    """Return the frame-frame cross-sim tensor C [N,T,N,T] and the arm score matrices."""
    g = mem["g"]                                    # [N, T, D]
    N, T, D = g.shape
    ghat = _l2norm(g, dim=-1)                        # zero-guard rows -> ~0 vectors
    G = ghat.reshape(N * T, D)
    Sff = (G @ G.t()).reshape(N, T, N, T).numpy().astype(np.float32)   # cos(frame,frame)
    # MeanMaxSim(i,j) = mean_q max_m Sff[i,q,j,m]
    mms = Sff.max(axis=3).mean(axis=1).astype(np.float64)              # [N, N]
    # POOLED: cosine of the (unnormalized) pooled frame means.
    pooled = _l2norm(g.mean(dim=1), dim=-1)                            # [N, D]
    spool = (pooled @ pooled.t()).numpy().astype(np.float64)          # [N, N]
    # PIPELINE-ANCHOR: banked img_feats cosine (already L2-normed).
    imn = _l2norm(mem["img"], dim=-1)
    spipe = (imn @ imn.t()).numpy().astype(np.float64)
    # shared text channel (WITH-TEXT).
    txn = _l2norm(mem["txt"], dim=-1)
    stext = (txn @ txn.t()).numpy().astype(np.float64)
    # single-frame max cosine per pair (near-dup transparency).
    maxframe = Sff.max(axis=(1, 3)).astype(np.float64)                # [N, N]
    return {"C": Sff, "mms": mms, "spool": spool, "spipe": spipe, "stext": stext,
            "maxframe": maxframe, "N": N, "T": T}


# ---------------------------------------------------------------------------
# Retrieval + REAL vote.
# ---------------------------------------------------------------------------
def run_vote(S, labels, k=TOPK, rank_only=False, exclude=None):
    """Top-k LOO retrieval by score matrix S, then the pipeline's real vote.
    S: [N,N] float64 (higher = closer). Diagonal self is excluded here.
    exclude: optional [N,N] bool; True entries are dropped from retrieval (A3).
    Returns (acc, macro_f1, votes[np.ndarray]).
    """
    N = S.shape[0]
    St = _tiebreak(S).copy()
    np.fill_diagonal(St, NEG_INF)                    # LOO self-exclusion
    if exclude is not None:
        St[exclude] = NEG_INF
    logging_dict = {}
    for i in range(N):
        row = St[i]
        topk_idx = np.argpartition(-row, k)[:k]
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]     # exact order (desc, tie by idx)
        # NB-a guard (S2S_CODE_REVIEW.md): never let an excluded/self NEG_INF entry enter the
        # vote as a neighbour (would multiply a label by ~-1e30). Drop them; the arithmetic
        # vote uses weight[:length] so a shorter list is handled correctly.
        topk_idx = topk_idx[row[topk_idx] > (NEG_INF / 2)]
        if topk_idx.size == 0:
            raise RuntimeError("degenerate retrieval: no finite neighbours for query {}".format(i))
        sims = (np.ones(topk_idx.size, dtype=np.float64) if rank_only
                else row[topk_idx].astype(np.float64))
        logging_dict[i] = {
            "retrieved_label": [int(labels[j]) for j in topk_idx],
            "retrieved_scores": list(sims),          # np.float64 scalars -> .item() in metrics
        }
    acc, roc, pre, rec, f1, votes, _lab, macro = compute_metrics_retrieval(
        logging_dict, torch.tensor(labels), majority_voting="arithmetic",
        topk=k, use_sim=True)
    return float(acc), float(macro["macro_f1"]), float(roc), np.asarray(votes, dtype=np.float64)


def _preds_from_votes(votes):
    return (1.0 / (1.0 + np.exp(-votes)) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Oracle ceiling (A4) — per-query frame selection, video-level gold ONLY.
# ---------------------------------------------------------------------------
def _single_query_vote_margins(C, labels, k=TOPK):
    """For every query i and candidate frame t, the pre-sigmoid vote margin v_t(i) using the
    single query-frame-to-memory-set score s_t(i,j)=max_m C[i,t,j,m]. This inlines the EXACT
    metrics.py:262-284 vote formula (the real metrics fn cannot return per-single-query
    margins without full aggregation; the FINAL oracle number below uses the real vote)."""
    N, T = C.shape[0], C.shape[1]
    weight = np.arange(1, k + 1, dtype=np.float64)[::-1]
    wsum = weight.sum()
    lab_signed = (labels.astype(np.float64) * 2.0 - 1.0)
    V = np.zeros((N, T), dtype=np.float64)
    for t in range(T):
        St = C[:, t, :, :].max(axis=2).astype(np.float64)     # s_t(i,j) [N,N]
        St = _tiebreak(St)
        np.fill_diagonal(St, NEG_INF)
        for i in range(N):
            row = St[i]
            idx = np.argpartition(-row, k)[:k]
            idx = idx[np.argsort(-row[idx])]
            labelmap = lab_signed[idx] * row[idx]
            V[i, t] = float((labelmap * weight).sum() / wsum)
    return V


def oracle_ceiling(C, labels, k=TOPK):
    """t*(i) = argmax_t (2 y_i - 1) v_t(i), tie-break smallest t; memory keeps FULL sets.
    Returns (acc, macro_f1, votes, tstar)."""
    N, T = C.shape[0], C.shape[1]
    V = _single_query_vote_margins(C, labels, k)
    signed = (labels.astype(np.float64) * 2.0 - 1.0)[:, None] * V   # [N, T]
    tstar = np.argmax(signed, axis=1)                               # smallest index on ties
    Sorc = np.empty((N, N), dtype=np.float64)
    for i in range(N):
        Sorc[i] = C[i, tstar[i], :, :].max(axis=1)                 # s_{t*(i)}(i, .)
    acc, mf1, roc, votes = run_vote(Sorc, labels, k)
    return acc, mf1, votes, tstar.tolist()


# ---------------------------------------------------------------------------
# Fano (N2) — +/-1 gold-label-agreement machine-validity arm.
# ---------------------------------------------------------------------------
def fano(labels, k=TOPK):
    N = labels.shape[0]
    same = (labels[:, None] == labels[None, :]).astype(np.float64)
    S = same * 2.0 - 1.0                              # +1 same-label, -1 diff-label
    acc, mf1, roc, votes = run_vote(S, labels, k)
    return acc


# ---------------------------------------------------------------------------
# Near-duplicate audit (A3).
# ---------------------------------------------------------------------------
def near_dup_audit(M, guard):
    """M has 'spool','mms','maxframe'. Flag distinct pairs with pooled_cos>=0.995 OR
    MMS>=0.995. Return the flag mask [N,N] and the distribution table."""
    N = M["N"]
    tri = np.triu(np.ones((N, N), dtype=bool), k=1)   # distinct unordered pairs
    valid = tri & ~(guard[:, None] | guard[None, :])  # ignore zero-guard rows
    dist = {}
    for th in NEAR_DUP_REPORT:
        dist["pooled>=%.3f" % th] = int(((M["spool"] >= th) & valid).sum())
        dist["mms>=%.3f" % th] = int(((M["mms"] >= th) & valid).sum())
        dist["maxframe>=%.3f" % th] = int(((M["maxframe"] >= th) & valid).sum())
    flag_u = ((M["spool"] >= NEAR_DUP_THRESH) | (M["mms"] >= NEAR_DUP_THRESH)) & valid
    flag = flag_u | flag_u.T                           # symmetric neighbour-drop mask
    return flag, dist, int(flag_u.sum())


# ---------------------------------------------------------------------------
# Permutation null (N1) — same permutation applied to both arms.
# ---------------------------------------------------------------------------
def permutation_null(M, labels, k=TOPK):
    """N1 permutation null for BOTH the sim-weighted primary AND the rank-only (A2)
    co-diagnostic. Same permutation applied to both arms within a seed (paired Δ preserved).
    B3 fix (S2S_CODE_REVIEW.md): the rank-only arm gets its OWN null so its corroboration
    can be judged on significance, not sign alone."""
    mms, spool = M["mms"], M["spool"]
    obs_set_acc, obs_set_f1, _, _ = run_vote(mms, labels, k)
    obs_pool_acc, obs_pool_f1, _, _ = run_vote(spool, labels, k)
    obs_set_acc_r, obs_set_f1_r, _, _ = run_vote(mms, labels, k, rank_only=True)
    obs_pool_acc_r, obs_pool_f1_r, _, _ = run_vote(spool, labels, k, rank_only=True)
    obs = {"dacc": obs_set_acc - obs_pool_acc, "df1": obs_set_f1 - obs_pool_f1,
           "dacc_rank": obs_set_acc_r - obs_pool_acc_r, "df1_rank": obs_set_f1_r - obs_pool_f1_r}
    nd = {"dacc": [], "df1": [], "dacc_rank": [], "df1_rank": []}
    for s in NULL_SEEDS:
        perm = np.random.default_rng(s).permutation(M["N"])
        ix = np.ix_(perm, perm)
        mms_s, spool_s = mms[ix], spool[ix]
        sa, sf, _, _ = run_vote(mms_s, labels, k)                      # SET (sim-weighted)
        pa, pf, _, _ = run_vote(spool_s, labels, k)                    # POOLED, SAME perm
        sar, sfr, _, _ = run_vote(mms_s, labels, k, rank_only=True)    # SET rank-only, SAME perm
        par, pfr, _, _ = run_vote(spool_s, labels, k, rank_only=True)  # POOLED rank-only, SAME perm
        nd["dacc"].append(sa - pa); nd["df1"].append(sf - pf)
        nd["dacc_rank"].append(sar - par); nd["df1_rank"].append(sfr - pfr)
    p95 = {kk: float(np.percentile(np.array(vv), 95)) for kk, vv in nd.items()}
    out = {"n_seeds": len(NULL_SEEDS)}
    for kk in ("dacc", "df1", "dacc_rank", "df1_rank"):
        out["obs_" + kk] = obs[kk]
        out["null_" + kk + "_p95"] = p95[kk]
        out["obs_" + kk + "_gt_p95"] = bool(obs[kk] > p95[kk])
    return out


def per_frame_null(mem, labels, n_seeds, k=TOPK):
    """Optional secondary null (design §6.6): shuffle individual frame vectors across all
    videos, destroying set structure, recompute C -> Delta. Separates 'alignment' from a
    generic 'richer-key' effect. Recomputes the matmul per seed (the heavy path)."""
    g = mem["g"]
    N, T, D = g.shape
    ghat = _l2norm(g, dim=-1).reshape(N * T, D)
    pooled = _l2norm(g.mean(dim=1), dim=-1)
    spool = (pooled @ pooled.t()).numpy().astype(np.float64)   # POOLED unchanged by frame shuffle
    obs_pool_acc, obs_pool_f1, _, _ = run_vote(spool, labels, k)
    dacc, df1 = [], []
    for s in range(n_seeds):
        perm = np.random.default_rng(10_000 + s).permutation(N * T)
        Gs = ghat[perm].reshape(N, T, D)
        Sff = (Gs.reshape(N * T, D) @ Gs.reshape(N * T, D).t()).reshape(N, T, N, T)
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
def bootstrap_delta(votes_set, votes_pool, labels, n_boot, seed=20260714):
    from sklearn.metrics import f1_score
    ps, pp = _preds_from_votes(votes_set), _preds_from_votes(votes_pool)
    y = labels.astype(int)
    N = len(y)
    rng = np.random.default_rng(seed)
    dacc, df1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        yy = y[idx]
        dacc[b] = (ps[idx] == yy).mean() - (pp[idx] == yy).mean()
        df1[b] = (f1_score(yy, ps[idx], average="macro", zero_division=0)
                  - f1_score(yy, pp[idx], average="macro", zero_division=0))
    return {"n_boot": n_boot,
            "dacc_p5": float(np.percentile(dacc, 5)), "dacc_p50": float(np.percentile(dacc, 50)),
            "dacc_p95": float(np.percentile(dacc, 95)),
            "df1_p5": float(np.percentile(df1, 5)), "df1_p50": float(np.percentile(df1, 50)),
            "df1_p95": float(np.percentile(df1, 95)),
            "dacc_p5_gt0": bool(np.percentile(dacc, 5) > 0),
            "df1_p5_gt0": bool(np.percentile(df1, 5) > 0)}


# ---------------------------------------------------------------------------
# Probe-side synthetic set-matching positive control (A1 spirit, CPU self-test).
# ---------------------------------------------------------------------------
def synthetic_set_control():
    """Plant a shared 'hateful segment' frame in two otherwise-different videos; MeanMaxSim
    must exceed POOLED for the planted pair (the set metric detects segment-sharing that
    pooling dilutes). HALT on failure -> a bug in the set-metric implementation."""
    rng = np.random.default_rng(7)
    D, T = 3584, 4
    shared = rng.standard_normal(D)
    A = rng.standard_normal((T, D)); A[0] = shared
    B = rng.standard_normal((T, D)); B[0] = shared    # A,B share frame 0, differ elsewhere
    Ah = torch.tensor(A) / torch.tensor(A).norm(dim=-1, keepdim=True)
    Bh = torch.tensor(B) / torch.tensor(B).norm(dim=-1, keepdim=True)
    mms = float((Ah @ Bh.t()).max(dim=1).values.mean().item())
    pa = A.mean(0); pb = B.mean(0)
    pooled = float(np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb)))
    if not (mms > pooled):
        raise RuntimeError("[synthetic set control] MMS {:.4f} !> POOLED {:.4f} — set-metric "
                           "bug".format(mms, pooled))
    _log("[probe self-test] synthetic shared-segment: MMS {:.4f} > POOLED {:.4f} OK".format(
        mms, pooled))
    return {"mms": mms, "pooled": pooled, "passed": True}


# ---------------------------------------------------------------------------
# Per-dataset probe.
# ---------------------------------------------------------------------------
def probe_dataset(dataset, frameset_dir, n_boot, n_perframe):
    mem = load_memory(dataset, frameset_dir)
    labels = mem["labels"]
    M = build_matrices(mem)

    arms = {}

    def add(name, S, rank_only=False, exclude=None):
        acc, mf1, roc, votes = run_vote(S, labels, TOPK, rank_only=rank_only, exclude=exclude)
        arms[name] = {"acc": acc, "macro_f1": mf1, "roc": roc, "votes": votes}
        return arms[name]

    a_pool = add("POOLED", M["spool"])
    a_set = add("SET", M["mms"])
    add("SET_CHAMFER", 0.5 * (M["mms"] + M["mms"].T))
    add("PIPELINE_ANCHOR", M["spipe"])
    add("WITH_TEXT_POOLED", M["spool"] + M["stext"])
    add("WITH_TEXT_SET", M["mms"] + M["stext"])
    # A2 rank-only co-diagnostic (retrieve by arm score, neutralise sim weight to 1.0).
    a_pool_r = add("POOLED_RANKONLY", M["spool"], rank_only=True)
    a_set_r = add("SET_RANKONLY", M["mms"], rank_only=True)

    # primary paired deltas.
    d_acc = a_set["acc"] - a_pool["acc"]
    d_f1 = a_set["macro_f1"] - a_pool["macro_f1"]
    d_acc_rank = a_set_r["acc"] - a_pool_r["acc"]
    d_f1_rank = a_set_r["macro_f1"] - a_pool_r["macro_f1"]

    # Fano (N2), oracle (A4), near-dup (A3), null (N1), bootstrap (D3), per-frame null (opt).
    fano_acc = fano(labels)
    orc_acc, orc_f1, orc_votes, tstar = oracle_ceiling(M["C"], labels)
    d_orc_acc = orc_acc - a_pool["acc"]
    d_orc_f1 = orc_f1 - a_pool["macro_f1"]

    flag, nd_dist, nd_pairs = near_dup_audit(M, mem["guard"])
    a_pool_x = add("POOLED_NEARDUP_EXCL", M["spool"], exclude=flag)
    a_set_x = add("SET_NEARDUP_EXCL", M["mms"], exclude=flag)
    d_acc_x = a_set_x["acc"] - a_pool_x["acc"]
    d_f1_x = a_set_x["macro_f1"] - a_pool_x["macro_f1"]

    null = permutation_null(M, labels)
    boot = bootstrap_delta(a_set["votes"], a_pool["votes"], labels, n_boot)
    # B3 fix: the rank-only arm gets its OWN bootstrap (paired SET_RANKONLY - POOLED_RANKONLY).
    boot_rank = bootstrap_delta(a_set_r["votes"], a_pool_r["votes"], labels, n_boot)
    pfn = per_frame_null(mem, labels, n_perframe) if n_perframe > 0 else None

    # B3 fix: pre-registered A2 credit rule (exp-s2s-r3.md:215-223) — the primary Δ is credited
    # only if the rank-only Δ matches its sign AND is itself significant (observed rank-only Δ >
    # 95th-pct rank-only permutation null AND rank-only bootstrap 5th-pct > 0). Significance is
    # judged on acc (mirroring the primary raw-bar gate); sign is required on BOTH acc and F1.
    rank_sign_ok = bool(np.sign(d_acc) == np.sign(d_acc_rank)
                        and np.sign(d_f1) == np.sign(d_f1_rank))
    rank_null_ok = bool(null["obs_dacc_rank_gt_p95"])
    rank_boot_ok = bool(boot_rank["dacc_p5_gt0"])
    rank_corroborates = bool(rank_sign_ok and rank_null_ok and rank_boot_ok)

    # strip the bulky vote arrays from the arm summaries for JSON.
    arm_summary = {k: {kk: v[kk] for kk in ("acc", "macro_f1", "roc")} for k, v in arms.items()}

    return {
        "dataset": dataset, "N": mem["N"], "T": mem["T"],
        "zero_guard": int(mem["guard"].sum()),
        "arms": arm_summary,
        "primary": {"d_acc": d_acc, "d_f1": d_f1,
                    "rankonly_d_acc": d_acc_rank, "rankonly_d_f1": d_f1_rank,
                    "rankonly_sign_ok": rank_sign_ok,
                    "rankonly_null_ok": rank_null_ok,
                    "rankonly_boot_ok": rank_boot_ok,
                    "rankonly_corroborates": rank_corroborates,
                    "rankonly_null_p95_acc": null["null_dacc_rank_p95"],
                    "rankonly_obs_dacc": null["obs_dacc_rank"],
                    "rankonly_boot_dacc_p5": boot_rank["dacc_p5"]},
        "bootstrap_rankonly": boot_rank,
        "fano_acc": fano_acc,
        "oracle": {"acc": orc_acc, "macro_f1": orc_f1, "d_acc": d_orc_acc, "d_f1": d_orc_f1},
        "near_dup": {"threshold": NEAR_DUP_THRESH, "flagged_pairs": nd_pairs,
                     "distribution": nd_dist,
                     "excluded_d_acc": d_acc_x, "excluded_d_f1": d_f1_x},
        "permutation_null": null,
        "bootstrap": boot,
        "per_frame_null": pfn,
        "stage_e_gatelog": read_stage_e_gatelog(dataset, frameset_dir),
    }


def read_stage_e_gatelog(dataset, frameset_dir):
    """Surface the Stage-E grid/G-decomp/G-recon numbers for the reviewer (train+dev_seen)."""
    out = {}
    for outname in ("train", "dev_seen"):
        p = os.path.join(REPO, "data/CLIP_Embedding", dataset, frameset_dir,
                         "{}_gatelog.json".format(outname))
        if os.path.exists(p):
            out[outname] = json.load(open(p))
    return out


# ---------------------------------------------------------------------------
# Mechanical gate check (§12 res 2: NOT the binding verdict).
# ---------------------------------------------------------------------------
def mechanical_gate_check(results):
    """Pre-registered threshold ARITHMETIC only. Prints PASS/KILL to stdout. Explicitly NOT
    the binding verdict — the independent verdict reviewer renders the ruling."""
    by_ds = {r["dataset"]: r for r in results}
    checks = []

    def rec(name, value, thr, op, note=""):
        if op == ">=":
            ok = value >= thr
        elif op == "<":
            ok = value < thr
        elif op == ">":
            ok = value > thr
        else:
            raise ValueError(op)
        checks.append({"gate": name, "value": value, "threshold": thr, "op": op,
                       "result": "ABOVE" if ok else "BELOW", "note": note})

    for ds, r in by_ds.items():
        rec("Fano[%s]" % ds, r["fano_acc"], FANO_BAR, ">=", "vote machine valid if ABOVE")
    oracle_all_below = all(r["oracle"]["d_acc"] < ORACLE_BAR for r in results)
    for ds, r in by_ds.items():
        rec("OracleDacc[%s]" % ds, r["oracle"]["d_acc"], ORACLE_BAR, ">=",
            "headroom if ABOVE (kill-switch: DEAD only if ALL datasets BELOW)")
    checks.append({"gate": "OracleKillSwitch(all-datasets)", "value": oracle_all_below,
                   "threshold": "all < %.2f" % ORACLE_BAR,
                   "result": "KILL(DEAD)" if oracle_all_below else "SURVIVES",
                   "note": "DEAD -> zero head GPU"})
    if "HateMM" in by_ds:
        h = by_ds["HateMM"]
        rec("RawDacc[HateMM]", h["primary"]["d_acc"], RAW_BAR, ">=")
        rec("RawDmF1[HateMM]", h["primary"]["d_f1"], RAW_BAR, ">=")
        # B3: full A2 corroboration = sign match AND rank-only own null-p95 AND rank-only bootstrap-5th>0.
        hp = h["primary"]
        checks.append({"gate": "RankOnlyCorroborates[HateMM] (A2)",
                       "value": hp["rankonly_corroborates"], "threshold": True,
                       "result": "ABOVE" if hp["rankonly_corroborates"] else "BELOW",
                       "note": "sign_ok={} null_ok={} boot_ok={}; primary credited only if "
                               "corroborated".format(hp["rankonly_sign_ok"], hp["rankonly_null_ok"],
                                                     hp["rankonly_boot_ok"])})
        rec("RankOnlyObsDacc>null95[HateMM] (A2)", hp["rankonly_obs_dacc"],
            hp["rankonly_null_p95_acc"], ">")
        rec("RankOnlyBoot5th>0[HateMM] (A2)", hp["rankonly_boot_dacc_p5"], 0.0, ">")
        rec("ObsDacc>null95[HateMM]", h["permutation_null"]["obs_dacc"],
            h["permutation_null"]["null_dacc_p95"], ">")
        rec("Bootstrap5th>0[HateMM]", h["bootstrap"]["dacc_p5"], 0.0, ">")
        rec("NearDupExclSurvives[HateMM] (A3)", h["near_dup"]["excluded_d_acc"], 0.0, ">")
    return checks


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------
def write_markdown(results, checks, out_md):
    L = []
    L.append("# S2S G0-cond Probe — RAW RESULTS (no pass/fail interpretation)\n")
    L.append("_Executor writes raw numbers only; the independent verdict reviewer renders "
             "the binding ruling (house rule; prereg §6.6, review §5.6/N6). The mechanical "
             "gate arithmetic is in `s2s_probe_results.json` and is NOT a verdict._\n")
    for r in results:
        ds = r["dataset"]
        L.append("\n## {}  (memory N={}, T={}, zero-guard rows={})\n".format(
            ds, r["N"], r["T"], r["zero_guard"]))
        L.append("| arm | acc | macro_f1 | roc |")
        L.append("|---|---|---|---|")
        for name, a in r["arms"].items():
            L.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                name, a["acc"], a["macro_f1"], a["roc"]))
        p = r["primary"]
        L.append("\n**Primary paired Δ(SET−POOLED):** acc {:+.4f}, macro_f1 {:+.4f}. "
                 "**Rank-only (A2):** acc {:+.4f}, macro_f1 {:+.4f}; obs Δacc {:+.4f} vs "
                 "rank-only null-95th {:+.4f}, rank-only bootstrap-5th {:+.4f} "
                 "(corroborates={}: sign={} null={} boot={}).".format(
                     p["d_acc"], p["d_f1"], p["rankonly_d_acc"], p["rankonly_d_f1"],
                     p["rankonly_obs_dacc"], p["rankonly_null_p95_acc"], p["rankonly_boot_dacc_p5"],
                     p["rankonly_corroborates"], p["rankonly_sign_ok"], p["rankonly_null_ok"],
                     p["rankonly_boot_ok"]))
        L.append("\n**Fano (±1 gold-label key) acc:** {:.4f}.".format(r["fano_acc"]))
        o = r["oracle"]
        L.append("**Oracle ceiling (A4):** acc {:.4f} (Δ vs POOLED acc {:+.4f}, "
                 "mF1 {:+.4f}).".format(o["acc"], o["d_acc"], o["d_f1"]))
        nd = r["near_dup"]
        L.append("**Near-dup (A3):** flagged pairs (≥{:.3f} pooled-OR-MMS) = {}; "
                 "excluded-retrieval Δ(SET−POOLED) acc {:+.4f}, mF1 {:+.4f}. Distribution: "
                 "{}.".format(nd["threshold"], nd["flagged_pairs"], nd["excluded_d_acc"],
                              nd["excluded_d_f1"], json.dumps(nd["distribution"])))
        nl = r["permutation_null"]
        L.append("**Permutation null (N1, {} seeds):** obs Δacc {:+.4f} vs null-95th {:+.4f}; "
                 "obs ΔmF1 {:+.4f} vs null-95th {:+.4f}.".format(
                     nl["n_seeds"], nl["obs_dacc"], nl["null_dacc_p95"], nl["obs_df1"],
                     nl["null_df1_p95"]))
        if r["per_frame_null"]:
            pf = r["per_frame_null"]
            L.append("**Per-frame null (optional, {} seeds):** Δacc-95th {:+.4f}, "
                     "ΔmF1-95th {:+.4f}.".format(pf["n_seeds"], pf["dacc_p95"], pf["df1_p95"]))
        b = r["bootstrap"]
        L.append("**Bootstrap ({} resamples):** Δacc [5/50/95]=[{:+.4f}/{:+.4f}/{:+.4f}]; "
                 "ΔmF1 [5/50/95]=[{:+.4f}/{:+.4f}/{:+.4f}].".format(
                     b["n_boot"], b["dacc_p5"], b["dacc_p50"], b["dacc_p95"],
                     b["df1_p5"], b["df1_p50"], b["df1_p95"]))
        ge = r["stage_e_gatelog"]
        if ge:
            L.append("**Stage-E gates:** " + "; ".join(
                "{}: decomp_max={} grecon_cos_min={} grecon_maxabs_max={}".format(
                    sp, g.get("decomp_res_max"), g.get("grecon_cos_min"),
                    g.get("grecon_maxabs_max")) for sp, g in ge.items()) + ".")
    L.append("\n## Mechanical gate arithmetic (NOT a verdict — see JSON)\n")
    L.append("| gate | value | threshold | op | result |")
    L.append("|---|---|---|---|---|")
    for c in checks:
        L.append("| {} | {} | {} | {} | {} |".format(
            c["gate"], c["value"], c["threshold"], c.get("op", ""), c["result"]))
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="S2S Stage-P CPU probe (zero training, zero test touch).")
    ap.add_argument("--datasets", type=str, default="HateMM,MHC")
    ap.add_argument("--frameset_dir", type=str, default="frameset_qwen7b_8f")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perframe_null", type=int, default=100,
                    help="optional secondary per-frame-shuffle null seeds (0 to skip).")
    ap.add_argument("--out_md", type=str, default=os.path.join(REPO, "refine-logs/S2S_PROBE_RESULTS.md"))
    ap.add_argument("--out_json", type=str, default=os.path.join(REPO, "refine-logs/s2s_probe_results.json"))
    args = ap.parse_args()

    # hard CPU-only; deterministic.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.manual_seed(20260714)
    np.random.seed(20260714)

    _log("=" * 78)
    _log("[S2S probe] datasets={} frameset_dir={} topk={} null_seeds={} n_boot={}".format(
        args.datasets, args.frameset_dir, TOPK, len(NULL_SEEDS), args.n_boot))
    _log("[S2S probe] N4 fail-closed: train+dev_seen ONLY; expected memory {}".format(EXPECTED_MEM))
    _log("=" * 78)

    synthetic_set_control()   # probe-side self-test (HALT on set-metric bug)

    t0 = time.time()
    results = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        _log("[S2S probe] --- {} ---".format(ds))
        results.append(probe_dataset(ds, args.frameset_dir, args.n_boot, args.n_perframe_null))
    _log("[S2S probe] probing done in {:.1f}s".format(time.time() - t0))

    checks = mechanical_gate_check(results)
    _log("\n[S2S probe] MECHANICAL gate arithmetic (pre-registered thresholds; NOT the binding "
         "verdict — the independent verdict reviewer rules):")
    for c in checks:
        _log("  GATE {:<34} value={} threshold={} {} -> {}".format(
            c["gate"], c["value"], c["threshold"], c.get("op", ""), c["result"]))

    out = {
        "meta": {"topk": TOPK, "frameset_dir": args.frameset_dir, "null_seeds": len(NULL_SEEDS),
                 "n_boot": args.n_boot, "expected_mem": EXPECTED_MEM,
                 "raw_bar": RAW_BAR, "oracle_bar": ORACLE_BAR, "fano_bar": FANO_BAR,
                 "near_dup_threshold": NEAR_DUP_THRESH,
                 "note": "Executor writes RAW numbers. The mechanical_gate_check block is "
                         "pre-registered threshold arithmetic ONLY and is NOT the binding "
                         "verdict; an independent verdict reviewer renders the ruling."},
        "results": results,
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
    write_markdown(results, checks, args.out_md)
    _log("[S2S probe] wrote {} and {}".format(args.out_md, args.out_json))


if __name__ == "__main__":
    main()
